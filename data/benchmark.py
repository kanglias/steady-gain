"""沪深 300 日线，给超额收益做基准。

这是指数行情，不是成分股名单，也不会变成交易指令。
缓存和个股日线分开存放。失败时调用方应退回「只和自己的买入持有比」。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data.bars import CacheMissError, FetchError
from data.cache import default_cache_dir
from data.normalize import parse_date

CSI300_CODE = "000300"
CSI300_NAME = "沪深300"
MISSING_CSI300_MESSAGE = "缺少沪深300缓存，暂无相对基准的超额。策略净值仍可查看。"


def get_csi300(
    start: str | date,
    end: str | date,
    *,
    refresh: bool = False,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    fetcher=None,
) -> pd.DataFrame:
    """读取沪深 300 日线。列：date, open, close。"""
    start_d = parse_date(start)
    end_d = parse_date(end)
    if start_d > end_d:
        raise ValueError(f"start 晚于 end: {start_d} > {end_d}")
    directory = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    cached, meta = _read(directory)
    covered = (
        not refresh
        and cached is not None
        and meta is not None
        and parse_date(meta["start"]) <= start_d
        and parse_date(meta["end"]) >= end_d
    )
    if covered and cached is not None:
        frame = _slice(cached, start_d, end_d)
        frame.attrs["source"] = meta.get("source", "cache")
        frame.attrs["cache_hit"] = True
        frame.attrs["name"] = CSI300_NAME
        return frame
    if offline:
        raise CacheMissError("沪深300 不在本地缓存里，本次不允许访问网络")
    fetch = fetcher or _fetch_csi300
    try:
        remote = _normalize_index(fetch(start_d, end_d))
    except CacheMissError:
        raise
    except Exception as exc:
        raise FetchError(f"沪深300 拉取失败: {type(exc).__name__}: {exc}") from exc
    if remote.empty:
        raise FetchError("沪深300 没有返回行情")
    _write(directory, remote, start_d, end_d)
    frame = _slice(remote, start_d, end_d)
    frame.attrs["source"] = "akshare.index_zh_a_hist"
    frame.attrs["cache_hit"] = False
    frame.attrs["name"] = CSI300_NAME
    return frame


def read_csi300_offline(
    start: str | date,
    end: str | date,
    *,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """只读本地沪深300缓存。没有文件、区间不够或读坏时返回 None，不访问网络。"""
    try:
        return get_csi300(start, end, offline=True, refresh=False, cache_dir=cache_dir)
    except (CacheMissError, FetchError, ValueError, OSError):
        return None


def csi300_excess(
    strategy_return: float,
    benchmark: pd.DataFrame | None,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
) -> dict[str, object]:
    """超额 = 策略累计收益 − 沪深300区间首尾收益。没有可用基准时不报错。"""
    strategy = pd.to_numeric(pd.Series([strategy_return]), errors="coerce").iloc[0]
    strategy_value = float("nan") if pd.isna(strategy) else float(strategy)
    bench_return = _benchmark_return(benchmark, start, end)
    if bench_return != bench_return:
        return {
            "available": False,
            "benchmark_name": CSI300_NAME,
            "benchmark_return": float("nan"),
            "strategy_return": strategy_value,
            "excess": float("nan"),
            "message": MISSING_CSI300_MESSAGE,
        }
    excess = strategy_value - float(bench_return) if strategy_value == strategy_value else float("nan")
    return {
        "available": True,
        "benchmark_name": CSI300_NAME,
        "benchmark_return": float(bench_return),
        "strategy_return": strategy_value,
        "excess": excess,
        "message": f"超额 = 策略收益 − {CSI300_NAME}区间收益。",
    }


def _benchmark_return(benchmark: pd.DataFrame | None, start, end) -> float:
    if benchmark is None or not isinstance(benchmark, pd.DataFrame) or benchmark.empty:
        return float("nan")
    if "date" not in benchmark.columns or "close" not in benchmark.columns:
        return float("nan")
    frame = benchmark.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    if start is not None:
        frame = frame.loc[frame["date"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end).normalize()]
    if len(frame) < 2 or float(frame["close"].iloc[0]) <= 0:
        return float("nan")
    return float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)


def _fetch_csi300(start: date, end: date) -> pd.DataFrame:
    import akshare as ak

    return ak.index_zh_a_hist(
        symbol=CSI300_CODE,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )


def _normalize_index(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    rename = {"日期": "date", "开盘": "open", "收盘": "close", "date": "date", "open": "open", "close": "close"}
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    if "date" not in frame.columns or "close" not in frame.columns:
        raise FetchError("沪深300 行情缺少日期或收盘价")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "open" in frame.columns:
        frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    else:
        frame["open"] = frame["close"]
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    return frame.loc[:, ["date", "open", "close"]].reset_index(drop=True)


def _slice(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chosen = frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].reset_index(drop=True)
    if chosen.empty:
        raise FetchError("沪深300 在请求区间内没有交易日")
    return chosen


def _paths(directory: Path) -> tuple[Path, Path]:
    return directory / "000300_index.parquet", directory / "000300_index.meta.json"


def _read(directory: Path) -> tuple[pd.DataFrame | None, dict | None]:
    data_path, meta_path = _paths(directory)
    if not data_path.exists() or not meta_path.exists():
        return None, None
    try:
        frame = pd.read_parquet(data_path)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None
    return frame, meta


def _write(directory: Path, frame: pd.DataFrame, start: date, end: date) -> None:
    data_path, meta_path = _paths(directory)
    frame.to_parquet(data_path, index=False)
    meta = {
        "symbol": CSI300_CODE,
        "name": CSI300_NAME,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": int(len(frame)),
        "source": "akshare.index_zh_a_hist",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
