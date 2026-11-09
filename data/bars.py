"""按代码和日期区间读取 A 股日线。有缓存且区间已覆盖时不再访问网络。"""

from __future__ import annotations

import inspect
import time
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from data.cache import cache_covers, default_cache_dir, load_cache, save_cache
from data.normalize import (
    exchange_prefix,
    normalize_adjust,
    normalize_hist,
    normalize_symbol,
    parse_date,
)

FetchFn = Callable[[str, date, date, str], pd.DataFrame]


class DataError(Exception):
    """行情读取或拉取失败。"""


class FetchError(DataError):
    """数据源没有返回可用行情。"""


class CacheMissError(DataError):
    """请求区间超出本地缓存，且本次不允许访问网络。"""


def get_bars(
    symbol: str,
    start: str | date,
    end: str | date,
    *,
    adjust: str = "qfq",
    refresh: bool = False,
    offline: bool = False,
    cache_dir: str | Path | None = None,
    fetcher: FetchFn | None = None,
) -> pd.DataFrame:
    """返回一只 A 股的日线 OHLCV。

    列至少包括 `date, symbol, open, high, low, close, volume`。
    默认前复权（`adjust="qfq"`）。东财成交量单位是手。

    缓存能盖住 `[start, end]` 时直接切片返回。否则拉取「已有覆盖区间」和
    本次请求的并集，并整表写回缓存。前复权历史价会在除权后变化，需要对齐
    最新价时传 `refresh=True`。

    `offline=True` 时只读缓存，缺口直接报 `CacheMissError`。
    """
    code = normalize_symbol(symbol)
    start_d = parse_date(start)
    end_d = parse_date(end)
    if start_d > end_d:
        raise ValueError(f"start 晚于 end: {start_d} > {end_d}")
    adjust_code = normalize_adjust(adjust)

    directory = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or fetch_a_share_daily

    cached = load_cache(directory, code, adjust_code)
    frame = cached[0] if cached else None
    meta = cached[1] if cached else None
    covered = (
        not refresh
        and frame is not None
        and meta is not None
        and cache_covers(meta, code, adjust_code, start_d, end_d)
    )
    if covered and frame is not None and meta is not None:
        return _annotate(
            _slice_bars(frame, start_d, end_d),
            source=str(meta.get("source", "cache")),
            adjust=adjust_code,
            cache_hit=True,
        )

    fetch_start, fetch_end = start_d, end_d
    if meta is not None:
        try:
            fetch_start = min(start_d, parse_date(meta["start"]))
            fetch_end = max(end_d, parse_date(meta["end"]))
        except (KeyError, TypeError, ValueError):
            fetch_start, fetch_end = start_d, end_d

    if offline:
        raise CacheMissError(
            f"{code} 的缓存未覆盖 {start_d.isoformat()} ~ {end_d.isoformat()}。"
            "去掉 offline，或先联网拉取一次。"
        )

    try:
        fresh = fetch(code, fetch_start, fetch_end, adjust_code)
    except DataError:
        raise
    except Exception as exc:
        raise FetchError(f"拉取 {code} 日线失败: {exc}") from exc

    source = str(fresh.attrs.get("source") or getattr(fetch, "__name__", "fetcher"))
    fresh = normalize_hist(fresh, code)
    _require_ohlcv(fresh, code)
    save_cache(directory, code, adjust_code, fresh, fetch_start, fetch_end, source)
    return _annotate(
        _slice_bars(fresh, start_d, end_d),
        source=source,
        adjust=adjust_code,
        cache_hit=False,
    )


def fetch_a_share_daily(symbol: str, start: date, end: date, adjust: str = "qfq") -> pd.DataFrame:
    """从 akshare 拉日线。先东财 `stock_zh_a_hist`，失败再试腾讯 `stock_zh_a_hist_tx`。"""
    code = normalize_symbol(symbol)
    start_d = parse_date(start)
    end_d = parse_date(end)
    adjust_code = normalize_adjust(adjust)
    errors: list[Exception] = []

    try:
        frame = _fetch_eastmoney(code, start_d, end_d, adjust_code)
        frame.attrs["source"] = "akshare.stock_zh_a_hist"
        return frame
    except Exception as exc:
        errors.append(exc)

    try:
        frame = _fetch_tencent(code, start_d, end_d, adjust_code)
        frame.attrs["source"] = "akshare.stock_zh_a_hist_tx"
        return frame
    except Exception as exc:
        errors.append(exc)

    detail = "; ".join(f"{type(err).__name__}: {err}" for err in errors)
    raise FetchError(f"{code} 日线拉取失败（东财与腾讯均不可用）: {detail}") from errors[-1]


def _fetch_eastmoney(symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame:
    import akshare as ak

    kwargs = {
        "symbol": symbol,
        "period": "daily",
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "adjust": adjust,
    }
    if "timeout" in inspect.signature(ak.stock_zh_a_hist).parameters:
        kwargs["timeout"] = 20

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = ak.stock_zh_a_hist(**kwargs)
            frame = normalize_hist(raw, symbol)
            return _require_ohlcv(frame, symbol)
        except FetchError:
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(1.2 * (attempt + 1))
    raise FetchError(f"东财日线拉取失败: {symbol}") from last_error


def _fetch_tencent(symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_zh_a_hist_tx(
        symbol=f"{exchange_prefix(symbol)}{symbol}",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    # 腾讯接口的 turnover 与东财「换手率(%)」单位不一致，丢掉以免混用。
    if raw is not None and "turnover" in getattr(raw, "columns", []):
        raw = raw.drop(columns=["turnover"])
    frame = normalize_hist(raw, symbol)
    return _require_ohlcv(frame, symbol)


def _require_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = ("date", "open", "high", "low", "close", "volume")
    missing = [col for col in required if col not in frame.columns]
    if missing or frame.empty:
        raise FetchError(f"{symbol} 没有返回可用日线，缺少 {missing or '全部行'}")
    return frame


def _slice_bars(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (out["date"] >= start_ts) & (out["date"] <= end_ts)
    out = out.loc[mask].sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def _annotate(frame: pd.DataFrame, *, source: str, adjust: str, cache_hit: bool) -> pd.DataFrame:
    out = frame.copy()
    out.attrs["source"] = source
    out.attrs["adjust"] = adjust
    out.attrs["cache_hit"] = cache_hit
    return out


# --- 后续市场（阶段 1 不实现） ---------------------------------------------
# 港股、美股不进本阶段主路径。以后若要加，返回列与 get_bars 保持一致
# （date, symbol, open, high, low, close, volume），并且仍然只供研究，不下单。
#
# def get_hk_bars(symbol: str, start, end, *, adjust: str = "qfq", refresh: bool = False):
#     """TODO: 港股日线。symbol 预计为 5 位代码，如 00700。数据源未选。"""
#
# def get_us_bars(symbol: str, start, end, *, adjust: str = "qfq", refresh: bool = False):
#     """TODO: 美股日线。symbol 预计为 ticker，如 AAPL。数据源未选。"""
