"""A 股交易日历（上证口径）：新浪 `tool_trade_date_hist_sina`，缓存于 `.cache/calendar/`。

QC 默认只读本地缓存，不隐式联网；显式调用 `get_trade_calendar(refresh=True)` 或
`scripts/refresh_trade_calendar.py` 时才会访问网络。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from data.bars import CacheMissError

CALENDAR_ID = "cn_sse_trade_dates"
CALENDAR_EXCHANGE = "SSE"
CALENDAR_SOURCE = "akshare.tool_trade_date_hist_sina"


def default_calendar_dir() -> Path:
    """`ASHARE_CALENDAR_DIR` 或仓库根下 `.cache/calendar`。"""
    env = os.environ.get("ASHARE_CALENDAR_DIR")
    if env:
        return Path(env).expanduser()
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate / ".cache" / "calendar"
    return start / ".cache" / "calendar"


def calendar_paths(calendar_dir: Path) -> tuple[Path, Path, Path]:
    """返回 (parquet, csv, meta.json) 路径。"""
    base = calendar_dir / CALENDAR_ID
    return (
        base.with_suffix(".parquet"),
        Path(str(base) + ".csv"),  # cn_sse_trade_dates.csv
        base.with_suffix(".meta.json"),
    )


def _normalize_trade_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "trade_date" not in out.columns:
        raise ValueError("日历缺少 trade_date 列")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates(
        "trade_date", keep="last"
    )
    return out.reset_index(drop=True)


def _fetch_sina_trade_calendar() -> pd.DataFrame:
    import akshare as ak

    raw = ak.tool_trade_date_hist_sina()
    if raw is None or raw.empty:
        raise RuntimeError("tool_trade_date_hist_sina 返回空表")
    if "trade_date" not in raw.columns:
        raise RuntimeError(f"意外列: {raw.columns.tolist()}")
    return _normalize_trade_date_column(raw)


def save_trade_calendar(
    calendar_dir: Path,
    frame: pd.DataFrame,
    *,
    source: str = CALENDAR_SOURCE,
    fetched_at: datetime | None = None,
) -> dict:
    """写入 parquet（失败则 csv）+ meta.json；返回 meta 字典。"""
    calendar_dir = Path(calendar_dir)
    calendar_dir.mkdir(parents=True, exist_ok=True)
    payload = _normalize_trade_date_column(frame)
    parquet_path, csv_path, meta_path = calendar_paths(calendar_dir)
    try:
        payload.to_parquet(parquet_path, index=False)
        csv_path.unlink(missing_ok=True)
    except Exception:
        payload.to_csv(csv_path, index=False)
        parquet_path.unlink(missing_ok=True)

    tmin = payload["trade_date"].min()
    tmax = payload["trade_date"].max()
    start_d = tmin.date() if hasattr(tmin, "date") else pd.Timestamp(tmin).date()
    end_d = tmax.date() if hasattr(tmax, "date") else pd.Timestamp(tmax).date()
    meta = {
        "calendar_id": CALENDAR_ID,
        "exchange": CALENDAR_EXCHANGE,
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "rows": int(len(payload)),
        "source": source,
        "fetched_at": (fetched_at or datetime.now(timezone.utc)).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_trade_calendar_disk(calendar_dir: Path) -> tuple[pd.DataFrame, dict] | None:
    """仅读本地：parquet 优先，否则 csv。无文件或损坏返回 None。"""
    calendar_dir = Path(calendar_dir)
    parquet_path, csv_path, meta_path = calendar_paths(calendar_dir)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    frame: pd.DataFrame | None = None
    if parquet_path.exists():
        try:
            frame = pd.read_parquet(parquet_path)
        except Exception:
            frame = None
    if frame is None and csv_path.exists():
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            frame = None
    if frame is None or frame.empty:
        return None
    try:
        frame = _normalize_trade_date_column(frame)
    except Exception:
        return None
    return frame, meta


def get_trade_calendar(
    *,
    offline: bool = False,
    refresh: bool = False,
    calendar_dir: str | Path | None = None,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """返回 (normalized DataFrame[trade_date], meta)。

    - ``refresh=True``：必拉取并落盘（``offline=True`` 时禁止网络，若无法读缓存则失败）。
    - ``refresh=False``：有缓存则只读盘；无缓存时非 offline 则拉取并落盘。
    """
    directory = Path(calendar_dir) if calendar_dir is not None else default_calendar_dir()
    fetch = fetcher or _fetch_sina_trade_calendar

    if refresh:
        if offline:
            loaded = load_trade_calendar_disk(directory)
            if loaded is None:
                raise CacheMissError("交易日历缓存不存在，且 offline 禁止刷新")
            return loaded
        remote = fetch()
        meta = save_trade_calendar(directory, remote)
        return _normalize_trade_date_column(remote), meta

    loaded = load_trade_calendar_disk(directory)
    if loaded is not None:
        return loaded

    if offline:
        raise CacheMissError("交易日历不在本地缓存（.cache/calendar），本次不允许访问网络")

    remote = fetch()
    meta = save_trade_calendar(directory, remote)
    return _normalize_trade_date_column(remote), meta
