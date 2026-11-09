"""日线 parquet / csv 缓存。

覆盖区间记的是「已经向数据源查询过的起止日期」，不是 K 线第一根和最后一根。
这样上市前的空白不会导致每次重复请求。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data.normalize import parse_date


def default_cache_dir() -> Path:
    """优先 `ASHARE_CACHE_DIR`，否则在当前目录向上找仓库根下的 `.cache/bars`。"""
    env = os.environ.get("ASHARE_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate / ".cache" / "bars"
    return start / ".cache" / "bars"


def _as_symbol(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def cache_key(symbol: str, adjust: str) -> str:
    return f"{symbol}_{adjust or 'raw'}"


def cache_paths(cache_dir: Path, symbol: str, adjust: str) -> tuple[Path, Path, Path]:
    key = cache_key(symbol, adjust)
    return (
        cache_dir / f"{key}.parquet",
        cache_dir / f"{key}.csv",
        cache_dir / f"{key}.meta.json",
    )


def load_cache(cache_dir: Path, symbol: str, adjust: str) -> tuple[pd.DataFrame, dict] | None:
    parquet_path, csv_path, meta_path = cache_paths(cache_dir, symbol, adjust)
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
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].map(_as_symbol)
    return frame.reset_index(drop=True), meta


def save_cache(
    cache_dir: Path,
    symbol: str,
    adjust: str,
    frame: pd.DataFrame,
    start: date,
    end: date,
    source: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path, csv_path, meta_path = cache_paths(cache_dir, symbol, adjust)
    payload = frame.copy()
    payload["date"] = pd.to_datetime(payload["date"]).dt.normalize()
    if "symbol" in payload.columns:
        payload["symbol"] = payload["symbol"].map(_as_symbol)
    try:
        payload.to_parquet(parquet_path, index=False)
        csv_path.unlink(missing_ok=True)
    except Exception:
        payload.to_csv(csv_path, index=False)
        parquet_path.unlink(missing_ok=True)

    meta = {
        "symbol": symbol,
        "adjust": adjust,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": int(len(payload)),
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_covers(meta: dict, symbol: str, adjust: str, start: date, end: date) -> bool:
    if meta.get("symbol") != symbol or meta.get("adjust") != adjust:
        return False
    try:
        cached_start = parse_date(meta["start"])
        cached_end = parse_date(meta["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return cached_start <= start and cached_end >= end
