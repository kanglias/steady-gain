"""Map main-chain `.cache/bars/{6digit}_qfq.parquet` into side-path raw schema.

Volume unit
-----------
Main-chain cache column ``volume`` is mapped 1:1 to side-path ``vol``.
The locked production decision (S8 Step B) is that this quantity is kept in
「手」 (lot / hand) **without** converting to shares. Downstream code that
expects 股 must multiply by 100 itself; this adapter does not.

Constant: ``VOLUME_UNIT = "hand"``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Locked: volume → vol stays in 「手」; do not convert to shares here.
VOLUME_UNIT = "hand"  # 「手」; assert/docs must keep this literal

# Fallbacks only. Do not treat a machine path as the production lock.
# Override with ASHARE_CACHE_BARS_DIR, QUANT_DASHBOARD_CACHE_BARS, or --cache-dir.
_CACHE_FALLBACKS = (
    Path("/workspace/.cache/bars"),
    Path("/workspace/quant-dashboard/.cache/bars"),
    Path(r"C:\Users\sousenw\quant-dashboard\.cache\bars"),
)


def _repo_root() -> Path:
    """This file lives at <repo>/a-share-features/adapters/from_ashare_cache.py."""
    return Path(__file__).resolve().parents[2]


def DEFAULT_CACHE_DIR() -> Path:
    """Resolve the main-chain bars cache directory.

    Order: ``ASHARE_CACHE_BARS_DIR`` or ``QUANT_DASHBOARD_CACHE_BARS``,
    then ``./.cache/bars`` (cwd, then this repo) if that directory exists,
    then a few local fallbacks. None of the fallbacks is required.
    """
    env = os.environ.get("ASHARE_CACHE_BARS_DIR") or os.environ.get(
        "QUANT_DASHBOARD_CACHE_BARS"
    )
    if env and str(env).strip():
        return Path(env).expanduser()
    candidates = [
        Path.cwd() / ".cache" / "bars",
        _repo_root() / ".cache" / "bars",
        *_CACHE_FALLBACKS,
    ]
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            return path
    return _repo_root() / ".cache" / "bars"


def symbol_to_ts_code(symbol: object) -> str:
    """Map 6-digit (or already-suffixed) symbol to ts_code."""
    if symbol is None or (isinstance(symbol, float) and np.isnan(symbol)):
        return ""
    s = str(symbol).strip().upper().replace("-", "").replace("_", "")
    if s.endswith(".SH") or s.endswith(".SZ"):
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[-6:]
    if not digits:
        return ""
    # ETF / stock exchange heuristic: 5/6/9 → SH, else SZ
    if digits.startswith(("5", "6", "9")):
        return f"{digits}.SH"
    return f"{digits}.SZ"


def _read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _one_symbol_frame(
    cache_dir: Path,
    symbol: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())[-6:]
    parquet = cache_dir / f"{digits}_{adjust}.parquet"
    meta_path = cache_dir / f"{digits}_{adjust}.meta.json"
    if not parquet.exists():
        raise FileNotFoundError(f"cache bar missing: {parquet}")

    bars = pd.read_parquet(parquet).reset_index(drop=True)
    meta = _read_meta(meta_path)
    meta_source = str(meta.get("source") or "unknown")

    if "date" not in bars.columns:
        raise ValueError(f"{parquet.name}: missing date column")
    if "close" not in bars.columns:
        raise ValueError(f"{parquet.name}: missing close column")

    ts_code = symbol_to_ts_code(digits)
    dates = pd.to_datetime(bars["date"], errors="coerce")

    out = pd.DataFrame(
        {
            "ts_code": ts_code,
            "trade_date": dates.dt.strftime("%Y%m%d"),
            "open": bars["open"].astype("float64") if "open" in bars.columns else np.nan,
            "high": bars["high"].astype("float64") if "high" in bars.columns else np.nan,
            "low": bars["low"].astype("float64") if "low" in bars.columns else np.nan,
            "close": bars["close"].astype("float64"),
            # volume → vol; unit locked as 「手」 (VOLUME_UNIT == "hand")
            "vol": (
                bars["volume"].astype("float64")
                if "volume" in bars.columns
                else np.nan
            ),
            "amount": (
                bars["amount"].astype("float64")
                if "amount" in bars.columns
                else np.nan
            ),
        }
    )

    # pre_close from prior close (within this symbol series)
    out["pre_close"] = out["close"].shift(1)

    if "pct_change" in bars.columns:
        # Main-chain pct_change is typically percent already (e.g. 1.23 == 1.23%)
        out["pct_chg"] = bars["pct_change"].astype("float64")
    else:
        out["pct_chg"] = np.nan
    need = out["pct_chg"].isna()
    valid = out["pre_close"].notna() & (out["pre_close"] != 0) & out["close"].notna()
    out.loc[need & valid, "pct_chg"] = (
        out.loc[need & valid, "close"] / out.loc[need & valid, "pre_close"] - 1.0
    ) * 100.0

    out["source"] = f"ashare_cache:{meta_source}"
    out = out[out["trade_date"].notna() & (out["trade_date"] != "NaT")].reset_index(
        drop=True
    )
    return out


def load_from_ashare_cache(
    symbols: Sequence[str],
    cache_dir: Optional[Union[str, Path]] = None,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Load and map one or more symbols from the main-chain bars cache.

    Parameters
    ----------
    symbols:
        6-digit codes or ts_codes (e.g. ``513010``, ``513010.SH``).
    cache_dir:
        Directory containing ``{symbol}_{adjust}.parquet`` (+ optional meta).
        Defaults to ``DEFAULT_CACHE_DIR()`` / env override.
    adjust:
        Adjustment suffix; production default is ``qfq``.

    Returns
    -------
    DataFrame with side-path raw columns including ``vol`` in 「手」
    (see ``VOLUME_UNIT``).
    """
    assert VOLUME_UNIT == "hand", "volume→vol unit must stay 「手」 (hand)"
    cdir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR()
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        frames.append(_one_symbol_frame(cdir, sym, adjust=adjust))
    if not frames:
        raise ValueError("no symbols requested")
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "VOLUME_UNIT",
    "DEFAULT_CACHE_DIR",
    "symbol_to_ts_code",
    "load_from_ashare_cache",
]
