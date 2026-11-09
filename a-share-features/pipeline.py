#!/usr/bin/env python3
"""A-share ETF feature engineering pipeline.

Reads daily bars (main-chain qfq cache by default, or explicit --raw-dir),
cleans, computes indicators, writes feature matrix.

Production defaults (Step B): 513010.SH + 159338.SZ, adj_mode=qfq, cache-first.
Synthetic raw is rejected unless --allow-synthetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_CODES = ("513010.SH", "159338.SZ")
ADJ_MODE = "qfq"

# Non-watch research code: aliases kept for explicit --codes only (not a 513010 alias).
RESEARCH_ONLY_CODES = frozenset({"513180.SH"})

# ---------------------------------------------------------------------------
# CLI defaults: Windows paths for the user; box paths for local dry-run
# ---------------------------------------------------------------------------
DEFAULT_RAW_DIR_WIN = r"C:\Users\sousenw\a-share-data\raw"
DEFAULT_FEATURES_DIR_WIN = r"C:\Users\sousenw\a-share-data\features"
DEFAULT_RAW_DIR_BOX = "/workspace/a-share-data/raw"
DEFAULT_FEATURES_DIR_BOX = "/workspace/a-share-data/features"

CODE_ALIASES = {
    # watch / production
    "sh513010": "513010.SH",
    "513010sh": "513010.SH",
    "513010.sh": "513010.SH",
    "513010": "513010.SH",
    "sz159338": "159338.SZ",
    "159338sz": "159338.SZ",
    "159338.sz": "159338.SZ",
    "159338": "159338.SZ",
    # research-only (NOT an alias of 513010)
    "sh513180": "513180.SH",
    "513180sh": "513180.SH",
    "513180.sh": "513180.SH",
    "513180": "513180.SH",
}

PRICE_COLS = ["open", "high", "low", "close", "pre_close"]
OPTIONAL_RAW_COLS = [
    "nav",
    "nav_date",
    "accum_nav",
    "iopv",
    "premium_to_nav_est",
    "bid1",
    "ask1",
    "bid1_vol",
    "ask1_vol",
]


def detect_default_dirs() -> tuple[str, str, str]:
    """Raw/features stay on the old box-or-Windows split.

    Cache uses ``adapters.from_ashare_cache.DEFAULT_CACHE_DIR``: env, then
    ``./.cache/bars`` when that directory exists, then local fallbacks.
    Returns (raw_dir, features_dir, cache_dir).
    """
    from adapters.from_ashare_cache import DEFAULT_CACHE_DIR

    cache = str(DEFAULT_CACHE_DIR())
    if Path(DEFAULT_RAW_DIR_BOX).exists() or Path("/workspace").exists():
        return DEFAULT_RAW_DIR_BOX, DEFAULT_FEATURES_DIR_BOX, cache
    return DEFAULT_RAW_DIR_WIN, DEFAULT_FEATURES_DIR_WIN, cache


def normalize_ts_code(code: object) -> str:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ""
    s = str(code).strip()
    key = s.lower().replace("-", "").replace("_", "").replace(".", "")
    if key in CODE_ALIASES:
        return CODE_ALIASES[key]
    # Already canonical like 513180.SH
    upper = s.upper()
    if upper.endswith(".SH") or upper.endswith(".SZ"):
        return upper
    # Bare 6-digit → known aliases (watch + research-only)
    digits = "".join(ch for ch in upper if ch.isdigit())
    if digits in ("513010", "159338", "513180"):
        return CODE_ALIASES.get(digits.lower(), upper)
    return upper


def parse_trade_date(series: pd.Series) -> pd.Series:
    """Parse YYYYMMDD (int/str) or datetime-like into datetime64[ns] normalized."""
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series, errors="coerce")
        return out.dt.normalize().astype("datetime64[ns]")
    # Prefer integer/string YYYYMMDD (float 20250520.0 from parquet ok)
    as_str = (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    # Treat <NA>/nan/None as missing
    as_str = as_str.replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA})
    mask = as_str.notna() & as_str.str.match(r"^\d{8}$")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if mask.any():
        out.loc[mask] = pd.to_datetime(as_str.loc[mask], format="%Y%m%d", errors="coerce")
    # Fallback for leftover non-null non-YYYYMMDD
    leftover = series.notna() & out.isna() & as_str.notna()
    if leftover.any():
        out.loc[leftover] = pd.to_datetime(series.loc[leftover], errors="coerce")
    return out.dt.normalize().astype("datetime64[ns]")


def trade_date_yyyymmdd(dt: pd.Series) -> pd.Series:
    return dt.dt.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Sample data generation
# ---------------------------------------------------------------------------

def _trading_days(n: int = 120, end: Optional[pd.Timestamp] = None) -> pd.DatetimeIndex:
    """Generate ~n weekdays ending at `end` (skip Sat/Sun; no holiday calendar)."""
    if end is None:
        end = pd.Timestamp("2025-10-31")
    days: List[pd.Timestamp] = []
    cur = end.normalize()
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= pd.Timedelta(days=1)
    return pd.DatetimeIndex(sorted(days))


def generate_sample_raw(raw_dir: Path, codes: Sequence[str], n_days: int = 120) -> List[Path]:
    """Write realistic synthetic hfq-ish daily bars as parquet per ticker."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dates = _trading_days(n_days)
    start_s = dates[0].strftime("%Y%m%d")
    end_s = dates[-1].strftime("%Y%m%d")
    rng = np.random.default_rng(42)

    written: List[Path] = []
    # Different base levels / vol for the two ETFs
    specs = {
        "513010.SH": {"base": 0.85, "vol": 0.018, "nav_base": 0.84},
        "513180.SH": {"base": 0.85, "vol": 0.018, "nav_base": 0.84},
        "159338.SZ": {"base": 1.05, "vol": 0.012, "nav_base": 1.04},
    }

    for code in codes:
        spec = specs.get(code, {"base": 1.0, "vol": 0.015, "nav_base": 1.0})
        rets = rng.normal(0.0003, spec["vol"], size=len(dates))
        # mild autocorrelation
        for i in range(1, len(rets)):
            rets[i] = 0.15 * rets[i - 1] + 0.85 * rets[i]
        close = spec["base"] * np.cumprod(1.0 + rets)
        open_ = np.empty_like(close)
        open_[0] = close[0] / (1.0 + rets[0])
        open_[1:] = close[:-1] * (1.0 + rng.normal(0, 0.002, size=len(dates) - 1))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.012, size=len(dates)))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.012, size=len(dates)))
        pre_close = np.empty_like(close)
        pre_close[0] = open_[0]
        pre_close[1:] = close[:-1]
        pct_chg = (close / pre_close - 1.0) * 100.0
        vol = rng.uniform(5e7, 3e8, size=len(dates))
        amount = vol * close * rng.uniform(0.95, 1.05, size=len(dates))

        # NAV: slightly lagged / smoothed vs close; some days missing nav update
        nav = spec["nav_base"] * np.cumprod(1.0 + rets * 0.98 + rng.normal(0, 0.001, size=len(dates)))
        accum_nav = nav * 1.02
        # nav_date often same day, sometimes T-1
        nav_lag = rng.choice([0, 0, 0, 1], size=len(dates))
        nav_date = dates - pd.to_timedelta(nav_lag, unit="D")
        # Drop ~5% of nav rows to simulate missing
        nav_mask = rng.random(len(dates)) > 0.05
        iopv = close * (1.0 + rng.normal(0, 0.0015, size=len(dates)))
        premium = close / nav - 1.0

        # Inject a few suspend / bad OHLC rows for pipeline to flag
        bad_idx = [10, 55] if len(dates) > 60 else []

        rows = []
        for i, d in enumerate(dates):
            row = {
                "ts_code": code if i % 17 != 0 else code.lower().replace(".", ""),  # alias mix
                "trade_date": int(d.strftime("%Y%m%d")),
                "source": "synthetic_qfq",
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "pre_close": float(pre_close[i]),
                "pct_chg": float(pct_chg[i]),
                "vol": float(vol[i]),
                "amount": float(amount[i]),
                "iopv": float(iopv[i]),
                "premium_to_nav_est": float(premium[i]),
            }
            if nav_mask[i]:
                row["nav"] = float(nav[i])
                row["accum_nav"] = float(accum_nav[i])
                row["nav_date"] = int(nav_date[i].strftime("%Y%m%d"))
            else:
                row["nav"] = np.nan
                row["accum_nav"] = np.nan
                row["nav_date"] = pd.NA
            if i in bad_idx:
                # break OHLC validity on purpose
                row["high"] = min(row["open"], row["close"]) * 0.99
                row["vol"] = -1.0
            rows.append(row)

        df = pd.DataFrame(rows)
        # Drop a couple of trading days for this ticker only (outer-align test)
        if code in ("513010.SH", "513180.SH") and len(df) > 40:
            df = df.drop(df.index[[20, 21]]).reset_index(drop=True)
        elif code == "159338.SZ" and len(df) > 40:
            df = df.drop(df.index[[70]]).reset_index(drop=True)

        out = raw_dir / f"{code}_{start_s}_{end_s}.parquet"
        df.to_parquet(out, index=False)
        written.append(out)
        print(f"[sample] wrote {out} shape={df.shape}")

    return written


# ---------------------------------------------------------------------------
# Load / clean
# ---------------------------------------------------------------------------

def warn_research_only_codes(codes: Sequence[str]) -> None:
    """Log when explicit research-only codes (e.g. 513180) are requested."""
    for c in codes:
        nc = normalize_ts_code(c)
        if nc in RESEARCH_ONLY_CODES:
            print(
                f"[warn] {nc} is research-only (NOT watch; NOT an alias of 513010.SH)",
                file=sys.stderr,
            )


def reject_synthetic_sources(raw: pd.DataFrame, *, allow_synthetic: bool) -> None:
    """Refuse frames whose source contains 'synthetic' unless explicitly allowed."""
    if "source" not in raw.columns:
        return
    src = raw["source"].astype("string").fillna("")
    mask = src.str.lower().str.contains("synthetic", regex=False)
    if not mask.any():
        return
    bad = sorted({str(x) for x in src[mask].unique()})
    if allow_synthetic:
        print(f"[warn] synthetic source allowed by flag: {bad}", file=sys.stderr)
        return
    raise ValueError(
        "raw source contains synthetic ("
        + ", ".join(bad)
        + "); refuse by default. Re-run with --allow-synthetic for fixtures/联调 only."
    )


def load_raw_parquets(
    raw_dir: Path,
    codes: Sequence[str],
    *,
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in raw dir: {raw_dir}")

    frames: List[pd.DataFrame] = []
    wanted = {normalize_ts_code(c) for c in codes}
    for fp in files:
        df = pd.read_parquet(fp)
        if "ts_code" not in df.columns or "trade_date" not in df.columns:
            print(f"[warn] skip {fp.name}: missing ts_code/trade_date", file=sys.stderr)
            continue
        df = df.copy()
        df["ts_code"] = df["ts_code"].map(normalize_ts_code)
        df = df[df["ts_code"].isin(wanted)]
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No rows for codes {sorted(wanted)} in {raw_dir} "
            f"({len(files)} parquet file(s) found)"
        )
    raw = pd.concat(frames, ignore_index=True)
    reject_synthetic_sources(raw, allow_synthetic=allow_synthetic)
    raw["trade_date_dt"] = parse_trade_date(raw["trade_date"])
    raw = raw.dropna(subset=["trade_date_dt"])
    # Dedup: last wins per (ts_code, trade_date)
    raw = raw.sort_values(["ts_code", "trade_date_dt"]).drop_duplicates(
        ["ts_code", "trade_date_dt"], keep="last"
    )
    return raw.reset_index(drop=True)


def load_bars_from_cache(
    cache_dir: Path,
    codes: Sequence[str],
    *,
    adjust: str = ADJ_MODE,
) -> pd.DataFrame:
    """Load main-chain cache via adapters.from_ashare_cache and normalize dates."""
    # Local import keeps pipeline importable if adapter path tweaks later
    from adapters.from_ashare_cache import load_from_ashare_cache

    raw = load_from_ashare_cache(codes, cache_dir=cache_dir, adjust=adjust)
    raw["trade_date_dt"] = parse_trade_date(raw["trade_date"])
    raw = raw.dropna(subset=["trade_date_dt"])
    raw = raw.sort_values(["ts_code", "trade_date_dt"]).drop_duplicates(
        ["ts_code", "trade_date_dt"], keep="last"
    )
    return raw.reset_index(drop=True)


def outer_align(raw: pd.DataFrame, codes: Sequence[str]) -> pd.DataFrame:
    """Outer-align all codes on the union of trade dates; no price ffill."""
    codes_n = [normalize_ts_code(c) for c in codes]
    all_dates = pd.DatetimeIndex(sorted(raw["trade_date_dt"].unique()))
    pieces: List[pd.DataFrame] = []
    for code in codes_n:
        sub = raw[raw["ts_code"] == code].set_index("trade_date_dt").sort_index()
        # Keep first occurrence columns; reindex dates
        aligned = sub.reindex(all_dates)
        aligned["ts_code"] = code
        aligned["trade_date_dt"] = aligned.index
        pieces.append(aligned.reset_index(drop=True))
    out = pd.concat(pieces, ignore_index=True)
    out["trade_date"] = trade_date_yyyymmdd(out["trade_date_dt"])
    out["adj_mode"] = ADJ_MODE
    return out


def asof_align_nav(df: pd.DataFrame) -> pd.DataFrame:
    """As-of align nav by nav_date onto trade_date within each ts_code.

    For rows that already have nav on the same row we keep it; for missing nav
    we pull the latest available nav whose nav_date <= trade_date.
    """
    df = df.copy()
    if "nav_date" not in df.columns:
        df["nav_date"] = pd.NaT
    if "nav" not in df.columns:
        df["nav"] = np.nan
    if "accum_nav" not in df.columns:
        df["accum_nav"] = np.nan

    df["nav_date_dt"] = parse_trade_date(df["nav_date"])

    result_parts: List[pd.DataFrame] = []
    for code, g in df.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date_dt").copy()
        # Build nav timeline from rows that have nav + nav_date
        nav_src = g.dropna(subset=["nav", "nav_date_dt"])[
            ["nav_date_dt", "nav", "accum_nav"]
        ].drop_duplicates("nav_date_dt", keep="last")
        nav_src = nav_src.sort_values("nav_date_dt")
        # unify datetime resolution for merge_asof (ns vs us mismatch)
        g["trade_date_dt"] = pd.to_datetime(g["trade_date_dt"]).astype("datetime64[ns]")
        nav_src["nav_date_dt"] = pd.to_datetime(nav_src["nav_date_dt"]).astype("datetime64[ns]")

        if nav_src.empty:
            g["nav_asof"] = np.nan
            g["accum_nav_asof"] = np.nan
            g["nav_date_asof"] = pd.NaT
        else:
            left = g[["trade_date_dt"]].reset_index(drop=True)
            right = nav_src.rename(
                columns={
                    "nav": "nav_asof",
                    "accum_nav": "accum_nav_asof",
                    "nav_date_dt": "nav_date_asof",
                }
            )
            merged = pd.merge_asof(
                left,
                right,
                left_on="trade_date_dt",
                right_on="nav_date_asof",
                direction="backward",
            )
            g["nav_asof"] = merged["nav_asof"].values
            g["accum_nav_asof"] = merged["accum_nav_asof"].values
            g["nav_date_asof"] = merged["nav_date_asof"].values

        # Prefer row-native nav if present, else asof (avoid np.where on datetimes)
        native_nav = g["nav"].notna()
        g["nav"] = g["nav"].where(native_nav, g["nav_asof"])
        g["accum_nav"] = g["accum_nav"].where(g["accum_nav"].notna(), g["accum_nav_asof"])
        native_nd = g["nav_date_dt"].notna()
        g["nav_date_dt"] = g["nav_date_dt"].where(native_nd, g["nav_date_asof"])
        g["nav_date_dt"] = pd.to_datetime(g["nav_date_dt"], errors="coerce").astype(
            "datetime64[ns]"
        )
        g["trade_date_dt"] = pd.to_datetime(g["trade_date_dt"], errors="coerce").astype(
            "datetime64[ns]"
        )

        # Calendar-day lag
        lag = (g["trade_date_dt"] - g["nav_date_dt"]).dt.days
        g["nav_lag_days"] = lag.astype("float")
        g.loc[g["nav"].isna() | g["nav_date_dt"].isna(), "nav_lag_days"] = np.nan

        g = g.drop(columns=["nav_asof", "accum_nav_asof", "nav_date_asof"], errors="ignore")
        result_parts.append(g)

    out = pd.concat(result_parts, ignore_index=True)
    nd = pd.to_datetime(out["nav_date_dt"], errors="coerce")
    out["nav_date"] = nd.dt.strftime("%Y%m%d")
    out.loc[nd.isna(), "nav_date"] = pd.NA
    return out


def ensure_pre_close_and_pct_chg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts: List[pd.DataFrame] = []
    for _, g in df.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date_dt").copy()
        prev_close = g["close"].shift(1)
        # Fill missing pre_close with previous close (not ffill of pre_close itself across gaps beyond 1)
        missing_pc = g["pre_close"].isna() if "pre_close" in g.columns else pd.Series(True, index=g.index)
        if "pre_close" not in g.columns:
            g["pre_close"] = np.nan
        g.loc[missing_pc, "pre_close"] = prev_close[missing_pc]

        if "pct_chg" not in g.columns:
            g["pct_chg"] = np.nan
        need_pct = g["pct_chg"].isna()
        valid = g["pre_close"].notna() & (g["pre_close"] != 0) & g["close"].notna()
        g.loc[need_pct & valid, "pct_chg"] = (
            g.loc[need_pct & valid, "close"] / g.loc[need_pct & valid, "pre_close"] - 1.0
        ) * 100.0
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    vol = df["vol"] if "vol" in df.columns else pd.Series(np.nan, index=df.index)
    amt = df["amount"] if "amount" in df.columns else pd.Series(np.nan, index=df.index)

    # Missing entire bar → not valid, but also tracked via missing_flags
    has_bar = o.notna() & h.notna() & l.notna() & c.notna()
    ohlc_ok = (
        (h >= np.maximum(np.maximum(o, c), l))
        & (l <= np.minimum(np.minimum(o, c), h))
    )
    vol_ok = vol.isna() | (vol >= 0)
    amt_ok = amt.isna() | (amt >= 0)
    df["is_valid"] = (has_bar & ohlc_ok & vol_ok & amt_ok).astype("int64")
    # Rows with no bar at all stay is_valid=0
    df.loc[~has_bar, "is_valid"] = 0
    return df


def build_missing_flags(df: pd.DataFrame) -> pd.Series:
    flags = []
    for _, row in df.iterrows():
        bits: List[str] = []
        if pd.isna(row.get("close")):
            bits.append("no_bar")
        for col in ("open", "high", "low", "close", "vol", "amount"):
            if col in df.columns and pd.isna(row.get(col)):
                bits.append(f"miss_{col}")
        if pd.isna(row.get("nav")):
            bits.append("miss_nav")
        if pd.isna(row.get("pre_close")):
            bits.append("miss_pre_close")
        if int(row.get("is_valid", 0)) == 0 and "no_bar" not in bits:
            bits.append("invalid_ohlc")
        flags.append("|".join(bits) if bits else "")
    return pd.Series(flags, index=df.index, dtype="object")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for _, g in df.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date_dt").copy()
        c = g["close"]

        g["ret_1d"] = c.pct_change(1)
        g["ret_5d"] = c.pct_change(5)
        g["ret_20d"] = c.pct_change(20)
        g["vol_5d"] = g["ret_1d"].rolling(5, min_periods=5).std()
        g["vol_20d"] = g["ret_1d"].rolling(20, min_periods=20).std()

        g["turnover_proxy"] = np.where(
            c.notna() & (c > 0) & g["amount"].notna(),
            g["amount"] / c,
            np.nan,
        )

        g["ma5"] = c.rolling(5, min_periods=5).mean()
        g["ma20"] = c.rolling(20, min_periods=20).mean()

        # ma_cross: golden(+1) / death(-1) vs prior day; else 0. NaN when ma not ready.
        above = g["ma5"] > g["ma20"]
        below = g["ma5"] < g["ma20"]
        prev_above = above.shift(1)
        prev_below = below.shift(1)
        cross = pd.Series(0, index=g.index, dtype="float")
        cross[above & prev_below] = 1.0
        cross[below & prev_above] = -1.0
        not_ready = g["ma5"].isna() | g["ma20"].isna() | prev_above.isna()
        cross[not_ready] = np.nan
        g["ma_cross"] = cross

        ema12 = _ema(c, 12)
        ema26 = _ema(c, 26)
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
        # Need enough history: EMA26 needs 26, then DEA needs 9 more conceptually;
        # with min_periods on ewm, early values are NaN automatically.
        g["macd_dif"] = dif
        g["macd_dea"] = dea
        g["macd_hist"] = 2.0 * (dif - dea)
        # Force NaN until both legs ready
        macd_ready = dif.notna() & dea.notna()
        g.loc[~macd_ready, ["macd_dif", "macd_dea", "macd_hist"]] = np.nan

        # RSI 14 (Wilder-style via ewm alpha=1/14)
        delta = c.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # When avg_loss==0 and avg_gain>0 → RSI 100
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
        g["rsi_14"] = rsi

        trend = pd.Series(0.0, index=g.index)
        trend[c > g["ma20"]] = 1.0
        trend[c < g["ma20"]] = -1.0
        trend[g["ma20"].isna() | c.isna()] = np.nan
        g["trend_dir"] = trend

        # premium_to_nav_est
        if "premium_to_nav_est" not in g.columns:
            g["premium_to_nav_est"] = np.nan
        need_prem = g["premium_to_nav_est"].isna()
        both = c.notna() & g["nav"].notna() & (g["nav"] != 0)
        g.loc[need_prem & both, "premium_to_nav_est"] = (
            c[need_prem & both] / g.loc[need_prem & both, "nav"] - 1.0
        )

        if "iopv" not in g.columns:
            g["iopv"] = np.nan
        if "accum_nav" not in g.columns:
            g["accum_nav"] = np.nan

        parts.append(g)
    return pd.concat(parts, ignore_index=True)


FEATURE_COLUMNS = [
    "ts_code",
    "trade_date",
    "adj_mode",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "vol",
    "amount",
    "nav",
    "accum_nav",
    "nav_date",
    "nav_lag_days",
    "iopv",
    "premium_to_nav_est",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "turnover_proxy",
    "ma5",
    "ma20",
    "ma_cross",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "rsi_14",
    "trend_dir",
    "is_valid",
    "missing_flags",
]


def run_pipeline(
    features_dir: Path,
    codes: Sequence[str],
    *,
    raw_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    allow_synthetic: bool = False,
    from_cache: bool = True,
) -> Path:
    """Run feature pipeline.

    Dual entry (Step B):
    - Default ``from_cache=True``: read main-chain ``.cache/bars/{6digit}_qfq.parquet``.
    - Explicit ``raw_dir`` with ``from_cache=False``: legacy side-path raw parquets.
    """
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    warn_research_only_codes(codes)

    if from_cache:
        cdir = Path(cache_dir) if cache_dir is not None else Path(detect_default_dirs()[2])
        print(f"[load] cache_dir={cdir} adjust={ADJ_MODE} codes={list(codes)}")
        raw = load_bars_from_cache(cdir, codes, adjust=ADJ_MODE)
    else:
        if raw_dir is None:
            raise ValueError("raw_dir required when from_cache=False")
        raw_dir = Path(raw_dir)
        print(f"[load] raw_dir={raw_dir} codes={list(codes)}")
        raw = load_raw_parquets(raw_dir, codes, allow_synthetic=allow_synthetic)

    print(f"[load] raw rows={len(raw)} codes={sorted(raw['ts_code'].unique())}")

    aligned = outer_align(raw, codes)
    print(f"[align] outer-aligned rows={len(aligned)} dates={aligned['trade_date'].nunique()}")

    aligned = asof_align_nav(aligned)
    aligned = ensure_pre_close_and_pct_chg(aligned)
    aligned = validate_ohlc(aligned)
    aligned = compute_indicators(aligned)
    aligned["missing_flags"] = build_missing_flags(aligned)

    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in aligned.columns:
            aligned[col] = np.nan if col not in ("ts_code", "trade_date", "adj_mode", "missing_flags") else ""

    features = aligned[FEATURE_COLUMNS].copy()
    features = features.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # Output naming — honest qfq (Step B)
    dmin = features["trade_date"].min()
    dmax = features["trade_date"].max()
    out_name = f"etf_features_qfq_{dmin}_{dmax}.parquet"
    out_path = features_dir / out_name
    features.to_parquet(out_path, index=False)

    latest = features_dir / "etf_features_qfq_latest.parquet"
    features.to_parquet(latest, index=False)

    print(f"[write] {out_path} shape={features.shape}")
    print(f"[write] {latest} shape={features.shape}")
    return out_path


def build_argparser() -> argparse.ArgumentParser:
    raw_default, feat_default, cache_default = detect_default_dirs()
    p = argparse.ArgumentParser(
        description="A-share ETF feature engineering pipeline (qfq; cache-first)"
    )
    p.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help=(
            "Legacy raw parquet dir. If set, disables cache-first and reads this "
            f"directory instead (box default would be {raw_default})."
        ),
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=cache_default,
        help=f"Main-chain bars cache dir (default: {cache_default})",
    )
    p.add_argument(
        "--features-dir",
        type=str,
        default=feat_default,
        help=f"Features output dir (default: {feat_default})",
    )
    p.add_argument(
        "--codes",
        type=str,
        nargs="*",
        default=list(DEFAULT_CODES),
        help="Tickers to process (default: 513010.SH 159338.SZ)",
    )
    p.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit raw source containing 'synthetic' (fixtures/联调 only)",
    )
    p.add_argument(
        "--generate-sample",
        action="store_true",
        help=(
            "Write synthetic raw bars then run via --raw-dir path. "
            "Requires --allow-synthetic; prefer writing under tests/fixtures/synthetic/"
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    codes = [normalize_ts_code(c) for c in args.codes]
    _, feat_default, cache_default = detect_default_dirs()
    features_dir = Path(args.features_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(cache_default)

    use_raw = args.raw_dir is not None or args.generate_sample
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(detect_default_dirs()[0])

    if args.generate_sample:
        if not args.allow_synthetic:
            print(
                "[error] --generate-sample writes synthetic raw; pass --allow-synthetic",
                file=sys.stderr,
            )
            return 2
        print(f"[sample] generating into {raw_dir}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        generate_sample_raw(raw_dir, codes)
        use_raw = True

    try:
        if use_raw:
            if not raw_dir.exists():
                print(f"[error] raw dir does not exist: {raw_dir}", file=sys.stderr)
                return 1
            out = run_pipeline(
                features_dir,
                codes,
                raw_dir=raw_dir,
                allow_synthetic=bool(args.allow_synthetic),
                from_cache=False,
            )
        else:
            if not cache_dir.exists():
                print(f"[error] cache dir does not exist: {cache_dir}", file=sys.stderr)
                return 1
            out = run_pipeline(
                features_dir,
                codes,
                cache_dir=cache_dir,
                from_cache=True,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    feat = pd.read_parquet(out)
    print("\n=== verification ===")
    print(f"shape: {feat.shape}")
    print(f"columns ({len(feat.columns)}): {list(feat.columns)}")
    print(f"adj_mode unique: {sorted(feat['adj_mode'].astype(str).unique())}")
    print(f"ts_code counts:\n{feat['ts_code'].value_counts().to_string()}")
    print(f"is_valid counts:\n{feat['is_valid'].value_counts().to_string()}")
    print("\nhead:")
    print(feat.head(10).to_string())
    print("\ntail:")
    print(feat.tail(6).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
