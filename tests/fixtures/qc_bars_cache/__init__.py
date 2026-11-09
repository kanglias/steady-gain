"""用于 `test_qc_bars_cache` 的小型合成 parquet + meta 构造器（非黄金标准）。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def _write_meta(
    path: Path,
    *,
    symbol: str,
    adjust: str,
    start: date,
    end: date,
    rows: int,
    source: str,
    fetched_at: datetime | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "adjust": adjust,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "rows": rows,
                "source": source,
                "fetched_at": (fetched_at or datetime.now(timezone.utc)).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_bar_file(
    directory: Path,
    symbol: str,
    adjust: str,
    frame: pd.DataFrame,
    *,
    meta_start: date,
    meta_end: date,
    source: str,
    fetched_at: datetime | None = None,
    wrong_row_count: bool = False,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    key = f"{symbol}_{adjust}"
    pq = directory / f"{key}.parquet"
    meta = directory / f"{key}.meta.json"
    frame.to_parquet(pq, index=False)
    rows = int(len(frame)) + (1 if wrong_row_count else 0)
    _write_meta(
        meta,
        symbol=symbol,
        adjust=adjust,
        start=meta_start,
        end=meta_end,
        rows=rows,
        source=source,
        fetched_at=fetched_at,
    )
    return pq, meta


def make_good_pair(directory: Path) -> None:
    days = pd.bdate_range("2024-01-02", periods=3, freq="C")
    price = 10.0
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "000001",
            "open": price,
            "high": price + 1.0,
            "low": price - 0.5,
            "close": price + 0.2,
            "volume": 1000.0,
            "amount": 1_000_000.0,
            "amplitude": 1.2,
            "pct_change": 0.1,
            "change": 0.1,
            "turnover": 0.5,
        }
    )
    write_bar_file(
        directory,
        "000001",
        "qfq",
        frame,
        meta_start=date(2024, 1, 1),
        meta_end=date(2024, 12, 31),
        source="testfixture.full",
    )


def make_bad_ohlc(directory: Path) -> None:
    """high 低于 max(open, close, low)。"""
    days = pd.bdate_range("2024-01-02", periods=2, freq="C")
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "000002",
            "open": [10.0, 10.0],
            "high": [9.0, 10.0],
            "low": [9.5, 9.5],
            "close": [9.8, 10.0],
            "volume": [100.0, 100.0],
            "amount": [1e6, 1e6],
        }
    )
    write_bar_file(
        directory,
        "000002",
        "qfq",
        frame,
        meta_start=date(2024, 1, 1),
        meta_end=date(2024, 12, 31),
        source="testfixture.bad_ohlc",
    )


def make_symbol_mismatch(directory: Path) -> None:
    days = pd.bdate_range("2024-01-02", periods=2, freq="C")
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "999999",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100.0,
            "amount": 1e6,
        }
    )
    write_bar_file(
        directory,
        "000003",
        "qfq",
        frame,
        meta_start=date(2024, 1, 1),
        meta_end=date(2024, 12, 31),
        source="testfixture.bad_symbol",
    )


def make_meta_row_mismatch(directory: Path) -> None:
    days = pd.bdate_range("2024-01-02", periods=2, freq="C")
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "000004",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100.0,
            "amount": 1e6,
        }
    )
    write_bar_file(
        directory,
        "000004",
        "qfq",
        frame,
        meta_start=date(2024, 1, 1),
        meta_end=date(2024, 12, 31),
        source="testfixture.meta_rows",
        wrong_row_count=True,
    )


def make_missing_day_pair(directory: Path) -> None:
    """两标的并集含 1/2、1/3、1/4；000005 缺 1/3。"""
    d_a = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frame_a = pd.DataFrame(
        {
            "date": d_a,
            "symbol": "000005",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100.0,
            "amount": 1e6,
        }
    )
    write_bar_file(
        directory,
        "000005",
        "qfq",
        frame_a,
        meta_start=date(2024, 1, 2),
        meta_end=date(2024, 1, 3),
        source="testfixture.A",
    )

    d_b = pd.to_datetime(["2024-01-02", "2024-01-04"])
    frame_b = pd.DataFrame(
        {
            "date": d_b,
            "symbol": "000006",
            "open": 20.0,
            "high": 21.0,
            "low": 19.0,
            "close": 20.5,
            "volume": 200.0,
            "amount": 2e6,
        }
    )
    write_bar_file(
        directory,
        "000006",
        "qfq",
        frame_b,
        meta_start=date(2024, 1, 2),
        meta_end=date(2024, 1, 4),
        source="testfixture.B",
    )


def make_stale_meta_warning(directory: Path, *, ref: datetime) -> None:
    days = pd.bdate_range("2024-01-02", periods=2, freq="C")
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "000007",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100.0,
            "amount": 1e6,
        }
    )
    old = ref - timedelta(days=20)
    write_bar_file(
        directory,
        "000007",
        "qfq",
        frame,
        meta_start=date(2024, 1, 1),
        meta_end=date(2024, 12, 31),
        source="testfixture.stale",
        fetched_at=old,
    )


def make_volume_zero_warning(directory: Path) -> None:
    days = pd.bdate_range("2024-01-02", periods=2, freq="C")
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "000008",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": [0.0, 100.0],
            "amount": [0.0, 1e6],
        }
    )
    write_bar_file(
        directory,
        "000008",
        "qfq",
        frame,
        meta_start=date(2024, 1, 1),
        meta_end=date(2024, 12, 31),
        source="testfixture.vol0",
    )


def make_close_nonpositive_error(directory: Path) -> None:
    days = pd.bdate_range("2024-01-02", periods=2, freq="C")
    frame = pd.DataFrame(
        {
            "date": days,
            "symbol": "000009",
            "open": [10.0, 0.0],
            "high": [11.0, 1.0],
            "low": [9.0, 0.0],
            "close": [10.5, -0.1],
            "volume": [100.0, 100.0],
            "amount": [1e6, 1e6],
        }
    )
    write_bar_file(
        directory,
        "000009",
        "qfq",
        frame,
        meta_start=date(2024, 1, 1),
        meta_end=date(2024, 12, 31),
        source="testfixture.bad_close",
    )
