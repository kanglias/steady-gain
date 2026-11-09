from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data.qc_bars_cache import discover_qfq_parquets, main_cli, run_bars_cache_qc
from tests.fixtures.qc_bars_cache import (
    write_bar_file,
    make_bad_ohlc,
    make_close_nonpositive_error,
    make_good_pair,
    make_meta_row_mismatch,
    make_missing_day_pair,
    make_stale_meta_warning,
    make_symbol_mismatch,
    make_volume_zero_warning,
)
from tests.fixtures.trade_calendar import write_fixture_calendar


def _empty_calendar_dir(parent: Path) -> Path:
    d = parent / "_no_cal"
    d.mkdir(exist_ok=True)
    return d


def test_discover_qfq_parquets_filters(tmp_path):
    (tmp_path / "000001_qfq.parquet").write_bytes(b"")
    (tmp_path / "000001_HFQ.parquet").write_bytes(b"")
    (tmp_path / "000001_qfq.meta.json").write_text("{}", encoding="utf-8")
    found = discover_qfq_parquets(tmp_path)
    assert len(found) == 1
    assert found[0].name == "000001_qfq.parquet"


def test_qc_good_only(tmp_path):
    make_good_pair(tmp_path)
    ref = datetime(2026, 1, 15, tzinfo=timezone.utc)
    report = run_bars_cache_qc(
        tmp_path, stale_days=7.0, reference_time=ref, calendar_dir=_empty_calendar_dir(tmp_path)
    )
    assert report["files_scanned"] == 1
    assert report["summary"]["errors"] == 0
    block = report["files"][0]
    assert block["errors"] == []
    assert set(block["optional_present"]) >= {"amount", "turnover"}


def test_qc_ohlc_violation(tmp_path):
    make_bad_ohlc(tmp_path)
    report = run_bars_cache_qc(tmp_path, calendar_dir=_empty_calendar_dir(tmp_path))
    assert report["summary"]["errors"] >= 1
    assert any("high < max" in m for m in report["files"][0]["errors"])


def test_qc_symbol_mismatch(tmp_path):
    make_symbol_mismatch(tmp_path)
    report = run_bars_cache_qc(tmp_path, calendar_dir=_empty_calendar_dir(tmp_path))
    assert any("symbol" in m for m in report["files"][0]["errors"])


def test_qc_meta_rows_mismatch(tmp_path):
    make_meta_row_mismatch(tmp_path)
    report = run_bars_cache_qc(tmp_path, calendar_dir=_empty_calendar_dir(tmp_path))
    assert any("meta.rows" in m for m in report["files"][0]["errors"])


def test_qc_missing_days_union(tmp_path):
    make_missing_day_pair(tmp_path)
    report = run_bars_cache_qc(tmp_path, calendar_dir=_empty_calendar_dir(tmp_path))
    assert report["files_scanned"] == 2
    assert report["calendar_active"] is False
    by_sym = {Path(f["parquet"]).name: f for f in report["files"]}
    b = by_sym["000006_qfq.parquet"]
    assert "2024-01-03" in b["missing_days"]
    assert "2024-01-03" in b["missing_vs_union"]


def test_qc_calendar_missing_day_warning(tmp_path):
    bars = tmp_path / "bars"
    cal = tmp_path / "cal"
    make_missing_day_pair(bars)
    write_fixture_calendar(
        cal,
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        calendar_end=date(2026, 12, 31),
    )
    report = run_bars_cache_qc(
        bars,
        calendar_dir=cal,
        reference_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert report["calendar_active"] is True
    by_sym = {Path(f["parquet"]).name: f for f in report["files"]}
    b = by_sym["000006_qfq.parquet"]
    assert "2024-01-03" in b["missing_days"]
    assert any("相对交易日历缺失" in w for w in b["warnings"])


def test_qc_extra_saturday_error_with_calendar(tmp_path):
    bars = tmp_path / "bars"
    cal = tmp_path / "cal"
    # 2024-01-06 为周六；日历仅含正常交易日
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-06"]),
            "symbol": "000010",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100.0,
            "amount": 1e6,
        }
    )
    write_bar_file(
        bars,
        "000010",
        "qfq",
        frame,
        meta_start=date(2024, 1, 2),
        meta_end=date(2024, 1, 10),
        source="testfixture.saturday",
    )
    write_fixture_calendar(cal, ["2024-01-02", "2024-01-03"], calendar_end=date(2026, 12, 31))
    report = run_bars_cache_qc(
        bars,
        calendar_dir=cal,
        reference_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    block = report["files"][0]
    assert "2024-01-06" in block["extra_non_trading_days"]
    assert any("非交易日出现行情" in m for m in block["errors"])


def test_qc_good_without_calendar_shows_fallback_info(tmp_path):
    make_good_pair(tmp_path)
    cal = tmp_path / "empty_cal"
    cal.mkdir()
    report = run_bars_cache_qc(tmp_path, calendar_dir=cal)
    assert report["calendar_active"] is False
    assert any("未使用官方交易日历" in m for m in report["info"])


def test_qc_stale_fetched_at_warning(tmp_path):
    ref = datetime(2026, 6, 1, tzinfo=timezone.utc)
    make_stale_meta_warning(tmp_path, ref=ref)
    report = run_bars_cache_qc(
        tmp_path, stale_days=7.0, reference_time=ref, calendar_dir=_empty_calendar_dir(tmp_path)
    )
    assert report["summary"]["errors"] == 0
    assert any("fetched_at" in w for w in report["files"][0]["warnings"])


def test_qc_volume_zero_warning(tmp_path):
    make_volume_zero_warning(tmp_path)
    report = run_bars_cache_qc(tmp_path, calendar_dir=_empty_calendar_dir(tmp_path))
    assert report["summary"]["errors"] == 0
    assert any("volume == 0" in w for w in report["files"][0]["warnings"])


def test_qc_close_nonpositive_error(tmp_path):
    make_close_nonpositive_error(tmp_path)
    report = run_bars_cache_qc(tmp_path, calendar_dir=_empty_calendar_dir(tmp_path))
    assert any("close <=" in m for m in report["files"][0]["errors"])


def test_main_cli_exit_code(tmp_path, capsys):
    make_bad_ohlc(tmp_path)
    cal = _empty_calendar_dir(tmp_path)
    code = main_cli(["--cache-dir", str(tmp_path), "--calendar-dir", str(cal), "--quiet"])
    assert code == 1
    captured = capsys.readouterr()
    assert "errors=" in captured.out
