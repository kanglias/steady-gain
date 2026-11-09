from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.bars import CacheMissError
from data.trade_calendar import get_trade_calendar, load_trade_calendar_disk
from tests.fixtures.trade_calendar import write_fixture_calendar


def test_get_trade_calendar_cache_hit_no_second_fetch(tmp_path, monkeypatch):
    calls: list[int] = []

    def fake_fetch() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"])})

    monkeypatch.setattr("data.trade_calendar._fetch_sina_trade_calendar", fake_fetch)

    d = tmp_path / "cal"
    first, meta1 = get_trade_calendar(calendar_dir=d, fetcher=fake_fetch)
    second, meta2 = get_trade_calendar(calendar_dir=d, fetcher=fake_fetch)
    assert len(first) == 2
    assert len(second) == 2
    assert meta1["rows"] == meta2["rows"]
    assert len(calls) == 1  # 第二次命中磁盘，不调 fetcher


def test_get_trade_calendar_refresh_refetches(tmp_path, monkeypatch):
    calls: list[int] = []

    def fake_fetch() -> pd.DataFrame:
        calls.append(1)
        day = "2024-01-02" if len(calls) == 1 else "2024-01-04"
        return pd.DataFrame({"trade_date": pd.to_datetime([day])})

    d = tmp_path / "cal"
    a, _ = get_trade_calendar(calendar_dir=d, fetcher=fake_fetch)
    assert len(a) == 1
    b, _ = get_trade_calendar(calendar_dir=d, refresh=True, fetcher=fake_fetch)
    assert len(b) == 1
    assert len(calls) == 2


def test_get_trade_calendar_offline_without_cache_raises(tmp_path, monkeypatch):
    def boom() -> pd.DataFrame:
        raise AssertionError("不应在 offline 且无缓存时访问网络")

    with pytest.raises(CacheMissError):
        get_trade_calendar(calendar_dir=tmp_path / "missing", offline=True, refresh=False, fetcher=boom)


def test_load_trade_calendar_disk_parquet_roundtrip(tmp_path):
    write_fixture_calendar(tmp_path, ["2024-01-02", "2024-01-03"])
    loaded = load_trade_calendar_disk(tmp_path)
    assert loaded is not None
    frame, meta = loaded
    assert len(frame) == 2
    assert meta.get("exchange") == "SSE"
