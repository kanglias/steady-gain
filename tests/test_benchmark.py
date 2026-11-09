from __future__ import annotations

import pandas as pd
import pytest

from data import csi300_excess, get_csi300, read_csi300_offline
from data.bars import CacheMissError


def test_csi300_uses_fetcher_and_then_cache(tmp_path):
    def fake(start, end):
        dates = pd.bdate_range(start, end)
        return pd.DataFrame({"日期": dates, "开盘": 1.0, "收盘": range(len(dates))})

    frame = get_csi300("2024-01-02", "2024-01-05", cache_dir=tmp_path, fetcher=fake)
    assert list(frame.columns) == ["date", "open", "close"]
    assert frame.attrs["cache_hit"] is False
    again = get_csi300("2024-01-02", "2024-01-05", cache_dir=tmp_path, fetcher=fake)
    assert again.attrs["cache_hit"] is True
    with pytest.raises(CacheMissError):
        get_csi300("2020-01-01", "2020-01-31", cache_dir=tmp_path, offline=True, fetcher=fake)


def test_excess_with_benchmark_and_graceful_without_cache(tmp_path):
    bench = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "close": [10.0, 12.0],
        }
    )
    got = csi300_excess(0.10, bench)
    assert got["available"] is True
    assert got["benchmark_return"] == pytest.approx(0.20)
    assert got["excess"] == pytest.approx(0.10 - 0.20)
    missing = csi300_excess(0.25, None)
    assert missing["available"] is False
    assert pd.isna(missing["excess"])
    assert "缺少沪深300缓存" in str(missing["message"])
    assert "策略净值" in str(missing["message"])
    assert csi300_excess(0.25, pd.DataFrame())["available"] is False
    assert read_csi300_offline("2024-01-02", "2024-01-05", cache_dir=tmp_path) is None
