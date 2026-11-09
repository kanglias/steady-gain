from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from data import CacheMissError, get_bars
from data.cache import default_cache_dir
from data.normalize import normalize_adjust, normalize_symbol, parse_date


def _fetcher(calls: list):
    def fetcher(symbol, start, end, adjust):
        calls.append((symbol, start, end, adjust))
        days = pd.bdate_range(start, end)
        price = 10 + (pd.Series(range(len(days))) % 5).to_numpy()
        return pd.DataFrame(
            {
                "日期": days,
                "开盘": price,
                "最高": price + 0.4,
                "最低": price - 0.4,
                "收盘": price + 0.1,
                "成交量": 1000,
                "成交额": 1_000_000,
            }
        )

    return fetcher


def test_symbol_and_date_helpers():
    assert normalize_symbol("sz000001") == "000001"
    assert normalize_symbol("600519.SH") == "600519"
    assert normalize_adjust("前复权") == "qfq"
    assert normalize_adjust("") == ""
    assert parse_date("20240102") == date(2024, 1, 2)
    assert parse_date("2024-01-02") == date(2024, 1, 2)
    assert parse_date(datetime(2024, 1, 2, 15, 0)) == date(2024, 1, 2)
    with pytest.raises(ValueError):
        normalize_symbol("平安银行")
    with pytest.raises(ValueError):
        normalize_adjust("adj")


def test_cache_hit_extend_refresh_and_offline(tmp_path, monkeypatch):
    calls: list = []
    fetcher = _fetcher(calls)
    kwargs = {"cache_dir": tmp_path, "fetcher": fetcher, "adjust": "qfq"}

    first = get_bars("SZ000001", "2024-01-02", "2024-03-29", **kwargs)
    assert calls == [("000001", date(2024, 1, 2), date(2024, 3, 29), "qfq")]
    assert first.attrs["cache_hit"] is False
    assert {"date", "symbol", "open", "high", "low", "close", "volume"} <= set(first.columns)
    assert first["symbol"].eq("000001").all()
    assert first["date"].min() >= pd.Timestamp("2024-01-02")
    assert first["date"].max() <= pd.Timestamp("2024-03-29")
    assert (tmp_path / "000001_qfq.parquet").exists()

    second = get_bars("000001", "2024-02-01", "2024-02-29", **kwargs)
    assert len(calls) == 1
    assert second.attrs["cache_hit"] is True
    assert second["date"].min() >= pd.Timestamp("2024-02-01")
    assert second["date"].max() <= pd.Timestamp("2024-02-29")

    wider = get_bars("000001", "2024-01-02", "2024-06-28", **kwargs)
    assert len(calls) == 2
    assert calls[1] == ("000001", date(2024, 1, 2), date(2024, 6, 28), "qfq")
    assert wider.attrs["cache_hit"] is False
    assert wider["date"].max() <= pd.Timestamp("2024-06-28")

    before = len(calls)
    with pytest.raises(CacheMissError):
        get_bars("000001", "2020-01-02", "2024-06-28", offline=True, **kwargs)
    assert len(calls) == before

    refreshed = get_bars("000001", "2024-02-01", "2024-02-29", refresh=True, **kwargs)
    assert len(calls) == before + 1
    # 刷新仍合并已覆盖区间，避免一次短请求把缓存裁短。
    assert calls[-1][1] == date(2024, 1, 2)
    assert calls[-1][2] == date(2024, 6, 28)
    assert refreshed["date"].min() >= pd.Timestamp("2024-02-01")


def test_csv_fallback_when_parquet_cannot_be_written(tmp_path, monkeypatch):
    calls: list = []

    def boom(self, *args, **kwargs):
        raise OSError("parquet 不可用")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    get_bars(
        "600519",
        "2024-01-02",
        "2024-01-31",
        cache_dir=tmp_path,
        fetcher=_fetcher(calls),
    )
    assert (tmp_path / "600519_qfq.csv").exists()
    assert not (tmp_path / "600519_qfq.parquet").exists()

    monkeypatch.undo()
    cached = get_bars(
        "600519",
        "2024-01-02",
        "2024-01-31",
        cache_dir=tmp_path,
        fetcher=_fetcher(calls),
    )
    assert len(calls) == 1
    assert cached.attrs["cache_hit"] is True


def test_invalid_range_does_not_fetch(tmp_path):
    def fetcher(*_args):
        raise AssertionError("不该访问数据源")

    with pytest.raises(ValueError):
        get_bars("000001", "2024-05-01", "2024-01-01", cache_dir=tmp_path, fetcher=fetcher)


def test_default_cache_dir_uses_env_or_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("ASHARE_CACHE_DIR", str(tmp_path / "custom"))
    assert default_cache_dir() == tmp_path / "custom"

    monkeypatch.delenv("ASHARE_CACHE_DIR")
    repo = tmp_path / "repo" / "notebooks"
    repo.mkdir(parents=True)
    (tmp_path / "repo" / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    assert default_cache_dir() == tmp_path / "repo" / ".cache" / "bars"
