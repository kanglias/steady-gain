"""Offline Step B tests: adapter mapping, defaults, synthetic gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.from_ashare_cache import (  # noqa: E402
    VOLUME_UNIT,
    load_from_ashare_cache,
    symbol_to_ts_code,
)
import pipeline as pl  # noqa: E402

FIXTURE_CACHE = ROOT / "tests" / "fixtures" / "cache_bars"
FIXTURE_SYN = ROOT / "tests" / "fixtures" / "synthetic"


def test_volume_unit_is_hand_documented():
    """Lock: volume→vol stays 「手」; constant must remain 'hand'."""
    assert VOLUME_UNIT == "hand"
    # Adapter module docstring must mention 手 / hand
    from adapters import from_ashare_cache as mod

    doc = (mod.__doc__ or "") + (mod.load_from_ashare_cache.__doc__ or "")
    assert "手" in doc or "hand" in doc.lower()


def test_default_codes_exclude_513180():
    assert "513180.SH" not in pl.DEFAULT_CODES
    assert pl.DEFAULT_CODES == ("513010.SH", "159338.SZ")
    assert pl.ADJ_MODE == "qfq"


def test_513180_alias_is_research_only_not_513010():
    assert pl.normalize_ts_code("513180") == "513180.SH"
    assert pl.normalize_ts_code("513010") == "513010.SH"
    assert pl.normalize_ts_code("513180") != pl.normalize_ts_code("513010")
    assert "513180.SH" in pl.RESEARCH_ONLY_CODES


def test_adapter_maps_fixture_cache():
    raw = load_from_ashare_cache(
        ["513010", "159338"],
        cache_dir=FIXTURE_CACHE,
        adjust="qfq",
    )
    assert set(raw["ts_code"].unique()) == {"513010.SH", "159338.SZ"}
    assert raw["trade_date"].str.match(r"^\d{8}$").all()
    for col in ("open", "high", "low", "close", "vol", "amount", "pre_close", "pct_chg", "source"):
        assert col in raw.columns
    assert raw["source"].str.startswith("ashare_cache:").all()
    # First row pre_close is NaN; subsequent rows filled from prior close
    g = raw[raw["ts_code"] == "513010.SH"].sort_values("trade_date")
    assert pd.isna(g.iloc[0]["pre_close"])
    if len(g) > 1:
        assert g.iloc[1]["pre_close"] == pytest.approx(g.iloc[0]["close"])
    assert symbol_to_ts_code("513010") == "513010.SH"
    assert symbol_to_ts_code("159338") == "159338.SZ"
    assert VOLUME_UNIT == "hand"


def test_synthetic_rejected_without_flag():
    with pytest.raises(ValueError, match="synthetic"):
        pl.load_raw_parquets(FIXTURE_SYN, ["513180.SH", "159338.SZ"], allow_synthetic=False)


def test_allow_synthetic_still_works():
    raw = pl.load_raw_parquets(
        FIXTURE_SYN,
        ["513180.SH", "159338.SZ"],
        allow_synthetic=True,
    )
    assert len(raw) > 0
    assert raw["source"].astype(str).str.contains("synthetic").any()


def test_run_pipeline_from_fixture_cache(tmp_path):
    out = pl.run_pipeline(
        tmp_path,
        ["513010.SH", "159338.SZ"],
        cache_dir=FIXTURE_CACHE,
        from_cache=True,
    )
    assert out.name.startswith("etf_features_qfq_")
    feat = pd.read_parquet(out)
    assert set(feat["adj_mode"].unique()) == {"qfq"}
    assert "513180.SH" not in set(feat["ts_code"].unique())
    latest = tmp_path / "etf_features_qfq_latest.parquet"
    assert latest.exists()
