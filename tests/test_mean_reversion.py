from __future__ import annotations

import pandas as pd

from strategy import MeanReversion, available_strategies, mean_reversion_sensitivity


def _bars(close):
    dates = pd.bdate_range("2024-03-01", periods=len(close))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "600519",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
    )


def _strategy(**kwargs):
    params = dict(
        window=3,
        entry_band=0.05,
        exit_band=0.0,
        confirm_days=2,
        min_hold_days=1,
        target_weight=0.4,
        stop_loss=None,
    )
    params.update(kwargs)
    return MeanReversion(**params)


def test_mean_reversion_is_registered_separately_from_dual_ma():
    assert "MeanReversion" in available_strategies()
    strategy = MeanReversion()
    assert strategy.name == "mean_reversion_20"
    assert strategy.window == 20
    assert strategy.target_weight == 0.5
    assert not strategy.name.startswith("dual_ma")


def test_two_day_stretch_buys_and_reversion_sells():
    # 3 日均线在 10 附近。连续两天明显低于均线后买入，回到均线后卖出。
    close = [10, 10, 10, 10, 9.2, 8.4, 8.4, 10, 10, 10]
    result = _strategy().run(_bars(close))
    assert list(result.signals["side"]) == ["buy", "sell"]
    assert result.signals.iloc[0]["weight"] == 0.4
    assert "低于" in result.signals.iloc[0]["reason"]
    assert result.positions["weight"].max() == 0.4
    assert "ma" in result.indicators.columns


def test_one_day_dip_does_not_enter():
    close = [10, 10, 10, 10, 8.0, 10, 10, 10]
    result = _strategy().run(_bars(close))
    assert result.signals.empty
    assert result.positions["weight"].eq(0).all()


def test_stop_can_exit_before_reversion():
    close = [10, 10, 10, 10, 9.0, 8.0, 7.0, 6.5]
    result = _strategy(stop_loss=0.08, min_hold_days=5, exit_band=0.0).run(_bars(close))
    sell = result.signals.loc[result.signals["side"] == "sell"]
    assert len(sell) == 1
    assert "止损" in sell.iloc[0]["reason"]


def test_sensitivity_grid_uses_next_open_and_does_not_claim_a_target():
    close = [10, 10, 10, 9, 8, 8, 10, 10]
    table = mean_reversion_sensitivity(_bars(close), windows=(3,), entry_bands=(0.05,))
    assert table.iloc[0]["fill"] == "next_open"
    assert "不是每月 3%" in table.iloc[0]["note"]
