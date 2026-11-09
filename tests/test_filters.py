from __future__ import annotations

import pandas as pd

from tests.test_dual_ma import bare_ma, sample_bars


def _bars(close: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(close))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [max(value - 0.2, 0.1) for value in close],
            "close": close,
            "volume": 1000,
        }
    )


def test_confirmation_delays_the_entry():
    bars = sample_bars()
    # 样本里价格在第 4 根首次上穿，第 5 根才凑满连续两日。
    result = bare_ma(2, 3, confirm_days=2).run(bars)
    assert result.signals.iloc[0]["side"] == "buy"
    assert result.signals.iloc[0]["date"] == bars["date"].iloc[5]
    assert result.positions["weight"].iloc[4] == 0
    assert result.positions["weight"].iloc[5] == 1


def test_min_hold_ignores_an_early_death_cross():
    bars = sample_bars()
    result = bare_ma(2, 3, min_hold_days=10).run(bars)
    assert list(result.signals["side"]) == ["buy"]
    assert result.positions["weight"].iloc[-1] == 1


def test_stop_loss_can_exit_before_min_hold():
    bars = sample_bars()
    result = bare_ma(2, 3, min_hold_days=10, stop_loss=0.08).run(bars)
    sell = result.signals.loc[result.signals["side"] == "sell"].iloc[0]
    assert "止损" in sell["reason"]
    assert sell["date"] == bars["date"].iloc[7]


def test_take_profit_hook():
    bars = _bars([5, 5, 5, 5, 8, 8, 12])
    result = bare_ma(2, 3, min_hold_days=10, take_profit=0.2).run(bars)
    sell = result.signals.loc[result.signals["side"] == "sell"].iloc[0]
    assert "止盈" in sell["reason"]


def test_target_weight_is_not_forced_to_full():
    bars = sample_bars()
    result = bare_ma(2, 3, target_weight=0.4).run(bars)
    assert result.signals.iloc[0]["weight"] == 0.4
    assert result.positions["weight"].max() == 0.4


def test_trend_filter_blocks_a_cross_below_the_longer_average():
    # 短均线上穿时长均线，但收盘仍低于更长的趋势均线。
    bars = _bars([10, 10, 10, 10, 9, 8, 8, 9, 10])
    blocked = bare_ma(2, 3, trend_window=8).run(bars)
    opened = bare_ma(2, 3, trend_window=None).run(bars)
    assert opened.signals["side"].eq("buy").any()
    assert not blocked.signals["side"].eq("buy").any()


def test_adx_threshold_can_block_entries():
    close = [float(10 + i) for i in range(40)]
    bars = _bars(close)
    blocked = bare_ma(2, 5, adx_window=5, adx_min=1000).run(bars)
    assert blocked.signals.empty
    assert blocked.indicators["adx"].notna().any()
