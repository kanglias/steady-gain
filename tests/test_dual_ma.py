from __future__ import annotations

import pandas as pd
import pytest

from strategy import DualMovingAverage, available_strategies


def sample_bars() -> pd.DataFrame:
    close = [1, 1, 1, 1, 3, 3, 3, 0, 0, 0]
    dates = pd.bdate_range("2024-01-02", periods=len(close))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": 1000,
        }
    )


def test_defaults_and_registry():
    strategy = DualMovingAverage()
    assert strategy.name == "dual_ma_10_30"
    assert strategy.confirm_days == 2
    assert strategy.min_hold_days == 5
    assert strategy.trend_window == 60
    assert strategy.target_weight == 0.5
    assert strategy.stop_loss == 0.08
    assert strategy.take_profit is None
    assert "DualMovingAverage" in available_strategies()
    with pytest.raises(ValueError):
        DualMovingAverage(10, 10)
    with pytest.raises(ValueError):
        DualMovingAverage(0, 5)
    with pytest.raises(TypeError):
        DualMovingAverage(True, 5)  # type: ignore[arg-type]


def bare_ma(short: int = 2, long: int = 3, **kwargs) -> DualMovingAverage:
    """关掉过滤和止损，用来对照裸交叉。日常示例不要这样用。"""
    params = dict(
        trend_window=None,
        confirm_days=1,
        min_hold_days=1,
        target_weight=1.0,
        stop_loss=None,
        take_profit=None,
        adx_window=None,
    )
    params.update(kwargs)
    return DualMovingAverage(short, long, **params)


def test_crossover_signals_and_position():
    bars = sample_bars()
    result = bare_ma(2, 3).run(bars)

    assert list(result.signals["side"]) == ["buy", "sell"]
    assert result.signals["date"].tolist() == [bars["date"].iloc[4], bars["date"].iloc[7]]
    assert result.signals["price"].tolist() == [3, 0]
    assert "高于" in result.signals.iloc[0]["reason"]
    assert "低于" in result.signals.iloc[1]["reason"]
    assert result.positions["weight"].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert len(result.indicators) == len(bars)
    assert pd.isna(result.indicators.loc[0, "ma_short"])
    assert pd.isna(result.indicators.loc[1, "ma_long"])
    assert pd.notna(result.indicators.loc[2, "ma_long"])


def test_flat_book_ignores_death_cross_until_a_buy():
    # 短均线一开始就在长均线之上，随后下穿。起始空仓，这次下穿不改变仓位。
    close = [1, 3, 5, 4, 2, 1, 1, 4, 6, 6]
    dates = pd.bdate_range("2024-01-02", periods=len(close))
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": 1000,
        }
    )
    result = bare_ma(2, 3).run(bars)
    assert result.signals.iloc[0]["side"] == "buy"
    assert set(result.signals["side"]).issubset({"buy", "sell"})
    assert result.positions["weight"].iloc[0] == 0


def test_rejects_mixed_symbols():
    bars = sample_bars()
    extra = bars.copy()
    extra["symbol"] = "600519"
    mixed = pd.concat([bars, extra], ignore_index=True)
    with pytest.raises(ValueError):
        DualMovingAverage(2, 3).run(mixed)
