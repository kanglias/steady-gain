from __future__ import annotations

import pandas as pd
import pytest

from backtest import FOLLOW_LABEL, compare_fills, fill_assumption_table, run_backtest
from strategy.base import StrategyResult


def _book(close, weight, open_=None):
    dates = pd.bdate_range("2024-01-02", periods=len(close))
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close if open_ is None else open_,
            "close": close,
        }
    )
    result = StrategyResult(
        signals=pd.DataFrame(columns=["date", "symbol", "side", "price", "reason"]),
        positions=pd.DataFrame({"date": dates, "symbol": "000001", "weight": weight}),
        indicators=pd.DataFrame({"date": dates}),
    )
    return bars, result


def test_close_fill_return_drawdown_and_turnover():
    bars, result = _book([10, 11, 12, 9, 9], [0, 1, 1, 0, 0])
    stats = run_backtest(bars, result, fill="close")
    assert stats.fill == "close"
    assert stats.total_return == pytest.approx(-2 / 11)
    assert stats.max_drawdown == pytest.approx(0.25)
    assert stats.turnover == pytest.approx(2)
    assert "滑点" in run_backtest.__doc__


def test_next_open_fill_differs_from_close_fill():
    bars, result = _book(
        [10, 11, 12, 9, 9],
        [0, 1, 1, 0, 0],
        open_=[10, 10.5, 11.5, 10, 9],
    )
    both = compare_fills(bars, result)
    assert both.next_open_fill.total_return == pytest.approx(-5 / 23)
    assert both.next_open_fill.total_return != pytest.approx(both.close_fill.total_return)
    assert both.next_open_fill.turnover == pytest.approx(both.close_fill.turnover)
    assert both.next_open_fill.max_drawdown == pytest.approx(0.25)


def test_partial_weight_turnover_and_flat_book():
    bars, result = _book([10, 10, 10, 10], [0, 0.5, 0.5, 0])
    stats = run_backtest(bars, result)
    assert stats.total_return == pytest.approx(0)
    assert stats.turnover == pytest.approx(1)

    cash_bars, cash = _book([10, 11, 12], [0, 0, 0])
    flat = run_backtest(cash_bars, cash)
    assert flat.total_return == pytest.approx(0)
    assert flat.max_drawdown == pytest.approx(0)
    assert flat.turnover == pytest.approx(0)


def test_rejects_non_positive_prices_and_missing_open():
    bars, result = _book([10, -1, 10], [0, 1, 1])
    with pytest.raises(ValueError):
        run_backtest(bars, result)
    no_open = bars.drop(columns=["open"])
    with pytest.raises(ValueError):
        run_backtest(no_open, result, fill="next_open")


def test_next_open_stop_gap_and_side_by_side_label():
    dates = pd.bdate_range("2024-01-02", periods=5)
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": [10, 10, 10, 7, 6],
            "close": [10, 10, 10, 8, 9],
        }
    )
    positions = pd.DataFrame({"date": dates, "symbol": "000001", "weight": [0, 1, 1, 0, 0]})
    indicators = pd.DataFrame({"date": dates})
    stopped = StrategyResult(
        signals=pd.DataFrame(
            {
                "date": [dates[3]],
                "symbol": ["000001"],
                "side": ["sell"],
                "price": [8.0],
                "reason": ["收盘较开仓价回撤达到止损 8%"],
            }
        ),
        positions=positions,
        indicators=indicators,
    )
    close_fill = run_backtest(bars, stopped, fill="close")
    stop_fill = run_backtest(bars, stopped, fill="next_open_stop")
    assert close_fill.total_return == pytest.approx(-0.2)
    assert stop_fill.total_return == pytest.approx(-0.4)
    plain = StrategyResult(
        signals=pd.DataFrame(
            {
                "date": [dates[3]],
                "symbol": ["000001"],
                "side": ["sell"],
                "price": [8.0],
                "reason": ["短均线低于长均线"],
            }
        ),
        positions=positions,
        indicators=indicators,
    )
    assert run_backtest(bars, plain, fill="next_open_stop").total_return == pytest.approx(-0.2)
    table = fill_assumption_table(compare_fills(bars, stopped), stop_fill)
    assert table["label"].tolist() == ["次日开盘成交", "收盘成交", "次日开盘止损"]
    assert table.iloc[0]["fill"] == "next_open"
    assert table.iloc[0]["follow"] == FOLLOW_LABEL
    assert (table["follow"] == FOLLOW_LABEL).sum() == 1
