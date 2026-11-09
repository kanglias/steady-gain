from __future__ import annotations

import pandas as pd
import pytest

from journal import compare_actual, format_report
from journal.log import FillLogError
from strategy.base import StrategyResult


def _book():
    dates = pd.bdate_range("2024-01-02", periods=4)
    close = [10.0, 11.0, 12.0, 13.0]
    open_ = [10.0, 10.5, 11.5, 12.5]
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": open_,
            "high": close,
            "low": close,
            "close": close,
        }
    )
    result = StrategyResult(
        signals=pd.DataFrame(
            {
                "date": [dates[0], dates[2]],
                "symbol": "000001",
                "side": ["buy", "sell"],
                "price": [10.0, 12.0],
                "weight": [0.5, 0.0],
                "reason": ["偏离", "回归"],
            }
        ),
        positions=pd.DataFrame(
            {"date": dates, "symbol": "000001", "weight": [0.5, 0.5, 0.0, 0.0]}
        ),
        indicators=pd.DataFrame({"date": dates}),
    )
    return bars, result


def _fills(buy_price=10.0, sell_price=12.0, buy_qty=50, sell_qty=50):
    return pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "side": ["buy", "sell"],
            "price": [buy_price, sell_price],
            "quantity": [buy_qty, sell_qty],
            "timestamp": ["2024-01-02T15:00:00", "2024-01-04T15:00:00"],
            "signal_date": ["2024-01-02", "2024-01-04"],
        }
    )


def test_slippage_is_positive_when_the_fill_is_worse():
    bars, result = _book()
    report = compare_actual(bars, result, _fills(buy_price=11.0, sell_price=11.0), equity=1000)
    buy = report.fills.iloc[0]
    sell = report.fills.iloc[1]
    assert buy["slippage_vs_close"] == pytest.approx(11 / 10 - 1)
    assert sell["slippage_vs_close"] == pytest.approx(12 / 11 - 1)
    assert buy["slippage_vs_next_open"] == pytest.approx(11 / 10.5 - 1)
    assert sell["slippage_vs_next_open"] == pytest.approx(12.5 / 11 - 1)
    assert report.summary["mean_slippage_vs_close"] == pytest.approx(
        ((11 / 10 - 1) + (12 / 11 - 1)) / 2
    )
    assert "backtest_close_return" in report.summary
    assert "backtest_next_open_return" in report.summary
    assert report.summary["backtest_close_return"] != report.summary["backtest_next_open_return"]


def test_full_follow_and_round_trip_pnl():
    bars, result = _book()
    report = compare_actual(bars, result, _fills(), equity=1000)
    assert report.summary["n_fully_followed"] == 2
    assert report.summary["total_pnl"] == pytest.approx(100.0)
    assert report.summary["realized_pnl"] == pytest.approx(100.0)
    assert report.summary["actual_return"] == pytest.approx(0.1)
    assert report.daily["daily_pnl"].sum() == pytest.approx(report.summary["total_pnl"])
    closed = report.round_trips.loc[report.round_trips["status"] == "closed"]
    assert closed.iloc[0]["pnl"] == pytest.approx(100.0)
    text = format_report(report)
    assert "不会发送委托" in text
    assert "000001" in text


def test_partial_fill_is_not_fully_followed():
    bars, result = _book()
    report = compare_actual(bars, result, _fills(buy_qty=20, sell_qty=20), equity=1000)
    assert report.signals.iloc[0]["fully_followed"] == False
    assert report.signals.iloc[0]["quantity_ratio"] == pytest.approx(20 / 50)
    assert report.summary["n_filled_signals"] == 2
    # 买入没有跟满；卖出只是把已经买到的 20 股清掉，相对「回到空仓」算跟满。
    assert report.signals.iloc[1]["fully_followed"] == True
    assert report.summary["n_fully_followed"] == 1


def test_without_equity_a_linked_fill_counts_as_followed():
    bars, result = _book()
    report = compare_actual(bars, result, _fills(buy_qty=1, sell_qty=1))
    assert report.summary["n_fully_followed"] == 2
    assert pd.isna(report.signals.iloc[0]["quantity_ratio"])
    assert report.summary["actual_return"] != report.summary["actual_return"]
    assert report.summary["total_pnl"] == pytest.approx(2.0)


def test_missing_fill_is_not_followed_and_unmatched_fill_is_counted():
    bars, result = _book()
    only_buy = _fills().iloc[:1]
    extra = pd.DataFrame(
        {
            "symbol": ["000001"],
            "side": ["buy"],
            "price": [11.0],
            "quantity": [1],
            "timestamp": ["2024-01-03T15:00:00"],
            "signal_date": ["2024-01-03"],
        }
    )
    report = compare_actual(bars, result, pd.concat([only_buy, extra], ignore_index=True), equity=1000)
    assert report.signals.iloc[1]["fully_followed"] == False
    assert report.summary["n_unmatched_fills"] == 1
    assert report.summary["unrealized_pnl"] > 0


def test_costs_reduce_actual_return():
    bars, result = _book()
    fills = _fills()
    fills["commission"] = [10.0, 10.0]
    fills["stamp_tax"] = [0.0, 5.0]
    report = compare_actual(bars, result, fills, equity=1000)
    assert report.summary["total_cost"] == pytest.approx(25.0)
    assert report.summary["total_pnl"] == pytest.approx(75.0)
    assert report.summary["actual_return"] == pytest.approx(0.075)


def test_excess_versus_buy_hold_and_benchmark():
    bars, result = _book()
    report = compare_actual(bars, result, _fills(), equity=1000)
    assert report.summary["buy_hold_return"] == pytest.approx(13 / 10 - 1)
    assert report.summary["excess_vs_buy_hold"] == pytest.approx(0.1 - (13 / 10 - 1))
    bench = bars[["date", "close"]].copy()
    bench["close"] = [10.0, 10.0, 10.0, 12.0]
    compared = compare_actual(bars, result, _fills(), equity=1000, benchmark=bench, benchmark_name="沪深300")
    assert compared.summary["benchmark_return"] == pytest.approx(12 / 10 - 1)
    assert compared.summary["excess_vs_benchmark"] == pytest.approx(0.1 - (12 / 10 - 1))
    assert compared.summary["benchmark_name"] == "沪深300"
    assert "excess_next_open_vs_benchmark" in compared.summary


def test_train_test_split_labels():
    bars, result = _book()
    split = bars["date"].iloc[2]
    report = compare_actual(bars, result, _fills(), equity=1000, split_date=split)
    assert set(report.daily["slice"]) == {"train", "test"}
    assert (report.daily["slice"] == "train").sum() == 2
    assert report.summary["split_date"] == pd.Timestamp(split).strftime("%Y-%m-%d")
    assert report.summary["buy_hold_return_train"] == pytest.approx(11 / 10 - 1)
    assert report.summary["actual_return_train"] == pytest.approx(report.daily.loc[report.daily["slice"].eq("train"), "equity"].iloc[-1] / 1000 - 1)
    assert report.summary["actual_return_test"] == pytest.approx(
        report.daily["equity"].iloc[-1] / report.daily.loc[report.daily["slice"].eq("train"), "equity"].iloc[-1] - 1
    )


def test_oversell_is_rejected():
    bars, result = _book()
    with pytest.raises(FillLogError):
        compare_actual(bars, result, _fills(buy_qty=10, sell_qty=11), equity=1000)
