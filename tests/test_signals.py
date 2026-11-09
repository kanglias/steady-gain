from __future__ import annotations

import pandas as pd
import pytest

from strategy import position_snapshot, signal_board
from strategy.base import StrategyResult
from viz import plot_signals


def _result():
    dates = pd.bdate_range("2024-01-02", periods=6)
    close = [10, 10, 11, 12, 9, 9]
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [1000] * len(close),
        }
    )
    result = StrategyResult(
        signals=pd.DataFrame(
            {
                "date": [dates[1], dates[3]],
                "symbol": ["000001", "000001"],
                "side": ["buy", "sell"],
                "price": [10.0, 12.0],
                "weight": [0.5, 0.0],
                "reason": ["确认上穿", "确认下穿"],
            }
        ),
        positions=pd.DataFrame(
            {"date": dates, "symbol": "000001", "weight": [0, 0.5, 0.5, 0.5, 0, 0]}
        ),
        indicators=pd.DataFrame({"date": dates, "ma_short": close, "ma_long": close}),
    )
    return bars, result


def test_board_columns_and_flip_flag():
    bars, result = _result()
    board = signal_board(result, strategy_name="dual_ma_10_30", start=bars["date"].iloc[0], end=bars["date"].iloc[-1], bars=bars, flip_days=5)
    assert list(board["side"]) == ["buy", "sell"]
    assert board["suggested_weight"].tolist() == [0.5, 0.0]
    sell = board.iloc[1]
    assert sell["holding_days"] == 2
    assert sell["flag"] == "短持翻脸"
    assert sell["consecutive_losses"] == 0
    assert sell["days_since_cross"] == 2
    assert board.iloc[0]["flag"] == "短持翻脸"
    assert board.iloc[0]["holding_days"] == 2
    today = signal_board(result, strategy_name="dual_ma_10_30", on=bars["date"].iloc[-1], bars=bars)
    assert today.empty


def test_open_leg_unrealized_pnl_and_chart_flag():
    dates = pd.bdate_range("2024-02-01", periods=4)
    close = [10, 10, 11, 12]
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "600519",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
    )
    result = StrategyResult(
        signals=pd.DataFrame(
            {
                "date": [dates[1]],
                "symbol": ["600519"],
                "side": ["buy"],
                "price": [10.0],
                "weight": [0.5],
                "reason": ["确认上穿"],
            }
        ),
        positions=pd.DataFrame({"date": dates, "symbol": "600519", "weight": [0, 0.5, 0.5, 0.5]}),
        indicators=pd.DataFrame({"date": dates}),
    )
    board = signal_board(result, strategy_name="dual_ma_10_30", start=dates[0], end=dates[-1], bars=bars, flip_days=1)
    assert board.iloc[0]["unrealized_pnl"] == pytest.approx(0.2)
    assert board.iloc[0]["holding_days"] == 2
    assert board.iloc[0]["flag"] == ""
    snap = position_snapshot(result, bars, strategy_name="dual_ma_10_30")
    assert snap.iloc[0]["state"] == "持有"
    assert snap.iloc[0]["unrealized_pnl"] == pytest.approx(0.2)
    assert snap.iloc[0]["suggested_weight"] == 0.5

    losing = result.signals.copy()
    losing["flag"] = "短持翻脸"
    marked = StrategyResult(signals=losing, positions=result.positions, indicators=result.indicators)
    # 上面这行把买入也标成翻脸，只为确认图例能单独标出。
    fig = plot_signals(bars, marked, title="标记")
    assert "短持翻脸" in [trace.name for trace in fig.data]


def test_events_panel_is_a_trace():
    dates = pd.bdate_range("2024-02-01", periods=4)
    close = [10, 10, 11, 12]
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "600519",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
    )
    result = StrategyResult(
        signals=pd.DataFrame(columns=["date", "symbol", "side", "price", "weight", "reason"]),
        positions=pd.DataFrame({"date": dates, "symbol": "600519", "weight": 0.0}),
        indicators=pd.DataFrame({"date": dates}),
    )
    events = pd.DataFrame(
        {
            "date": [dates[1]],
            "title": ["示例事件"],
            "region": ["CN"],
            "impact": ["high"],
            "summary": ["只用于确认面板"],
            "sectors": [None],
            "symbols": [None],
        }
    )
    fig = plot_signals(bars, result, events=events, title="事件")
    assert "政策与事件" in [trace.name for trace in fig.data]
