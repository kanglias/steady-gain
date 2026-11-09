from __future__ import annotations

import pandas as pd
import pytest

from strategy import DualMovingAverage
from tests.test_dual_ma import bare_ma, sample_bars
from viz import plot_signals, save_chart


def test_chart_has_candles_mas_and_markers():
    bars = sample_bars()
    result = bare_ma(2, 3).run(bars)
    fig = plot_signals(bars, result, title="000001 测试")

    names = [trace.name for trace in fig.data]
    assert "K线" in names
    assert "短均线" in names
    assert "长均线" in names
    assert "买入" in names
    assert "卖出" in names
    assert "成交量" in names
    buy = next(trace for trace in fig.data if trace.name == "买入")
    sell = next(trace for trace in fig.data if trace.name == "卖出")
    assert len(list(buy.x)) == 1
    assert len(list(sell.x)) == 1
    assert "000001 测试" in fig.layout.title.text
    assert "不是券商成交" in fig.layout.title.text


def test_chart_allows_empty_signals(tmp_path):
    bars = sample_bars()
    result = bare_ma(2, 3).run(bars.iloc[:3])
    assert result.signals.empty
    fig = plot_signals(bars.iloc[:3], result, show_volume=False)
    names = [trace.name for trace in fig.data]
    assert "K线" in names
    assert "买入" not in names
    path = save_chart(fig, tmp_path / "empty.html")
    text = path.read_text(encoding="utf-8")
    assert "plotly" in text.lower()
    assert path.stat().st_size > 1000


def test_chart_requires_ohlc():
    with pytest.raises(ValueError):
        plot_signals(pd.DataFrame({"date": [], "close": []}), DualMovingAverage().run)
