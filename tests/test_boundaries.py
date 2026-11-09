from __future__ import annotations

import pandas as pd

import backtest
import journal
import orders


def test_backtest_requires_a_result_and_documents_costs():
    with __import__("pytest").raises(TypeError):
        backtest.run_backtest(pd.DataFrame({"date": [], "close": []}))
    assert "滑点" in backtest.run_backtest.__doc__
    assert "next_open" in backtest.run_backtest.__doc__


def test_orders_has_no_broker_surface():
    banned = {
        "place_order",
        "submit_order",
        "auto_trade",
        "send_order",
        "buy",
        "sell",
    }
    assert banned.isdisjoint(set(dir(orders)))
    assert banned.isdisjoint(set(dir(journal)))
