from __future__ import annotations

import pandas as pd

from data import load_events
from strategy.base import StrategyResult
from strategy.events import EventGuard, event_switch_samples


def test_local_calendar_loads_without_network():
    events = load_events()
    assert {"date", "title", "region", "impact", "summary", "sectors", "symbols"} <= set(events.columns)
    assert not events.empty
    assert events.attrs["source"] == "local"
    assert (events["impact"] == "high").all()


def test_fetch_failure_falls_back_to_local():
    def boom():
        raise RuntimeError("down")

    events = load_events(fetch=True, fetcher=boom)
    assert events.attrs["source"] == "local"
    assert "down" in events.attrs["fetch_error"]
    assert len(events) == len(load_events())


def test_fetch_success_merges_remote_rows():
    def extra():
        return pd.DataFrame(
            [
                {
                    "date": "2024-06-20",
                    "title": "LPR 调整",
                    "region": "CN",
                    "impact": "high",
                    "summary": "测试用报价",
                    "sectors": ["银行"],
                    "symbols": None,
                }
            ]
        )

    events = load_events(fetch=True, fetcher=extra, start="2024-06-01", end="2024-06-30")
    assert events.attrs["source"] == "local+akshare.macro_china_lpr"
    assert "LPR 调整" in set(events["title"])


def test_guard_blocks_a_partial_entry_and_off_is_a_noop():
    dates = pd.bdate_range("2024-01-02", periods=4)
    result = StrategyResult(
        signals=pd.DataFrame(
            {
                "date": [dates[1]],
                "symbol": ["000001"],
                "side": ["buy"],
                "price": [10.0],
                "weight": [0.5],
                "reason": ["上穿"],
            }
        ),
        positions=pd.DataFrame({"date": dates, "symbol": "000001", "weight": [0, 0.5, 0.5, 0.5]}),
        indicators=pd.DataFrame({"date": dates}),
    )
    events = pd.DataFrame(
        {
            "date": [dates[1]],
            "title": ["两会"],
            "region": ["CN"],
            "impact": ["high"],
            "summary": ["测试"],
            "sectors": [None],
            "symbols": [None],
        }
    )
    same = EventGuard(events, mode="off").apply(result)
    assert same.positions["weight"].tolist() == [0, 0.5, 0.5, 0.5]

    blocked = EventGuard(events, mode="block_entries").apply(result)
    assert blocked.positions["weight"].tolist() == [0, 0.0, 0.5, 0.5]
    assert blocked.signals.iloc[0]["date"] == dates[2]
    assert blocked.signals.iloc[0]["side"] == "buy"
    assert blocked.signals.iloc[0]["weight"] == 0.5


def test_event_switch_samples_are_reproducible():
    samples = event_switch_samples()
    entry = samples["entry_on_event"]
    day = entry.loc[entry["event"]].iloc[0]
    assert day["off"] == 0.5
    assert day["block_entries"] == 0.0
    assert day["reduce"] == 0.0
    holding = samples["holding_through_event"]
    mid = holding.loc[holding["event"]].iloc[0]
    assert mid["off"] == 0.5
    assert mid["block_entries"] == 0.5
    assert mid["reduce"] == 0.0
    after = holding.shift(-1).loc[holding["event"]].iloc[0]
    assert after["reduce"] == 0.5
