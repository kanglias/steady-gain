from __future__ import annotations

import pandas as pd
import pytest

from journal import FILL_OUTPUT_COLUMNS, append_fills, load_fills, save_fills, simulate_follow_fills
from journal.log import FillLogError


def _rows():
    return pd.DataFrame(
        {
            "symbol": ["sz000001", "600519"],
            "side": ["buy", "sell"],
            "price": [11.2, 1400],
            "quantity": [100, 10],
            "timestamp": ["2024-01-03 09:30", "2024-01-08T14:55:00"],
            "signal_date": ["2024-01-02", "2024-01-05"],
        }
    )


def test_save_and_load_csv_and_json(tmp_path):
    csv_path, json_path = save_fills(_rows(), tmp_path / "fills")
    from_csv = load_fills(csv_path)
    from_json = load_fills(json_path)
    assert list(from_csv.columns) == FILL_OUTPUT_COLUMNS
    assert from_csv["commission"].tolist() == [0.0, 0.0]
    assert from_csv["is_simulated"].tolist() == [False, False]
    assert from_csv["symbol"].tolist() == ["000001", "600519"]
    assert from_csv["timestamp"].tolist() == ["2024-01-03T09:30:00", "2024-01-08T14:55:00"]
    assert from_json["signal_date"].tolist() == ["2024-01-02", "2024-01-05"]
    assert from_json["price"].tolist() == [11.2, 1400.0]
    assert "place_order" not in json_path.read_text(encoding="utf-8")


def test_simulated_fill_uses_next_open_but_sizes_on_the_signal():
    dates = pd.bdate_range("2024-01-02", periods=3)
    bars = pd.DataFrame(
        {"date": dates, "symbol": "000001", "open": [10.0, 12.0, 11.0], "close": [10.0, 11.0, 11.0]}
    )
    signals = pd.DataFrame(
        {
            "date": [dates[0]],
            "symbol": ["000001"],
            "side": ["buy"],
            "price": [10.0],
            "weight": [0.5],
        }
    )
    fills = simulate_follow_fills(bars, signals, equity=1000)
    assert fills.iloc[0]["price"] == 12
    assert fills.iloc[0]["quantity"] == 50
    assert fills.iloc[0]["signal_date"] == "2024-01-02"
    assert bool(fills.iloc[0]["is_simulated"]) is True


def test_simulated_flag_is_forced_on_save(tmp_path):
    frame = _rows()
    frame["is_simulated"] = False
    csv_path, _ = save_fills(frame, tmp_path / "demo", simulated=True)
    loaded = load_fills(csv_path)
    assert loaded["is_simulated"].tolist() == [True, True]
    assert "true" in csv_path.read_text(encoding="utf-8").lower()
    demo = simulate_follow_fills(
        pd.DataFrame(
            {
                "date": pd.bdate_range("2024-02-01", periods=2),
                "symbol": "000001",
                "open": [10.0, 10.5],
                "close": [10.0, 10.4],
            }
        ),
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-02-01")],
                "symbol": ["000001"],
                "side": ["buy"],
                "price": [10.0],
                "weight": [1.0],
            }
        ),
        equity=1000,
    )
    assert bool(demo.iloc[0]["is_simulated"]) is True


def test_append_keeps_existing_rows(tmp_path):
    save_fills(_rows().iloc[:1], tmp_path / "fills")
    append_fills(_rows().iloc[1:], tmp_path / "fills")
    loaded = load_fills(tmp_path / "fills")
    assert len(loaded) == 2
    assert loaded["side"].tolist() == ["buy", "sell"]


@pytest.mark.parametrize(
    "patch",
    [
        {"side": "hold"},
        {"price": 0},
        {"quantity": -1},
        {"symbol": ""},
        {"timestamp": "not-a-time"},
        {"signal_date": ""},
    ],
)
def test_rejects_bad_rows(patch):
    frame = _rows().iloc[:1].copy()
    column, value = next(iter(patch.items()))
    frame[column] = value
    with pytest.raises(FillLogError):
        save_fills(frame, "/tmp/ignored-fills")
