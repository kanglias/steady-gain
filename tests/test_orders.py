from __future__ import annotations

import pandas as pd
import pytest

from orders import BannedSymbolError, export_order_list, load_banned


def _board():
    return pd.DataFrame(
        {
            "symbol": ["000001", "600519"],
            "side": ["buy", "sell"],
            "date": ["2024-04-03", "2024-04-12"],
            "strategy": ["dual_ma_10_30", "dual_ma_10_30"],
            "reason": ["确认上穿", "止损"],
            "price": [8.29, 7.89],
            "suggested_weight": [0.5, 0.0],
            "holding_days": [0, 4],
            "unrealized_pnl": [-0.02, pd.NA],
            "consecutive_losses": [0, 1],
            "days_since_cross": [pd.NA, 7],
            "flag": ["", "短持翻脸"],
        }
    )


def test_export_writes_csv_and_markdown(tmp_path):
    csv_path, md_path = export_order_list(_board(), tmp_path / "orders", banned=set())
    text = csv_path.read_text(encoding="utf-8")
    assert "suggested_weight" in text
    assert "holding_days" in text
    assert "unrealized_pnl" in text
    assert "consecutive_losses" in text
    assert "days_since_cross" in text
    assert "短持翻脸" in text
    markdown = md_path.read_text(encoding="utf-8")
    assert "不会发送任何委托" in markdown
    assert "建议仓位" in markdown
    assert "短持翻脸" in markdown
    assert "000001" in markdown
    assert "止损" in text
    assert "确认上穿" in text
    assert "pause_advice" in text


def test_pause_advice_when_losses_or_flips_repeat(tmp_path):
    board = _board()
    board.loc[0, "consecutive_losses"] = 3
    csv_path, md_path = export_order_list(board, tmp_path / "paused", banned=set())
    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    assert "建议暂停加仓" in csv_text
    assert "仍持有，不是新开仓" in csv_text
    assert "建议暂停加仓" in md_text
    assert "仍持有，不是新开仓" in md_text
    quiet = _board()
    quiet["flag"] = ""
    quiet["consecutive_losses"] = 1
    _, md_quiet = export_order_list(quiet, tmp_path / "quiet", banned=set(), pause_after_losses=3)
    assert "建议暂停加仓" not in md_quiet.read_text(encoding="utf-8")
    flips = _board()
    flips["consecutive_losses"] = 0
    extra = flips.iloc[[1]].copy()
    extra["date"] = "2024-04-20"
    flips = pd.concat([flips, extra], ignore_index=True)
    _, md_flips = export_order_list(flips, tmp_path / "flips", banned=set(), pause_after_losses=9)
    text = md_flips.read_text(encoding="utf-8")
    assert "建议暂停加仓" in text
    assert "短持翻脸" in text


def test_pause_stamps_only_the_latest_row(tmp_path):
    board = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000001"],
            "side": ["buy", "buy", "sell"],
            "date": ["2024-01-02", "2024-03-01", "2024-04-03"],
            "strategy": ["dual_ma_10_30", "dual_ma_10_30", "dual_ma_10_30"],
            "reason": ["旧买", "再买", "卖出"],
            "price": [10.0, 11.0, 9.0],
            "suggested_weight": [0.5, 0.5, 0.0],
            "consecutive_losses": [1, 2, 3],
            "flag": ["", "", ""],
        }
    )
    csv_path, md_path = export_order_list(board, tmp_path / "latest", banned=set())
    frame = pd.read_csv(csv_path)
    pauses = frame["pause_advice"].fillna("").astype(str).tolist()
    assert pauses[0] == ""
    assert pauses[1] == ""
    assert "建议暂停加仓" in pauses[2]
    assert "当前空仓" in pauses[2]
    markdown = md_path.read_text(encoding="utf-8")
    assert "**建议暂停加仓**" in markdown
    assert "当前空仓" in markdown
    assert markdown.count("建议暂停加仓") == 2


def test_banned_symbol_writes_nothing(tmp_path):
    with pytest.raises(BannedSymbolError):
        export_order_list(_board(), tmp_path / "orders", banned={"SZ000001"})
    assert list(tmp_path.glob("orders*")) == []


def test_default_banned_file_is_empty():
    assert load_banned() == set()


def test_banned_file_rejects_the_whole_export(tmp_path, monkeypatch):
    banned_file = tmp_path / "banned.txt"
    banned_file.write_text("# 注释\n000001\n", encoding="utf-8")
    monkeypatch.setattr("orders.export.DEFAULT_BANNED_FILE", banned_file)
    with pytest.raises(BannedSymbolError):
        export_order_list(_board(), tmp_path / "orders")
    assert list(tmp_path.glob("orders*")) == []
