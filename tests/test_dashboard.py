from __future__ import annotations

import pandas as pd
import pytest

from dashboard.view import (
    EVENT_FOOTNOTE,
    PAGE_BANNER,
    aggregate_weekly_bars,
    backtest_cards,
    chart_heading,
    dashboard_chart_layout,
    estimate_buy_shares,
    event_card_html,
    event_impact_badge,
    event_region_label,
    event_tags,
    filter_events,
    format_holding_days,
    make_strategy,
    order_preview,
    signal_table,
    strategies_for,
    weekly_fee_note,
    weekly_order_preview,
    weekly_order_rows,
    weekly_rule_note,
    win_rate_from_signals,
)
from strategy.base import StrategyResult
from backtest import FOLLOW_LABEL


def _bars(close):
    index = pd.bdate_range("2024-01-02", periods=len(close))
    frame = pd.DataFrame(
        {
            "date": index,
            "open": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "symbol": "000001",
        }
    )
    return frame


def test_debounce_and_risk_toggles_drop_filters():
    params = {
        "short_window": 3,
        "long_window": 5,
        "trend_window": 8,
        "confirm_days": 2,
        "min_hold_days": 4,
        "target_weight": 0.5,
        "stop_loss": 0.08,
        "adx_window": 14,
        "adx_min": 20,
    }
    off = make_strategy("dual_ma", debounce=False, risk=False, params=params)
    assert off.confirm_days == 1
    assert off.min_hold_days == 1
    assert off.stop_loss is None
    assert off.trend_window is None
    assert off.adx_window is None
    on = make_strategy("dual_ma", debounce=True, risk=True, params=params)
    assert on.confirm_days == 2
    assert on.stop_loss == 0.08


def test_event_filters_are_facets():
    events = pd.DataFrame(
        [
            {
                "date": "2024-03-05",
                "title": "全国两会开幕",
                "region": "CN",
                "impact": "high",
                "summary": "宏观政策",
                "symbols": None,
            },
            {
                "date": "2024-06-12",
                "title": "美联储议息",
                "region": "US",
                "impact": "high",
                "summary": "海外利率",
                "symbols": None,
            },
            {
                "date": "2024-08-30",
                "title": "贵州茅台中报",
                "region": "CN",
                "impact": "medium",
                "summary": "业绩说明",
                "symbols": ["600519"],
            },
        ]
    )
    assert "宏观" in event_tags(events.iloc[0])
    assert filter_events(events, "海外")["title"].tolist() == ["美联储议息"]
    assert filter_events(events, "财报")["title"].tolist() == ["贵州茅台中报"]
    related = filter_events(events, "全部", symbol="600519", related_only=True)
    assert set(related["title"]) == {"全国两会开幕", "美联储议息", "贵州茅台中报"}
    only = filter_events(events, "财报", symbol="000001", related_only=True)
    assert only.empty


def test_win_rate_ignores_open_trade_and_follow_view_is_next_open():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-08", "2024-01-15", "2024-01-20"]),
            "side": ["buy", "sell", "buy", "sell"],
            "price": [10.0, 12.0, 10.0, 9.0],
        }
    )
    assert win_rate_from_signals(signals) == 0.5
    close = [10, 10, 11, 11, 12, 12, 9, 9, 10, 10, 10, 10]
    bars = _bars(close)
    strategy = make_strategy(
        "mean_reversion",
        debounce=False,
        risk=False,
        params={
            "window": 3,
            "entry_band": 0.05,
            "confirm_days": 1,
            "min_hold_days": 1,
            "target_weight": 1.0,
            "stop_loss": 0.2,
        },
    )
    cards = backtest_cards(bars, strategy.run(bars))
    assert cards["follow_label"] == FOLLOW_LABEL
    assert cards["next_open"]["fill"] == "next_open"
    assert "total_return" in cards["close"]


def test_order_preview_keeps_weight_as_percent_and_estimates_shares():
    board = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "side": ["buy", "sell"],
            "suggested_weight": [0.5, 0.0],
            "price": [10.0, 11.0],
            "reason": ["确认", "离场"],
            "flag": ["", "短持翻脸"],
            "pause_advice": ["", "建议暂停加仓；当前空仓"],
        }
    )
    preview = order_preview(board, {"000001": "平安银行"}, equity=10_000)
    assert preview.iloc[0]["建议仓位"] == "50%"
    assert estimate_buy_shares(10_000, 0.5, 10) == 500
    assert preview.iloc[0]["估算股数"] == "500"
    assert preview.iloc[0]["约合手数"] == "5"
    assert preview.iloc[1]["估算股数"] == ""
    assert "不代算" in preview.iloc[1]["备注"]
    assert "短持翻脸" in preview.iloc[1]["备注"]


def test_high_price_buy_says_not_enough_for_one_lot():
    board = pd.DataFrame(
        {
            "symbol": ["600519"],
            "side": ["buy"],
            "suggested_weight": [0.5],
            "price": [1500.0],
            "reason": ["确认"],
            "flag": [""],
            "pause_advice": [""],
        }
    )
    preview = order_preview(board, {"600519": "贵州茅台"}, equity=100_000)
    assert preview.iloc[0]["操作"] == "买入"
    assert preview.iloc[0]["建议仓位"] == "50%"
    assert preview.iloc[0]["估算股数"] == "资金不足一手"
    assert preview.iloc[0]["约合手数"] == "资金不足一手"
    assert "资金不足一手" in preview.iloc[0]["备注"]
    assert preview.iloc[0]["预估金额"] == ""


def test_signal_table_holding_days_are_integers():
    assert format_holding_days(6.0) == "6天"
    assert format_holding_days("6.000000") == "6天"
    assert format_holding_days(pd.NA) == ""
    board = pd.DataFrame(
        {
            "symbol": ["000001"],
            "side": ["buy"],
            "suggested_weight": [0.5],
            "holding_days": [6.0],
            "unrealized_pnl": [0.01],
            "price": [10.0],
            "flag": [""],
        }
    )
    table = signal_table(board, {"000001": "平安银行"})
    assert table.iloc[0]["持有天数"] == "6天"
    assert "000000" not in table.iloc[0]["持有天数"]
    preview = order_preview(board, {"000001": "平安银行"}, equity=10_000)
    assert preview.iloc[0]["持有天数"] == "6天"


def test_chart_heading_is_human_and_legend_stays_off_the_title():
    strategy = make_strategy(
        "dual_ma",
        debounce=True,
        risk=True,
        params={
            "short_window": 10,
            "long_window": 30,
            "trend_window": 60,
            "confirm_days": 2,
            "min_hold_days": 5,
            "target_weight": 0.5,
            "stop_loss": 0.08,
            "adx_window": 14,
            "adx_min": 20,
        },
    )
    assert chart_heading("600519", "贵州茅台", strategy) == "600519 贵州茅台 · DualMA 10/30"
    assert "dual_ma_" not in chart_heading("600519", "贵州茅台", strategy)
    layout = dashboard_chart_layout()
    assert layout["title"]["text"] == ""
    assert layout["legend"]["xanchor"] == "right"
    assert layout["margin"]["t"] >= 80


def test_event_card_uses_calendar_fields_and_skips_empty_pills():
    card = event_card_html(
        {
            "date": "2024-09-24",
            "title": "央行宣布一揽子金融政策",
            "region": "CN",
            "impact": "high",
            "summary": "降准同日释放。",
            "sectors": ["银行", "房地产"],
            "symbols": None,
        }
    )
    assert "2024-09-24" in card
    assert "<b>央行宣布一揽子金融政策</b>" in card
    assert event_region_label("CN") == "国内"
    assert event_impact_badge("high") == "HIGH"
    assert "HIGH" in card
    assert "银行" in card
    assert "关联" not in card
    assert EVENT_FOOTNOTE.startswith("本地 calendars/cn_events.json")
    assert PAGE_BANNER.startswith("研究看板。系统不会自动下单")


def _weekly_params():
    return {
        "window": 20,
        "z_entry": 1.0,
        "z_exit": 0.25,
        "max_names": 10,
        "max_hold_weeks": 8,
        "w_name_max": 0.10,
        "w_industry_max": 0.30,
        "w_gross_max": 0.80,
        "preview_capital": 100_000,
    }


def test_frequency_switches_strategy_menu_and_weekly_ignores_daily_toggles():
    assert [key for key, _ in strategies_for("daily")] == ["dual_ma", "mean_reversion"]
    assert [key for key, _ in strategies_for("weekly")] == ["weekly_mr"]
    try:
        strategies_for("intraday")
    except KeyError:
        pass
    else:
        raise AssertionError("未知频率应拒绝")
    params = _weekly_params()
    off = make_strategy("weekly_mr", debounce=False, risk=False, params=params)
    on = make_strategy("weekly_mr", debounce=True, risk=True, params=params)
    assert off.name == on.name == "weekly_mean_reversion"
    assert off.w_name_max == on.w_name_max == 0.10
    assert off.w_gross_max == 0.80
    note = weekly_rule_note(on)
    assert "next_open" in note
    assert "单票 10%" in note and "行业 30%" in note and "总仓 80%" in note
    assert "不会自动下单" in note
    fee = weekly_fee_note()
    assert "max(名义×0.00025, 5元)" in fee
    assert "印花税只在卖出" in fee
    assert "仍未扣费" in fee


def test_weekly_bars_use_last_session_and_orders_use_next_open():
    dates = pd.to_datetime(
        ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-15", "2024-01-16"]
    )
    bars = pd.DataFrame(
        {
            "date": dates,
            "open": [10, 11, 12, 13, 14, 15],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5, 16],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
            "close": [10.2, 11.2, 12.2, 13.2, 14.2, 15.2],
            "volume": [1, 1, 1, 1, 2, 2],
            "symbol": "000001",
        }
    )
    weekly = aggregate_weekly_bars(bars)
    assert weekly["date"].tolist() == list(pd.to_datetime(["2024-01-11", "2024-01-16"]))
    assert weekly.iloc[0]["open"] == 10
    assert weekly.iloc[0]["high"] == 13.5
    assert weekly.iloc[0]["low"] == 9.5
    assert weekly.iloc[0]["close"] == 13.2
    assert weekly.iloc[0]["volume"] == 4
    strategy = make_strategy("weekly_mr", debounce=True, risk=True, params=_weekly_params())
    result = StrategyResult(
        signals=pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-11"), pd.Timestamp("2024-01-16")],
                "symbol": ["000001", "000001"],
                "side": ["buy", "sell"],
                "price": [13.2, 15.2],
                "reason": ["周频超卖", "周频离场"],
            }
        ),
        positions=pd.DataFrame(
            {
                "date": dates,
                "symbol": "000001",
                "weight": [0, 0, 0, 0.1, 0.1, 0],
            }
        ),
        indicators=pd.DataFrame({"date": dates, "z": [None, None, None, -1.4, -0.2, 0.1]}),
    )
    orders = weekly_order_rows(bars, result, strategy)
    assert orders["signal"].tolist() == ["enter", "exit"]
    assert orders.iloc[0]["asof_date"] == "2024-01-11"
    assert orders.iloc[0]["trade_date"] == "2024-01-15"
    assert orders.iloc[0]["target_weight"] == pytest.approx(0.10)
    assert orders.iloc[0]["raw_weight"] == pytest.approx(1.0)
    assert orders.iloc[0]["next_open_ref"] == 14
    assert orders.iloc[0]["preview_notional"] == pytest.approx(10_000)
    assert orders.iloc[0]["z_score"] == pytest.approx(-1.4)
    assert orders.iloc[1]["trade_date"] == "2024-01-17"
    assert pd.isna(orders.iloc[1]["next_open_ref"])
    assert orders.iloc[1]["target_weight"] == 0
    preview = weekly_order_preview(orders.iloc[[0]], {"000001": "平安银行"}, 100_000)
    assert preview.iloc[0]["成交日"] == "2024-01-15"
    assert preview.iloc[0]["开盘参考"] == "14.00"
    assert preview.iloc[0]["目标权重"] == "10%"
    assert preview.iloc[0]["约束前"] == "100%"
    rich = orders.iloc[[0]].copy()
    rich["next_open_ref"] = 1500.0
    rich_preview = weekly_order_preview(rich, {"000001": "平安银行"}, 100_000)
    assert rich_preview.iloc[0]["估算股数"] == "资金不足一手"


def test_rendered_banner_includes_fee_disclaimer():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=90)
    app.run()
    assert list(app.exception) == []
    text = "\n".join(item.value for item in list(app.markdown) + list(app.caption))
    assert PAGE_BANNER in text
    assert "跟单看 next_open" in text
    assert "<b>回测未扣费；对账看成交日志</b>" in text
    assert "000001 平安银行 · DualMA 10/30" in text
    assert "dual_ma_10_30" not in text
    assert EVENT_FOOTNOTE in text
    labels = [item.label for item in app.checkbox]
    assert "仅当前标的相关" in labels
    toggles = [item for item in app.toggle if item.label == "事件日降仓提示"]
    assert len(toggles) == 1
    assert toggles[0].value is False
    source = Path(__file__).resolve().parents[1].joinpath("dashboard", "app.py").read_text(encoding="utf-8")
    assert "events=None" in source


def test_follow_banner_states_backtest_is_before_fees():
    from pathlib import Path

    from backtest import FEE_DISCLAIMER, FOLLOW_LABEL

    root = Path(__file__).resolve().parents[1]
    source = (root / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert FEE_DISCLAIMER == "回测未扣费；对账看成交日志"
    assert "FOLLOW_LABEL" in source
    assert "<b>{FEE_DISCLAIMER}</b>" in source
    assert "follow-lock" in source
    script = (root / "scripts" / "plot_demo.py").read_text(encoding="utf-8")
    assert "FEE_DISCLAIMER" in script
    assert FOLLOW_LABEL == "跟单看 next_open"


def _select(app, label):
    found = [item for item in app.selectbox if item.label == label]
    assert len(found) == 1
    return found[0]


def test_app_switches_frequency_and_strategy():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=90)
    app.run()
    assert list(app.exception) == []
    assert _select(app, "频率").value == "daily"
    assert _select(app, "策略").value == "dual_ma"
    _select(app, "策略").set_value("mean_reversion")
    app.run()
    assert list(app.exception) == []
    assert _select(app, "策略").value == "mean_reversion"
    _select(app, "频率").set_value("weekly")
    app.run()
    assert list(app.exception) == []
    strategy = _select(app, "策略")
    assert strategy.value == "weekly_mr"
    assert "周频均值回归" in list(strategy.options)
    text = "\n".join(item.value for item in list(app.markdown) + list(app.caption))
    assert "周频规则" in text
    assert "next_open" in text
    assert "单票 10%" in text
    assert "max(名义×0.00025, 5元)" in text
    assert "系统不会自动下单" in text
    assert "回测未扣费；对账看成交日志" in text
    assert "周K" in text
    _select(app, "频率").set_value("daily")
    app.run()
    assert list(app.exception) == []
    restored = _select(app, "策略")
    assert restored.value == "mean_reversion"
    assert "周频均值回归" not in list(restored.options)
    assert "均值回归" in list(restored.options)
