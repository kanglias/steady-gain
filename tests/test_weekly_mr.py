from __future__ import annotations

import pandas as pd
import pytest

from backtest import cost_sensitivity, format_weekly_report, run_weekly_backtest
from scripts.weekly_mr import parse_args, selected_symbols
from backtest.weekly import _trade_cost
from orders import export_weekly_orders
from strategy import WeeklyMeanReversion, available_strategies
from strategy.weekly_mr import is_beijing, project_weights, rolling_zscore, week_end_dates


def test_zscore_uses_sample_std_and_eps():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    score = rolling_zscore(close, window=3, eps=1e-8)
    # [1,2,3] 均值 2，样本标准差 1，z = (3-2)/(1+eps)
    assert score.iloc[2] == pytest.approx(1.0 / (1.0 + 1e-8))
    flat = rolling_zscore(pd.Series([4.0, 4.0, 4.0, 4.0]), window=3, eps=1e-8)
    assert flat.iloc[-1] == pytest.approx(0.0)


def test_week_end_is_last_session_not_a_missing_friday():
    dates = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-15"])
    ends = week_end_dates(dates)
    assert pd.Timestamp("2024-01-11") in ends
    assert pd.Timestamp("2024-01-12") not in ends
    assert pd.Timestamp("2024-01-15") in ends


def test_project_weights_caps_name_industry_and_gross():
    names = [f"{i:06d}" for i in range(1, 5)]
    industries = {name: "银行" for name in names}
    score = {name: float(4 - i) for i, name in enumerate(names)}
    weights, degraded = project_weights(
        names,
        score,
        industries,
        raw_each=0.25,
        w_name_max=0.10,
        w_industry_max=0.30,
        w_gross_max=0.80,
    )
    assert degraded is False
    assert sum(weights.values()) == pytest.approx(0.30)
    assert max(weights.values()) <= 0.10 + 1e-12
    assert names[-1] not in weights

    wide = [f"{i:06d}" for i in range(10)]
    weights, _ = project_weights(
        wide,
        {name: 1.0 for name in wide},
        {name: f"行业{i}" for i, name in enumerate(wide)},
        raw_each=0.10,
        w_name_max=0.10,
        w_industry_max=0.30,
        w_gross_max=0.80,
    )
    assert sum(weights.values()) == pytest.approx(0.80)
    assert max(weights.values()) == pytest.approx(0.08)


def test_missing_industry_degrades_to_name_and_gross(caplog):
    names = ["000001", "000002", "000003", "000004"]
    weights, degraded = project_weights(
        names,
        {name: 1.0 for name in names},
        {name: None for name in names},
        raw_each=0.25,
        w_name_max=0.10,
        w_industry_max=0.30,
        w_gross_max=0.80,
    )
    assert degraded is True
    assert sum(weights.values()) == pytest.approx(0.40)
    strategy = _loose(max_names=4)
    book = strategy.run_portfolio(_panel(names, close_path=_friday_dips(), industry=None))
    assert book.industry_degraded is True
    assert book.notes
    assert "降级" in book.notes[0]


def test_entry_exit_and_hold_limit_and_filters():
    dipped = _panel(["000001"], close_path=_crash_then_flat())
    entered = _loose(max_hold_weeks=8, max_names=1).run_portfolio(dipped)
    assert "enter" in set(entered.orders["signal"])
    assert (entered.orders["signal"] == "enter").sum() >= 1
    held_out = _loose(max_hold_weeks=2, max_names=1).run_portfolio(dipped)
    exits = held_out.orders.loc[held_out.orders["signal"].eq("exit")]
    assert not exits.empty

    blocked = _panel(["830001", "000001"], close_path=_crash_then_flat())
    blocked.loc[blocked["symbol"].eq("000002"), "name"] = "*ST测试"
    st = _panel(["000002"], close_path=_crash_then_flat(), name="*ST测试")
    suspended = _panel(["000001"], close_path=_crash_then_flat(), suspend=True)
    bj = _loose(max_names=5).run_portfolio(blocked)
    assert is_beijing("830001")
    assert "830001" not in set(bj.orders["code"])
    assert _loose().run_portfolio(st).orders.empty or set(_loose().run_portfolio(st).orders["signal"]) <= {"exit"}
    assert _loose().run_portfolio(suspended).orders.empty


def test_weekly_cost_matches_journal_per_fill():
    rates = dict(commission_rate=0.00025, stamp_tax_rate=0.0005, min_commission=5.0)
    buy_small, _ = _trade_cost({}, {"000001": 0.1}, 100_000, **rates)
    assert buy_small == pytest.approx(5.0)
    buy_large, _ = _trade_cost({}, {"000001": 1.0}, 100_000, **rates)
    assert buy_large == pytest.approx(25.0)
    sell, _ = _trade_cost({"000001": 0.1}, {}, 100_000, **rates)
    assert sell == pytest.approx(10.0)
    two_buys, _ = _trade_cost({}, {"000001": 0.01, "000002": 0.01}, 100_000, **rates)
    assert two_buys == pytest.approx(10.0)


def test_cost_tiers_and_report_labels():
    panel = _panel(["000001", "000002"], close_path=_crash_then_flat())
    strategy = _loose(max_names=2)
    table = cost_sensitivity(panel, strategy, start="2024-01-01", split="2024-02-01")
    assert table["cost_mult"].tolist() == [0.5, 1.0, 2.0]
    cheap = table.loc[table["cost_mult"].eq(0.5), "total_return"].iloc[0]
    base = table.loc[table["cost_mult"].eq(1.0), "total_return"].iloc[0]
    rich = table.loc[table["cost_mult"].eq(2.0), "total_return"].iloc[0]
    assert cheap >= base >= rich
    result = run_weekly_backtest(panel, strategy, start="2024-01-01", split="2024-02-01", benchmark=None)
    text = format_weekly_report(result)
    assert "无风险利率按 0" in text
    assert "前复权" in text
    assert not result.equity.empty
    assert result.benchmark["available"] is False
    assert "缺少沪深300缓存" in text
    dates = pd.to_datetime(panel["date"]).drop_duplicates().sort_values()
    bench = pd.DataFrame({"date": dates, "close": [10.0 + index * 0.1 for index in range(len(dates))]})
    marked = run_weekly_backtest(panel, strategy, start="2024-01-01", split="2024-02-01", benchmark=bench)
    assert marked.benchmark["available"] is True
    assert marked.benchmark["excess"] == pytest.approx(marked.stats["total_return"] - marked.benchmark["total_return"])
    assert "相对超额" in format_weekly_report(marked)


def test_weekly_order_export_uses_six_digit_code(tmp_path):
    orders = pd.DataFrame(
        [
            {
                "asof_date": "2024-01-12",
                "trade_date": "2024-01-15",
                "code": "600519.SH",
                "z_score": -1.4,
                "signal": "enter",
                "target_weight": 0.08,
                "raw_weight": 0.10,
                "industry": "食品饮料",
                "next_open_ref": 1500.0,
                "preview_notional": 8000.0,
            }
        ]
    )
    csv_path, md_path = export_weekly_orders(orders, tmp_path / "weekly")
    text = csv_path.read_text(encoding="utf-8")
    assert "600519" in text
    assert "600519.SH" not in text
    assert "asof_date" in text and "next_open_ref" in text
    assert "不会发送任何委托" in md_path.read_text(encoding="utf-8")


def test_sample_watch_resolves_etfs_and_unknown_sample_exits(capsys):
    args = parse_args(["--sample", "watch"])
    assert selected_symbols(args) == ["159338", "513010"]
    with pytest.raises(SystemExit) as exclusive:
        selected_symbols(parse_args(["--sample", "watch", "--symbols", "000001"]))
    assert "只能选一个" in str(exclusive.value)
    with pytest.raises(SystemExit) as unknown:
        parse_args(["--sample", "not-a-sample"])
    assert unknown.value.code != 0
    error = capsys.readouterr().err
    assert "not-a-sample" in error
    assert "watch" in error


def test_strategy_is_registered_for_one_symbol():
    assert "WeeklyMeanReversion" in available_strategies()
    dates = pd.bdate_range("2024-01-02", periods=30)
    close = [10.0] * 20 + [6.0] * 10
    bars = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": close,
            "close": close,
        }
    )
    result = _loose().run(bars)
    assert "z" in result.indicators.columns
    assert set(result.positions["weight"].unique()) <= {0.0, 0.10}


def _loose(**overrides) -> WeeklyMeanReversion:
    params = dict(
        window=5,
        min_listed_days=5,
        min_adv20=1.0,
        min_close=1.0,
        max_names=10,
        max_hold_weeks=8,
        w_name_max=0.10,
        w_industry_max=0.30,
        w_gross_max=0.80,
    )
    params.update(overrides)
    return WeeklyMeanReversion(**params)


def _friday_dips(n: int = 50) -> list[float]:
    """工作日收在 10，周五收到 6，使信号日的 z 保持在入场线以下。"""
    dates = pd.bdate_range("2024-01-02", periods=n)
    return [6.0 if day.day_of_week == 4 else 10.0 for day in dates]


def _crash_then_flat() -> list[float]:
    return _friday_dips()


def _panel(symbols, close_path=None, industry="银行", name="", suspend=False, amount=30_000_000.0):
    path = close_path or [10.0] * 40
    dates = pd.bdate_range("2024-01-02", periods=len(path))
    rows = []
    for code in symbols:
        for day, close in zip(dates, path, strict=True):
            rows.append(
                {
                    "date": day,
                    "symbol": code,
                    "open": close,
                    "close": close,
                    "amount": amount,
                    "industry": industry,
                    "listed_date": "2010-01-04",
                    "is_st": False,
                    "suspend": suspend,
                    "name": name,
                }
            )
    return pd.DataFrame(rows)
