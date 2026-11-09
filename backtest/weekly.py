"""周频组合回测。信号日收盘决定权重，下一交易日开盘换仓。

费用和 journal.apply_cost_schedule 同一公式：每一笔买卖的佣金是
max(名义金额 × 费率, 最低佣金)，印花税默认只按卖出名义金额 × 税率另计。
三档敏感性把佣金费率、印花税率和最低佣金一起乘 0.5、1、2。
Sharpe 的无风险利率按 0 计算，报告里会标明。
这里的收益不是承诺，也不拿日频「每月约 3%」来验收。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.benchmark import csi300_excess
from data.normalize import normalize_symbol
from strategy.weekly_mr import WeeklyMeanReversion, WeeklyPortfolio

DEFAULT_COMMISSION = 0.00025
DEFAULT_STAMP = 0.0005
DEFAULT_MIN_COMMISSION = 5.0
TRADING_DAYS = 252
WEEKS_PER_YEAR = 52


@dataclass
class WeeklyBacktest:
    equity: pd.DataFrame
    stats: dict[str, float]
    train: dict[str, float]
    test: dict[str, float]
    buy_hold: dict[str, float]
    benchmark: dict[str, float]
    portfolio: WeeklyPortfolio
    cost_mult: float
    notes: list[str] = field(default_factory=list)


def run_weekly_backtest(
    panel: pd.DataFrame,
    strategy: WeeklyMeanReversion | None = None,
    *,
    start: str = "2018-01-01",
    end: str | None = None,
    split: str = "2022-01-01",
    cost_mult: float = 1.0,
    equity0: float = 100_000,
    benchmark: pd.DataFrame | None = None,
    commission_rate: float = DEFAULT_COMMISSION,
    stamp_tax_rate: float = DEFAULT_STAMP,
    min_commission: float = DEFAULT_MIN_COMMISSION,
) -> WeeklyBacktest:
    """从 start 跑到行情最后一天。训练段是 split 之前，测试段从 split 当天起。"""
    book = strategy or WeeklyMeanReversion()
    end_ts = pd.Timestamp(end).normalize() if end is not None else pd.to_datetime(panel["date"]).max()
    portfolio = book.run_portfolio(panel, start=start, end=end_ts)
    equity, turnovers = _mark_to_market(
        panel,
        portfolio.orders,
        start=pd.Timestamp(start).normalize(),
        end=end_ts,
        equity0=float(equity0),
        commission_rate=float(commission_rate) * float(cost_mult),
        stamp_tax_rate=float(stamp_tax_rate) * float(cost_mult),
        min_commission=float(min_commission) * float(cost_mult),
    )
    stats = _stats(equity, turnovers)
    train = _stats(*_slice(equity, turnovers, end=pd.Timestamp(split).normalize()))
    test = _stats(*_slice(equity, turnovers, start=pd.Timestamp(split).normalize()))
    hold = _buy_hold(panel, pd.Timestamp(start).normalize(), end_ts)
    compared = csi300_excess(stats["total_return"], benchmark, start=pd.Timestamp(start), end=end_ts)
    bench = {
        "total_return": float(compared["benchmark_return"]),
        "excess": float(compared["excess"]),
        "available": bool(compared["available"]),
    }
    notes = list(portfolio.notes)
    notes.append("Sharpe 的无风险利率按 0 计算。")
    notes.append("价格沿用传入行情的复权；仓库默认日线是前复权。")
    if not bench["available"]:
        notes.append(str(compared["message"]))
    return WeeklyBacktest(
        equity=equity,
        stats=stats,
        train=train,
        test=test,
        buy_hold=hold,
        benchmark=bench,
        portfolio=portfolio,
        cost_mult=float(cost_mult),
        notes=notes,
    )


def cost_sensitivity(
    panel: pd.DataFrame,
    strategy: WeeklyMeanReversion | None = None,
    *,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    start: str = "2018-01-01",
    end: str | None = None,
    split: str = "2022-01-01",
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """默认费率的 0.5 倍、1 倍、2 倍。每档给出测试期年化、回撤和年化换手。"""
    rows = []
    for mult in multipliers:
        result = run_weekly_backtest(
            panel,
            strategy,
            start=start,
            end=end,
            split=split,
            cost_mult=mult,
            benchmark=benchmark,
        )
        rows.append(
            {
                "cost_mult": mult,
                "total_return": result.stats["total_return"],
                "test_annual_return": result.test["annual_return"],
                "test_max_drawdown": result.test["max_drawdown"],
                "test_turnover_annual": result.test["turnover_annual"],
                "test_sharpe": result.test["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def format_weekly_report(result: WeeklyBacktest) -> str:
    """给人看的短报告。收益不是承诺。"""
    full = result.stats
    lines = [
        "# 周频均值回归",
        "",
        "信号在当周最后交易日收盘，下一交易日开盘换仓。不会自动下单。",
        "Sharpe 的无风险利率按 0 计算。粗算结果不是每月 3% 的验收。",
        "",
        f"- 全样本累计 {_pct(full['total_return'])}，年化 {_pct(full['annual_return'])}，"
        f"Sharpe {_num(full['sharpe'])}，最大回撤 {_pct(full['max_drawdown'])}，"
        f"Calmar {_num(full['calmar'])}",
        f"- 单边周换手 {_pct(full['turnover_weekly'])}，年化换手 {_pct(full['turnover_annual'])}",
        f"- 训练段（切分日之前）累计 {_pct(result.train['total_return'])}，"
        f"年化 {_pct(result.train['annual_return'])}，回撤 {_pct(result.train['max_drawdown'])}",
        f"- 测试段（切分日起）累计 {_pct(result.test['total_return'])}，"
        f"年化 {_pct(result.test['annual_return'])}，回撤 {_pct(result.test['max_drawdown'])}，"
        f"年化换手 {_pct(result.test['turnover_annual'])}",
        f"- 等权买入持有 {_pct(result.buy_hold['total_return'])}",
        _benchmark_line(result),
        f"- 本档费用倍数 {result.cost_mult:g}",
    ]
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _mark_to_market(
    panel: pd.DataFrame,
    orders: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    equity0: float,
    commission_rate: float,
    stamp_tax_rate: float,
    min_commission: float,
) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, float]]]:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)]
    calendar = list(pd.DatetimeIndex(sorted(frame["date"].unique())))
    prices: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}
    for row in frame.itertuples(index=False):
        prices[(str(row.symbol), pd.Timestamp(row.date))] = (float(row.open), float(row.close))
    switches: dict[pd.Timestamp, dict[str, float]] = {}
    if orders is not None and not orders.empty:
        for trade_date, group in orders.groupby("trade_date", sort=True):
            book = {
                str(row.code): float(row.target_weight)
                for row in group.itertuples(index=False)
                if float(row.target_weight) > 0
            }
            switches[pd.Timestamp(trade_date)] = book
    equity = float(equity0)
    weight: dict[str, float] = {}
    prev_close: dict[str, float] = {}
    rows = [{"date": calendar[0] - pd.Timedelta(days=1), "equity": equity}] if calendar else []
    # 起点单独记在第一天之前不方便切片。下面每个交易日收盘记权益，第一天开盘前是 equity0。
    rows = []
    turnovers: list[tuple[pd.Timestamp, float]] = []
    last_equity = equity
    for day in calendar:
        gap_ret = 0.0
        session_ret = 0.0
        if day in switches:
            new_weight = switches[day]
            gap_ret = _book_return(weight, prev_close, prices, day, leg="gap")
            equity *= 1.0 + gap_ret
            cost, one_way = _trade_cost(
                weight,
                new_weight,
                equity,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax_rate,
                min_commission=min_commission,
            )
            equity = max(0.0, equity - cost)
            turnovers.append((day, one_way))
            weight = new_weight
            session_ret = _book_return(weight, prev_close, prices, day, leg="session")
            equity *= 1.0 + session_ret
        else:
            session_ret = _book_return(weight, prev_close, prices, day, leg="close")
            equity *= 1.0 + session_ret
        for (code, stamp), (open_, close) in prices.items():
            if stamp == day:
                prev_close[code] = close
        daily = 0.0 if last_equity == 0 else equity / last_equity - 1.0
        last_equity = equity
        rows.append({"date": day, "equity": equity, "daily_return": daily})
    curve = pd.DataFrame(rows)
    if curve.empty:
        curve = pd.DataFrame(columns=["date", "equity", "daily_return"])
    return curve, turnovers


def _book_return(weight, prev_close, prices, day, *, leg: str) -> float:
    total = 0.0
    for code, held in weight.items():
        if held == 0 or code not in prev_close:
            continue
        quote = prices.get((code, day))
        if quote is None:
            continue
        open_, close = quote
        base = prev_close[code]
        if base <= 0:
            continue
        if leg == "gap":
            total += held * (open_ / base - 1.0)
        elif leg == "session":
            if open_ <= 0:
                continue
            total += held * (close / open_ - 1.0)
        else:
            total += held * (close / base - 1.0)
    return float(total)


def _trade_cost(old, new, equity, *, commission_rate, stamp_tax_rate, min_commission) -> tuple[float, float]:
    """和手工成交同一公式。每只股票的加减仓算一笔，佣金取 max(名义×费率, 最低佣金)。"""
    rows = []
    changed = 0.0
    for code in set(old) | set(new):
        delta = float(new.get(code, 0.0)) - float(old.get(code, 0.0))
        changed += abs(delta)
        if delta > 1e-12:
            rows.append(_cost_row(code, "buy", delta * equity))
        elif delta < -1e-12:
            rows.append(_cost_row(code, "sell", -delta * equity))
    one_way = changed / 2.0
    if not rows:
        return 0.0, float(one_way)
    # 放到函数里，避免 backtest 包初始化时和 journal.report 互相导入。
    from journal.log import apply_cost_schedule

    fills = apply_cost_schedule(
        pd.DataFrame(rows),
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        min_commission=min_commission,
        stamp_on="sell",
    )
    cost = float(fills["commission"].sum() + fills["stamp_tax"].sum())
    return cost, float(one_way)


def _cost_row(code: str, side: str, notional: float) -> dict[str, object]:
    return {
        "symbol": code,
        "side": side,
        "price": 1.0,
        "quantity": float(notional),
        "timestamp": "2024-01-02T09:30:00",
        "signal_date": "2024-01-02",
    }


def _slice(equity: pd.DataFrame, turnovers, *, start=None, end=None):
    if equity.empty:
        return equity, []
    frame = equity
    if start is not None:
        frame = frame.loc[frame["date"] >= start]
    if end is not None:
        frame = frame.loc[frame["date"] < end]
    picked = [(day, value) for day, value in turnovers if (start is None or day >= start) and (end is None or day < end)]
    return frame.reset_index(drop=True), picked


def _stats(equity: pd.DataFrame, turnovers) -> dict[str, float]:
    empty = {
        "total_return": float("nan"),
        "annual_return": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "calmar": float("nan"),
        "turnover_weekly": float("nan"),
        "turnover_annual": float("nan"),
    }
    if equity is None or equity.empty or "equity" not in equity.columns:
        return empty
    values = equity["equity"].astype(float)
    if len(values) < 1 or float(values.iloc[0]) < 0:
        return empty
    returns = equity["daily_return"].astype(float).fillna(0.0) if "daily_return" in equity.columns else values.pct_change().fillna(0.0)
    total = float((1.0 + returns).prod() - 1.0)
    days = max(1, len(returns))
    annual = float((1.0 + total) ** (TRADING_DAYS / days) - 1.0) if total > -1 else float("nan")
    std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std and std == std and std > 0 else float("nan")
    peak = values.cummax()
    ratio = values / peak - 1.0
    drawdown = float(max(0.0, -ratio.min())) if len(ratio) else float("nan")
    calmar = float(annual / drawdown) if drawdown and drawdown == drawdown and drawdown > 0 else float("nan")
    weekly = [value for _, value in turnovers]
    turnover_weekly = float(np.mean(weekly)) if weekly else 0.0
    return {
        "total_return": total,
        "annual_return": annual,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "turnover_weekly": turnover_weekly,
        "turnover_annual": turnover_weekly * WEEKS_PER_YEAR,
    }


def _buy_hold(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)]
    if frame.empty:
        return {"total_return": float("nan")}
    pivot = frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    base = pivot.iloc[0]
    usable = base[base > 0].index
    if len(usable) == 0 or len(pivot) < 2:
        return {"total_return": float("nan")}
    relative = pivot[usable].div(base[usable])
    path = relative.mean(axis=1)
    return {"total_return": float(path.iloc[-1] / path.iloc[0] - 1.0)}


def _benchmark_line(result: WeeklyBacktest) -> str:
    bench = result.benchmark
    if bench.get("available"):
        return (
            f"- 沪深300 {_pct(bench['total_return'])}，"
            f"相对超额 {_pct(bench['excess'])}"
        )
    return "- 沪深300 —。缺少沪深300缓存，暂无超额。策略净值不受影响。"


def _pct(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:.2%}"


def _num(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:.2f}"
