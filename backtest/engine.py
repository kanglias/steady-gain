"""单标的收盘仓位回测。没有滑点、费用、涨跌停，也没有成交记录。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategy.base import Strategy, StrategyResult


@dataclass
class BacktestResult:
    """研究用绩效，不是实盘回报，也不是阶段 3 的成交对照。

    fill: `close` 表示信号在当日收盘价成交；`next_open` 表示次日开盘价成交。
    total_return: 期末权益 / 期初权益 - 1。期初权益记为 1。
    max_drawdown: 权益从峰值到谷值的最大跌幅，用正数表示。0.2 是回撤 20%。
    turnover: 目标权重绝对变化之和。满仓进出一轮记为 2；半仓进出一轮记为 1。
        两种成交假设共用同一条权重，换手数相同，收益不同。
    """

    fill: str
    total_return: float
    max_drawdown: float
    turnover: float
    equity: pd.DataFrame
    symbol: str
    n_bars: int

    def rough_monthly_return(self) -> float:
        """把整段收益粗算成单利复利到「每月」。约 21 个交易日算一个月。

        这只是把这一次样本换一种说法，不是稳定月收益，更不是 3% 的验收结果。
        """
        months = self.n_bars / 21
        if months <= 0 or self.total_return <= -1:
            return float("nan")
        return float((1 + self.total_return) ** (1 / months) - 1)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "symbol": self.symbol,
            "fill": self.fill,
            "n_bars": self.n_bars,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "rough_monthly_return": self.rough_monthly_return(),
        }


@dataclass
class FillComparison:
    """同一条目标仓位、两种成交假设。都没有滑点、费用和涨跌停。"""

    close_fill: BacktestResult
    next_open_fill: BacktestResult


def run_backtest(
    bars: pd.DataFrame,
    result: StrategyResult | None = None,
    *,
    strategy: Strategy | None = None,
    fill: str = "close",
) -> BacktestResult:
    """按目标权重做研究回测，并显式选择成交假设。

    `fill="close"`：当天收盘才知道的权重，按收盘价成交，从下一根 K 线的
    收盘到收盘收益开始生效。不会用当日收盘去赚当日已经发生的涨跌。

    `fill="next_open"`：同样在收盘出信号，但成交价改成下一交易日开盘。
    隔夜缺口由旧仓位承担，开盘之后的日内涨跌由新仓位承担。
    跟单时看这个假设，不把收盘价当成已经能成交的价格。

    `fill="next_open_stop"`：止损仍用收盘价触发，不是盘中触发。触发日收盘后仓位还在，
    下一交易日开盘才按旧仓位承担隔夜缺口并退出，当天剩余涨跌不再持有。
    其他买卖仍按收盘成交假设。最后一根 K 线上的止损没有下一根开盘，当天收益与收盘假设相同。

    权重可以是 0 到 1 之间的建议仓位，不要求只能是满仓或空仓。
    这里没有滑点、佣金、印花税，也不处理涨跌停和停牌。前复权价格必须为正。
    这里的「成交」只是价格假设，不会记成成交单。手工成交在 journal 里对照。
    """
    if fill not in {"close", "next_open", "next_open_stop"}:
        raise ValueError("fill 只能是 close、next_open 或 next_open_stop")
    if result is None:
        if strategy is None:
            raise TypeError("传入 StrategyResult，或传入 strategy 以便调用 run")
        result = strategy.run(bars)
    merged = _align(bars, result, need_open=fill in {"next_open", "next_open_stop"})
    weight = merged["weight"]
    if fill == "next_open":
        strategy_return = _next_open_return(merged)
        held = weight.shift(1).fillna(0.0)
    elif fill == "next_open_stop":
        held = weight.shift(1).fillna(0.0)
        strategy_return = held * merged["close"].pct_change().fillna(0.0)
        strategy_return = _apply_next_open_stops(merged, result, strategy_return)
    else:
        held = weight.shift(1).fillna(0.0)
        strategy_return = held * merged["close"].pct_change().fillna(0.0)
    equity = (1.0 + strategy_return).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    changes = weight.diff()
    changes.iloc[0] = weight.iloc[0]
    symbol = _one_symbol(merged)
    curve = pd.DataFrame(
        {
            "date": merged["date"].to_numpy(),
            "close": merged["close"].to_numpy(),
            "weight": weight.to_numpy(),
            "held_weight": held.to_numpy(),
            "strategy_return": strategy_return.to_numpy(),
            "equity": equity.to_numpy(),
        }
    )
    return BacktestResult(
        fill=fill,
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        max_drawdown=float(max(0.0, -drawdown.min())),
        turnover=float(changes.abs().sum()),
        equity=curve,
        symbol=symbol,
        n_bars=int(len(curve)),
    )


def compare_fills(
    bars: pd.DataFrame,
    result: StrategyResult | None = None,
    *,
    strategy: Strategy | None = None,
) -> FillComparison:
    """同一策略结果，分别按收盘成交和次日开盘成交各跑一遍。"""
    if result is None:
        if strategy is None:
            raise TypeError("传入 StrategyResult，或传入 strategy 以便调用 run")
        result = strategy.run(bars)
    return FillComparison(
        close_fill=run_backtest(bars, result, fill="close"),
        next_open_fill=run_backtest(bars, result, fill="next_open"),
    )


FOLLOW_LABEL = "跟单看 next_open"
FEE_DISCLAIMER = "回测未扣费；对账看成交日志"
FILL_LABELS = {
    "close": "收盘成交",
    "next_open": "次日开盘成交",
    "next_open_stop": "次日开盘止损",
}


def fill_assumption_table(
    compared: FillComparison,
    stop_fill: BacktestResult | None = None,
) -> pd.DataFrame:
    """跟单视图锁定为次日开盘。第一行是 next_open，收盘成交只作对照。

    传入 `stop_fill` 时再加一行次日开盘止损。它只改止损的离场假设。
    """
    items = [compared.next_open_fill, compared.close_fill]
    if stop_fill is not None:
        items.append(stop_fill)
    rows = []
    for item in items:
        row = item.as_dict()
        row["label"] = FILL_LABELS.get(item.fill, item.fill)
        row["follow"] = FOLLOW_LABEL if item.fill == "next_open" else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_next_open_stops(
    merged: pd.DataFrame,
    result: StrategyResult,
    strategy_return: pd.Series,
) -> pd.Series:
    """止损日的下一根，用开盘缺口替换收盘到收盘收益。"""
    adjusted = strategy_return.copy()
    stops = _stop_positions(merged, result)
    weight = merged["weight"]
    for i in stops:
        if i + 1 >= len(merged):
            continue
        prev_weight = float(weight.iloc[i - 1]) if i > 0 else 0.0
        gap = float(merged["open"].iloc[i + 1] / merged["close"].iloc[i] - 1.0)
        adjusted.iloc[i + 1] = prev_weight * gap
    return adjusted


def _stop_positions(merged: pd.DataFrame, result: StrategyResult) -> list[int]:
    signals = result.signals
    if signals is None or signals.empty or "reason" not in signals.columns or "date" not in signals.columns:
        return []
    frame = signals.copy()
    if "side" not in frame.columns:
        return []
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    side = frame["side"].astype(str).str.lower()
    reason = frame["reason"].astype(str)
    stopped = frame.loc[side.eq("sell") & reason.str.contains("止损", na=False), "date"]
    days = {pd.Timestamp(day) for day in stopped if pd.notna(day)}
    return [i for i, day in enumerate(merged["date"]) if pd.Timestamp(day) in days]


def _next_open_return(merged: pd.DataFrame) -> pd.Series:
    close = merged["close"]
    open_ = merged["open"]
    weight = merged["weight"].to_numpy(dtype=float)
    prev_close = close.shift(1)
    gap = (open_ / prev_close - 1.0).fillna(0.0)
    session = (close / open_ - 1.0).fillna(0.0)
    overnight = pd.Series(0.0, index=merged.index)
    intraday = pd.Series(0.0, index=merged.index)
    if len(weight) >= 2:
        intraday.iloc[1:] = weight[:-1]
    if len(weight) >= 3:
        overnight.iloc[2:] = weight[:-2]
    return (1.0 + overnight * gap) * (1.0 + intraday * session) - 1.0


def _align(bars: pd.DataFrame, result: StrategyResult, *, need_open: bool) -> pd.DataFrame:
    if "date" not in bars.columns or "close" not in bars.columns:
        raise ValueError("行情需要 date 和 close 列")
    if need_open and "open" not in bars.columns:
        raise ValueError("次日开盘成交需要 open 列")
    positions = result.positions
    if "date" not in positions.columns or "weight" not in positions.columns:
        raise ValueError("仓位需要 date 和 weight 列")
    left = bars.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    left["close"] = pd.to_numeric(left["close"], errors="coerce")
    keep = ["date", "close"]
    if "symbol" in left.columns:
        keep.append("symbol")
    if need_open:
        left["open"] = pd.to_numeric(left["open"], errors="coerce")
        keep.insert(1, "open")
    left = left.loc[:, keep].dropna(subset=["date", "close"])
    right = positions.copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    right["weight"] = pd.to_numeric(right["weight"], errors="coerce")
    right = right.dropna(subset=["date", "weight"])
    merged = left.merge(right[["date", "weight"]], on="date", how="inner")
    merged = merged.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if merged.empty:
        raise ValueError("行情和仓位没有重叠的交易日")
    if need_open and merged["open"].isna().any():
        raise ValueError("次日开盘成交遇到缺失的开盘价")
    if merged["close"].le(0).any() or (need_open and merged["open"].le(0).any()):
        raise ValueError("回测要求价格为正。前复权出现非正价格时，这套收益率不适用。")
    if merged["weight"].lt(0).any() or merged["weight"].gt(1).any():
        raise ValueError("weight 需要在 0 到 1 之间")
    return merged


def _one_symbol(merged: pd.DataFrame) -> str:
    if "symbol" not in merged.columns or not merged["symbol"].notna().any():
        return ""
    symbols = [str(item) for item in merged["symbol"].dropna().unique().tolist()]
    if len(symbols) > 1:
        raise ValueError("回测按单只标的运行，请先按 symbol 拆开")
    return symbols[0]
