"""给信号补上持仓天数、浮动盈亏、连亏和短持翻脸标记。

这些是研究字段，不是成交回报。浮动盈亏用信号价和最近收盘价估算，没有费用。
"""

from __future__ import annotations

import pandas as pd

from strategy.base import StrategyResult

FLIP_LABEL = "短持翻脸"
DIAGNOSTIC_COLUMNS = [
    "suggested_weight",
    "holding_days",
    "unrealized_pnl",
    "consecutive_losses",
    "days_since_cross",
    "flag",
]


def apply_diagnostics(
    result: StrategyResult,
    bars: pd.DataFrame | None = None,
    *,
    flip_days: int = 5,
) -> StrategyResult:
    """返回带诊断列的信号副本。仓位和指标不动。"""
    signals = annotate_signals(result, bars, flip_days=flip_days)
    return StrategyResult(
        signals=signals,
        positions=result.positions.copy(),
        indicators=result.indicators.copy(),
    )


def annotate_signals(
    result: StrategyResult,
    bars: pd.DataFrame | None = None,
    *,
    flip_days: int = 5,
) -> pd.DataFrame:
    signals = result.signals.copy() if result.signals is not None else pd.DataFrame()
    base_columns = ["date", "symbol", "side", "price", "weight", "reason", *DIAGNOSTIC_COLUMNS]
    if signals.empty:
        return pd.DataFrame(columns=base_columns)
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["side"] = frame["side"].astype(str).str.lower()
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "price" not in frame.columns:
        frame["price"] = pd.NA
    if "reason" not in frame.columns:
        frame["reason"] = ""
    if "symbol" not in frame.columns:
        frame["symbol"] = ""

    index_of, weight_of = _position_lookup(result.positions)
    last_close = _last_close(bars)
    holding_days: list[float] = []
    unrealized: list[float] = []
    losses: list[float] = []
    since: list[float] = []
    flags: list[str] = []
    suggested: list[float] = []
    loss_streak = 0
    open_pos: int | None = None
    open_price: float | None = None
    prev_pos: int | None = None

    for loc in range(len(frame)):
        day = pd.Timestamp(frame["date"].iloc[loc])
        pos = index_of.get(day)
        side = frame["side"].iloc[loc]
        price = pd.to_numeric(pd.Series([frame["price"].iloc[loc]]), errors="coerce").iloc[0]
        gap = float(pos - prev_pos) if pos is not None and prev_pos is not None else pd.NA
        since.append(gap)
        if pos is not None:
            prev_pos = pos
        if side == "buy":
            open_pos = pos
            open_price = float(price) if pd.notna(price) else None
            suggested.append(_suggested(frame, loc, weight_of, day, default=pd.NA))
            holding_days.append(0)
            unrealized.append(pd.NA)
            losses.append(loss_streak)
            flags.append("")
        elif side == "sell" and open_pos is not None and pos is not None:
            held = float(pos - open_pos)
            entry = open_price
            leg = (float(price) / entry - 1.0) if entry not in (None, 0) and pd.notna(price) else pd.NA
            if pd.notna(leg) and leg < 0:
                loss_streak += 1
            elif pd.notna(leg):
                loss_streak = 0
            flag = FLIP_LABEL if pd.notna(held) and held <= flip_days else ""
            for back in range(loc - 1, -1, -1):
                if frame["side"].iloc[back] == "buy":
                    holding_days[back] = held
                    if flag:
                        flags[back] = FLIP_LABEL
                    break
            suggested.append(0.0)
            holding_days.append(held)
            unrealized.append(pd.NA)
            losses.append(loss_streak)
            flags.append(flag)
            open_pos = None
            open_price = None
        else:
            suggested.append(_suggested(frame, loc, weight_of, day, default=pd.NA))
            holding_days.append(pd.NA)
            unrealized.append(pd.NA)
            losses.append(loss_streak)
            flags.append("")

    if open_pos is not None and open_price not in (None, 0) and last_close is not None and index_of:
        last_pos = max(index_of.values())
        held_now = float(last_pos - open_pos)
        pnl_now = last_close / open_price - 1.0
        for loc in range(len(frame) - 1, -1, -1):
            if frame["side"].iloc[loc] == "buy":
                holding_days[loc] = held_now
                unrealized[loc] = pnl_now
                break

    frame["suggested_weight"] = suggested
    frame["holding_days"] = holding_days
    frame["unrealized_pnl"] = unrealized
    frame["consecutive_losses"] = losses
    frame["days_since_cross"] = since
    frame["flag"] = flags
    return frame


def position_snapshot(
    result: StrategyResult,
    bars: pd.DataFrame | None,
    *,
    strategy_name: str,
    flip_days: int = 5,
) -> pd.DataFrame:
    """每个标的一行当前状态，方便在没有新信号的日子看浮动盈亏。"""
    annotated = annotate_signals(result, bars, flip_days=flip_days)
    positions = result.positions
    if positions is None or positions.empty:
        return pd.DataFrame()
    last = positions.iloc[-1]
    symbol = str(last["symbol"]) if "symbol" in positions.columns else ""
    as_of = pd.Timestamp(last["date"])
    weight = float(last["weight"])
    open_row = None
    if not annotated.empty:
        buys = annotated.loc[annotated["side"] == "buy"]
        sells = annotated.loc[annotated["side"] == "sell"]
        if weight > 0 and not buys.empty:
            open_row = buys.iloc[-1]
        last_flag = ""
        if not sells.empty and str(sells.iloc[-1]["flag"]) == FLIP_LABEL:
            last_flag = FLIP_LABEL
        days_since = pd.NA
        if not annotated.empty:
            index_of, _ = _position_lookup(positions)
            last_signal_day = pd.Timestamp(annotated["date"].iloc[-1])
            if last_signal_day in index_of and as_of in index_of:
                days_since = float(index_of[pd.Timestamp(as_of)] - index_of[last_signal_day])
        losses = float(annotated["consecutive_losses"].iloc[-1])
    else:
        last_flag = ""
        days_since = pd.NA
        losses = 0
    if open_row is not None:
        holding = open_row["holding_days"]
        pnl = open_row["unrealized_pnl"]
        state = "持有"
    else:
        holding = 0
        pnl = pd.NA
        state = "空仓"
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "state": state,
                "date": as_of,
                "strategy": strategy_name,
                "suggested_weight": weight,
                "holding_days": holding,
                "unrealized_pnl": pnl,
                "consecutive_losses": losses,
                "days_since_cross": days_since,
                "flag": last_flag,
            }
        ]
    )


def _position_lookup(positions: pd.DataFrame) -> tuple[dict[pd.Timestamp, int], dict[pd.Timestamp, float]]:
    if positions is None or positions.empty or "date" not in positions.columns:
        return {}, {}
    frame = positions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).drop_duplicates("date")
    index_of = {pd.Timestamp(day): i for i, day in enumerate(frame["date"])}
    weights = {}
    if "weight" in frame.columns:
        weights = {
            pd.Timestamp(day): float(weight)
            for day, weight in zip(frame["date"], frame["weight"], strict=True)
            if pd.notna(weight)
        }
    return index_of, weights


def _suggested(frame: pd.DataFrame, loc: int, weight_of: dict, day: pd.Timestamp, default):
    if "weight" in frame.columns and pd.notna(frame["weight"].iloc[loc]):
        return float(frame["weight"].iloc[loc])
    return weight_of.get(day, default)


def _last_close(bars: pd.DataFrame | None) -> float | None:
    if bars is None or "close" not in getattr(bars, "columns", []):
        return None
    close = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])
