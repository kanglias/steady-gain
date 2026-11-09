"""事件日约束。默认关闭，不改双均线自己的仓位。

`block_entries`：事件日不允许从空仓新开仓，已有仓位仍可按策略卖出。
`reduce`：事件日把目标仓位降到 0。下一天若策略仍给出持有，会再开仓。
这是研究用的开关，不是风控引擎，也没有滑点和涨跌停。
"""

from __future__ import annotations

import pandas as pd

from data.events import IMPACT_RANK
from data.normalize import normalize_symbol, parse_date
from strategy.base import StrategyResult


class EventGuard:
    def __init__(
        self,
        events: pd.DataFrame | None,
        *,
        mode: str = "off",
        min_impact: str = "high",
        reduce_to: float = 0.0,
    ):
        if mode not in {"off", "block_entries", "reduce"}:
            raise ValueError("mode 只能是 off、block_entries 或 reduce")
        if min_impact not in IMPACT_RANK:
            raise ValueError("min_impact 只能是 low、medium 或 high")
        if reduce_to != 0.0:
            raise ValueError("阶段 2 的降仓只支持降到 0")
        self.events = events if events is not None else pd.DataFrame()
        self.mode = mode
        self.min_impact = min_impact
        self.reduce_to = reduce_to

    def apply(self, result: StrategyResult, bars: pd.DataFrame | None = None) -> StrategyResult:
        if self.mode == "off":
            return StrategyResult(
                signals=result.signals.copy(),
                positions=result.positions.copy(),
                indicators=result.indicators.copy(),
            )
        positions = _prepare_positions(result.positions)
        symbol = ""
        if "symbol" in positions.columns and positions["symbol"].notna().any():
            symbol = str(positions["symbol"].dropna().iloc[0])
        blocked = _event_titles(self.events, symbol, self.min_impact)
        weights: list[float] = []
        reasons_for_change: list[str] = []
        previous = 0.0
        for day, raw_weight in zip(positions["date"], positions["weight"], strict=True):
            weight = float(raw_weight)
            title = blocked.get(pd.Timestamp(day).normalize(), "")
            forced = False
            if title and self.mode == "reduce":
                if weight > self.reduce_to:
                    forced = True
                weight = min(weight, self.reduce_to)
            elif title and self.mode == "block_entries" and previous == 0.0 and weight > previous:
                weight = previous
                forced = True
            note = ""
            if forced and self.mode == "reduce" and weight < float(raw_weight):
                note = f"事件日降低仓位：{title}"
            elif forced and self.mode == "block_entries":
                note = f"事件日不新开仓：{title}"
            weights.append(weight)
            reasons_for_change.append(note)
            previous = weight

        adjusted = positions.copy()
        adjusted["weight"] = weights
        signals = _signals_from_weights(
            adjusted,
            original=result.signals,
            notes=reasons_for_change,
            prices=_price_map(bars),
        )
        return StrategyResult(
            signals=signals,
            positions=adjusted.reset_index(drop=True),
            indicators=result.indicators.copy(),
        )


def event_switch_samples() -> dict[str, pd.DataFrame]:
    """两段可重复的仓位路径，用来看 `block_entries` 和 `reduce` 的差别。

    不拉行情。`entry_on_event`：空仓在事件日想开仓。
    `holding_through_event`：事件日前已经持有。
    """
    return {
        "entry_on_event": _switch_case([0.0, 0.5, 0.5, 0.5], 1, "开仓日遇事件"),
        "holding_through_event": _switch_case([0.0, 0.5, 0.5, 0.5, 0.5], 2, "持仓中遇事件"),
    }


def _switch_case(weights: list[float], event_offset: int, title: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-06-03", periods=len(weights))
    events = pd.DataFrame(
        [
            {
                "date": dates[event_offset],
                "title": title,
                "region": "CN",
                "impact": "high",
                "summary": "可重复样本，不是公告原文",
                "sectors": None,
                "symbols": None,
            }
        ]
    )
    result = StrategyResult(
        signals=pd.DataFrame(columns=["date", "symbol", "side", "price", "weight", "reason"]),
        positions=pd.DataFrame({"date": dates, "symbol": "000001", "weight": weights}),
        indicators=pd.DataFrame({"date": dates}),
    )
    table = {"date": dates, "case": title}
    for mode in ("off", "block_entries", "reduce"):
        adjusted = EventGuard(events, mode=mode).apply(result)
        table[mode] = adjusted.positions["weight"].to_numpy()
    frame = pd.DataFrame(table)
    frame["event"] = frame["date"].eq(pd.Timestamp(dates[event_offset]).normalize())
    return frame


def _prepare_positions(positions: pd.DataFrame) -> pd.DataFrame:
    if "date" not in positions.columns or "weight" not in positions.columns:
        raise ValueError("仓位需要 date 和 weight 列")
    frame = positions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["date", "weight"]).sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError("仓位为空")
    return frame.reset_index(drop=True)


def _event_titles(events: pd.DataFrame, symbol: str, min_impact: str) -> dict[pd.Timestamp, str]:
    if events is None or events.empty:
        return {}
    needed = {"date", "title", "impact"}
    if not needed <= set(events.columns):
        raise ValueError("事件表需要 date、title、impact 列")
    floor = IMPACT_RANK[min_impact]
    code = normalize_symbol(symbol) if symbol else ""
    grouped: dict[pd.Timestamp, list[str]] = {}
    for row in events.to_dict(orient="records"):
        impact = str(row.get("impact", "")).lower()
        if IMPACT_RANK.get(impact, 0) < floor:
            continue
        if not _applies_to_symbol(row.get("symbols"), code):
            continue
        try:
            day = pd.Timestamp(parse_date(row["date"]))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(day, []).append(str(row["title"]))
    return {day: "；".join(titles) for day, titles in grouped.items()}


def _applies_to_symbol(symbols: object, code: str) -> bool:
    if symbols is None or (isinstance(symbols, float) and pd.isna(symbols)):
        return True
    if isinstance(symbols, list):
        if not symbols:
            return True
        return code in {normalize_symbol(item) for item in symbols}
    text = str(symbols).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return True
    return code in {normalize_symbol(part) for part in text.replace("、", "|").split("|") if part.strip()}


def _price_map(bars: pd.DataFrame | None) -> dict[pd.Timestamp, float]:
    if bars is None or "date" not in bars.columns or "close" not in bars.columns:
        return {}
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    return {pd.Timestamp(day): float(price) for day, price in zip(frame["date"], frame["close"], strict=True)}


def _signals_from_weights(
    positions: pd.DataFrame,
    *,
    original: pd.DataFrame,
    notes: list[str],
    prices: dict[pd.Timestamp, float],
) -> pd.DataFrame:
    original_lookup = _original_lookup(original)
    symbol = ""
    if "symbol" in positions.columns and positions["symbol"].notna().any():
        symbol = str(positions["symbol"].dropna().iloc[0])
    rows = []
    previous = 0.0
    for index, weight in enumerate(positions["weight"].tolist()):
        day = pd.Timestamp(positions["date"].iloc[index])
        side = ""
        signed_weight = float(weight)
        if previous == 0.0 and signed_weight > 0:
            side = "buy"
        elif previous > 0 and signed_weight == 0.0:
            side = "sell"
        previous = signed_weight
        if not side:
            continue
        known = original_lookup.get((day, side))
        note = notes[index]
        if known and not note:
            reason = known["reason"]
            price = known["price"]
        else:
            reason = note or "目标仓位变化"
            price = prices.get(day, pd.NA)
        rows.append(
            {
                "date": day,
                "symbol": symbol,
                "side": side,
                "price": price,
                "weight": signed_weight,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=["date", "symbol", "side", "price", "weight", "reason"])


def _original_lookup(signals: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], dict]:
    if signals is None or signals.empty or "date" not in signals.columns or "side" not in signals.columns:
        return {}
    lookup = {}
    for row in signals.to_dict(orient="records"):
        try:
            day = pd.Timestamp(parse_date(row["date"]))
        except (TypeError, ValueError):
            continue
        side = str(row["side"]).lower()
        lookup[(day, side)] = {
            "reason": str(row.get("reason", "")),
            "price": row.get("price", pd.NA),
        }
    return lookup
