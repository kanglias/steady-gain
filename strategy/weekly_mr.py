"""周频自身价格 z-score 均值回归。

价格沿用数据层的复权口径。仓库日线默认是前复权（qfq），不是规格稿里的后复权。
调用方传入什么收盘价，z 就用什么。行业只参与持仓上限，不进入信号。

`run` 仍是单标的接口，供策略登记。选股、行业和总仓约束在 `run_portfolio`。
不会下单。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from data.normalize import normalize_symbol
from strategy.base import Strategy, StrategyResult, prepare_bars, register

logger = logging.getLogger(__name__)

ORDER_COLUMNS = [
    "asof_date",
    "trade_date",
    "code",
    "z_score",
    "signal",
    "target_weight",
    "raw_weight",
    "industry",
    "next_open_ref",
    "preview_notional",
]


def rolling_zscore(close: pd.Series, window: int = 20, eps: float = 1e-8) -> pd.Series:
    """z = (P − MA(W)) / (STD(W) + ε)。STD 是样本标准差（ddof=1）。"""
    if window < 2:
        raise ValueError("window 至少为 2")
    if eps <= 0:
        raise ValueError("eps 需要是正数")
    price = pd.to_numeric(close, errors="coerce")
    mean = price.rolling(window, min_periods=window).mean()
    std = price.rolling(window, min_periods=window).std(ddof=1)
    return (price - mean) / (std + eps)


def week_end_dates(dates: pd.Series) -> pd.DatetimeIndex:
    """每个 ISO 周里，给定交易日中的最后一天。节假日没有的周五不会被造出来。"""
    day = pd.Series(pd.to_datetime(dates, errors="coerce")).dt.normalize().dropna().drop_duplicates()
    if day.empty:
        return pd.DatetimeIndex([])
    frame = pd.DataFrame({"date": day.sort_values()})
    iso = frame["date"].dt.isocalendar()
    frame["year"] = iso.year.astype(int)
    frame["week"] = iso.week.astype(int)
    last = frame.groupby(["year", "week"], sort=True)["date"].max()
    return pd.DatetimeIndex(last.to_list())


def is_beijing(code: str) -> bool:
    """北交所代码不进股票池。含 4/8 开头和 92 新代码。"""
    try:
        norm = normalize_symbol(code)
    except ValueError:
        return True
    return norm.startswith(("4", "8", "92"))


def is_st_name(name: object, is_st: object = False) -> bool:
    if is_st is True or str(is_st).strip().lower() in {"1", "true", "yes", "y", "是"}:
        return True
    text = "" if name is None or (isinstance(name, float) and pd.isna(name)) else str(name).upper()
    return "ST" in text


def project_weights(
    names: list[str],
    score: dict[str, float],
    industries: dict[str, str | None],
    *,
    raw_each: float,
    w_name_max: float,
    w_industry_max: float,
    w_gross_max: float,
    frozen: dict[str, float] | None = None,
) -> tuple[dict[str, float], bool]:
    """按 |z| 从大到小贪心截断，再在超过总仓时把可交易部分等比缩小。

    行业字段缺失时，整本账退回单票和总仓，并返回 degraded=True。
    `frozen` 是停牌无法改的旧仓，占用单票、行业和总仓额度，本身不被缩放。
    """
    locked = {key: float(value) for key, value in (frozen or {}).items() if float(value) > 0}
    degraded = False
    industry_on = True
    if any(_blank_industry(industries.get(name)) for name in names):
        industry_on = False
        degraded = True
    used: dict[str, float] = {}
    if industry_on:
        for name, weight in locked.items():
            ind = industries.get(name)
            if _blank_industry(ind):
                industry_on = False
                degraded = True
                used = {}
                break
            used[str(ind)] = used.get(str(ind), 0.0) + weight
    order = sorted(names, key=lambda name: (-float(score.get(name, 0.0)), name))
    weights: dict[str, float] = {}
    for name in order:
        room_name = w_name_max
        if industry_on:
            ind = str(industries.get(name))
            room_ind = max(0.0, w_industry_max - used.get(ind, 0.0))
        else:
            ind = None
            room_ind = math.inf
        weight = min(float(raw_each), room_name, room_ind)
        if weight <= 0:
            continue
        weights[name] = weight
        if industry_on and ind is not None:
            used[ind] = used.get(ind, 0.0) + weight
    frozen_sum = sum(locked.values())
    active_sum = sum(weights.values())
    total = frozen_sum + active_sum
    if total > w_gross_max + 1e-12 and active_sum > 0:
        room = max(0.0, w_gross_max - frozen_sum)
        scale = room / active_sum
        weights = {name: weight * scale for name, weight in weights.items() if weight * scale > 1e-12}
    return weights, degraded


def _blank_industry(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == "" or str(value).strip().lower() == "nan"


@dataclass
class WeeklyPortfolio:
    orders: pd.DataFrame
    targets: pd.DataFrame
    industry_degraded: bool
    notes: list[str] = field(default_factory=list)


@register
class WeeklyMeanReversion(Strategy):
    """周频做多超卖。单票 10%、行业 30%、总仓 80%，都可改。"""

    def __init__(
        self,
        window: int = 20,
        eps: float = 1e-8,
        z_entry: float = 1.0,
        z_exit: float = 0.25,
        max_names: int = 10,
        max_hold_weeks: int = 8,
        w_name_max: float = 0.10,
        w_industry_max: float = 0.30,
        w_gross_max: float = 0.80,
        preview_capital: float = 100_000,
        min_listed_days: int = 60,
        min_adv20: float = 20_000_000,
        min_close: float = 2.0,
        universe: str = "all",
        constituents: set[str] | list[str] | None = None,
    ):
        if int(window) < 2:
            raise ValueError("window 至少为 2")
        if float(eps) <= 0:
            raise ValueError("eps 需要是正数")
        if float(z_entry) <= 0 or float(z_exit) < 0 or float(z_exit) >= float(z_entry):
            raise ValueError("需要 z_entry > z_exit ≥ 0，这样离场线在入场线之上")
        if int(max_names) < 1 or int(max_hold_weeks) < 1:
            raise ValueError("max_names 和 max_hold_weeks 至少为 1")
        for label, value in (
            ("w_name_max", w_name_max),
            ("w_industry_max", w_industry_max),
            ("w_gross_max", w_gross_max),
        ):
            if not 0 < float(value) <= 1:
                raise ValueError(f"{label} 需要在 (0, 1] 内")
        if universe not in {"all", "csi300"}:
            raise ValueError("universe 只能是 all 或 csi300")
        self.window = int(window)
        self.eps = float(eps)
        self.z_entry = float(z_entry)
        self.z_exit = float(z_exit)
        self.max_names = int(max_names)
        self.max_hold_weeks = int(max_hold_weeks)
        self.w_name_max = float(w_name_max)
        self.w_industry_max = float(w_industry_max)
        self.w_gross_max = float(w_gross_max)
        self.preview_capital = float(preview_capital)
        self.min_listed_days = int(min_listed_days)
        self.min_adv20 = float(min_adv20)
        self.min_close = float(min_close)
        self.universe = universe
        self.constituents = None if constituents is None else {normalize_symbol(item) for item in constituents}
        if universe == "csi300" and not self.constituents:
            raise ValueError("切到沪深300时需要传入成分代码，本模块不联网拉成分")
        self.name = "weekly_mean_reversion"

    def run(self, bars: pd.DataFrame) -> StrategyResult:
        """单标的周频进出。权重用单票上限，不含跨票排序。"""
        frame = prepare_bars(bars, "close")
        symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns and frame["symbol"].notna().any() else ""
        if symbol:
            symbol = normalize_symbol(symbol)
        z = rolling_zscore(frame["close"], self.window, self.eps)
        weeks = set(week_end_dates(frame["date"]))
        weight = 0.0
        weeks_held = 0
        weights: list[float] = []
        signals: list[dict] = []
        for pos, day in enumerate(frame["date"]):
            stamp = pd.Timestamp(day)
            if stamp not in weeks:
                weights.append(weight)
                continue
            if weight > 0:
                weeks_held += 1
            value = z.iloc[pos]
            exit_now = weight > 0 and (
                weeks_held >= self.max_hold_weeks or (pd.notna(value) and float(value) >= -self.z_exit)
            )
            enter_now = weight == 0 and pd.notna(value) and float(value) <= -self.z_entry
            if exit_now:
                weight = 0.0
                weeks_held = 0
                signals.append(
                    {
                        "date": stamp,
                        "symbol": symbol,
                        "side": "sell",
                        "price": float(frame["close"].iloc[pos]),
                        "reason": f"周频离场 z={float(value):.2f}" if pd.notna(value) else "周频持有到期",
                    }
                )
            elif enter_now:
                weight = self.w_name_max
                weeks_held = 0
                signals.append(
                    {
                        "date": stamp,
                        "symbol": symbol,
                        "side": "buy",
                        "price": float(frame["close"].iloc[pos]),
                        "reason": f"周频超卖 z={float(value):.2f}",
                    }
                )
            weights.append(weight)
        positions = pd.DataFrame({"date": frame["date"], "symbol": symbol, "weight": weights})
        indicators = pd.DataFrame({"date": frame["date"], "z": z.to_numpy()})
        signal_frame = pd.DataFrame(signals)
        if signal_frame.empty:
            signal_frame = pd.DataFrame(columns=["date", "symbol", "side", "price", "reason"])
        return StrategyResult(signals=signal_frame, positions=positions, indicators=indicators)

    def run_portfolio(
        self,
        panel: pd.DataFrame,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> WeeklyPortfolio:
        """跨股票周频调仓。信号在当周最后交易日收盘，成交日是下一交易日。"""
        frame = _prepare_panel(panel, self.window, self.eps)
        start_ts = pd.Timestamp(start).normalize() if start is not None else frame["date"].min()
        end_ts = pd.Timestamp(end).normalize() if end is not None else frame["date"].max()
        calendar = pd.DatetimeIndex(sorted(frame["date"].unique()))
        lookup = _lookup(frame)
        held_weight: dict[str, float] = {}
        held_weeks: dict[str, int] = {}
        orders: list[dict] = []
        notes: list[str] = []
        degraded = False
        logged_degrade = False
        for signal_date in week_end_dates(calendar):
            if signal_date < start_ts or signal_date > end_ts:
                continue
            trade_date = _next_day(calendar, signal_date)
            if trade_date is None:
                continue
            frozen: dict[str, float] = {}
            carry: list[str] = []
            exiting: list[str] = []
            for code in list(held_weight):
                held_weeks[code] = held_weeks.get(code, 0) + 1
                row = lookup.get((code, signal_date))
                trade = lookup.get((code, trade_date))
                if row is None or not _can_trade(row, trade):
                    frozen[code] = held_weight[code]
                    continue
                value = row["z"]
                fail_filter = not self._entry_filters(code, row, trade, signal_date, lookup)
                timed_out = held_weeks[code] >= self.max_hold_weeks
                reverted = pd.notna(value) and float(value) >= -self.z_exit
                if fail_filter or timed_out or reverted:
                    exiting.append(code)
                else:
                    carry.append(code)
            slots = self.max_names - len(frozen) - len(carry)
            candidates: list[tuple[float, str]] = []
            if slots > 0:
                for code in sorted({item[0] for item in lookup}):
                    if code in frozen or code in carry or code in exiting or code in held_weight:
                        continue
                    row = lookup.get((code, signal_date))
                    trade = lookup.get((code, trade_date))
                    if row is None or not _can_trade(row, trade):
                        continue
                    if not self._entry_filters(code, row, trade, signal_date, lookup):
                        continue
                    value = row["z"]
                    if pd.isna(value) or float(value) > -self.z_entry:
                        continue
                    candidates.append((float(value), code))
            candidates.sort()
            fresh = [code for _, code in candidates[:slots]]
            active = carry + fresh
            industries = {}
            scores = {}
            for code in active:
                row = lookup[(code, signal_date)]
                industries[code] = _industry_of(row)
                value = row["z"]
                scores[code] = abs(float(value)) if pd.notna(value) else 0.0
            for code, weight in frozen.items():
                row = lookup.get((code, signal_date))
                industries[code] = _industry_of(row) if row is not None else None
            raw_each = (1.0 / len(active)) if active else 0.0
            projected, step_degraded = project_weights(
                active,
                scores,
                industries,
                raw_each=raw_each,
                w_name_max=self.w_name_max,
                w_industry_max=self.w_industry_max,
                w_gross_max=self.w_gross_max,
                frozen=frozen,
            )
            if step_degraded:
                degraded = True
                if not logged_degrade:
                    message = "行业字段缺失，周频约束降级为单票和总仓"
                    logger.warning(message)
                    notes.append(message)
                    logged_degrade = True
            new_weight = dict(frozen)
            new_weight.update({code: weight for code, weight in projected.items() if weight > 1e-12})
            for code in exiting:
                self._append_order(
                    orders,
                    code=code,
                    signal_date=signal_date,
                    trade_date=trade_date,
                    lookup=lookup,
                    signal="exit",
                    target=0.0,
                    raw=0.0,
                )
            for code in active:
                target = float(new_weight.get(code, 0.0))
                if target <= 1e-12:
                    continue
                kind = "enter" if code in fresh else "hold"
                self._append_order(
                    orders,
                    code=code,
                    signal_date=signal_date,
                    trade_date=trade_date,
                    lookup=lookup,
                    signal=kind,
                    target=target,
                    raw=raw_each,
                )
                if code in fresh:
                    held_weeks[code] = 0
            for code in list(held_weeks):
                if code not in new_weight:
                    held_weeks.pop(code, None)
            held_weight = new_weight
        order_frame = pd.DataFrame(orders, columns=ORDER_COLUMNS)
        targets = order_frame.loc[order_frame["target_weight"] > 0, ["asof_date", "code", "target_weight"]].copy()
        return WeeklyPortfolio(orders=order_frame, targets=targets, industry_degraded=degraded, notes=notes)

    def _entry_filters(self, code: str, row: pd.Series, trade: pd.Series | None, signal_date, lookup) -> bool:
        if is_beijing(code):
            return False
        if self.universe == "csi300" and self.constituents is not None and code not in self.constituents:
            return False
        if is_st_name(row.get("name"), row.get("is_st")):
            return False
        close = float(row["close"])
        if not close >= self.min_close:
            return False
        adv = row.get("adv20")
        if pd.isna(adv) or float(adv) < self.min_adv20:
            return False
        age = row.get("listed_age")
        if pd.isna(age) or int(age) < self.min_listed_days:
            return False
        if trade is None:
            return False
        return True

    def _append_order(self, orders, *, code, signal_date, trade_date, lookup, signal, target, raw) -> None:
        row = lookup.get((code, signal_date))
        trade = lookup.get((code, trade_date))
        z_value = None if row is None else row.get("z")
        orders.append(
            {
                "asof_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
                "code": code,
                "z_score": None if z_value is None or pd.isna(z_value) else float(z_value),
                "signal": signal,
                "target_weight": float(target),
                "raw_weight": float(raw),
                "industry": "" if row is None or _blank_industry(_industry_of(row)) else str(_industry_of(row)),
                "next_open_ref": None if trade is None or pd.isna(trade.get("open")) else float(trade["open"]),
                "preview_notional": float(self.preview_capital) * float(target),
            }
        )


def _prepare_panel(panel: pd.DataFrame, window: int, eps: float) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("行情面板需要是 pandas.DataFrame")
    required = {"date", "symbol", "open", "close"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError("行情面板缺少列: " + ", ".join(sorted(missing)))
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = frame["symbol"].map(lambda value: normalize_symbol(value))
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "amount" not in frame.columns:
        frame["amount"] = pd.NA
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol", "open", "close"])
    frame = frame.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    pieces = []
    for _, group in frame.groupby("symbol", sort=False):
        item = group.copy()
        item["z"] = rolling_zscore(item["close"], window, eps).to_numpy()
        item["adv20"] = item["amount"].shift(1).rolling(20, min_periods=20).mean().to_numpy()
        item["listed_age"] = _listed_age(item)
        item["first_bar"] = item["date"].iloc[0]
        pieces.append(item)
    if not pieces:
        raise ValueError("行情面板为空")
    return pd.concat(pieces, ignore_index=True)


def _lookup(frame: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], pd.Series]:
    out = {}
    for row in frame.itertuples(index=False):
        out[(str(row.symbol), pd.Timestamp(row.date).normalize())] = pd.Series(row._asdict())
    return out


def _next_day(calendar: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    later = calendar[calendar > day]
    if len(later) == 0:
        return None
    return pd.Timestamp(later[0])


def _can_trade(row: pd.Series | None, trade: pd.Series | None) -> bool:
    if row is None or trade is None:
        return False
    return not _flag(row.get("suspend")) and not _flag(trade.get("suspend"))


def _flag(value: object) -> bool:
    if value is True:
        return True
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _industry_of(row: pd.Series | None) -> str | None:
    if row is None or "industry" not in row.index:
        return None
    value = row.get("industry")
    if _blank_industry(value):
        return None
    return str(value)


def _listed_age(item: pd.DataFrame) -> list[int]:
    """到每一天为止的上市交易日数。没有上市日时，用这段行情自身的长度。"""
    dates = pd.to_datetime(item["date"])
    if "listed_date" not in item.columns or item["listed_date"].isna().all():
        return list(range(1, len(item) + 1))
    listed = pd.to_datetime(item["listed_date"], errors="coerce").iloc[0]
    if pd.isna(listed):
        return list(range(1, len(item) + 1))
    listed = pd.Timestamp(listed).normalize()
    if listed > dates.iloc[0]:
        after = (dates >= listed).cumsum()
        return [int(value) for value in after]
    before = max(0, len(pd.bdate_range(listed, dates.iloc[0])) - 1)
    return [before + offset for offset in range(1, len(item) + 1)]
