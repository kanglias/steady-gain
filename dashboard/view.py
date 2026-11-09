"""看板用的纯计算。不依赖 Streamlit，方便测试。

这里不连券商，也不把示例收益当成每月 3% 的验收。
"""

from __future__ import annotations

import pandas as pd

from backtest import FOLLOW_LABEL, compare_fills, run_backtest
from backtest.weekly import DEFAULT_COMMISSION, DEFAULT_MIN_COMMISSION, DEFAULT_STAMP
from data.normalize import normalize_symbol
from strategy import DualMovingAverage, MeanReversion, WeeklyMeanReversion
from strategy.base import StrategyResult
from strategy.weekly_mr import ORDER_COLUMNS, week_end_dates

FREQUENCIES = (
    ("daily", "日频"),
    ("weekly", "周频"),
)
STRATEGY_CHOICES = (
    ("dual_ma", "双均线"),
    ("mean_reversion", "均值回归"),
)
WEEKLY_CHOICES = (("weekly_mr", "周频均值回归"),)
EVENT_FILTERS = ("全部", "国内", "海外", "财报", "宏观")
PAGE_BANNER = "研究看板。系统不会自动下单，也没有券商接口。"
EVENT_FOOTNOTE = "本地 calendars/cn_events.json · 可插拔占位，非 Finance API"
EVENT_EMPTY_NOTE = "这个筛选下没有事件。海外和财报需要自行补进本地日历。"
DOMESTIC_REGIONS = {"CN", "CHN", "中国", "国内"}
_EARNINGS_WORDS = ("财报", "业绩", "年报", "季报", "中报")
_MACRO_WORDS = ("宏观", "央行", "LPR", "会议", "政策", "经济", "两会", "全会", "降准", "利率")


def strategies_for(freq: str) -> tuple[tuple[str, str], ...]:
    """日频是双均线和均值回归。周频只有周频均值回归。"""
    if freq == "daily":
        return STRATEGY_CHOICES
    if freq == "weekly":
        return WEEKLY_CHOICES
    raise KeyError(f"未知频率: {freq}")


def weekly_rule_note(strategy: WeeklyMeanReversion) -> str:
    """清单和信号预览共用的周频口径。"""
    return (
        "周频规则：信号在当周最后交易日收盘，计划在下一交易日开盘成交（next_open）。"
        f"约束：单票 {strategy.w_name_max:.0%}、行业 {strategy.w_industry_max:.0%}、总仓 {strategy.w_gross_max:.0%}。"
        "当前清单只预览这一只股票，行业上限要在多票组合里才会截断。"
        "不会自动下单。"
    )


def weekly_fee_note() -> str:
    """和 journal、周频回测同一套研究费率。本页摘要数字仍未扣费。"""
    return (
        "研究用费用：买卖佣金都是 max(名义×"
        f"{DEFAULT_COMMISSION:g}, {DEFAULT_MIN_COMMISSION:g}元)，"
        f"印花税只在卖出，按名义×{DEFAULT_STAMP:g}。"
        "本页回测摘要仍未扣费，扣费后的组合结果在周频回测脚本里。"
    )


def make_strategy(choice: str, *, debounce: bool, risk: bool, params: dict):
    """按顶栏开关组装策略。关掉防抖或风控时，对应阈值不参与计算。

    周频均值回归不用日频的确认天数、止损和 ADX。这两个开关对它不起作用。
    """
    if choice == "weekly_mr":
        return WeeklyMeanReversion(
            window=int(params["window"]),
            z_entry=float(params["z_entry"]),
            z_exit=float(params["z_exit"]),
            max_names=int(params["max_names"]),
            max_hold_weeks=int(params["max_hold_weeks"]),
            w_name_max=float(params["w_name_max"]),
            w_industry_max=float(params["w_industry_max"]),
            w_gross_max=float(params["w_gross_max"]),
            preview_capital=float(params["preview_capital"]),
        )
    weight = float(params["target_weight"])
    if choice == "dual_ma":
        return DualMovingAverage(
            short_window=int(params["short_window"]),
            long_window=int(params["long_window"]),
            trend_window=int(params["trend_window"]) if risk else None,
            confirm_days=int(params["confirm_days"]) if debounce else 1,
            min_hold_days=int(params["min_hold_days"]) if debounce else 1,
            target_weight=weight,
            stop_loss=float(params["stop_loss"]) if risk else None,
            adx_window=int(params["adx_window"]) if risk else None,
            adx_min=float(params["adx_min"]) if risk else None,
        )
    if choice == "mean_reversion":
        return MeanReversion(
            window=int(params["window"]),
            entry_band=float(params["entry_band"]),
            confirm_days=int(params["confirm_days"]) if debounce else 1,
            min_hold_days=int(params["min_hold_days"]) if debounce else 1,
            target_weight=weight,
            stop_loss=float(params["stop_loss"]) if risk else None,
        )
    raise KeyError(f"未知策略: {choice}")


def event_tags(row: dict | pd.Series) -> set[str]:
    """一张卡片可以同时是国内和宏观。筛选是切面，不是互斥分类。"""
    region = str(row.get("region", "")).strip().upper()
    if region in {item.upper() for item in DOMESTIC_REGIONS} or str(row.get("region", "")).strip() in DOMESTIC_REGIONS:
        tags = {"国内"}
    else:
        tags = {"海外"}
    text = f"{row.get('title', '')} {row.get('summary', '')}"
    if any(word in text for word in _EARNINGS_WORDS):
        tags.add("财报")
    if any(word in text for word in _MACRO_WORDS):
        tags.add("宏观")
    return tags


def filter_events(
    events: pd.DataFrame,
    chip: str,
    *,
    symbol: str = "",
    related_only: bool = False,
) -> pd.DataFrame:
    if chip not in EVENT_FILTERS:
        raise ValueError(f"未知事件筛选: {chip}")
    if events is None or events.empty:
        return events.iloc[0:0].copy() if isinstance(events, pd.DataFrame) else pd.DataFrame()
    frame = events.copy()
    if chip != "全部":
        keep = [chip in event_tags(row) for row in frame.to_dict(orient="records")]
        frame = frame.loc[keep]
    if related_only and symbol and not frame.empty:
        wanted = str(symbol)
        related = []
        for symbols in frame["symbols"].tolist() if "symbols" in frame.columns else [None] * len(frame):
            if symbols is None or (isinstance(symbols, float) and pd.isna(symbols)):
                related.append(True)
            elif isinstance(symbols, (list, tuple, set)):
                related.append(wanted in {str(item) for item in symbols} or len(symbols) == 0)
            else:
                related.append(wanted in str(symbols))
        frame = frame.loc[related]
    return frame.reset_index(drop=True)


def win_rate_from_signals(signals: pd.DataFrame) -> float:
    """已平仓来回里，卖出信号价高于买入信号价的比例。未平仓不计入，也不扣费用。"""
    if signals is None or signals.empty or "side" not in signals.columns or "price" not in signals.columns:
        return float("nan")
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["side"] = frame["side"].astype(str).str.lower()
    frame = frame.dropna(subset=["date", "price"]).sort_values("date")
    wins = 0
    closed = 0
    entry: float | None = None
    for side, price in zip(frame["side"], frame["price"], strict=True):
        if side == "buy" and entry is None:
            entry = float(price)
        elif side == "sell" and entry is not None:
            closed += 1
            if float(price) > entry:
                wins += 1
            entry = None
    if closed == 0:
        return float("nan")
    return wins / closed


def backtest_cards(bars: pd.DataFrame, result) -> dict:
    """跟单数字来自 next_open。收盘成交只作对照。胜率两种假设共用信号价。"""
    compared = compare_fills(bars, result)
    stop = run_backtest(bars, result, fill="next_open_stop")
    rate = win_rate_from_signals(result.signals)
    return {
        "follow_label": FOLLOW_LABEL,
        "win_rate": rate,
        "next_open": compared.next_open_fill.as_dict(),
        "close": compared.close_fill.as_dict(),
        "next_open_stop": stop.as_dict(),
    }


def format_holding_days(value: object) -> str:
    """信号预览和导出预览用整数天，避免 6.000000。"""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"{int(number)}天"


def display_strategy_name(strategy) -> str:
    """图外标题用人话，不用 dual_ma_10_30 这种内部 id。"""
    if isinstance(strategy, DualMovingAverage):
        return f"DualMA {int(strategy.short_window)}/{int(strategy.long_window)}"
    if isinstance(strategy, MeanReversion):
        return f"均值回归 {int(strategy.window)}"
    if isinstance(strategy, WeeklyMeanReversion):
        return "周频均值回归"
    return str(getattr(strategy, "name", "") or "策略")


def chart_heading(symbol: str, security_name: str, strategy) -> str:
    label = " ".join(part for part in (str(symbol).strip(), str(security_name).strip()) if part)
    return f"{label} · {display_strategy_name(strategy)}"


def dashboard_chart_layout() -> dict:
    """标题放在图外。图例在图内右上，顶边距让开 K 线。"""
    return {
        "title": {"text": ""},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "right",
            "x": 1,
        },
        "margin": {"l": 48, "r": 16, "t": 88, "b": 28},
    }


def event_region_label(region: object) -> str:
    text = "" if region is None or (isinstance(region, float) and pd.isna(region)) else str(region).strip()
    if text.upper() in {"CN", "CHN"} or text in {"中国", "国内"}:
        return "国内"
    return "海外"


def event_impact_badge(impact: object) -> str:
    text = "" if impact is None or (isinstance(impact, float) and pd.isna(impact)) else str(impact).strip().lower()
    return {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}.get(text, text.upper())


def event_card_html(row: dict | pd.Series) -> str:
    """右栏卡片。空的行业和关联标的不画。"""
    import html

    day = pd.Timestamp(row.get("date"))
    date_text = "" if pd.isna(day) else f"{day:%Y-%m-%d}"
    title = html.escape(str(row.get("title") or ""))
    summary = html.escape(str(row.get("summary") or ""))
    region = html.escape(event_region_label(row.get("region")))
    impact = html.escape(event_impact_badge(row.get("impact")))
    sectors = _event_labels(row.get("sectors"))
    symbols = _event_labels(row.get("symbols"))
    pills = "".join(f'<span class="event-pill">{html.escape(item)}</span>' for item in sectors)
    symbol_line = ""
    if symbols:
        symbol_line = f'<div class="meta">关联 {"、".join(html.escape(item) for item in symbols)}</div>'
    sector_line = f'<div class="event-pills">{pills}</div>' if pills else ""
    return (
        '<div class="event-card">'
        f'<div class="meta">{date_text} · {region} · <span class="event-badge">{impact}</span></div>'
        f"<div><b>{title}</b></div>"
        f"<div>{summary}</div>"
        f"{sector_line}{symbol_line}"
        "</div>"
    )


def _event_labels(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        labels = []
        for item in value:
            text = "" if item is None or (isinstance(item, float) and pd.isna(item)) else str(item).strip()
            if text and text.lower() != "nan":
                labels.append(text)
        return labels
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [text]


def aggregate_weekly_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """把日线收成周 K。一根的日期是该周最后一个交易日，和信号日对齐。"""
    required = ("date", "open", "high", "low", "close")
    missing = [column for column in required if column not in bars.columns]
    if missing:
        raise ValueError("周 K 缺少列: " + ", ".join(missing))
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    ends = {pd.Timestamp(day).normalize() for day in week_end_dates(frame["date"])}
    iso = frame["date"].dt.isocalendar()
    frame["_year"] = iso.year.astype(int)
    frame["_week"] = iso.week.astype(int)
    rows = []
    for _, group in frame.groupby(["_year", "_week"], sort=True):
        last = pd.Timestamp(group["date"].iloc[-1]).normalize()
        if last not in ends:
            continue
        row = {
            "date": last,
            "open": float(group["open"].iloc[0]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group["close"].iloc[-1]),
        }
        if "volume" in group.columns and group["volume"].notna().any():
            row["volume"] = float(group["volume"].sum(min_count=1))
        if "symbol" in group.columns and group["symbol"].notna().any():
            row["symbol"] = group["symbol"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def weekly_order_rows(
    bars: pd.DataFrame,
    result: StrategyResult,
    strategy: WeeklyMeanReversion,
    *,
    start=None,
    end=None,
) -> pd.DataFrame:
    """把单只股票的周频买卖收成调仓清单。

    成交日优先用行情里的下一交易日。样本停在信号日当天时，先记下一工作日，开盘参考留空。
    约束前权重按这一只等权记为 1，目标权重再截到单票和总仓上限。
    """
    dates = pd.to_datetime(bars["date"], errors="coerce").dt.normalize().dropna()
    calendar = pd.DatetimeIndex(sorted(dates.unique()))
    opens = _opens_by_day(bars)
    z_of = _z_by_day(result)
    symbol = _result_symbol(bars, result)
    cap = min(1.0, float(strategy.w_name_max), float(strategy.w_gross_max))
    start_ts = None if start is None else pd.Timestamp(start).normalize()
    end_ts = None if end is None else pd.Timestamp(end).normalize()
    rows: list[dict] = []
    seen_enter: set[pd.Timestamp] = set()
    signals = result.signals if result.signals is not None else pd.DataFrame()
    if not signals.empty:
        ordered = signals.copy()
        ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce").dt.normalize()
        ordered = ordered.dropna(subset=["date"]).sort_values("date")
        for record in ordered.to_dict(orient="records"):
            asof = pd.Timestamp(record["date"]).normalize()
            if not _in_window(asof, start_ts, end_ts):
                continue
            side = str(record.get("side", "")).lower()
            if side == "buy":
                kind = "enter"
                raw = 1.0
                target = cap
                seen_enter.add(asof)
            elif side == "sell":
                kind = "exit"
                raw = 0.0
                target = 0.0
            else:
                continue
            trade = _planned_trade_day(calendar, asof)
            rows.append(_weekly_row(symbol, asof, trade, z_of, opens, strategy, kind, target, raw))
    hold_day = _latest_week_end(calendar)
    if hold_day is not None and hold_day not in seen_enter and _weight_on(result, hold_day) > 0:
        if _in_window(hold_day, start_ts, end_ts):
            trade = _planned_trade_day(calendar, hold_day)
            rows.append(_weekly_row(symbol, hold_day, trade, z_of, opens, strategy, "hold", cap, 1.0))
    if not rows:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def weekly_signal_table(
    board: pd.DataFrame,
    names: dict[str, str],
    indicators: pd.DataFrame | None,
) -> pd.DataFrame:
    """信号预览在日频列后加一列信号日 z。"""
    table = signal_table(board, names)
    if "z" in table.columns:
        return table
    scores = _z_by_day_frame(indicators)
    if board is None or board.empty or "date" not in board.columns:
        table.insert(2, "z", [])
        return table
    values = []
    for day in pd.to_datetime(board["date"], errors="coerce"):
        score = scores.get(pd.Timestamp(day).normalize()) if pd.notna(day) else None
        values.append("" if score is None or pd.isna(score) else f"{float(score):.2f}")
    table.insert(2, "z", values)
    return table


def weekly_order_preview(orders: pd.DataFrame, names: dict[str, str], capital: float) -> pd.DataFrame:
    """周频清单预览。参考价是计划成交日的开盘价，不是信号日收盘价。"""
    columns = ["信号日", "成交日", "代码", "名称", "信号", "z", "目标权重", "约束前", "开盘参考", "预览金额", "估算股数", "备注"]
    labels = {"enter": "调入", "hold": "继续持有", "exit": "调出", "flat": "空仓"}
    if orders is None or orders.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in orders.to_dict(orient="records"):
        code = str(row.get("code", ""))
        kind = str(row.get("signal", ""))
        weight = float(row.get("target_weight") or 0.0)
        price = pd.to_numeric(pd.Series([row.get("next_open_ref")]), errors="coerce").iloc[0]
        note = "下一交易日开盘"
        if kind in {"enter", "hold"} and weight > 0 and pd.notna(price):
            shares = estimate_buy_shares(capital, weight, float(price))
            share_text = "资金不足一手" if shares <= 0 else str(shares)
            if shares <= 0:
                note = "下一交易日开盘；资金不足一手"
        elif kind in {"enter", "hold"} and weight > 0:
            share_text = ""
            note = "下一交易日开盘；样本里还没有这一天的开盘价"
        else:
            share_text = ""
            if kind == "exit":
                note = "下一交易日开盘；卖出数量以持仓为准，这里不代算"
        z_value = row.get("z_score")
        rows.append(
            {
                "信号日": row.get("asof_date", ""),
                "成交日": row.get("trade_date", ""),
                "代码": code,
                "名称": names.get(code, ""),
                "信号": labels.get(kind, kind),
                "z": "" if z_value is None or pd.isna(z_value) else f"{float(z_value):.2f}",
                "目标权重": f"{weight:.0%}",
                "约束前": f"{float(row.get('raw_weight') or 0.0):.0%}",
                "开盘参考": "" if pd.isna(price) else f"{float(price):.2f}",
                "预览金额": f"{float(row.get('preview_notional') or 0.0):.2f}",
                "估算股数": share_text,
                "备注": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _opens_by_day(bars: pd.DataFrame) -> dict[pd.Timestamp, float]:
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    out = {}
    for day, price in zip(frame["date"], frame["open"], strict=True):
        if pd.notna(day) and pd.notna(price):
            out[pd.Timestamp(day)] = float(price)
    return out


def _z_by_day(result: StrategyResult) -> dict[pd.Timestamp, float]:
    return _z_by_day_frame(result.indicators)


def _z_by_day_frame(indicators: pd.DataFrame | None) -> dict[pd.Timestamp, float]:
    if indicators is None or indicators.empty or "date" not in indicators.columns or "z" not in indicators.columns:
        return {}
    frame = indicators.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["z"] = pd.to_numeric(frame["z"], errors="coerce")
    out = {}
    for day, score in zip(frame["date"], frame["z"], strict=True):
        if pd.notna(day) and pd.notna(score):
            out[pd.Timestamp(day)] = float(score)
    return out


def _result_symbol(bars: pd.DataFrame, result: StrategyResult) -> str:
    for frame in (result.positions, result.signals, bars):
        if frame is not None and "symbol" in frame.columns and frame["symbol"].notna().any():
            return normalize_symbol(frame["symbol"].dropna().iloc[0])
    return ""


def _latest_week_end(calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    if len(calendar) == 0:
        return None
    ends = week_end_dates(pd.Series(calendar))
    if len(ends) == 0:
        return None
    return pd.Timestamp(ends[-1]).normalize()


def _weight_on(result: StrategyResult, day: pd.Timestamp) -> float:
    positions = result.positions
    if positions is None or positions.empty or "weight" not in positions.columns:
        return 0.0
    frame = positions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    matched = frame.loc[frame["date"] == day, "weight"]
    if matched.empty:
        return 0.0
    value = pd.to_numeric(pd.Series([matched.iloc[-1]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return 0.0
    return float(value)


def _in_window(day: pd.Timestamp, start: pd.Timestamp | None, end: pd.Timestamp | None) -> bool:
    if start is not None and day < start:
        return False
    if end is not None and day > end:
        return False
    return True


def _planned_trade_day(calendar: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp:
    later = calendar[calendar > day]
    if len(later) == 0:
        return (pd.Timestamp(day) + pd.offsets.BDay(1)).normalize()
    return pd.Timestamp(later[0]).normalize()


def _weekly_row(symbol, asof, trade, z_of, opens, strategy, kind, target, raw) -> dict:
    price = opens.get(pd.Timestamp(trade).normalize())
    score = z_of.get(pd.Timestamp(asof).normalize())
    return {
        "asof_date": pd.Timestamp(asof).strftime("%Y-%m-%d"),
        "trade_date": pd.Timestamp(trade).strftime("%Y-%m-%d"),
        "code": symbol,
        "z_score": score,
        "signal": kind,
        "target_weight": float(target),
        "raw_weight": float(raw),
        "industry": "",
        "next_open_ref": price,
        "preview_notional": float(strategy.preview_capital) * float(target),
    }


def signal_table(board: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """看板信号表。持有天数是整数字符串。"""
    columns = ["标的", "方向", "建议仓位", "持有天数", "浮动盈亏", "信号价", "状态"]
    if board is None or board.empty:
        return pd.DataFrame(columns=columns)
    view = pd.DataFrame(
        {
            "标的": board["symbol"].map(lambda code: f"{code} {names.get(str(code), '')}".strip()),
            "方向": board["side"].map(lambda side: "买入" if side == "buy" else "卖出"),
            "建议仓位": pd.to_numeric(board["suggested_weight"], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value):.0%}"
            ),
            "持有天数": board["holding_days"].map(format_holding_days),
            "浮动盈亏": pd.to_numeric(board["unrealized_pnl"], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
            ),
            "信号价": pd.to_numeric(board["price"], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
            ),
            "状态": board["flag"].fillna("").replace({"": "—"}),
        }
    )
    return view.loc[:, columns]


def estimate_buy_shares(equity: float, weight: float, price: float, *, lot: int = 100) -> int:
    if equity <= 0 or weight <= 0 or price <= 0:
        return 0
    return int(equity * float(weight) / float(price) / lot) * lot


def order_preview(board: pd.DataFrame, names: dict[str, str], equity: float) -> pd.DataFrame:
    """预览用。建议仓位仍是占总资产的比例。估算股数只对买入、且按参考资金向下取整到 100 股。"""
    columns = ["操作", "代码", "名称", "建议仓位", "持有天数", "估算股数", "约合手数", "参考价", "预估金额", "备注"]
    if board is None or board.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in board.to_dict(orient="records"):
        code = str(row["symbol"])
        side = str(row["side"])
        weight = pd.to_numeric(pd.Series([row.get("suggested_weight")]), errors="coerce").iloc[0]
        price = pd.to_numeric(pd.Series([row.get("price")]), errors="coerce").iloc[0]
        note = str(row.get("reason") or "")
        flag = str(row.get("flag") or "")
        pause = str(row.get("pause_advice") or "")
        extra = "；".join(part for part in (flag, pause) if part and part != "nan")
        if extra:
            note = f"{note}；{extra}" if note else extra
        if side == "buy" and pd.notna(weight) and pd.notna(price) and float(weight) > 0:
            shares = estimate_buy_shares(equity, float(weight), float(price))
            if shares <= 0:
                share_text = "资金不足一手"
                hand_text = "资金不足一手"
                amount_text = ""
                note = (note + "；" if note else "") + "资金不足一手"
            else:
                amount = shares * float(price)
                share_text = str(shares)
                hand_text = f"{shares / 100:.0f}"
                amount_text = f"{amount:.2f}"
        else:
            share_text = ""
            hand_text = ""
            amount_text = ""
            note = (note + "；" if note else "") + "卖出数量以持仓为准，这里不代算"
        rows.append(
            {
                "操作": "买入" if side == "buy" else "卖出",
                "代码": code,
                "名称": names.get(code, ""),
                "建议仓位": "" if pd.isna(weight) else f"{float(weight):.0%}",
                "持有天数": format_holding_days(row.get("holding_days")),
                "估算股数": share_text,
                "约合手数": hand_text,
                "参考价": "" if pd.isna(price) else f"{float(price):.2f}",
                "预估金额": amount_text,
                "备注": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)
