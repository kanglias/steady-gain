"""日常看板。只读研究和导出清单，不会连接券商。

启动：在仓库根目录执行
    streamlit run dashboard/app.py --server.port 8766 --server.address 127.0.0.1
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from backtest import FEE_DISCLAIMER, FOLLOW_LABEL
from dashboard.view import (
    EVENT_EMPTY_NOTE,
    EVENT_FILTERS,
    EVENT_FOOTNOTE,
    FREQUENCIES,
    PAGE_BANNER,
    aggregate_weekly_bars,
    backtest_cards,
    chart_heading,
    dashboard_chart_layout,
    event_card_html,
    filter_events,
    make_strategy,
    order_preview,
    signal_table,
    strategies_for,
    weekly_fee_note,
    weekly_order_preview,
    weekly_order_rows,
    weekly_rule_note,
    weekly_signal_table,
)
from data import (
    available_universes,
    csi300_excess,
    get_bars,
    load_events,
    read_csi300_offline,
    universe_names,
    universe_note,
    universe_symbols,
)
from journal import compare_actual, format_report, load_fills
from orders import BannedSymbolError, export_order_list, export_weekly_orders
from strategy import EventGuard, apply_diagnostics, position_snapshot, signal_board
from viz import plot_signals

OUTPUT = ROOT / "output"
CACHE = ROOT / ".cache" / "bars"

st.set_page_config(page_title="A 股研究看板", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; max-width: 1440px; }
      .follow-lock {
        background: #fff7ed; border: 1px solid #fdba74; color: #9a3412;
        font-weight: 700; padding: 8px 12px; border-radius: 8px; margin-bottom: 8px;
      }
      .warn-banner {
        display: block; width: 100%; box-sizing: border-box;
        background: #fef3c7; border: 1px solid #f59e0b; color: #78350f;
        padding: 14px 18px; border-radius: 8px; font-weight: 600;
        line-height: 1.5; margin: 0 0 14px 0;
      }
      .event-pill {
        display: inline-block; border: 1px solid #e5e7eb; border-radius: 999px;
        padding: 1px 8px; margin: 2px 4px 0 0; font-size: 12px; color: #374151;
      }
      .event-badge { font-weight: 700; letter-spacing: 0.04em; }
      .event-card {
        border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 10px; margin-bottom: 8px;
        background: #ffffff;
      }
      .event-card .meta { color: #6b7280; font-size: 12px; }
      div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e5e7eb; padding: 8px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _pct(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "—"
    return f"{float(value):.2%}"


@st.cache_data(show_spinner=False)
def _bars(symbol: str, start: str, end: str, adjust: str, offline: bool, refresh: bool) -> pd.DataFrame:
    return get_bars(symbol, start, end, adjust=adjust, offline=offline, refresh=refresh, cache_dir=CACHE)


@st.cache_data(show_spinner=False)
def _events(start: str, end: str) -> pd.DataFrame:
    return load_events(start=start, end=end, fetch=False)


def _params_from_widgets(choice: str) -> dict:
    if choice == "weekly_mr":
        return {
            "window": st.session_state["w_window"],
            "z_entry": st.session_state["w_z_entry"],
            "z_exit": st.session_state["w_z_exit"],
            "max_names": st.session_state["w_names"],
            "max_hold_weeks": st.session_state["w_hold"],
            "w_name_max": st.session_state["w_name_max"],
            "w_industry_max": st.session_state["w_ind_max"],
            "w_gross_max": st.session_state["w_gross_max"],
            "preview_capital": st.session_state["w_capital"],
        }
    if choice == "dual_ma":
        return {
            "short_window": st.session_state["p_short"],
            "long_window": st.session_state["p_long"],
            "trend_window": st.session_state["p_trend"],
            "confirm_days": st.session_state["p_confirm"],
            "min_hold_days": st.session_state["p_hold"],
            "target_weight": st.session_state["p_weight"],
            "stop_loss": st.session_state["p_stop"],
            "adx_window": st.session_state["p_adx"],
            "adx_min": st.session_state["p_adx_min"],
        }
    return {
        "window": st.session_state["m_window"],
        "entry_band": st.session_state["m_band"],
        "confirm_days": st.session_state["m_confirm"],
        "min_hold_days": st.session_state["m_hold"],
        "target_weight": st.session_state["m_weight"],
        "stop_loss": st.session_state["m_stop"],
    }


def _color_side(value: str) -> str:
    if value == "买入":
        return "color: #b91c1c; font-weight: 700"
    if value == "卖出":
        return "color: #15803d; font-weight: 700"
    return ""


def _highlight_flip(row: pd.Series) -> list[str]:
    flag = str(row.get("状态", ""))
    if flag == "短持翻脸":
        return ["background-color: #f3e8ff"] * len(row)
    return [""] * len(row)


@st.dialog("周频调仓清单", width="large")
def _weekly_order_dialog(orders: pd.DataFrame, names: dict[str, str], strategy) -> None:
    st.markdown(
        '<div class="warn-banner">系统不会自动下单。这份清单只供复制到券商客户端，由你自己确认。</div>',
        unsafe_allow_html=True,
    )
    st.markdown(weekly_rule_note(strategy))
    st.markdown(weekly_fee_note())
    try:
        csv_path, md_path = export_weekly_orders(orders, OUTPUT / "weekly_orders")
    except Exception as exc:
        st.error(str(exc))
        return
    preview = weekly_order_preview(orders, names, strategy.preview_capital)
    if preview.empty:
        st.caption("这个窗口里没有下一交易日可以成交的调仓。")
    else:
        st.dataframe(preview, width="stretch", hide_index=True)
    csv_bytes = csv_path.read_bytes()
    md_text = md_path.read_text(encoding="utf-8")
    left, right = st.columns(2)
    left.download_button("导出 CSV", data=csv_bytes, file_name="weekly_orders.csv", mime="text/csv", type="primary", key="weekly_csv")
    right.download_button(
        "导出 Markdown",
        data=md_text.encode("utf-8"),
        file_name="weekly_orders.md",
        mime="text/markdown",
        key="weekly_md",
    )
    st.caption(f"文件已写到 {csv_path} 和 {md_path}。下面这段 Markdown 可以直接复制。")
    st.text_area("复制 Markdown", md_text, height=220, key="weekly_copy")


@st.dialog("下单清单", width="large")
def _order_dialog(board: pd.DataFrame, names: dict[str, str], equity: float, holding: dict[str, bool]) -> None:
    st.markdown(
        '<div class="warn-banner">系统不会自动下单。这份清单只供复制到券商客户端，由你自己确认。</div>',
        unsafe_allow_html=True,
    )
    st.caption("建议仓位是占总资产的比例。买入股数按参考资金估算，向下取整到 100 股（1 手）。不够一手时写「资金不足一手」，这行建议还在。卖出不代算数量。")
    try:
        csv_path, md_path = export_order_list(board, OUTPUT / "orders", holding=holding)
    except BannedSymbolError as exc:
        st.error(str(exc))
        st.caption("禁投名单命中时不会写 output/orders.csv 或 orders.md。")
        return
    exported = pd.read_csv(csv_path, dtype={"symbol": str})
    preview = order_preview(exported, names, equity)
    st.dataframe(preview, width="stretch", hide_index=True)
    csv_bytes = csv_path.read_bytes()
    md_text = md_path.read_text(encoding="utf-8")
    left, right = st.columns(2)
    left.download_button("导出 CSV", data=csv_bytes, file_name="orders.csv", mime="text/csv", type="primary")
    right.download_button("导出 Markdown", data=md_text.encode("utf-8"), file_name="orders.md", mime="text/markdown")
    st.caption(f"文件已写到 {csv_path} 和 {md_path}。下面这段 Markdown 可以直接复制。")
    st.text_area("复制 Markdown", md_text, height=220)


def _render_events(events: pd.DataFrame) -> None:
    st.markdown("**政策与事件**")
    chip = st.segmented_control("筛选", EVENT_FILTERS, default="全部", key="event_chip")
    related = st.checkbox(
        "仅当前标的相关",
        value=False,
        help="写了代码的事件按当前自选过滤。没有代码的宏观事件仍然显示。",
    )
    chosen = filter_events(events, chip or "全部", symbol=st.session_state.get("symbol", ""), related_only=related)
    if chosen.empty:
        st.caption(EVENT_EMPTY_NOTE)
    else:
        with st.container(height=560):
            for row in chosen.sort_values("date").to_dict(orient="records"):
                st.markdown(event_card_html(row), unsafe_allow_html=True)
    st.caption(EVENT_FOOTNOTE)
    st.toggle(
        "事件日降仓提示",
        key="reduce_events",
        help="默认关闭。打开后只把高影响事件日的目标仓位降到 0，不会自动下单。",
    )


def _render_backtest(cards: dict, excess: dict | None = None) -> None:
    st.markdown(
        f'<div class="follow-lock">【跟单视图锁定为 next_open】{FOLLOW_LABEL}'
        f'　<b>{FEE_DISCLAIMER}</b></div>',
        unsafe_allow_html=True,
    )
    st.caption("第一组数字才是跟单口径。收盘成交只说明「如果收盘就能成交」会差多少。回测不含费用、涨跌停。胜率按信号价的已平仓来回计算，两种假设相同，也不是每月 3% 的验收。")
    follow = cards["next_open"]
    other = cards["close"]
    rate = _pct(cards["win_rate"])
    a, b, c, d = st.columns(4)
    a.metric("收益 · 次日开盘", _pct(follow["total_return"]))
    b.metric("最大回撤 · 次日开盘", _pct(follow["max_drawdown"]))
    c.metric("换手 · 次日开盘", f"{float(follow['turnover']):.2f}")
    d.metric("胜率 · 信号价", rate)
    e, f, g, h = st.columns(4)
    e.metric("收益 · 收盘对照", _pct(other["total_return"]))
    f.metric("最大回撤 · 收盘对照", _pct(other["max_drawdown"]))
    g.metric("换手 · 收盘对照", f"{float(other['turnover']):.2f}")
    h.metric("粗算月收益 · 次日开盘", _pct(follow["rough_monthly_return"]))
    st.caption(
        f"次日开盘止损（仍不是盘中）收益 {_pct(cards['next_open_stop']['total_return'])}。"
        "粗算月收益只是把整段结果折成单月，不是承诺。"
    )
    if excess:
        if excess.get("available"):
            st.metric("超额 · 相对沪深300", _pct(excess["excess"]))
            st.caption(
                f"沪深300 {_pct(excess['benchmark_return'])}。"
                "超额 = 上面的次日开盘收益 − 指数区间收益。不是承诺。"
            )
        else:
            st.caption(str(excess.get("message") or "缺少沪深300缓存，暂无相对基准的超额。策略净值仍可查看。"))


def _signal_table(board: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    return signal_table(board, names)


def main() -> None:
    st.markdown(f'<div class="warn-banner">{PAGE_BANNER}</div>', unsafe_allow_html=True)
    end_default = date.today()
    start_default = end_default - timedelta(days=365 * 5)

    top = st.columns([0.7, 1.15, 1.0, 1.0, 0.7, 0.55, 0.7])
    freq = top[0].selectbox("频率", [key for key, _ in FREQUENCIES], format_func=dict(FREQUENCIES).__getitem__, key="freq")
    strategy_options = strategies_for(freq)
    strategy_labels = [key for key, _ in strategy_options]
    memory_key = f"strategy_memory_{freq}"
    widget_key = f"strategy_{freq}"
    remembered = st.session_state.get(memory_key, strategy_labels[0])
    if remembered not in strategy_labels:
        remembered = strategy_labels[0]
    if widget_key not in st.session_state:
        st.session_state[widget_key] = remembered
    choice = top[1].selectbox(
        "策略",
        strategy_labels,
        format_func=dict(strategy_options).__getitem__,
        key=widget_key,
    )
    st.session_state[memory_key] = choice
    start = top[2].date_input("开始", value=start_default)
    end = top[3].date_input("结束", value=end_default)
    sample = top[4].selectbox("样本", available_universes(), index=0)
    debounce = top[5].toggle("防抖", value=True, help="日频：打开时使用确认天数和最短持有。周频不使用这个开关。")
    risk = top[6].toggle("风控", value=True, help="日频：打开时使用止损；双均线还会使用趋势均线和 ADX。周频不使用这个开关。")
    weekly = freq == "weekly"
    if weekly:
        st.caption("周频不看防抖和风控开关。确认天数、止损和 ADX 只属于日频。")

    with st.expander("策略参数", expanded=False):
        if weekly:
            st.caption("单票、行业、总仓是约束上限。参考资金只用来预览金额，不会下单。")
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("z 窗口", 2, 120, 20, key="w_window")
            c2.number_input("入场 z", 0.1, 5.0, 1.0, step=0.05, key="w_z_entry")
            c3.number_input("离场 z", 0.0, 4.9, 0.25, step=0.05, key="w_z_exit")
            c4.number_input("最多持有周数", 1, 52, 8, key="w_hold")
            d1, d2, d3, d4 = st.columns(4)
            d1.number_input("最多持股", 1, 30, 10, key="w_names")
            d2.number_input("单票上限", 0.01, 1.0, 0.10, step=0.01, key="w_name_max")
            d3.number_input("行业上限", 0.01, 1.0, 0.30, step=0.01, key="w_ind_max")
            d4.number_input("总仓上限", 0.01, 1.0, 0.80, step=0.01, key="w_gross_max")
            st.number_input("参考资金", 10000, 10_000_000, 100000, step=10000, key="w_capital")
        elif choice == "dual_ma":
            st.caption("建议仓位是占总资产的比例，不是手数。关掉上方开关后，对应参数保留在这里，但不参与计算。")
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("短均线", 2, 60, 10, key="p_short")
            c2.number_input("长均线", 3, 120, 30, key="p_long")
            c3.number_input("趋势均线", 10, 250, 60, key="p_trend")
            c4.number_input("ADX 窗口", 2, 60, 14, key="p_adx")
            d1, d2, d3, d4 = st.columns(4)
            d1.number_input("确认天数", 1, 10, 2, key="p_confirm")
            d2.number_input("最短持有", 1, 30, 5, key="p_hold")
            d3.number_input("建议仓位", 0.05, 1.0, 0.5, step=0.05, key="p_weight")
            d4.number_input("止损", 0.01, 0.5, 0.08, step=0.01, key="p_stop")
            st.number_input("ADX 下限", 0.0, 80.0, 20.0, key="p_adx_min")
        else:
            st.caption("建议仓位是占总资产的比例，不是手数。关掉上方开关后，对应参数保留在这里，但不参与计算。")
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("均线窗口", 2, 120, 20, key="m_window")
            c2.number_input("入场偏离", 0.01, 0.3, 0.05, step=0.01, key="m_band")
            c3.number_input("确认天数", 1, 10, 2, key="m_confirm")
            c4.number_input("最短持有", 1, 30, 3, key="m_hold")
            d1, d2 = st.columns(2)
            d1.number_input("建议仓位", 0.05, 1.0, 0.5, step=0.05, key="m_weight")
            d2.number_input("止损", 0.01, 0.5, 0.08, step=0.01, key="m_stop")

    names = universe_names(sample)
    symbols = universe_symbols(sample)
    if st.session_state.get("symbol") not in symbols:
        st.session_state["symbol"] = symbols[0]
    st.caption(f"{sample}：{universe_note(sample)}")

    left, center, right = st.columns([1.05, 3.3, 1.35], gap="small")
    with left:
        st.markdown("**自选**")
        query = st.text_input("搜索代码或名称", "")
        shown = [
            code
            for code in symbols
            if query.strip() in code or query.strip() in names.get(code, "")
        ]
        if not shown:
            st.caption("没有匹配的代码。")
        else:
            current = st.session_state.get("symbol")
            index = shown.index(current) if current in shown else 0
            st.session_state["symbol"] = st.radio(
                "标的",
                shown,
                index=index,
                format_func=lambda code: f"{code}  {names.get(code, '')}",
                label_visibility="collapsed",
            )
    symbol = st.session_state["symbol"]

    reduce_events = bool(st.session_state.get("reduce_events", False))
    offline = st.toggle("只用本地行情", value=True)

    error = ""
    try:
        bars = _bars(symbol, start.isoformat(), end.isoformat(), "qfq", offline, False)
        events = _events(start.isoformat(), end.isoformat())
        strategy = make_strategy(choice, debounce=debounce, risk=risk, params=_params_from_widgets(choice))
        guarded = EventGuard(events, mode="reduce" if reduce_events else "off").apply(strategy.run(bars), bars)
        shown_result = apply_diagnostics(guarded, bars)
    except Exception as exc:
        bars = None
        events = pd.DataFrame()
        strategy = None
        guarded = None
        shown_result = None
        error = f"{type(exc).__name__}: {exc}"

    with center:
        if error:
            st.error(error)
            st.caption("默认不访问网络。若缓存没有覆盖这段区间，关掉「只用本地行情」后再试。")
        else:
            st.markdown(f"**{chart_heading(symbol, names.get(symbol, ''), strategy)}**")
            chart_bars = aggregate_weekly_bars(bars) if weekly else bars
            fig = plot_signals(chart_bars, shown_result, title=None, events=None)
            fig.update_layout(height=640, **dashboard_chart_layout())
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
            if weekly:
                st.caption(
                    "周K：一根是一个交易周，日期是该周最后一个交易日。红涨绿跌。"
                    "买卖标记是该日收盘信号，计划成交是下一交易日开盘，不是已经发出的委托。"
                )
            else:
                st.caption("K 线红涨绿跌。图上买入标记为红、卖出标记为绿，和涨跌同色；方向以文字为准。标记是收盘信号，不是成交。")

    with right:
        if error:
            st.caption("行情没有载入，事件先按本地日历能读到的部分显示。")
        _render_events(events if isinstance(events, pd.DataFrame) else pd.DataFrame())

    if error or shown_result is None:
        return

    window_start = pd.Timestamp(end) - pd.Timedelta(days=90)
    window = signal_board(
        {symbol: shown_result},
        strategy_name=strategy.name,
        start=window_start.date(),
        end=end,
        bars={symbol: bars},
    )
    snap = position_snapshot(guarded, bars, strategy_name=strategy.name)
    holding = {}
    if not snap.empty:
        holding[symbol] = str(snap.iloc[0]["state"]) == "持有"

    bottom_left, bottom_right = st.columns([1.25, 1])
    with bottom_left:
        st.markdown("**回测摘要**")
        cards = backtest_cards(bars, guarded)
        bench = read_csi300_offline(bars["date"].min(), bars["date"].max(), cache_dir=CACHE)
        excess = csi300_excess(cards["next_open"]["total_return"], bench)
        _render_backtest(cards, excess)
        if weekly:
            st.markdown(weekly_rule_note(strategy))
            st.markdown(weekly_fee_note())
    with bottom_right:
        st.markdown("**信号预览**")
        if weekly:
            st.caption("近 90 个自然日里的周频信号。z 是信号日收盘的分数。建议仓位是单票上限内的比例。")
            preview = weekly_signal_table(window.tail(8), names, shown_result.indicators)
        else:
            st.caption("近 90 个自然日。建议仓位是比例。短持翻脸用底色标出。")
            preview = _signal_table(window.tail(8), names)
        if preview.empty:
            st.caption("这段窗口里没有新的买卖。")
        else:
            styled = preview.style.map(_color_side, subset=["方向"]).apply(_highlight_flip, axis=1)
            st.dataframe(styled, width="stretch", hide_index=True)
        if st.button("生成下单清单", type="primary"):
            if weekly:
                orders = weekly_order_rows(bars, shown_result, strategy, start=window_start.date(), end=end)
                _weekly_order_dialog(orders, names, strategy)
            else:
                _order_dialog(window, names, equity=100_000, holding=holding)
        with st.expander("完整信号"):
            full = weekly_signal_table(window, names, shown_result.indicators) if weekly else _signal_table(window, names)
            if full.empty:
                st.caption("窗口内没有信号。")
            else:
                st.dataframe(
                    full.style.map(_color_side, subset=["方向"]).apply(_highlight_flip, axis=1),
                    width="stretch",
                    hide_index=True,
                )
            if not snap.empty:
                st.caption(
                    f"当前腿 {snap.iloc[0]['state']}，建议仓位 {_pct(snap.iloc[0]['suggested_weight'])}。"
                    "暂停加仓若出现，会写明仍持有还是空仓，不是新开仓。"
                )

    with st.expander("阶段 4 · 费用、超额、样本外"):
        st.caption(
            "回测摘要没有扣费。实际收益只在成交日志里扣佣金、印花税和其他费用。"
            "演示日志带 is_simulated=true，不是实盘业绩。大约每月 3% 不是验收线。"
        )
        split = st.text_input("train / test 分界（当天及之后为 test）", value="2024-01-01")
        fills_path = OUTPUT / "fills.csv"
        if bench is None:
            st.caption("缺少沪深300缓存，暂无相对基准的超额。策略净值仍可查看。成交日志里的超额只和买入持有比。")
        else:
            st.caption(f"沪深300 缓存 {len(bench)} 根")
        if not fills_path.exists():
            st.caption("还没有 output/fills.csv。先跑 python scripts/plot_demo.py --offline，或把自己的成交存成同一格式。")
        else:
            fills = load_fills(fills_path)
            report = compare_actual(
                bars,
                guarded,
                fills,
                equity=100_000,
                benchmark=bench,
                split_date=split,
            )
            st.text(format_report(report))


if __name__ == "__main__":
    main()
