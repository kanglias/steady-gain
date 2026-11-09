#!/usr/bin/env python3
"""阶段 4 演示：费用、超额、样本外分段、锁定的次日开盘跟单。

与 notebooks/daily_entry.ipynb 同一条路径。不会连接券商。
跟单视图锁定为 next_open。收盘成交只作对照。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backtest import FEE_DISCLAIMER, FOLLOW_LABEL, compare_fills, fill_assumption_table, run_backtest  # noqa: E402
from data import (  # noqa: E402
    available_universes,
    get_bars,
    get_csi300,
    load_events,
    universe_names,
    universe_note,
    universe_symbols,
)
from journal import (  # noqa: E402
    apply_cost_schedule,
    compare_actual,
    format_report,
    save_fills,
    simulate_follow_fills,
)
from orders import PAUSE_LABEL, BannedSymbolError, export_order_list  # noqa: E402
from strategy import (  # noqa: E402
    DualMovingAverage,
    EventGuard,
    MeanReversion,
    apply_diagnostics,
    mean_reversion_sensitivity,
    position_snapshot,
    signal_board,
)
from viz import plot_signals, save_chart  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A 股研究示例：信号、清单、成交对照")
    parser.add_argument("--sample", default="demo", choices=available_universes())
    parser.add_argument("--symbols", nargs="*", default=None, help="覆盖样本名单")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD，默认大约 5 年前")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD，默认今天")
    parser.add_argument("--short", type=int, default=10)
    parser.add_argument("--long", type=int, default=30)
    parser.add_argument("--adjust", default="qfq", help="qfq / hfq / 空字符串表示不复权")
    parser.add_argument("--refresh", action="store_true", help="忽略已覆盖的缓存，重新拉取")
    parser.add_argument("--offline", action="store_true", help="只用本地行情缓存")
    parser.add_argument("--event-mode", default="off", choices=["off", "block_entries", "reduce"])
    parser.add_argument("--fetch-events", action="store_true", help="额外尝试拉取 LPR 变动日")
    parser.add_argument("--order-days", type=int, default=90, help="清单覆盖最近多少个自然日")
    parser.add_argument("--equity", type=float, default=100_000, help="演示成交使用的参考资金（元）")
    parser.add_argument("--pause-after-losses", type=int, default=3)
    parser.add_argument("--split", default="2024-01-01", help="这一天及之后记为 test，之前记为 train")
    parser.add_argument("--commission-rate", type=float, default=0.00025, help="研究用佣金费率，不是券商价目")
    parser.add_argument("--stamp-tax-rate", type=float, default=0.0005, help="研究用印花税率，默认只计在卖出")
    parser.add_argument("--min-commission", type=float, default=5.0, help="单笔最低佣金（元）")
    parser.add_argument("--output", default=str(ROOT / "output"))
    return parser.parse_args(argv)


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _num(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(int(value)) if float(value).is_integer() else f"{float(value):.4g}"


def _print_assumptions(table: pd.DataFrame) -> None:
    print("【跟单视图锁定为 next_open】第一行才是跟单假设。收盘成交只作对照。")
    print(f"{FOLLOW_LABEL}。{FEE_DISCLAIMER}。")
    view = table.copy()
    for column in ("total_return", "max_drawdown", "rough_monthly_return"):
        view[column] = view[column].map(_pct)
    view["turnover"] = view["turnover"].map(lambda value: f"{float(value):.2f}")
    columns = ["label", "follow", "total_return", "max_drawdown", "turnover", "rough_monthly_return"]
    print(view.loc[:, columns].to_string(index=False))
    print("粗算月收益不是每月 3% 的验收。止损默认看收盘；「次日开盘止损」只改止损的离场假设。")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=365 * 5)
    names = universe_names(args.sample)
    symbols = args.symbols or universe_symbols(args.sample)
    strategy = DualMovingAverage(short_window=args.short, long_window=args.long)
    cache_dir = ROOT / ".cache" / "bars"
    out_dir = Path(args.output)
    print(
        "说明：这是研究示例，不是下单指令。"
        "大约每月 3% 是方向，不是本示例的承诺。"
        "从阶段 4 起，对照里会扣研究用费用，并给出相对买入持有和沪深 300 的超额，以及 train/test 分段。"
        "示例策略的收益仍然不是每月 3% 的验收。"
    )
    print(
        "试用记录（只描述那次过滤后的示例，不是核定结论）："
        "工具链能用；过滤有帮助，但粗算月收益仍大约在 -0.09% 到 -0.26%，"
        "离大约 3%/月还有约 3 个百分点。跟单优先看 next_open。"
    )
    print(f"样本 {args.sample}：{universe_note(args.sample)}  代码 {', '.join(symbols)}")
    print(
        f"双均线 {strategy.name}：确认 {strategy.confirm_days} 日，最短持有 {strategy.min_hold_days} 日，"
        f"趋势 {strategy.trend_window}，ADX {strategy.adx_window}/{strategy.adx_min}，"
        f"建议仓位 {strategy.target_weight:.0%}，止损 {strategy.stop_loss}（收盘触发）。"
    )
    print(f"区间 {start} ~ {end} · 复权 {args.adjust or '不复权'} · 事件约束 {args.event_mode}")

    events = load_events(start=start, end=end, fetch=args.fetch_events)
    if events.attrs.get("fetch_error"):
        print(f"事件外源失败，已退回本地日历: {events.attrs['fetch_error']}")
    print(f"事件 {len(events)} 条 · 来源 {events.attrs.get('source')}")

    books: dict[str, pd.DataFrame] = {}
    guarded: dict[str, object] = {}
    shown: dict[str, object] = {}
    holding: dict[str, bool] = {}
    for symbol in symbols:
        bars = get_bars(
            symbol,
            start,
            end,
            adjust=args.adjust,
            refresh=args.refresh,
            offline=args.offline,
            cache_dir=cache_dir,
        )
        code = str(bars["symbol"].iloc[0])
        books[code] = bars
        raw = strategy.run(bars)
        guarded[code] = EventGuard(events, mode=args.event_mode).apply(raw, bars)
        shown[code] = apply_diagnostics(guarded[code], bars)
        name = names.get(code, "")
        label = f"{code} {name}".strip()
        signals = shown[code].signals
        buys = int((signals["side"] == "buy").sum()) if not signals.empty else 0
        sells = int((signals["side"] == "sell").sum()) if not signals.empty else 0
        flips = int(signals["flag"].eq("短持翻脸").sum()) if not signals.empty and "flag" in signals.columns else 0
        weight = float(shown[code].positions.iloc[-1]["weight"])
        hit = "缓存" if bars.attrs.get("cache_hit") else "网络"
        print(
            f"{label}  {len(bars)} 根  {bars['date'].min():%Y-%m-%d}~{bars['date'].max():%Y-%m-%d}  "
            f"来源 {bars.attrs.get('source')} ({hit})  买入 {buys}  卖出 {sells}  "
            f"短持翻脸标记 {flips}  最新建议仓位 {weight:.0%}"
        )
        compared = compare_fills(bars, guarded[code])
        stop_fill = run_backtest(bars, guarded[code], fill="next_open_stop")
        _print_assumptions(fill_assumption_table(compared, stop_fill))
        snap = position_snapshot(guarded[code], bars, strategy_name=strategy.name)
        if not snap.empty:
            row = snap.iloc[0]
            holding[code] = str(row["state"]) == "持有"
            print(
                f"  当前腿: {row['state']}  建议 {_pct(row['suggested_weight'])}  "
                f"持有 {_num(row['holding_days'])} 日  浮动 {_pct(row['unrealized_pnl'])}  "
                f"连亏 {_num(row['consecutive_losses'])}  距上次交叉 {_num(row['days_since_cross'])} 日  {row['flag']}"
            )
        title = f"{label} · {strategy.name} · 建议仓位 {strategy.target_weight:.0%}"
        fig = plot_signals(bars, shown[code], events=events, title=title)
        path = save_chart(fig, out_dir / f"{code}_{strategy.name}.html")
        print(f"  图: {path}")

    window_start = end - timedelta(days=args.order_days)
    window = signal_board(shown, strategy_name=strategy.name, start=window_start, end=end, bars=books)
    latest_day = max(frame["date"].max() for frame in books.values())
    today = signal_board(shown, strategy_name=strategy.name, on=latest_day, bars=books)
    print(f"最近交易日 {pd.Timestamp(latest_day):%Y-%m-%d} 的新信号 {len(today)} 条")
    if today.empty:
        print("  当天没有新的买卖。仓位状态见上面的当前腿。")
    else:
        view = today.copy()
        view["date"] = view["date"].dt.strftime("%Y-%m-%d")
        print(view.to_string(index=False))
    print(f"清单窗口 {window_start} ~ {end}，共 {len(window)} 条")
    try:
        csv_path, md_path = export_order_list(
            window,
            out_dir / "orders",
            pause_after_losses=args.pause_after_losses,
            holding=holding,
        )
    except BannedSymbolError as exc:
        print(f"拒绝导出: {exc}")
        print("清单文件没有写出。从 orders/banned.txt 去掉这些代码后才能导出。")
        return 2
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")
    if PAUSE_LABEL in md_path.read_text(encoding="utf-8"):
        print(f"  {PAUSE_LABEL}：连续亏损或短持翻脸反复出现，并写明仍持有还是空仓。这不是自动停手，也不是新开仓。")
    print("  清单只给人看。不会发送任何委托。")

    print(
        f"登记演示成交：参考资金 {args.equity:.0f} 元，价格用信号日之后的开盘（{FOLLOW_LABEL}）。"
        "每笔强制 is_simulated=true，不能当成实盘业绩。"
        f"研究用费用：佣金费率 {args.commission_rate}，最低 {args.min_commission:.0f} 元，"
        f"卖出印花税 {args.stamp_tax_rate}。这不是券商价目。"
    )
    simulated = [
        simulate_follow_fills(books[code], shown[code].signals, equity=args.equity)
        for code in books
    ]
    fills = pd.concat(simulated, ignore_index=True) if simulated else pd.DataFrame()
    fills = apply_cost_schedule(
        fills,
        commission_rate=args.commission_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        min_commission=args.min_commission,
    )
    fill_csv, fill_json = save_fills(fills, out_dir / "fills", simulated=True)
    print(f"  成交 {len(fills)} 笔  is_simulated=true  CSV: {fill_csv}  JSON: {fill_json}")
    benchmark = _load_benchmark(start, end, refresh=args.refresh, offline=args.offline, cache_dir=cache_dir)
    for code, bars in books.items():
        report = compare_actual(
            bars,
            guarded[code],
            fills,
            equity=args.equity,
            benchmark=benchmark,
            split_date=args.split,
        )
        print(format_report(report))

    print("切换到第二种策略：均值回归。窗口默认 20，不是把双均线的短窗改短。")
    second = MeanReversion()
    for code, bars in books.items():
        result = EventGuard(events, mode=args.event_mode).apply(second.run(bars), bars)
        compared = compare_fills(bars, result)
        print(f"{code} {names.get(code, '')} · {second.name}  信号 {len(result.signals)} 条")
        _print_assumptions(fill_assumption_table(compared, run_backtest(bars, result, fill="next_open_stop")))
        fig = plot_signals(
            bars,
            apply_diagnostics(result, bars),
            events=events,
            title=f"{code} {names.get(code, '')} · {second.name}",
        )
        path = save_chart(fig, out_dir / f"{code}_{second.name}.html")
        print(f"  图: {path}")
        grid = mean_reversion_sensitivity(bars)
        view = grid.copy()
        view["total_return"] = view["total_return"].map(_pct)
        view["max_drawdown"] = view["max_drawdown"].map(_pct)
        print(view.loc[:, ["window", "entry_band", "fill", "total_return", "max_drawdown", "n_signals", "note"]].to_string(index=False))
    return 0


def _load_benchmark(start, end, *, refresh: bool, offline: bool, cache_dir: Path):
    try:
        frame = get_csi300(start, end, refresh=refresh, offline=offline, cache_dir=cache_dir)
    except Exception as exc:
        print(f"沪深300 不可用，超额只和买入持有比: {type(exc).__name__}: {exc}")
        return None
    hit = "缓存" if frame.attrs.get("cache_hit") else "网络"
    print(f"基准 沪深300  {len(frame)} 根  ({hit})")
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
