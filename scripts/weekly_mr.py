#!/usr/bin/env python3
"""周频均值回归研究入口。默认用合成数据，不访问网络。

全 A、2018 至今的实盘口径需要本地日线。本脚本不调用行情接口。
价格口径与仓库一致，用调用方已经准备好的前复权收盘价。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backtest import cost_sensitivity, format_weekly_report, run_weekly_backtest  # noqa: E402
from data.benchmark import read_csi300_offline  # noqa: E402
from data.normalize import normalize_symbol  # noqa: E402
from data.universe import available_universes, universe_note, universe_symbols  # noqa: E402
from orders import export_weekly_orders  # noqa: E402
from strategy import WeeklyMeanReversion  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="周频均值回归研究。不连接券商，默认不联网。")
    parser.add_argument("--demo", action="store_true", help="用一段合成行情演示因子、约束和费用三档")
    parser.add_argument(
        "--sample",
        default=None,
        choices=available_universes(),
        help="与看板相同的样本名，含 watch。只读本地缓存。不要和 --symbols 同时使用，同时给出则退出。",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="只读本地缓存里的这些代码。不要和 --sample 同时使用，同时给出则退出。",
    )
    parser.add_argument("--offline", action="store_true", help="只用 .cache/bars，缺失就停")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--split", default="2022-01-01")
    parser.add_argument("--output", default=str(ROOT / "output"))
    return parser.parse_args(argv)


def selected_symbols(args: argparse.Namespace) -> list[str] | None:
    """返回要读缓存的代码。都没给时返回 None，走合成行情。"""
    symbols = [str(item) for item in (args.symbols or [])]
    sample = args.sample
    if symbols and sample:
        raise SystemExit("--sample 与 --symbols 只能选一个。")
    if sample:
        return universe_symbols(sample)
    if symbols:
        return symbols
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = selected_symbols(args)
    if symbols:
        panel = _load_cache(symbols)
        strategy = WeeklyMeanReversion()
        if args.sample:
            print(f"样本 {args.sample}：{universe_note(args.sample)} 只读本地缓存，不是全 A。")
        else:
            print("样本是本地缓存里这几只股票，不是全 A，也不是 2018 至今的完整股票池。")
    else:
        panel = _demo_panel()
        strategy = WeeklyMeanReversion(
            window=5,
            min_listed_days=5,
            min_adv20=1.0,
            min_close=1.0,
            max_names=4,
        )
        print("这是合成行情，用来核对周频链路。周五收在 6、其他交易日收在 10，收益率是人为价格的结果，不是历史业绩。")
    print("复权口径：与仓库日线一致，研究默认前复权。规格稿写过后复权，这里不另换一套价格。")
    range_start = args.start if symbols else "2024-01-01"
    benchmark = _cached_csi300(range_start, pd.to_datetime(panel["date"]).max())
    result = run_weekly_backtest(
        panel,
        strategy,
        start=range_start,
        split=args.split if symbols else "2024-02-01",
        benchmark=benchmark,
    )
    print(format_weekly_report(result))
    table = cost_sensitivity(
        panel,
        strategy,
        start=range_start,
        split=args.split if symbols else "2024-02-01",
        benchmark=benchmark,
    )
    view = table.copy()
    for column in ("total_return", "test_annual_return", "test_max_drawdown", "test_turnover_annual"):
        view[column] = view[column].map(lambda value: "—" if pd.isna(value) else f"{float(value):.2%}")
    print("费用三档（默认费率 ×0.5 / ×1 / ×2）。测试期数字只描述这一次样本。")
    print(view.to_string(index=False))
    out = Path(args.output)
    csv_path, md_path = export_weekly_orders(result.portfolio.orders, out / "weekly_orders")
    print(f"清单 {csv_path}")
    print(f"说明 {md_path}")
    print("不会发送任何委托。")
    return 0


def _demo_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=40)
    rows = []
    for code in ["000001", "000002", "600000", "600519"]:
        for day in dates:
            close = 6.0 if day.day_of_week == 4 else 10.0
            rows.append(
                {
                    "date": day,
                    "symbol": code,
                    "open": close,
                    "close": close,
                    "amount": 30_000_000.0,
                    "industry": "银行" if code.startswith("0") else "食品饮料",
                    "listed_date": "2010-01-04",
                    "is_st": False,
                    "suspend": False,
                    "name": code,
                }
            )
    return pd.DataFrame(rows)


def _load_cache(symbols: list[str]) -> pd.DataFrame:
    frames = []
    cache = ROOT / ".cache" / "bars"
    for raw in symbols:
        code = normalize_symbol(raw)
        path = cache / f"{code}_qfq.parquet"
        if not path.exists():
            raise SystemExit(f"本地没有 {code} 的前复权缓存：{path}。本脚本不联网补数。")
        frame = pd.read_parquet(path)
        frame["symbol"] = code
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    if "amount" not in panel.columns:
        panel["amount"] = pd.NA
    panel["listed_date"] = "2010-01-04"
    panel["is_st"] = False
    panel["suspend"] = False
    panel["industry"] = pd.NA
    print("缓存行情没有行业字段，约束会降级为单票和总仓。")
    return panel


def _cached_csi300(start, end) -> pd.DataFrame | None:
    frame = read_csi300_offline(start, end, cache_dir=ROOT / ".cache" / "bars")
    if frame is None:
        print("缺少沪深300缓存，暂无相对基准的超额。策略净值仍会计算，不请求指数接口。")
        return None
    print(f"沪深300使用本地缓存 {len(frame)} 根。")
    return frame


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["--demo"]))
