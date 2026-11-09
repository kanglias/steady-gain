"""把手工成交和信号、两种回测假设放在一起看。

滑点用正数表示比假设更差：买入买贵了，或卖出卖便宜了。
没有费用和涨跌停。这不是审计报告，也不是每月 3% 的验收。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest import compare_fills
from data.normalize import normalize_symbol
from journal.log import FillLogError, normalize_fills
from strategy.base import Strategy, StrategyResult


@dataclass
class ActualReport:
    """一只标的的对照结果。

    fills: 每一笔成交，以及相对收盘假设、次日开盘假设的滑点。
    signals: 每一条信号有没有被同方向的成交跟上，股数差多少。
    round_trips: 按移动平均成本结转的已实现盈亏；末尾若仍持仓，另有一行浮动。
    daily: 按日的持仓、盈亏，以及在给出参考资金时的实际权重。
    summary: 汇总。收益缺口只有在提供 equity 时才有意义。
    """

    fills: pd.DataFrame
    signals: pd.DataFrame
    round_trips: pd.DataFrame
    daily: pd.DataFrame
    summary: dict[str, float | int | str]


def compare_actual(
    bars: pd.DataFrame,
    result: StrategyResult | None = None,
    fills: pd.DataFrame | None = None,
    *,
    strategy: Strategy | None = None,
    equity: float | None = None,
    quantity_tolerance: float = 0.01,
    benchmark: pd.DataFrame | None = None,
    benchmark_name: str = "沪深300",
    split_date=None,
) -> ActualReport:
    """对照信号建议和记下来的成交。

    `equity` 是参考资金（元）。给出之后，目标股数 = 资金 × 建议仓位 / 信号价。
    成交股数与目标股数的差不超过 `max(1, tolerance × 目标股数)` 时，视为跟满。
    不给资金时，只判断这条信号有没有同方向的成交，算不出股数是否跟满，
    `fully_followed` 这时表示「有成交」，`actual_return` 留空。

    回测收益来自同一条目标权重的收盘成交假设和次日开盘假设，不含费用。
    实际盈亏用成交价和收盘价做现金加持仓，再减去佣金、印花税和其他费用。
    `actual_return` 是扣费后的收益。示例策略的这个数字不是每月 3% 的验收。

    `benchmark` 是基准的日线，至少要有 date 和 close。不传时仍给出同一标的的买入持有。
    `split_date` 之前的交易日标 `train`，当天及之后标 `test`。测试段收益从分段前的权益接着算。
    """
    if equity is not None and (isinstance(equity, bool) or not isinstance(equity, (int, float)) or equity <= 0):
        raise ValueError("equity 需要是正数")
    if result is None:
        if strategy is None:
            raise TypeError("传入 StrategyResult，或传入 strategy 以便调用 run")
        result = strategy.run(bars)
    book = _prepare_book(bars)
    symbol = _one_symbol(book)
    logged = _fills_for_symbol(fills, symbol)
    assumptions = _assumptions(book)
    enriched = _enrich_fills(logged, assumptions, _signal_keys(result))
    signal_rows, trips, daily, position_end = _walk(
        book,
        result,
        enriched,
        equity=None if equity is None else float(equity),
        quantity_tolerance=float(quantity_tolerance),
    )
    signal_rows = _apply_slice(signal_rows, "date", split_date)
    daily = _apply_slice(daily, "date", split_date)
    summary = _summary(
        book,
        result,
        enriched,
        signal_rows,
        trips,
        daily,
        position_end,
        equity=None if equity is None else float(equity),
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        split_date=split_date,
    )
    return ActualReport(
        fills=enriched,
        signals=signal_rows,
        round_trips=trips,
        daily=daily,
        summary=summary,
    )


def format_report(report: ActualReport) -> str:
    """短的文字报告，给笔记本和命令行看。"""
    item = report.summary
    lines = [
        f"# 成交对照 {item['symbol']}",
        "",
        "手工成交相对信号。正的滑点表示比假设更差。不会发送委托。",
        "",
        f"- 信号 {item['n_signals']} 条，有成交 {item['n_filled_signals']} 条，跟满 {item['n_fully_followed']} 条",
        f"- 成交 {item['n_fills']} 笔，其中对不上信号的 {item['n_unmatched_fills']} 笔",
        f"- 盈亏 {item['total_pnl']:.2f} 元（已实现 {item['realized_pnl']:.2f}，浮动 {item['unrealized_pnl']:.2f}，费用 {item['total_cost']:.2f}）",
        f"- 扣费后的实际收益 {_pct(item['actual_return'])}。跟单对照看次日开盘假设 {_pct(item['backtest_next_open_return'])}",
        f"- 平均滑点 相对收盘 {_pct(item['mean_slippage_vs_close'])}，相对次日开盘 {_pct(item['mean_slippage_vs_next_open'])}",
        f"- 买入持有 {_pct(item['buy_hold_return'])}，实际超额 {_pct(item['excess_vs_buy_hold'])}，次日开盘假设超额 {_pct(item['excess_next_open_vs_buy_hold'])}",
        f"- {item['benchmark_name'] or '基准'} {_pct(item['benchmark_return'])}，实际超额 {_pct(item['excess_vs_benchmark'])}，次日开盘假设超额 {_pct(item['excess_next_open_vs_benchmark'])}",
        f"- 分段 {item['split_date'] or '未分'}：训练实际 {_pct(item['actual_return_train'])} / 测试实际 {_pct(item['actual_return_test'])}",
        "",
        "跟单视图锁定为 next_open。收盘假设只作对照。",
        "实际收益已扣除成交记录里的佣金、印花税和其他费用。它和回测不是同一套账。",
        "示例策略的收益不是每月 3% 的验收。",
    ]
    if item.get("has_simulated"):
        lines.insert(2, "这些成交带有 is_simulated=true，是演示记录，不是实盘业绩。")
    missed = report.signals.loc[~report.signals["fully_followed"]] if not report.signals.empty else report.signals
    if not missed.empty:
        lines.extend(["", "未跟满的信号："])
        for row in missed.head(8).itertuples(index=False):
            lines.append(
                f"- {row.date} {row.side} 建议仓位 {_pct(row.suggested_weight)} "
                f"成交 {row.filled_quantity:g} 股 / 目标 {_num(row.target_quantity)}"
            )
    return "\n".join(lines) + "\n"


def _walk(
    book: pd.DataFrame,
    result: StrategyResult,
    fills: pd.DataFrame,
    *,
    equity: float | None,
    quantity_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cash = 0.0 if equity is None else float(equity)
    start_cash = cash
    shares = 0.0
    avg_cost = 0.0
    realized = 0.0
    costs = 0.0
    assigned = _assign_fills(book, fills)
    weights = _weight_on(book, result)
    daily_rows: list[dict] = []
    trip_rows: list[dict] = []
    prev_equity = start_cash
    open_signal = ""

    for pos in range(len(book)):
        day = pd.Timestamp(book["date"].iloc[pos])
        day_realized = 0.0
        for fill in assigned.get(day, []):
            quantity = float(fill.quantity)
            price = float(fill.price)
            cost = _fill_cost(fill)
            costs += cost
            if fill.side == "buy":
                avg_cost = (avg_cost * shares + price * quantity + cost) / (shares + quantity)
                cash -= price * quantity + cost
                shares += quantity
                if not open_signal:
                    open_signal = str(fill.signal_date)
            else:
                if quantity > shares + 1e-6:
                    raise FillLogError(
                        f"{day:%Y-%m-%d} 卖出 {quantity:g} 股，超过当时持仓 {shares:g} 股"
                    )
                trade_pnl = (price - avg_cost) * quantity - cost
                day_realized += trade_pnl
                realized += trade_pnl
                cash += price * quantity - cost
                shares -= quantity
                trip_rows.append(
                    {
                        "symbol": fill.symbol,
                        "status": "closed",
                        "quantity": quantity,
                        "avg_cost": avg_cost,
                        "exit_price": price,
                        "pnl": trade_pnl,
                        "signal_date": fill.signal_date,
                        "opened_signal_date": open_signal,
                        "timestamp": fill.timestamp,
                    }
                )
                if shares <= 1e-6:
                    shares = 0.0
                    avg_cost = 0.0
                    open_signal = ""
        close = float(book["close"].iloc[pos])
        end_equity = cash + shares * close
        actual_weight = pd.NA
        if equity is not None and end_equity != 0:
            actual_weight = shares * close / end_equity
        daily_rows.append(
            {
                "date": day,
                "close": close,
                "target_weight": float(weights[pos]),
                "shares": shares,
                "cash": cash,
                "equity": end_equity,
                "daily_pnl": end_equity - prev_equity,
                "actual_weight": actual_weight,
            }
        )
        prev_equity = end_equity

    last_close = float(book["close"].iloc[-1]) if len(book) else 0.0
    unrealized = shares * (last_close - avg_cost) if shares else 0.0
    if shares > 0:
        trip_rows.append(
            {
                "symbol": _one_symbol(book),
                "status": "open",
                "quantity": shares,
                "avg_cost": avg_cost,
                "exit_price": last_close,
                "pnl": unrealized,
                "signal_date": "",
                "opened_signal_date": open_signal,
                "timestamp": "",
            }
        )
    signal_rows = _score_signals(
        result,
        fills,
        equity=equity,
        quantity_tolerance=quantity_tolerance,
    )
    position_end = {
        "start_cash": start_cash,
        "end_equity": prev_equity,
        "realized": realized,
        "unrealized": unrealized,
        "shares": shares,
        "total_cost": costs,
    }
    return (
        signal_rows,
        pd.DataFrame(trip_rows),
        pd.DataFrame(daily_rows),
        position_end,
    )


def _score_signals(
    result: StrategyResult,
    fills: pd.DataFrame,
    *,
    equity: float | None,
    quantity_tolerance: float,
) -> pd.DataFrame:
    signals = _signal_frame(result)
    rows: list[dict] = []
    shares = 0.0
    grouped = _fills_by_signal(fills)
    for signal in signals.itertuples(index=False):
        key = (str(signal.date), signal.side)
        matched = grouped.get(key, [])
        filled = float(sum(item.quantity for item in matched))
        price = float(signal.price)
        weight = float(signal.weight)
        target_quantity = pd.NA
        quantity_ratio = pd.NA
        if equity is None:
            fully = filled > 0
        else:
            target_shares = float(equity) * weight / price if price > 0 else 0.0
            target_quantity = abs(target_shares - shares)
            slack = max(1.0, quantity_tolerance * float(target_quantity))
            fully = abs(filled - float(target_quantity)) <= slack
            if float(target_quantity) > 0 and filled <= 0:
                fully = False
            quantity_ratio = filled / float(target_quantity) if float(target_quantity) > 1e-9 else (1.0 if filled == 0 else pd.NA)
        signed = filled if signal.side == "buy" else -filled
        shares += signed
        if shares < -1e-6:
            shares = 0.0
        rows.append(
            {
                "date": signal.date,
                "symbol": signal.symbol,
                "side": signal.side,
                "price": price,
                "suggested_weight": weight,
                "filled_quantity": filled,
                "target_quantity": target_quantity,
                "quantity_ratio": quantity_ratio,
                "fully_followed": bool(fully),
            }
        )
    return pd.DataFrame(rows)


def _summary(
    book: pd.DataFrame,
    result: StrategyResult,
    fills: pd.DataFrame,
    signals: pd.DataFrame,
    trips: pd.DataFrame,
    daily: pd.DataFrame,
    position_end: dict[str, float],
    *,
    equity: float | None,
    benchmark: pd.DataFrame | None,
    benchmark_name: str,
    split_date,
) -> dict[str, float | int | str | bool]:
    matched = fills.loc[fills["matched"]] if not fills.empty else fills
    close_slip = matched["slippage_vs_close"] if not matched.empty else pd.Series(dtype=float)
    open_slip = matched["slippage_vs_next_open"] if not matched.empty else pd.Series(dtype=float)
    close_return, next_return = _backtest_returns(book, result)
    total_pnl = float(position_end["end_equity"] - position_end["start_cash"])
    actual_return = total_pnl / equity if equity else float("nan")
    n_signals = int(len(signals))
    n_filled = int(signals["filled_quantity"].gt(0).sum()) if n_signals else 0
    n_full = int(signals["fully_followed"].sum()) if n_signals else 0
    buy_hold = _price_return(book["close"])
    bench = _aligned_benchmark(book, benchmark)
    bench_return = _price_return(bench["close"]) if bench is not None else float("nan")
    train_actual, test_actual, train_hold, test_hold = _slice_returns(daily, position_end["start_cash"], equity)
    train_bench, test_bench = _slice_benchmark(bench, split_date)
    has_simulated = bool(fills["is_simulated"].map(_is_true).any()) if not fills.empty and "is_simulated" in fills.columns else False
    return {
        "symbol": _one_symbol(book),
        "n_signals": n_signals,
        "n_filled_signals": n_filled,
        "n_fully_followed": n_full,
        "n_fills": int(len(fills)),
        "n_unmatched_fills": int((~fills["matched"]).sum()) if not fills.empty else 0,
        "total_pnl": total_pnl,
        "realized_pnl": float(position_end["realized"]),
        "unrealized_pnl": float(position_end["unrealized"]),
        "total_cost": float(position_end["total_cost"]),
        "mean_slippage_vs_close": float(close_slip.dropna().mean()) if close_slip.notna().any() else float("nan"),
        "mean_slippage_vs_next_open": float(open_slip.dropna().mean()) if open_slip.notna().any() else float("nan"),
        "backtest_close_return": close_return,
        "backtest_next_open_return": next_return,
        "actual_return": float(actual_return),
        "return_gap_vs_close": float(actual_return - close_return) if equity else float("nan"),
        "return_gap_vs_next_open": float(actual_return - next_return) if equity and pd.notna(next_return) else float("nan"),
        "buy_hold_return": buy_hold,
        "excess_vs_buy_hold": _excess(actual_return, buy_hold),
        "excess_next_open_vs_buy_hold": _excess(next_return, buy_hold),
        "benchmark_name": benchmark_name if benchmark is not None else "",
        "benchmark_return": bench_return,
        "excess_vs_benchmark": _excess(actual_return, bench_return),
        "excess_next_open_vs_benchmark": _excess(next_return, bench_return),
        "split_date": "" if split_date is None else pd.Timestamp(split_date).strftime("%Y-%m-%d"),
        "actual_return_train": train_actual,
        "actual_return_test": test_actual,
        "buy_hold_return_train": train_hold,
        "buy_hold_return_test": test_hold,
        "excess_vs_buy_hold_train": _excess(train_actual, train_hold),
        "excess_vs_buy_hold_test": _excess(test_actual, test_hold),
        "benchmark_return_train": train_bench,
        "benchmark_return_test": test_bench,
        "has_simulated": has_simulated,
        "follow_fill": "next_open",
        "equity": float(equity) if equity else float("nan"),
        "n_round_trips": int(trips["status"].eq("closed").sum()) if not trips.empty and "status" in trips.columns else 0,
        "n_days": int(len(daily)),
    }


def _backtest_returns(book: pd.DataFrame, result: StrategyResult) -> tuple[float, float]:
    positions = result.positions.copy()
    positions["date"] = pd.to_datetime(positions["date"]).dt.normalize()
    dummy = StrategyResult(signals=result.signals, positions=positions, indicators=result.indicators)
    has_open = "open" in book.columns and book["open"].notna().all()
    try:
        compared = compare_fills(book, dummy) if has_open else None
    except (TypeError, ValueError):
        compared = None
    if compared is None:
        from backtest import run_backtest

        close_return = float(run_backtest(book, dummy, fill="close").total_return)
        return close_return, float("nan")
    return float(compared.close_fill.total_return), float(compared.next_open_fill.total_return)


def _signal_keys(result: StrategyResult) -> set[tuple[str, str]]:
    frame = _signal_frame(result)
    return {(str(row.date), str(row.side)) for row in frame.itertuples(index=False)}


def _enrich_fills(fills: pd.DataFrame, assumptions: pd.DataFrame, signal_keys: set[tuple[str, str]]) -> pd.DataFrame:
    if fills.empty:
        frame = fills.copy()
        for column in ("matched", "close_assumption", "next_open_assumption", "slippage_vs_close", "slippage_vs_next_open"):
            frame[column] = pd.Series(dtype="object")
        return frame
    lookup = assumptions.set_index("date")
    rows: list[dict] = []
    for fill in fills.itertuples(index=False):
        day = pd.Timestamp(fill.signal_date)
        close_price = pd.NA
        next_open = pd.NA
        if day in lookup.index:
            close_price = lookup.at[day, "close"]
            next_open = lookup.at[day, "next_open"]
        close_value = float(close_price) if pd.notna(close_price) else float("nan")
        next_value = float(next_open) if pd.notna(next_open) else float("nan")
        rows.append(
            {
                "symbol": fill.symbol,
                "side": fill.side,
                "price": float(fill.price),
                "quantity": float(fill.quantity),
                "timestamp": fill.timestamp,
                "signal_date": fill.signal_date,
                "matched": (str(fill.signal_date), fill.side) in signal_keys,
                "close_assumption": close_value,
                "next_open_assumption": next_value,
                "slippage_vs_close": _worse(fill.side, float(fill.price), close_value),
                "slippage_vs_next_open": _worse(fill.side, float(fill.price), next_value),
                "commission": _amount(fill, "commission"),
                "stamp_tax": _amount(fill, "stamp_tax"),
                "other_cost": _amount(fill, "other_cost"),
                "is_simulated": _is_true(getattr(fill, "is_simulated", False)),
            }
        )
    return pd.DataFrame(rows)


def _fill_cost(fill) -> float:
    return _amount(fill, "commission") + _amount(fill, "stamp_tax") + _amount(fill, "other_cost")


def _amount(fill, name: str) -> float:
    value = getattr(fill, name, 0.0)
    if value is None or value != value:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value != value:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _apply_slice(frame: pd.DataFrame, date_col: str, split_date) -> pd.DataFrame:
    labeled = frame.copy()
    if labeled.empty or date_col not in labeled.columns or split_date is None:
        labeled["slice"] = ""
        return labeled
    split_ts = pd.Timestamp(split_date).normalize()
    days = pd.to_datetime(labeled[date_col], errors="coerce").dt.normalize()
    labeled["slice"] = "test"
    labeled.loc[days < split_ts, "slice"] = "train"
    labeled.loc[days.isna(), "slice"] = ""
    return labeled


def _price_return(close: pd.Series) -> float:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if len(values) < 2 or float(values.iloc[0]) <= 0:
        return float("nan")
    return float(values.iloc[-1] / values.iloc[0] - 1.0)


def _excess(left: float, right: float) -> float:
    if left != left or right != right:
        return float("nan")
    return float(left - right)


def _aligned_benchmark(book: pd.DataFrame, benchmark: pd.DataFrame | None) -> pd.DataFrame | None:
    if benchmark is None or benchmark.empty or "date" not in benchmark.columns or "close" not in benchmark.columns:
        return None
    right = benchmark.copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    right["close"] = pd.to_numeric(right["close"], errors="coerce")
    right = right.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    merged = book[["date"]].merge(right[["date", "close"]], on="date", how="inner")
    if len(merged) < 2:
        return None
    return merged.reset_index(drop=True)


def _slice_returns(daily: pd.DataFrame, start_cash: float, equity: float | None) -> tuple[float, float, float, float]:
    if daily.empty or "slice" not in daily.columns or not daily["slice"].isin(["train", "test"]).any():
        return (float("nan"),) * 4
    train = daily.loc[daily["slice"].eq("train")]
    test = daily.loc[daily["slice"].eq("test")]
    train_hold = _price_return(train["close"]) if not train.empty else float("nan")
    if train.empty:
        test_hold = _price_return(test["close"]) if not test.empty else float("nan")
    elif test.empty:
        test_hold = float("nan")
    else:
        base = float(train["close"].iloc[-1])
        test_hold = float(test["close"].iloc[-1] / base - 1.0) if base > 0 else float("nan")
    if not equity or start_cash == 0:
        return float("nan"), float("nan"), train_hold, test_hold
    train_actual = float(train["equity"].iloc[-1] / start_cash - 1.0) if not train.empty else float("nan")
    if not test.empty and not train.empty and float(train["equity"].iloc[-1]) != 0:
        test_actual = float(test["equity"].iloc[-1] / float(train["equity"].iloc[-1]) - 1.0)
    elif not test.empty:
        test_actual = float(test["equity"].iloc[-1] / start_cash - 1.0)
    else:
        test_actual = float("nan")
    return train_actual, test_actual, train_hold, test_hold


def _slice_benchmark(bench: pd.DataFrame | None, split_date) -> tuple[float, float]:
    if bench is None or split_date is None:
        return float("nan"), float("nan")
    labeled = _apply_slice(bench, "date", split_date)
    train = labeled.loc[labeled["slice"].eq("train"), "close"]
    test = labeled.loc[labeled["slice"].eq("test"), "close"]
    train_return = _price_return(train)
    if train.empty or test.empty:
        return train_return, _price_return(test)
    base = float(train.iloc[-1])
    test_return = float(test.iloc[-1] / base - 1.0) if base > 0 else float("nan")
    return train_return, test_return


def _worse(side: str, fill_price: float, assumption: float) -> float:
    if assumption != assumption or assumption <= 0:
        return float("nan")
    if side == "buy":
        return fill_price / assumption - 1.0
    return assumption / fill_price - 1.0


def _assumptions(book: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"date": book["date"], "close": book["close"].astype(float)})
    if "open" in book.columns:
        frame["next_open"] = book["open"].shift(-1)
    else:
        frame["next_open"] = pd.NA
    return frame


def _assign_fills(book: pd.DataFrame, fills: pd.DataFrame) -> dict[pd.Timestamp, list]:
    dates = [pd.Timestamp(day) for day in book["date"]]
    date_set = set(dates)
    assigned: dict[pd.Timestamp, list] = {day: [] for day in dates}
    if fills.empty:
        return assigned
    ordered = fills.copy()
    ordered["_ts"] = pd.to_datetime(ordered["timestamp"], format="mixed")
    ordered = ordered.sort_values("_ts")
    for fill in ordered.itertuples(index=False):
        day = pd.Timestamp(fill.timestamp).normalize()
        if day not in date_set:
            later = [item for item in dates if item >= day]
            if not later:
                raise FillLogError(f"成交时间 {fill.timestamp} 晚于行情最后一天")
            day = later[0]
        assigned[day].append(fill)
    return assigned


def _fills_for_symbol(fills: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    frame = normalize_fills(fills if fills is not None else pd.DataFrame(columns=["symbol", "side", "price", "quantity", "timestamp", "signal_date"]))
    if frame.empty or not symbol:
        return frame
    return frame.loc[frame["symbol"] == symbol].reset_index(drop=True)


def _fills_by_signal(fills: pd.DataFrame) -> dict[tuple[str, str], list]:
    grouped: dict[tuple[str, str], list] = {}
    if fills.empty:
        return grouped
    for fill in fills.itertuples(index=False):
        grouped.setdefault((str(fill.signal_date), fill.side), []).append(fill)
    return grouped


def _signal_frame(result: StrategyResult) -> pd.DataFrame:
    signals = result.signals.copy() if result.signals is not None else pd.DataFrame()
    columns = ["date", "symbol", "side", "price", "weight"]
    if signals.empty:
        return pd.DataFrame(columns=columns)
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["side"] = frame["side"].astype(str).str.lower()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if "weight" not in frame.columns:
        frame["weight"] = pd.NA
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].map(lambda value: normalize_symbol(value) if str(value).strip() else "")
    else:
        frame["symbol"] = ""
    frame = frame.dropna(subset=["date", "price", "weight"])
    return frame.loc[:, columns].reset_index(drop=True)


def _weight_on(book: pd.DataFrame, result: StrategyResult) -> list[float]:
    positions = result.positions.copy()
    positions["date"] = pd.to_datetime(positions["date"], errors="coerce").dt.normalize()
    weights = {
        pd.Timestamp(day): float(weight)
        for day, weight in zip(positions["date"], positions["weight"], strict=False)
        if pd.notna(day) and pd.notna(weight)
    }
    return [weights.get(pd.Timestamp(day), 0.0) for day in book["date"]]


def _prepare_book(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("行情需要是 pandas.DataFrame")
    required = {"date", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError("行情缺少列: " + ", ".join(sorted(missing)))
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "open" in frame.columns:
        frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError("行情为空")
    if (frame["close"] <= 0).any():
        raise ValueError("收盘价必须为正")
    if "symbol" in frame.columns and frame["symbol"].notna().any():
        symbols = {normalize_symbol(value) for value in frame["symbol"].dropna().unique()}
        if len(symbols) > 1:
            raise ValueError("成交对照按单只标的计算，请先按 symbol 拆开")
        frame["symbol"] = frame["symbol"].map(lambda value: normalize_symbol(value))
    return frame.reset_index(drop=True)


def _one_symbol(book: pd.DataFrame) -> str:
    if "symbol" in book.columns and book["symbol"].notna().any():
        return str(book["symbol"].dropna().iloc[0])
    return ""


def _pct(value: object) -> str:
    if value is None or value != value:
        return "—"
    return f"{float(value):.2%}"


def _num(value: object) -> str:
    if value is None or value != value:
        return "—"
    return f"{float(value):.2f}"
