"""把一只或多只标的的策略信号收成建议表。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from data.normalize import parse_date
from strategy.base import StrategyResult
from strategy.diagnostics import DIAGNOSTIC_COLUMNS, annotate_signals

BOARD_COLUMNS = [
    "symbol",
    "side",
    "direction",
    "date",
    "strategy",
    "reason",
    "price",
    *DIAGNOSTIC_COLUMNS,
]
_DIRECTION = {"buy": "买入", "sell": "卖出"}


def signal_board(
    results: StrategyResult | Mapping[str, StrategyResult] | Iterable[StrategyResult],
    *,
    strategy_name: str,
    on=None,
    start=None,
    end=None,
    bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    flip_days: int = 5,
) -> pd.DataFrame:
    """买卖建议。

    除了代码、方向、日期、策略和原因，还有建议仓位、持仓天数、当前这笔的浮动盈亏、
    连续亏损段数、距上次交叉的交易日，以及「短持翻脸」标记。
    `side` 仍是 `buy` / `sell`，`direction` 是买入 / 卖出。建议仓位可以小于 1。

    `on` 只看某一天（日常入口用最近一个交易日，不是自然日的「今天」）。
    `start` / `end` 看闭区间。不要同时传 `on` 和区间。
    三个日期都不传时，使用这些结果里仓位表的最后一天，因此当天没有交叉时表是空的。
    """
    if on is not None and (start is not None or end is not None):
        raise ValueError("on 与 start/end 只能选一种")
    frames = [
        _signals_of(item, strategy_name, _bars_for(item, bars), flip_days) for item in _iter_results(results)
    ]
    board = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BOARD_COLUMNS)
    if board.empty:
        return pd.DataFrame(columns=BOARD_COLUMNS)
    board["date"] = pd.to_datetime(board["date"], errors="coerce").dt.normalize()
    board = board.dropna(subset=["date", "symbol", "side"])
    if on is not None:
        day = pd.Timestamp(parse_date(on))
        board = board.loc[board["date"] == day]
    else:
        if start is None and end is None:
            last = _last_session(results)
            if last is None:
                board = board.iloc[0:0]
            else:
                board = board.loc[board["date"] == last]
        if start is not None:
            board = board.loc[board["date"] >= pd.Timestamp(parse_date(start))]
        if end is not None:
            board = board.loc[board["date"] <= pd.Timestamp(parse_date(end))]
    board = board.sort_values(["date", "symbol", "side"]).reset_index(drop=True)
    return board.loc[:, BOARD_COLUMNS]


def _iter_results(results):
    if isinstance(results, StrategyResult):
        yield results
        return
    if isinstance(results, Mapping):
        yield from results.values()
        return
    yield from results


def _signals_of(
    result: StrategyResult,
    strategy_name: str,
    bars: pd.DataFrame | None,
    flip_days: int,
) -> pd.DataFrame:
    signals = annotate_signals(result, bars, flip_days=flip_days)
    if signals.empty:
        return pd.DataFrame(columns=BOARD_COLUMNS)
    frame = signals.copy()
    frame["side"] = frame["side"].astype(str).str.lower()
    frame = frame.loc[frame["side"].isin(["buy", "sell"])]
    if "symbol" not in frame.columns:
        frame["symbol"] = ""
    if "reason" not in frame.columns:
        frame["reason"] = ""
    if "price" not in frame.columns:
        frame["price"] = pd.NA
    frame["direction"] = frame["side"].map(_DIRECTION)
    frame["strategy"] = strategy_name
    frame["symbol"] = frame["symbol"].astype(str)
    for column in DIAGNOSTIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, BOARD_COLUMNS]


def _bars_for(result: StrategyResult, bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None):
    if bars is None or isinstance(bars, pd.DataFrame):
        return bars
    positions = result.positions
    if positions is None or positions.empty or "symbol" not in positions.columns:
        return None
    symbol = str(positions["symbol"].dropna().iloc[0])
    return bars.get(symbol)


def _last_session(results) -> pd.Timestamp | None:
    last = None
    for result in _iter_results(results):
        positions = result.positions
        if positions is None or positions.empty or "date" not in positions.columns:
            continue
        day = pd.to_datetime(positions["date"], errors="coerce").max()
        if pd.isna(day):
            continue
        day = pd.Timestamp(day).normalize()
        if last is None or day > last:
            last = day
    return last
