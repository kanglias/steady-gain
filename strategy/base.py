"""策略统一接口。

策略只产出信号和目标仓位，收益在 backtest 里算。新增策略时继承 `Strategy`，
实现 `run`，并用 `@register` 登记。`run` 一次只接受一只标的。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """按类名登记策略。重复导入同一类会覆盖为同一个对象。"""
    REGISTRY[cls.__name__] = cls
    return cls


def available_strategies() -> list[str]:
    return sorted(REGISTRY)


@dataclass
class StrategyResult:
    """一次策略运行的结果。

    signals: 只在出现买卖动作的交易日有行。列：date, symbol, side, price, reason。
        side 取 `buy` 或 `sell`。
    positions: 每个交易日一行的目标仓位。列：date, symbol, weight。
        权重在 0 到 1 之间，表示建议仓位，不是下单股数。
    indicators: 与行情对齐的指标，供画图。至少含 date。
    """

    signals: pd.DataFrame
    positions: pd.DataFrame
    indicators: pd.DataFrame


class Strategy(ABC):
    name: str

    @abstractmethod
    def run(self, bars: pd.DataFrame) -> StrategyResult:
        """输入日线，输出信号、目标仓位和指标。"""


def prepare_bars(bars: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("行情需要是 pandas.DataFrame")
    if "date" not in bars.columns:
        raise ValueError("行情需要 date 列")
    if price_col not in bars.columns:
        raise ValueError(f"行情缺少价格列 {price_col}")

    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
    frame = frame.dropna(subset=["date", price_col])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("行情为空")

    if "symbol" in frame.columns:
        symbols = [str(item) for item in frame["symbol"].dropna().unique().tolist()]
        if len(symbols) > 1:
            raise ValueError("策略按单只标的运行，请先按 symbol 拆开")
    return frame
