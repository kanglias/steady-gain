"""收盘价相对自身均线的简单均值回归。

价格连续几天明显低于均线才开仓，回到均线附近或触发止损才离开。
这不是双均线，也不靠把短均线改短来增加交易。示例不承诺收益。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import Strategy, StrategyResult, prepare_bars, register


@register
class MeanReversion(Strategy):
    """低于均线一定幅度买入，回到均线后卖出。

    `window` 是用来衡量偏离的均线长度，默认 20，不是另一套 10/30 交叉，
    也不要把窗口改短去凑每月 3%。建议仓位默认 0.5。止损默认 8%，看收盘价，
    不是盘中触发。最短持有默认 3 个交易日。
    """

    def __init__(
        self,
        window: int = 20,
        *,
        entry_band: float = 0.05,
        exit_band: float = 0.0,
        confirm_days: int = 2,
        min_hold_days: int = 3,
        target_weight: float = 0.5,
        stop_loss: float | None = 0.08,
        price_col: str = "close",
    ):
        self.window = _window("window", window)
        self.entry_band = _band("entry_band", entry_band)
        self.exit_band = _band("exit_band", exit_band, allow_zero=True)
        if self.exit_band > 0:
            raise ValueError("exit_band 不能大于 0，回到均线或均线之上才算回归")
        self.confirm_days = _window("confirm_days", confirm_days)
        self.min_hold_days = _window("min_hold_days", min_hold_days)
        if isinstance(target_weight, bool) or not isinstance(target_weight, (int, float)):
            raise TypeError("target_weight 需要是数字")
        if not 0 < float(target_weight) <= 1:
            raise ValueError("target_weight 需要在 (0, 1] 内")
        self.target_weight = float(target_weight)
        self.stop_loss = None if stop_loss is None else _band("stop_loss", stop_loss)
        self.price_col = price_col
        self.name = f"mean_reversion_{self.window}"

    def run(self, bars: pd.DataFrame) -> StrategyResult:
        frame = prepare_bars(bars, self.price_col)
        price = frame[self.price_col].astype(float)
        average = price.rolling(self.window, min_periods=self.window).mean()
        distance = price / average - 1.0
        stretched = distance.notna() & (distance <= -self.entry_band)
        reverted = distance.notna() & (distance >= self.exit_band)
        confirmed = _streak(stretched) == self.confirm_days

        symbol = ""
        if "symbol" in frame.columns and frame["symbol"].notna().any():
            symbol = str(frame["symbol"].dropna().iloc[0])
        weight = np.zeros(len(frame), dtype=float)
        rows: list[dict] = []
        holding = 0.0
        entry_index: int | None = None
        entry_price: float | None = None

        for i in range(len(frame)):
            close_i = float(price.iloc[i])
            if holding == 0.0:
                if bool(confirmed.iloc[i]):
                    holding = self.target_weight
                    entry_index = i
                    entry_price = close_i
                    rows.append(
                        {
                            "date": frame["date"].iloc[i],
                            "symbol": symbol,
                            "side": "buy",
                            "price": close_i,
                            "weight": holding,
                            "reason": (
                                f"收盘价低于 {self.window} 日均线 {self.entry_band:.0%} 已连续 {self.confirm_days} 日，"
                                f"建议仓位 {self.target_weight:.0%}"
                            ),
                        }
                    )
            else:
                assert entry_index is not None and entry_price is not None
                held_days = i - entry_index
                change = close_i / entry_price - 1.0
                reason = ""
                if self.stop_loss is not None and change <= -self.stop_loss:
                    reason = f"收盘较开仓价回撤达到止损 {self.stop_loss:.0%}"
                elif held_days >= self.min_hold_days and bool(reverted.iloc[i]):
                    reason = (
                        f"收盘价回到 {self.window} 日均线附近，"
                        f"且持仓已满 {self.min_hold_days} 日"
                    )
                if reason:
                    holding = 0.0
                    entry_index = None
                    entry_price = None
                    rows.append(
                        {
                            "date": frame["date"].iloc[i],
                            "symbol": symbol,
                            "side": "sell",
                            "price": close_i,
                            "weight": 0.0,
                            "reason": reason,
                        }
                    )
            weight[i] = holding

        signals = pd.DataFrame(rows, columns=["date", "symbol", "side", "price", "weight", "reason"])
        positions = pd.DataFrame({"date": frame["date"].to_numpy(), "symbol": symbol, "weight": weight})
        indicators = pd.DataFrame(
            {
                "date": frame["date"].to_numpy(),
                "ma": average.to_numpy(dtype="float64"),
                "distance": distance.to_numpy(dtype="float64"),
            }
        )
        return StrategyResult(signals=signals, positions=positions, indicators=indicators)


def _window(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 需要是不小于 1 的整数")
    return value


def _band(name: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 需要是数字")
    if allow_zero:
        if float(value) < 0:
            raise ValueError(f"{name} 不能为负")
    elif float(value) <= 0:
        raise ValueError(f"{name} 需要大于 0")
    return float(value)


def mean_reversion_sensitivity(
    bars: pd.DataFrame,
    windows: tuple[int, ...] = (20, 40),
    entry_bands: tuple[float, ...] = (0.05, 0.08),
) -> pd.DataFrame:
    """几组参数在次日开盘假设下的收益。敏感度表，不是寻优，也不是每月 3% 的验收。"""
    from backtest import run_backtest

    rows = []
    for window in windows:
        for band in entry_bands:
            strategy = MeanReversion(window=int(window), entry_band=float(band))
            result = strategy.run(bars)
            stats = run_backtest(bars, result, fill="next_open")
            rows.append(
                {
                    "window": strategy.window,
                    "entry_band": strategy.entry_band,
                    "fill": "next_open",
                    "total_return": stats.total_return,
                    "max_drawdown": stats.max_drawdown,
                    "n_signals": int(len(result.signals)),
                    "note": "敏感度，不是每月 3% 的验收",
                }
            )
    return pd.DataFrame(rows)


def _streak(mask: pd.Series) -> pd.Series:
    started = mask & ~mask.shift(fill_value=False)
    groups = started.cumsum()
    streak = mask.groupby(groups).cumcount() + 1
    return streak.where(mask, 0)
