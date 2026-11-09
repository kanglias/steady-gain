"""双均线，带防抖动和简单风险钩子。

示例默认仍是 10/30，但不会只靠把均线改短来减少假交叉。默认还会要求：
连续两日确认、收盘价站上更长的趋势均线、ADX 类强度达标、最短持仓 5 个交易日。
建议仓位默认 0.5，不是写死的满仓。止损默认 8%，用收盘价相对入场收盘价判断，
不是盘中触发。盘中止损不在本阶段。回测可以另选 `next_open_stop`：
触发仍看收盘，假定下一交易日开盘才卖出。止盈钩子默认关闭。

这是说明用的规则，不是交易指令，也不承诺收益。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import Strategy, StrategyResult, prepare_bars, register


def _window(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 需要是整数")
    if value < 1:
        raise ValueError(f"{name} 至少为 1")
    return value


def _optional_window(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _window(name, value)


def adx_like(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Wilder 平滑的 ADX 近似，用来挡震荡里的假交叉。缺高低价时不要调用。"""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = _wilder(true_range, window)
    plus_di = 100 * _wilder(plus_dm, window) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, window) / atr.replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return _wilder(dx, window)


def _wilder(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _streak(mask: pd.Series) -> pd.Series:
    """连续为真的天数。刚变成真的那一天记为 1。"""
    started = mask & ~mask.shift(fill_value=False)
    groups = started.cumsum()
    streak = mask.groupby(groups).cumcount() + 1
    return streak.where(mask, 0)


@register
class DualMovingAverage(Strategy):
    """收盘价上的均线交叉，加上确认、趋势、最短持仓和止损钩子。

    把 `confirm_days=1`、`min_hold_days=1`、`trend_window=None`、`adx_window=None`、
    `target_weight=1`、`stop_loss=None` 设上，就退回「裸的 0/1 交叉」。
    日常示例不要这么做：假交叉要靠这些过滤来减，而不是把短均线再改短。
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        *,
        trend_window: int | None = 60,
        confirm_days: int = 2,
        min_hold_days: int = 5,
        target_weight: float = 0.5,
        stop_loss: float | None = 0.08,
        take_profit: float | None = None,
        adx_window: int | None = 14,
        adx_min: float | None = 20.0,
        price_col: str = "close",
    ):
        self.short_window = _window("short_window", short_window)
        self.long_window = _window("long_window", long_window)
        if self.short_window >= self.long_window:
            raise ValueError("短均线窗口必须小于长均线窗口")
        self.trend_window = _optional_window("trend_window", trend_window)
        if self.trend_window is not None and self.trend_window <= self.long_window:
            raise ValueError("趋势均线窗口必须长于长均线，过滤才有单独的意义")
        self.confirm_days = _window("confirm_days", confirm_days)
        self.min_hold_days = _window("min_hold_days", min_hold_days)
        if isinstance(target_weight, bool) or not isinstance(target_weight, (int, float)):
            raise TypeError("target_weight 需要是数字")
        if not 0 < float(target_weight) <= 1:
            raise ValueError("target_weight 需要在 (0, 1] 内，1 表示满仓建议")
        self.target_weight = float(target_weight)
        self.stop_loss = _positive_rate("stop_loss", stop_loss)
        self.take_profit = _positive_rate("take_profit", take_profit)
        self.adx_window = _optional_window("adx_window", adx_window)
        if adx_min is not None and (isinstance(adx_min, bool) or not isinstance(adx_min, (int, float))):
            raise TypeError("adx_min 需要是数字")
        if adx_min is not None and float(adx_min) < 0:
            raise ValueError("adx_min 不能为负")
        self.adx_min = None if adx_min is None else float(adx_min)
        self.price_col = price_col
        self.name = f"dual_ma_{self.short_window}_{self.long_window}"

    def run(self, bars: pd.DataFrame) -> StrategyResult:
        frame = prepare_bars(bars, self.price_col)
        price = frame[self.price_col].astype(float)
        ma_short = price.rolling(self.short_window, min_periods=self.short_window).mean()
        ma_long = price.rolling(self.long_window, min_periods=self.long_window).mean()
        ma_trend = (
            price.rolling(self.trend_window, min_periods=self.trend_window).mean()
            if self.trend_window
            else pd.Series(np.nan, index=frame.index)
        )
        adx = _maybe_adx(frame, price, self.adx_window)
        above = ma_short.notna() & ma_long.notna() & (ma_short > ma_long)
        below = ma_short.notna() & ma_long.notna() & (ma_short < ma_long)
        confirmed_up = _streak(above) == self.confirm_days
        confirmed_down = _streak(below) == self.confirm_days

        symbol = _symbol_of(frame)
        use_adx = self.adx_window is not None and bool(adx.notna().any())
        weight = np.zeros(len(frame), dtype=float)
        signal_rows: list[dict] = []
        holding = 0.0
        entry_index: int | None = None
        entry_price: float | None = None

        for i in range(len(frame)):
            close_i = float(price.iloc[i])
            if holding == 0.0:
                if bool(confirmed_up.iloc[i]) and _entry_allowed(
                    close_i,
                    ma_trend.iloc[i],
                    adx.iloc[i],
                    self.trend_window,
                    self.adx_min if use_adx else None,
                ):
                    holding = self.target_weight
                    entry_index = i
                    entry_price = close_i
                    signal_rows.append(
                        _signal(
                            frame,
                            i,
                            symbol,
                            "buy",
                            close_i,
                            holding,
                            self._buy_reason(),
                        )
                    )
            else:
                assert entry_index is not None and entry_price is not None
                held_days = i - entry_index
                change = close_i / entry_price - 1.0
                exit_reason = self._exit_reason(change, held_days, bool(confirmed_down.iloc[i]))
                if exit_reason:
                    holding = 0.0
                    entry_index = None
                    entry_price = None
                    signal_rows.append(_signal(frame, i, symbol, "sell", close_i, 0.0, exit_reason))
            weight[i] = holding

        signals = pd.DataFrame(
            signal_rows,
            columns=["date", "symbol", "side", "price", "weight", "reason"],
        )
        positions = pd.DataFrame(
            {"date": frame["date"].to_numpy(), "symbol": symbol, "weight": weight}
        )
        indicators = pd.DataFrame(
            {
                "date": frame["date"].to_numpy(),
                "ma_short": ma_short.to_numpy(dtype="float64"),
                "ma_long": ma_long.to_numpy(dtype="float64"),
                "ma_trend": ma_trend.to_numpy(dtype="float64"),
                "adx": adx.to_numpy(dtype="float64"),
            }
        )
        return StrategyResult(signals=signals, positions=positions, indicators=indicators)

    def _buy_reason(self) -> str:
        parts = [f"短均线高于长均线已连续 {self.confirm_days} 日"]
        if self.trend_window:
            parts.append(f"收盘价在 {self.trend_window} 日趋势均线之上")
        if self.adx_window and self.adx_min is not None:
            parts.append(f"ADX 近似不低于 {self.adx_min:.0f}")
        parts.append(f"建议仓位 {self.target_weight:.0%}")
        return "，".join(parts)

    def _exit_reason(self, change: float, held_days: int, death_confirmed: bool) -> str:
        if self.stop_loss is not None and change <= -self.stop_loss:
            return f"收盘较开仓价回撤达到止损 {self.stop_loss:.0%}"
        if self.take_profit is not None and change >= self.take_profit:
            return f"收盘较开仓价达到止盈 {self.take_profit:.0%}"
        if death_confirmed and held_days >= self.min_hold_days:
            return (
                f"短均线低于长均线已连续 {self.confirm_days} 日，"
                f"且持仓已满 {self.min_hold_days} 日"
            )
        return ""


def _positive_rate(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 需要是数字")
    if float(value) <= 0:
        raise ValueError(f"{name} 需要大于 0")
    return float(value)


def _maybe_adx(frame: pd.DataFrame, close: pd.Series, window: int | None) -> pd.Series:
    empty = pd.Series(np.nan, index=frame.index, dtype="float64")
    if window is None or "high" not in frame.columns or "low" not in frame.columns:
        return empty
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    if high.isna().all() or low.isna().all():
        return empty
    return adx_like(high, low, close, window)


def _entry_allowed(close: float, trend: float, adx: float, trend_window: int | None, adx_min: float | None) -> bool:
    if trend_window is not None and (pd.isna(trend) or close <= float(trend)):
        return False
    if adx_min is not None and (pd.isna(adx) or float(adx) < adx_min):
        return False
    return True


def _symbol_of(frame: pd.DataFrame) -> str:
    if "symbol" in frame.columns and frame["symbol"].notna().any():
        return str(frame["symbol"].dropna().iloc[0])
    return ""


def _signal(frame: pd.DataFrame, index: int, symbol: str, side: str, price: float, weight: float, reason: str) -> dict:
    return {
        "date": frame["date"].iloc[index],
        "symbol": symbol,
        "side": side,
        "price": price,
        "weight": weight,
        "reason": reason,
    }
