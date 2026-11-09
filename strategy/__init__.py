"""可插拔策略。输入单只标的的日线，输出信号和目标仓位。"""

from strategy.base import Strategy, StrategyResult, available_strategies, register
from strategy.board import signal_board
from strategy.diagnostics import apply_diagnostics, position_snapshot
from strategy.dual_ma import DualMovingAverage
from strategy.events import EventGuard, event_switch_samples
from strategy.mean_reversion import MeanReversion, mean_reversion_sensitivity
from strategy.weekly_mr import WeeklyMeanReversion

__all__ = [
    "DualMovingAverage",
    "EventGuard",
    "MeanReversion",
    "Strategy",
    "StrategyResult",
    "apply_diagnostics",
    "available_strategies",
    "event_switch_samples",
    "mean_reversion_sensitivity",
    "position_snapshot",
    "register",
    "WeeklyMeanReversion",
    "signal_board",
]
