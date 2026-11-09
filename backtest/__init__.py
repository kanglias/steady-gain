"""单标的研究回测：收益、最大回撤、换手。

没有滑点、费用、涨跌停。这里的成交是价格假设。手工成交记在 journal。
"""

from backtest.weekly import cost_sensitivity, format_weekly_report, run_weekly_backtest
from backtest.engine import (
    FEE_DISCLAIMER,
    FOLLOW_LABEL,
    BacktestResult,
    FillComparison,
    compare_fills,
    fill_assumption_table,
    run_backtest,
)

__all__ = [
    "FEE_DISCLAIMER",
    "FOLLOW_LABEL",
    "BacktestResult",
    "FillComparison",
    "compare_fills",
    "cost_sensitivity",
    "fill_assumption_table",
    "format_weekly_report",
    "run_backtest",
    "run_weekly_backtest",
]
