"""手工成交日志，以及和信号、回测假设的对照。

不连接券商，也不提交委托。演示成交只在 `simulate_follow_fills`，标明是虚构的。
"""

from journal.log import (
    COST_COLUMNS,
    FILL_COLUMNS,
    FILL_OUTPUT_COLUMNS,
    FillLogError,
    append_fills,
    apply_cost_schedule,
    load_fills,
    save_fills,
)
from journal.report import ActualReport, compare_actual, format_report
from journal.simulate import simulate_follow_fills

__all__ = [
    "COST_COLUMNS",
    "FILL_COLUMNS",
    "FILL_OUTPUT_COLUMNS",
    "ActualReport",
    "FillLogError",
    "append_fills",
    "apply_cost_schedule",
    "compare_actual",
    "format_report",
    "load_fills",
    "save_fills",
    "simulate_follow_fills",
]
