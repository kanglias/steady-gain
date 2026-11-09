#!/usr/bin/env python3
"""打印可重复的事件开关样本：off、block_entries、reduce。

不拉行情，也不下单。同一条目标仓位在三种模式下的权重并排看。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy import event_switch_samples  # noqa: E402


def main() -> int:
    print("事件约束样本。默认研究路径仍是 off。下面只说明开关打开后权重会怎样变。")
    for name, frame in event_switch_samples().items():
        view = frame.copy()
        view["date"] = view["date"].dt.strftime("%Y-%m-%d")
        print(f"\n{name} · {view['case'].iloc[0]}")
        print(view.drop(columns=["case"]).to_string(index=False))
    print("\n开仓日遇事件：block_entries 和 reduce 都会把当天新开仓压住。")
    print("持仓中遇事件：block_entries 不砍已有仓，reduce 当天降到 0，下一天若策略仍持有会回来。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
