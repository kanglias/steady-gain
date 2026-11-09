#!/usr/bin/env python3
"""只读：A 股日线缓存（`*_qfq.parquet` + `.meta.json`）数据质量检查 CLI。

不要用合成行情或 features 侧数据作为本检查的黄金标准；本工具面向真实落地缓存。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.qc_bars_cache import main_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_cli())
