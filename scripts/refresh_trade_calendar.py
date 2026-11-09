#!/usr/bin/env python3
"""从新浪拉取 A 股交易日历并写入 `.cache/calendar/`（需联网）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trade_calendar import default_calendar_dir, get_trade_calendar  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="刷新上证交易日历缓存（akshare 新浪）")
    p.add_argument(
        "--calendar-dir",
        type=Path,
        default=None,
        help="输出目录（默认 ASHARE_CALENDAR_DIR 或 .cache/calendar）",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="不访问网络，仅读取已有缓存（用于检查）",
    )
    p.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="将 meta JSON 打印到路径（默认 stdout 打印摘要）",
    )
    ns = p.parse_args(argv)
    directory = ns.calendar_dir if ns.calendar_dir is not None else default_calendar_dir()
    try:
        frame, meta = get_trade_calendar(
            offline=ns.offline, refresh=not ns.offline, calendar_dir=directory
        )
    except Exception as exc:
        print(f"失败: {exc}", file=sys.stderr)
        return 1
    summary = {
        "calendar_id": meta.get("calendar_id"),
        "rows": len(frame),
        "start": meta.get("start"),
        "end": meta.get("end"),
        "source": meta.get("source"),
        "fetched_at": meta.get("fetched_at"),
        "calendar_dir": str(directory.resolve()),
    }
    if ns.out_meta:
        ns.out_meta.parent.mkdir(parents=True, exist_ok=True)
        ns.out_meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
