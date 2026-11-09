"""测试用合成交易日历（fixture only，非生产金标准）。"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data.trade_calendar import CALENDAR_ID, save_trade_calendar


def write_fixture_calendar(
    directory: Path,
    trade_dates: list[date | str],
    *,
    calendar_end: date | None = None,
) -> Path:
    """写入最小 parquet + meta；``trade_dates`` 为开市日列表。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parsed = [pd.Timestamp(d).normalize() for d in trade_dates]
    frame = pd.DataFrame({"trade_date": parsed})
    ends = max(parsed)
    end_d = calendar_end or (ends.date() if hasattr(ends, "date") else pd.Timestamp(ends).date())
    meta = save_trade_calendar(
        directory,
        frame,
        source="testfixture.synthetic_calendar",
        fetched_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    # 覆盖 end 以便测试 as_of 截断
    meta_path = directory / f"{CALENDAR_ID}.meta.json"
    meta["end"] = end_d.isoformat()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path
