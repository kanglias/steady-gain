"""政策与宏观事件日历。

默认只读本地 JSON/CSV。`fetch=True` 时尝试用 akshare 的 LPR 调整日补充，
失败则退回本地文件，不让日历把日常流程打断。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from data.normalize import normalize_symbol, parse_date

DEFAULT_CALENDAR = Path(__file__).resolve().parent / "calendars" / "cn_events.json"
EVENT_COLUMNS = ["date", "title", "region", "impact", "summary", "sectors", "symbols"]
IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}
_IMPACT_ALIASES = {
    "low": "low",
    "l": "low",
    "低": "low",
    "medium": "medium",
    "mid": "medium",
    "m": "medium",
    "中": "medium",
    "high": "high",
    "h": "high",
    "高": "high",
}


def load_events(
    path: str | Path | None = None,
    *,
    start=None,
    end=None,
    fetch: bool = False,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """读取事件日历。

    本地文件是主体，可按自己的研究改 `data/calendars/cn_events.json`。
    `symbols` 为空表示全市场；写了代码则只和这些标的有关。`sectors` 只作说明，
    阶段 2 没有行业成分表，不会按板块自动展开成股票。

    `fetch=True` 时额外请求 akshare `macro_china_lpr`，只保留 1 年期或 5 年期
    利率发生变化的报价日。请求失败时退回本地日历，并在 `attrs["fetch_error"]` 留下原因。
    """
    calendar = Path(path) if path is not None else DEFAULT_CALENDAR
    frame = _read_calendar(calendar)
    source = "local"
    fetch_error = ""
    if fetch:
        try:
            remote = fetcher() if fetcher is not None else _fetch_lpr_events()
            frame = _merge_events(frame, _normalize_frame(remote))
            source = "local+akshare.macro_china_lpr"
        except Exception as exc:
            fetch_error = f"{type(exc).__name__}: {exc}"
            source = "local"
    frame = _filter_span(frame, start, end)
    frame.attrs["source"] = source
    frame.attrs["fetch_error"] = fetch_error
    frame.attrs["calendar"] = str(calendar)
    return frame.reset_index(drop=True)


def _read_calendar(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到事件日历: {path}")
    if path.suffix.lower() == ".json":
        raw = pd.read_json(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError("事件日历只支持 .json 或 .csv")
    if raw.empty:
        return _empty_events()
    return _normalize_frame(raw)


def _normalize_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_events()
    frame = raw.copy()
    rename = {
        "日期": "date",
        "标题": "title",
        "地区": "region",
        "影响": "impact",
        "摘要": "summary",
        "板块": "sectors",
        "相关板块": "sectors",
        "代码": "symbols",
        "相关代码": "symbols",
    }
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    missing = [col for col in ("date", "title") if col not in frame.columns]
    if missing:
        raise ValueError(f"事件日历缺少列: {', '.join(missing)}")
    records: list[dict] = []
    for row in frame.to_dict(orient="records"):
        title = str(row.get("title", "")).strip()
        if not title or title.lower() == "nan":
            continue
        try:
            day = parse_date(row["date"])
        except (TypeError, ValueError):
            continue
        impact = _normalize_impact(row.get("impact", "medium"))
        records.append(
            {
                "date": pd.Timestamp(day),
                "title": title,
                "region": _text(row.get("region"), default="CN"),
                "impact": impact,
                "summary": _text(row.get("summary"), default=""),
                "sectors": _as_list(row.get("sectors")),
                "symbols": _as_symbols(row.get("symbols")),
            }
        )
    if not records:
        return _empty_events()
    out = pd.DataFrame(records, columns=EVENT_COLUMNS)
    return out.sort_values(["date", "title"]).drop_duplicates(["date", "title"]).reset_index(drop=True)


def _fetch_lpr_events() -> pd.DataFrame:
    """把 LPR 报价变化记成宏观事件。接口失败由 load_events 吃掉。"""
    import akshare as ak

    raw = ak.macro_china_lpr()
    if raw is None or raw.empty:
        raise RuntimeError("LPR 接口返回空表")
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["TRADE_DATE"], errors="coerce")
    frame["lpr1y"] = pd.to_numeric(frame["LPR1Y"], errors="coerce")
    frame["lpr5y"] = pd.to_numeric(frame["LPR5Y"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    changed = frame["lpr1y"].diff().ne(0) | frame["lpr5y"].diff().ne(0)
    changed.iloc[0] = False
    picked = frame.loc[changed]
    rows = []
    for row in picked.itertuples(index=False):
        rows.append(
            {
                "date": row.date,
                "title": "LPR 调整",
                "region": "CN",
                "impact": "high",
                "summary": f"1年期 LPR {row.lpr1y:.2f}%，5年期 LPR {row.lpr5y:.2f}%",
                "sectors": ["银行", "房地产"],
                "symbols": None,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _merge_events(local: pd.DataFrame, remote: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([local, remote], ignore_index=True)
    if merged.empty:
        return _empty_events()
    return merged.sort_values(["date", "title"]).drop_duplicates(["date", "title"], keep="first").reset_index(drop=True)


def _filter_span(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    out = frame.copy()
    if start is not None:
        out = out.loc[out["date"] >= pd.Timestamp(parse_date(start))]
    if end is not None:
        out = out.loc[out["date"] <= pd.Timestamp(parse_date(end))]
    return out.reset_index(drop=True)


def _normalize_impact(value: object) -> str:
    text = "medium" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip().lower()
    if text not in _IMPACT_ALIASES:
        raise ValueError(f"impact 只能是 low / medium / high，收到: {value}")
    return _IMPACT_ALIASES[text]


def _text(value: object, *, default: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    if text.lower() == "nan":
        return default
    return text


def _as_list(value: object) -> list[str] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    items = [part.strip() for part in text.replace("、", "|").replace(",", "|").split("|")]
    items = [part for part in items if part]
    return items or None


def _as_symbols(value: object) -> list[str] | None:
    items = _as_list(value)
    if not items:
        return None
    return [normalize_symbol(item) for item in items]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)
