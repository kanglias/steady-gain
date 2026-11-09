"""把 akshare 的日线表和调用参数收成稳定的内部格式。"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

# 东财 stock_zh_a_hist 的中文列。英文列（腾讯兜底接口）保持原名。
COLUMN_MAP = {
    "日期": "date",
    "交易日": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

DESIRED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change",
    "turnover",
]

_ADJUST_ALIASES = {
    "": "",
    "none": "",
    "raw": "",
    "bfq": "",
    "不复权": "",
    "qfq": "qfq",
    "前复权": "qfq",
    "hfq": "hfq",
    "后复权": "hfq",
}


def normalize_symbol(symbol: str) -> str:
    """接受 `000001`、`sz000001`、`600519.SH` 这类写法，返回 6 位数字。"""
    raw = str(symbol).strip().lower().replace(" ", "").replace(".", "")
    for token in ("sh", "sz", "bj"):
        if raw.startswith(token):
            raw = raw[len(token) :]
            break
        if raw.endswith(token):
            raw = raw[: -len(token)]
            break
    if not (len(raw) == 6 and raw.isdigit()):
        raise ValueError(f"A 股代码需为 6 位数字（可带 sh/sz/bj 前缀），收到: {symbol}")
    return raw


def exchange_prefix(symbol: str) -> str:
    """给腾讯日线兜底接口用的交易所前缀。阶段 1 不处理港股美股。"""
    code = normalize_symbol(symbol)
    if code.startswith(("5", "6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    if code.startswith(("0", "1", "2", "3")):
        return "sz"
    raise ValueError(f"无法判断 {code} 的交易所")


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("日期为空")
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"无法解析日期: {value}")
    return parsed.date()


def normalize_adjust(adjust: str | None) -> str:
    value = "" if adjust is None else str(adjust).strip().lower()
    if value not in _ADJUST_ALIASES:
        raise ValueError("adjust 只能是 qfq（前复权）、hfq（后复权）或空字符串（不复权）")
    return _ADJUST_ALIASES[value]


def normalize_hist(raw: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    """整理成 date/symbol/OHLCV。成交量保持数据源原单位（东财日线为手）。"""
    empty = pd.DataFrame(columns=DESIRED_COLUMNS)
    if raw is None or len(raw) == 0:
        return empty

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "_".join(str(part) for part in col if str(part) != "").strip("_")
            for col in frame.columns
        ]

    existing = set(map(str, frame.columns))
    rename: dict[str, str] = {}
    for src, dst in COLUMN_MAP.items():
        if src in existing and dst not in existing and dst not in rename.values():
            rename[src] = dst
    frame = frame.rename(columns=rename)

    if "date" not in frame.columns:
        return empty

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = normalize_symbol(symbol)
    for col in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    keep = [col for col in DESIRED_COLUMNS if col in frame.columns]
    return frame.loc[:, keep].reset_index(drop=True)
