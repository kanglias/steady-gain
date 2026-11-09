"""手工成交日志。写成 CSV 和 JSON，供之后和信号对照。

这里只记你从券商客户端抄下来的成交。没有下单、没有回报接入。
数量单位是股，不是行情里的手。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.normalize import normalize_symbol

FILL_COLUMNS = ["symbol", "side", "price", "quantity", "timestamp", "signal_date"]
COST_COLUMNS = ["commission", "stamp_tax", "other_cost"]
FILL_OUTPUT_COLUMNS = [*FILL_COLUMNS, *COST_COLUMNS, "is_simulated", "order_id", "account"]


class FillLogError(ValueError):
    """成交记录无法写入或无法解释。"""


def save_fills(
    fills: pd.DataFrame,
    path: str | Path,
    *,
    simulated: bool | None = None,
) -> tuple[Path, Path]:
    """把整份日志写成 `path.csv` 和 `path.json`，覆盖同名文件。

    `simulated=True` 时强制每行 `is_simulated=true`，避免演示成交被看成实盘业绩。
    """
    frame = normalize_fills(fills, force_simulated=bool(simulated))
    csv_path, json_path = _paths(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    payload = frame.to_dict(orient="records")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def append_fills(fills: pd.DataFrame, path: str | Path) -> tuple[Path, Path]:
    """追加到已有日志。文件不存在时新建。"""
    csv_path, _ = _paths(path)
    existing = load_fills(csv_path) if csv_path.exists() else normalize_fills(pd.DataFrame(columns=FILL_COLUMNS))
    incoming = normalize_fills(fills, force_simulated=False)
    combined = pd.concat([existing, incoming], ignore_index=True)
    return save_fills(combined, path)


def load_fills(path: str | Path) -> pd.DataFrame:
    """读 CSV 或 JSON。传入不带后缀的路径时优先读 CSV。"""
    file = Path(path)
    if file.suffix.lower() not in {".csv", ".json"}:
        csv_path, json_path = _paths(file)
        file = csv_path if csv_path.exists() else json_path
    if not file.exists():
        raise FillLogError(f"找不到成交日志: {file}")
    if file.suffix.lower() == ".json":
        raw = json.loads(file.read_text(encoding="utf-8"))
        frame = pd.DataFrame(raw)
    else:
        frame = pd.read_csv(file, dtype={"symbol": str, "side": str})
    return normalize_fills(frame)


def apply_cost_schedule(
    fills: pd.DataFrame,
    *,
    commission_rate: float = 0.0,
    stamp_tax_rate: float = 0.0,
    min_commission: float = 0.0,
    stamp_on: str = "sell",
) -> pd.DataFrame:
    """按成交金额写入佣金和印花税。这是研究用费率，不是券商价目。

    佣金 = max(价格 × 数量 × commission_rate, min_commission)，买卖都收。
    印花税默认只在卖出时按 stamp_tax_rate 计。已有的 `other_cost` 保留。
    """
    if stamp_on not in {"sell", "buy", "both"}:
        raise ValueError("stamp_on 只能是 sell、buy 或 both")
    frame = normalize_fills(fills)
    if frame.empty:
        return frame
    notional = frame["price"] * frame["quantity"]
    commission = (notional * float(commission_rate)).clip(lower=float(min_commission))
    if commission_rate == 0 and min_commission == 0:
        commission = frame["commission"]
    elif min_commission > 0 and commission_rate == 0:
        commission = pd.Series(float(min_commission), index=frame.index)
    frame["commission"] = commission.astype(float)
    if float(stamp_tax_rate) != 0:
        stamp = notional * float(stamp_tax_rate)
        if stamp_on == "sell":
            stamp = stamp.where(frame["side"].eq("sell"), 0.0)
        elif stamp_on == "buy":
            stamp = stamp.where(frame["side"].eq("buy"), 0.0)
        frame["stamp_tax"] = stamp.astype(float)
    return frame


def normalize_fills(fills: pd.DataFrame | None, *, force_simulated: bool = False) -> pd.DataFrame:
    """整理列，并检查代码、方向、价格、数量、时间和关联的信号日。

    佣金、印花税、其他费用缺省为 0，单位是元。`is_simulated` 缺省为 false。
    `force_simulated=True` 时一律写成 true。
    """
    if fills is None:
        raise FillLogError("成交表为空对象")
    if not isinstance(fills, pd.DataFrame):
        fills = pd.DataFrame(fills)
    missing = [column for column in FILL_COLUMNS if column not in fills.columns]
    if missing and not fills.empty:
        raise FillLogError("成交表缺少列: " + ", ".join(missing))
    frame = fills.copy()
    for column in FILL_OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    if frame.empty:
        empty = pd.DataFrame(columns=FILL_OUTPUT_COLUMNS)
        return empty

    symbols: list[str] = []
    for value in frame["symbol"].tolist():
        text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
        if not text or text.lower() == "nan":
            raise FillLogError("成交记录里有空的股票代码")
        symbols.append(normalize_symbol(text))
    frame["symbol"] = symbols
    frame["side"] = frame["side"].astype(str).str.lower()
    if not frame["side"].isin(["buy", "sell"]).all():
        raise FillLogError("side 只能是 buy 或 sell")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    if frame["price"].isna().any() or (frame["price"] <= 0).any():
        raise FillLogError("成交价必须是正数")
    if frame["quantity"].isna().any() or (frame["quantity"] <= 0).any():
        raise FillLogError("成交数量必须是正数，单位是股")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", format="mixed")
    if timestamps.isna().any():
        raise FillLogError("成交时间无法解析")
    signal_dates = pd.to_datetime(frame["signal_date"], errors="coerce", format="mixed")
    if signal_dates.isna().any():
        raise FillLogError("signal_date 无法解析")
    frame["timestamp"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S")
    frame["signal_date"] = signal_dates.dt.strftime("%Y-%m-%d")
    frame["price"] = frame["price"].astype(float)
    frame["quantity"] = frame["quantity"].astype(float)
    for column in COST_COLUMNS:
        amounts = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        if (amounts < 0).any():
            raise FillLogError(f"{column} 不能为负")
        frame[column] = amounts.astype(float)
    if force_simulated:
        frame["is_simulated"] = True
    else:
        frame["is_simulated"] = frame["is_simulated"].map(_as_bool)
    for column in ("order_id", "account"):
        frame[column] = frame[column].map(_as_text)
    return frame.loc[:, FILL_OUTPUT_COLUMNS].reset_index(drop=True)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _as_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _paths(path: str | Path) -> tuple[Path, Path]:
    target = Path(path)
    if target.suffix.lower() in {".csv", ".json"}:
        target = target.with_suffix("")
    return target.with_suffix(".csv"), target.with_suffix(".json")
