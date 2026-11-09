"""周频调仓清单。字段按研究规格的调仓输出，代码仍是 6 位数字。

只写出文件，不连接券商。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.normalize import normalize_symbol
from strategy.weekly_mr import ORDER_COLUMNS

_SIGNAL = {"enter": "调入", "hold": "继续持有", "exit": "调出", "flat": "空仓"}


def export_weekly_orders(orders: pd.DataFrame, path: str | Path) -> tuple[Path, Path]:
    """写出 `path.csv` 和 `path.md`。"""
    frame = _prepare(orders)
    target = Path(path)
    if target.suffix.lower() in {".csv", ".md"}:
        target = target.with_suffix("")
    target.parent.mkdir(parents=True, exist_ok=True)
    csv_path = target.with_suffix(".csv")
    md_path = target.with_suffix(".md")
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_markdown(frame), encoding="utf-8")
    return csv_path, md_path


def _prepare(orders: pd.DataFrame) -> pd.DataFrame:
    if orders is None:
        raise ValueError("周频清单为空对象")
    if not isinstance(orders, pd.DataFrame):
        orders = pd.DataFrame(orders)
    missing = [column for column in ORDER_COLUMNS if column not in orders.columns]
    if missing and not orders.empty:
        raise ValueError("周频清单缺少列: " + ", ".join(missing))
    frame = orders.copy()
    if frame.empty:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    frame["code"] = frame["code"].map(normalize_symbol)
    frame["asof_date"] = pd.to_datetime(frame["asof_date"]).dt.strftime("%Y-%m-%d")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["signal"] = frame["signal"].astype(str)
    if not frame["signal"].isin(_SIGNAL).all():
        raise ValueError("signal 只能是 enter、hold、exit 或 flat")
    return frame.loc[:, ORDER_COLUMNS].reset_index(drop=True)


def _markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# 周频调仓清单",
        "",
        "信号日收盘生成，计划在下一交易日开盘调仓。请到券商客户端自行决定。这里不会发送任何委托。",
        "",
        "| 信号日 | 成交日 | 代码 | 信号 | z | 目标权重 | 约束前 | 行业 | 开盘参考 | 预览金额 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if frame.empty:
        lines.append("|  |  |  |  |  |  |  |  | 当周没有调仓 |  |")
    for row in frame.itertuples(index=False):
        lines.append(
            "| {asof} | {trade} | {code} | {signal} | {z} | {target} | {raw} | {industry} | {open_} | {notional} |".format(
                asof=row.asof_date,
                trade=row.trade_date,
                code=row.code,
                signal=_SIGNAL.get(row.signal, row.signal),
                z="" if row.z_score is None or pd.isna(row.z_score) else f"{float(row.z_score):.2f}",
                target=f"{float(row.target_weight):.2%}",
                raw=f"{float(row.raw_weight):.2%}",
                industry="" if row.industry is None or pd.isna(row.industry) else row.industry,
                open_="" if row.next_open_ref is None or pd.isna(row.next_open_ref) else f"{float(row.next_open_ref):.2f}",
                notional=f"{float(row.preview_notional):.2f}",
            )
        )
    lines.append("")
    return "\n".join(lines)
