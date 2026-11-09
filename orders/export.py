"""把买卖建议写成 CSV 和 Markdown，供人工复制。

不会连接券商，也没有提交、报单或自动交易。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.normalize import normalize_symbol

DEFAULT_BANNED_FILE = Path(__file__).resolve().parent / "banned.txt"
_REQUIRED = ["symbol", "side", "date", "strategy", "reason"]
_EXTRA = [
    "suggested_weight",
    "holding_days",
    "unrealized_pnl",
    "consecutive_losses",
    "days_since_cross",
    "flag",
]
_DIRECTION = {"buy": "买入", "sell": "卖出"}
PAUSE_LABEL = "建议暂停加仓"


class BannedSymbolError(ValueError):
    """清单里出现了禁投代码。文件不会被写出。"""


def load_banned(path: str | Path | None = None) -> set[str]:
    """读取禁投代码。空行和 # 注释会被忽略。"""
    file = Path(path) if path is not None else DEFAULT_BANNED_FILE
    if not file.exists():
        return set()
    banned: set[str] = set()
    for line in file.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            banned.add(normalize_symbol(text))
    return banned


def export_order_list(
    suggestions: pd.DataFrame,
    path: str | Path,
    *,
    banned: set[str] | list[str] | None = None,
    pause_after_losses: int = 3,
    pause_after_flips: int = 2,
    holding: dict[str, bool] | None = None,
) -> tuple[Path, Path]:
    """写出 `path.csv` 与 `path.md`。

    默认带上诊断列和信号原因（止损、均线确认都在 `reason` 里）。
    同一代码连续亏损达到 `pause_after_losses`（默认 3），或短持翻脸的卖出达到
    `pause_after_flips`（默认 2，表示反复出现），清单标「建议暂停加仓」，
    并写明当前是仍持有还是空仓，避免把提示看成新开仓。
    `holding` 用代码映射到是否仍持有；不传时看该代码在这份清单里最后一行的建议仓位。
    这只是研究提示，不会自动停手，也不会下单。

    `banned is None` 时使用 `orders/banned.txt`。传入集合则只用该集合，方便测试。
    任一代码命中禁投名单时抛出 `BannedSymbolError`，两个文件都不会创建。
    表是空的也可以导出，方便确认「这一天没有建议」。
    """
    frame = _apply_pause(
        _prepare(suggestions),
        pause_after_losses=pause_after_losses,
        pause_after_flips=pause_after_flips,
        holding=holding,
    )
    denied = set(banned) if banned is not None else load_banned()
    denied = {normalize_symbol(item) for item in denied}
    hit = sorted({code for code in frame["symbol"].tolist() if code in denied})
    if hit:
        raise BannedSymbolError(
            "这些代码在禁投名单里，清单没有写出: " + ", ".join(hit)
        )
    target = Path(path)
    if target.suffix.lower() in {".csv", ".md"}:
        target = target.with_suffix("")
    target.parent.mkdir(parents=True, exist_ok=True)
    csv_path = target.with_suffix(".csv")
    md_path = target.with_suffix(".md")
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_markdown(frame), encoding="utf-8")
    return csv_path, md_path


def _prepare(suggestions: pd.DataFrame) -> pd.DataFrame:
    if suggestions is None:
        raise ValueError("建议表为空对象")
    missing = [col for col in _REQUIRED if col not in suggestions.columns]
    if missing:
        raise ValueError(f"建议表缺少列: {', '.join(missing)}")
    frame = suggestions.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=[*_REQUIRED, "direction", "price", *_EXTRA])
    frame["symbol"] = frame["symbol"].map(lambda value: normalize_symbol(value) if str(value).strip() else "")
    if frame["symbol"].eq("").any():
        raise ValueError("建议表里有空的股票代码")
    frame["side"] = frame["side"].astype(str).str.lower()
    if not frame.empty and not frame["side"].isin(["buy", "sell"]).all():
        raise ValueError("side 只能是 buy 或 sell")
    frame["direction"] = frame["side"].map(_DIRECTION)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if frame["date"].isna().any():
        raise ValueError("建议表里有无法解析的日期")
    frame["strategy"] = frame["strategy"].astype(str)
    frame["reason"] = frame["reason"].fillna("").astype(str)
    if "price" not in frame.columns:
        frame["price"] = pd.NA
    for column in _EXTRA:
        if column not in frame.columns:
            frame[column] = pd.NA
    columns = ["symbol", "side", "direction", "date", "strategy", "reason", "price", *_EXTRA, "pause_advice"]
    frame["pause_advice"] = ""
    return frame.loc[:, columns].reset_index(drop=True)


def _apply_pause(
    frame: pd.DataFrame,
    *,
    pause_after_losses: int,
    pause_after_flips: int,
    holding: dict[str, bool] | None = None,
) -> pd.DataFrame:
    if isinstance(pause_after_losses, bool) or not isinstance(pause_after_losses, int) or pause_after_losses < 1:
        raise ValueError("pause_after_losses 需要是不小于 1 的整数")
    if isinstance(pause_after_flips, bool) or not isinstance(pause_after_flips, int) or pause_after_flips < 1:
        raise ValueError("pause_after_flips 需要是不小于 1 的整数")
    notes: dict[str, str] = {}
    if not frame.empty:
        for symbol, group in frame.groupby("symbol", sort=False):
            reasons: list[str] = []
            losses = pd.to_numeric(group["consecutive_losses"], errors="coerce")
            if losses.notna().any() and float(losses.max()) >= pause_after_losses:
                reasons.append(f"连续亏损达到 {pause_after_losses}")
            flips = group.loc[
                group["side"].astype(str).eq("sell") & group["flag"].astype(str).eq("短持翻脸")
            ]
            if len(flips) >= pause_after_flips:
                reasons.append(f"短持翻脸卖出出现 {len(flips)} 次")
            if reasons:
                code = str(symbol)
                held = holding[code] if holding is not None and code in holding else _still_holding(group)
                status = "仍持有，不是新开仓" if held else "当前空仓"
                notes[code] = "；".join(reasons) + "；" + status
    frame = frame.copy()
    frame["pause_advice"] = ""
    # 只标该代码日期最晚的一行。更早的买入保留原样，避免整段历史看起来都在暂停。
    # 汇总条用 notes，写在 Markdown 开头。
    if notes:
        days = pd.to_datetime(frame["date"], errors="coerce")
        for code, text in notes.items():
            mask = frame["symbol"].astype(str).eq(str(code))
            if not mask.any():
                continue
            latest = None
            latest_key = None
            for idx in frame.index[mask]:
                day = days.loc[idx]
                valid = pd.notna(day)
                key = (1 if valid else 0, pd.Timestamp(day) if valid else pd.Timestamp.min, idx)
                if latest_key is None or key > latest_key:
                    latest_key = key
                    latest = idx
            status = text.split("；")[-1]
            frame.loc[latest, "pause_advice"] = f"{PAUSE_LABEL}；{status}"
    frame.attrs["pause_notes"] = notes
    return frame


def _still_holding(group: pd.DataFrame) -> bool:
    ordered = group.copy()
    ordered["_day"] = pd.to_datetime(ordered["date"], errors="coerce")
    last = ordered.sort_values("_day").iloc[-1]
    weight = pd.to_numeric(pd.Series([last["suggested_weight"]]), errors="coerce").iloc[0]
    if pd.notna(weight):
        return float(weight) > 0
    return str(last["side"]) == "buy"


def _markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# 下单清单",
        "",
        "这份清单只给人对照。请到券商客户端自行决定是否下单。这里不会发送任何委托。",
        "原因列保留止损和交叉说明。诊断列默认都在。",
        "",
    ]
    notes = frame.attrs.get("pause_notes") or {}
    if notes:
        detail = "；".join(f"{code}（{text}）" for code, text in notes.items())
        lines.extend(
            [
                f"**{PAUSE_LABEL}**：{detail}。",
                "这是研究提示，不是自动停止加仓，也不是交易指令。",
                "",
            ]
        )
    lines.extend(
        [
        "| 代码 | 方向 | 日期 | 策略 | 建议仓位 | 持仓天数 | 浮动盈亏 | 连亏 | 距上次交叉 | 标记 | 暂停 | 原因 | 参考价 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if frame.empty:
        lines.append("|  |  |  |  |  |  |  |  |  |  |  | 区间内没有买卖建议 |  |")
    for row in frame.itertuples(index=False):
        lines.append(
            "| {symbol} | {direction} | {date} | {strategy} | {weight} | {held} | {pnl} | {losses} | {since} | {flag} | {pause} | {reason} | {price} |".format(
                symbol=row.symbol,
                direction=row.direction,
                date=row.date,
                strategy=_cell(row.strategy),
                weight=_pct(row.suggested_weight),
                held=_num(row.holding_days, 0),
                pnl=_pct(row.unrealized_pnl),
                losses=_num(row.consecutive_losses, 0),
                since=_num(row.days_since_cross, 0),
                flag=_cell(row.flag),
                pause=_cell(row.pause_advice),
                reason=_cell(row.reason),
                price=_num(row.price, 2),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).replace("|", "/").replace("\n", " ")


def _num(value: object, digits: int) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.0%}"
