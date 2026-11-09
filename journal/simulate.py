"""按信号虚构「次日开盘跟随」成交，只为演示日志和对照。

不是券商回报。写出的每一行都强制 `is_simulated=true`。
真实成交用 `save_fills` 记入自己的价格、数量、费用和时间，不要沿用这个函数的结果冒充实盘。
"""

from __future__ import annotations

import math

import pandas as pd

from data.normalize import normalize_symbol


def simulate_follow_fills(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    equity: float,
    lot_size: int = 1,
) -> pd.DataFrame:
    """用信号日后的下一根开盘价生成演示成交。

    买入股数约为 `equity * 建议仓位 / 信号价`，再按 `lot_size` 向下取整。
    成交价仍用下一根开盘，这样股数跟的是建议，价差留在滑点里。
    卖出则把演示账本调到信号给出的目标股数。没有下一根 K 线时，用信号日收盘价。
    数量单位是股。默认不按 100 股一手取整，避免高价股在演示资金下变成 0。
    """
    if equity <= 0:
        raise ValueError("演示资金必须为正")
    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size < 1:
        raise ValueError("lot_size 需要是不小于 1 的整数")
    book = _bars(bars)
    orders = _signals(signals)
    columns = [
        "symbol",
        "side",
        "price",
        "quantity",
        "timestamp",
        "signal_date",
        "commission",
        "stamp_tax",
        "other_cost",
        "is_simulated",
        "order_id",
        "account",
    ]
    if orders.empty:
        return pd.DataFrame(columns=columns)

    index_of = {day: pos for pos, day in enumerate(book["date"])}
    shares = 0
    rows: list[dict] = []
    for order in orders.itertuples(index=False):
        pos = index_of.get(pd.Timestamp(order.date))
        if pos is None:
            continue
        fill_pos = pos + 1 if pos + 1 < len(book) else pos
        price = _fill_price(book, fill_pos, used_next_bar=fill_pos != pos)
        if not price or price <= 0:
            continue
        size_price = float(order.price) if pd.notna(order.price) and float(order.price) > 0 else price
        target = int(math.floor(float(equity) * float(order.weight) / size_price / lot_size) * lot_size)
        delta = target - shares
        if delta == 0:
            continue
        side = "buy" if delta > 0 else "sell"
        if side != order.side:
            continue
        shares = target
        clock = "09:30:00" if fill_pos != pos else "15:00:00"
        rows.append(
            {
                "symbol": order.symbol,
                "side": side,
                "price": float(price),
                "quantity": float(abs(delta)),
                "timestamp": f"{book['date'].iloc[fill_pos]:%Y-%m-%d}T{clock}",
                "signal_date": f"{pd.Timestamp(order.date):%Y-%m-%d}",
                "commission": 0.0,
                "stamp_tax": 0.0,
                "other_cost": 0.0,
                "is_simulated": True,
                "order_id": "",
                "account": "demo",
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    frame["is_simulated"] = True
    return frame


def _bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "open" in frame.columns:
        frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    if "symbol" in frame.columns and frame["symbol"].notna().any():
        frame["symbol"] = frame["symbol"].map(lambda value: normalize_symbol(value))
    return frame.reset_index(drop=True)


def _signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame(columns=["date", "symbol", "side", "weight"])
    frame = signals.copy()
    if "weight" not in frame.columns and "suggested_weight" in frame.columns:
        frame["weight"] = frame["suggested_weight"]
    if "weight" not in frame.columns:
        raise ValueError("信号需要 weight 或 suggested_weight")
    if "price" not in frame.columns:
        frame["price"] = pd.NA
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["side"] = frame["side"].astype(str).str.lower()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["date", "weight"])
    frame = frame.loc[frame["side"].isin(["buy", "sell"])]
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].map(lambda value: normalize_symbol(value) if str(value).strip() else "")
    else:
        frame["symbol"] = ""
    return frame.loc[:, ["date", "symbol", "side", "weight", "price"]].sort_values("date").reset_index(drop=True)


def _fill_price(book: pd.DataFrame, pos: int, *, used_next_bar: bool) -> float:
    if used_next_bar and "open" in book.columns and pd.notna(book["open"].iloc[pos]):
        return float(book["open"].iloc[pos])
    close = book["close"].iloc[pos]
    if pd.isna(close):
        return 0.0
    return float(close)
