"""用 plotly 画 A 股习惯的红涨绿跌 K 线，并叠上均线、买卖标记和政策事件。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategy.base import StrategyResult

_IMPACT_LABEL = {"high": "高", "medium": "中", "low": "低"}


def plot_signals(
    bars: pd.DataFrame,
    result: StrategyResult,
    *,
    title: str | None = None,
    show_volume: bool = True,
    events: pd.DataFrame | None = None,
) -> go.Figure:
    """画 K 线、均线、买卖标记。传入 events 时在图下方加「政策与事件」表。"""
    plot_df = _prepare_bars(bars)
    indicators = _align_indicators(plot_df, result.indicators)
    merged = plot_df.merge(indicators, on="date", how="left")
    markers = _marker_frame(plot_df, result.signals)
    has_volume = bool(
        show_volume and "volume" in merged.columns and merged["volume"].notna().any()
    )
    show_events = events is not None
    fig, volume_row = _make_figure(has_volume=has_volume, show_events=show_events)

    fig.add_trace(
        go.Candlestick(
            x=merged["date"],
            open=merged["open"],
            high=merged["high"],
            low=merged["low"],
            close=merged["close"],
            name="K线",
            increasing_line_color="#dc2626",
            increasing_fillcolor="#dc2626",
            decreasing_line_color="#16a34a",
            decreasing_fillcolor="#16a34a",
        ),
        row=1,
        col=1,
    )
    for column, label, color in (
        ("ma_short", "短均线", "#2563eb"),
        ("ma_long", "长均线", "#d97706"),
        ("ma_trend", "趋势均线", "#64748b"),
        ("ma", "均值", "#7c3aed"),
    ):
        if column not in merged.columns or not merged[column].notna().any():
            continue
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged[column],
                mode="lines",
                name=label,
                line=dict(color=color, width=1.4),
                hovertemplate=f"{label} %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    _add_markers(fig, markers)
    _add_event_markers(fig, merged, events)
    if has_volume and volume_row is not None:
        colors = [
            "#dc2626" if close >= open_ else "#16a34a"
            for open_, close in zip(merged["open"], merged["close"], strict=True)
        ]
        fig.add_trace(
            go.Bar(
                x=merged["date"],
                y=merged["volume"],
                name="成交量",
                marker_color=colors,
                showlegend=False,
                hovertemplate="成交量 %{y:,.0f}<extra></extra>",
            ),
            row=volume_row,
            col=1,
        )
        fig.update_yaxes(title_text="成交量", row=volume_row, col=1)
    if show_events:
        _add_event_table(fig, events, merged["date"].min(), merged["date"].max(), row=_event_row(has_volume))

    heading = title or _default_title(plot_df)
    note = "红涨绿跌 · 三角为策略在收盘时给出的信号，不是券商成交"
    fig.update_layout(
        title=dict(text=f"{heading}<br><sup>{note}</sup>", x=0, xanchor="left"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=56, r=24, t=88, b=40),
        height=980 if show_events else 720,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)
    if volume_row is not None:
        fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])], row=volume_row, col=1)
    return fig


def save_chart(fig: go.Figure, path: str | Path, *, embed_js: bool = True) -> Path:
    """把图写成单个 HTML。默认内嵌 plotly.js，离线也能打开。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        target,
        include_plotlyjs=True if embed_js else "cdn",
        full_html=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    return target


def _make_figure(*, has_volume: bool, show_events: bool) -> tuple[go.Figure, int | None]:
    specs: list[list[dict]] = [[{"type": "xy"}]]
    heights = [1.0]
    volume_row = None
    if has_volume and show_events:
        specs = [[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]]
        heights = [0.56, 0.16, 0.24]
        volume_row = 2
    elif has_volume:
        specs = [[{"type": "xy"}], [{"type": "xy"}]]
        heights = [0.74, 0.26]
        volume_row = 2
    elif show_events:
        specs = [[{"type": "xy"}], [{"type": "table"}]]
        heights = [0.68, 0.28]
    titles = [""] * len(specs)
    if show_events:
        titles[-1] = "政策与事件"
    fig = make_subplots(
        rows=len(specs),
        cols=1,
        shared_xaxes=bool(has_volume),
        vertical_spacing=0.035,
        row_heights=heights,
        specs=specs,
        subplot_titles=tuple(titles),
    )
    return fig, volume_row


def _event_row(has_volume: bool) -> int:
    return 3 if has_volume else 2


def _add_event_markers(fig: go.Figure, bars: pd.DataFrame, events: pd.DataFrame | None) -> None:
    if events is None or events.empty or "date" not in events.columns:
        return
    marked = events.copy()
    marked["date"] = _naive_dates(marked["date"])
    marked = marked.dropna(subset=["date"]).merge(bars[["date", "high"]], on="date", how="inner")
    if marked.empty:
        return
    grouped = (
        marked.groupby("date", as_index=False)
        .agg(high=("high", "max"), title=("title", lambda values: "；".join(str(item) for item in values)))
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["date"],
            y=grouped["high"].astype(float) * 1.03,
            mode="markers",
            name="政策事件",
            marker=dict(symbol="diamond", size=9, color="#7c3aed", line=dict(color="white", width=1)),
            customdata=grouped["title"],
            hovertemplate="政策事件 %{x|%Y-%m-%d}<br>%{customdata}<extra></extra>",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )


def _add_event_table(fig: go.Figure, events: pd.DataFrame, start, end, *, row: int) -> None:
    view = _event_rows(events, start, end)
    fig.add_trace(
        go.Table(
            name="政策与事件",
            header=dict(
                values=["日期", "标题", "地区", "影响", "摘要", "相关"],
                fill_color="#1f2937",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[view[col] for col in view.columns],
                fill_color="#f8fafc",
                align="left",
                font=dict(size=11),
                height=24,
            ),
        ),
        row=row,
        col=1,
    )


def _event_rows(events: pd.DataFrame, start, end) -> pd.DataFrame:
    columns = ["日期", "标题", "地区", "影响", "摘要", "相关"]
    if events is None or events.empty:
        return pd.DataFrame([["—", "区间内没有政策或宏观事件", "—", "—", "可编辑本地日历", "—"]], columns=columns)
    frame = events.copy()
    frame["date"] = _naive_dates(frame["date"])
    frame = frame.dropna(subset=["date"])
    frame = frame.loc[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
    frame = frame.sort_values("date")
    if frame.empty:
        return pd.DataFrame([["—", "区间内没有政策或宏观事件", "—", "—", "可编辑本地日历", "—"]], columns=columns)
    if len(frame) > 18:
        frame = frame.tail(18)
    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            [
                pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                str(row.get("title", "")),
                str(row.get("region", "")),
                _IMPACT_LABEL.get(str(row.get("impact", "")), str(row.get("impact", ""))),
                str(row.get("summary", "")),
                _related_text(row),
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def _related_text(row: dict) -> str:
    parts = []
    sectors = row.get("sectors")
    symbols = row.get("symbols")
    if isinstance(sectors, list) and sectors:
        parts.append("板块 " + "、".join(str(item) for item in sectors))
    if isinstance(symbols, list) and symbols:
        parts.append("代码 " + "、".join(str(item) for item in symbols))
    return "；".join(parts) if parts else "全市场"


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = ("date", "open", "high", "low", "close")
    missing = [col for col in required if col not in bars.columns]
    if missing:
        raise ValueError(f"画图缺少列: {', '.join(missing)}")
    frame = bars.copy()
    frame["date"] = _naive_dates(frame["date"])
    for col in ("open", "high", "low", "close", "volume"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError("没有可以绘制的 K 线")
    return frame.reset_index(drop=True)


def _align_indicators(bars: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    if indicators is None or not isinstance(indicators, pd.DataFrame):
        raise TypeError("策略结果需要 indicators 表")
    frame = indicators.copy()
    if "date" not in frame.columns:
        if len(frame) != len(bars):
            raise ValueError("indicators 需要 date 列，或与行情行数一致")
        frame.insert(0, "date", bars["date"].to_numpy())
    frame["date"] = _naive_dates(frame["date"])
    columns = ["date"]
    for col in ("ma_short", "ma_long"):
        if col in frame.columns:
            columns.append(col)
    extra = [col for col in frame.columns if col not in columns and col != "date"]
    return frame.loc[:, columns + extra].drop_duplicates("date", keep="last")


def _marker_frame(bars: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["date", "side", "price", "high", "low"])
    if signals is None or not isinstance(signals, pd.DataFrame) or signals.empty:
        return empty
    if "date" not in signals.columns or "side" not in signals.columns:
        raise ValueError("signals 需要 date 和 side 列")
    frame = signals.copy()
    frame["date"] = _naive_dates(frame["date"])
    frame["side"] = frame["side"].astype(str).str.lower()
    if "price" not in frame.columns:
        frame["price"] = pd.NA
    if "flag" not in frame.columns:
        frame["flag"] = ""
    marked = frame.merge(bars[["date", "high", "low"]], on="date", how="inner")
    marked = marked[marked["side"].isin(["buy", "sell"])]
    return marked.dropna(subset=["high", "low"]).reset_index(drop=True)


def _add_markers(fig: go.Figure, markers: pd.DataFrame) -> None:
    specs = (
        ("buy", "买入", "triangle-up", "#dc2626", "low", 0.99),
        ("sell", "卖出", "triangle-down", "#16a34a", "high", 1.01),
    )
    for side, label, symbol, color, anchor, scale in specs:
        subset = markers[markers["side"] == side]
        if subset.empty:
            continue
        flip = subset["flag"].astype(str).eq("短持翻脸") if "flag" in subset.columns else pd.Series(False, index=subset.index)
        _add_marker_trace(fig, subset.loc[~flip], label, symbol, color, anchor, scale)
        _add_marker_trace(fig, subset.loc[flip], "短持翻脸", "x", "#7c3aed", anchor, scale)


def _add_marker_trace(fig, subset, label, symbol, color, anchor, scale) -> None:
    if subset.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset[anchor].astype(float) * scale,
            mode="markers",
            name=label,
            marker=dict(symbol=symbol, size=12, color=color, line=dict(color="white", width=1)),
            customdata=pd.to_numeric(subset["price"], errors="coerce"),
            hovertemplate=f"{label} %{{x|%Y-%m-%d}}<br>信号价 %{{customdata:.2f}}<extra></extra>",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )


def _default_title(bars: pd.DataFrame) -> str:
    if "symbol" in bars.columns and bars["symbol"].notna().any():
        return f"{bars['symbol'].iloc[0]} 日线"
    return "日线"


def _naive_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    return dates.dt.normalize()
