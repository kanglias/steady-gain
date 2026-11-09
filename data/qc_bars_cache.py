"""只读：扫描 `.cache/bars` 下 A 股日线 parquet + meta，输出数据质量报告。

注意：本模块用于检查**真实缓存**（如 akshare 落地）。不要用合成行情或
features 项目的“标准答案”对照来当做本 QC 的黄金标准；合成数据只适用于
单元测试里构造通过/失败样例。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data.cache import default_cache_dir
from data.normalize import normalize_symbol
from data.trade_calendar import CALENDAR_ID, default_calendar_dir, load_trade_calendar_disk

REQUIRED_BAR_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
OPTIONAL_BAR_COLUMNS = ("amount", "amplitude", "pct_change", "change", "turnover")

# 文件名：`{6位代码}_{adjust}.parquet`
FILE_KEY_RE = re.compile(r"^(\d{6})_([^.]+)\.parquet$", re.IGNORECASE)


def _cell_symbol(value: object) -> str:
    """与 `data.cache.load_cache` 中 symbol 列规范化一致。"""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def _parse_filename_stem(path: Path) -> tuple[str, str] | None:
    match = FILE_KEY_RE.match(path.name)
    if not match:
        return None
    return match.group(1), match.group(2).lower()


def discover_qfq_parquets(cache_dir: Path) -> list[Path]:
    """列出目录下 `*_qfq.parquet`（大小写不敏感），不含子目录。"""
    if not cache_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(cache_dir.iterdir()):
        if not p.is_file():
            continue
        parsed = _parse_filename_stem(p)
        if parsed is None:
            continue
        _, adj = parsed
        if adj != "qfq":
            continue
        out.append(p)
    return out


def _parse_iso_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    ts = pd.Timestamp(parsed)
    return ts.date()


def _parse_iso_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    if hasattr(parsed, "to_pydatetime"):
        dt = parsed.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _dates_series(frame: pd.DataFrame) -> pd.Series:
    s = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return s


@dataclass
class FileReport:
    parquet_path: str
    meta_path: str
    symbol_from_filename: str
    adjust_from_filename: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    rows_in_file: int = 0
    optional_columns_present: list[str] = field(default_factory=list)
    optional_columns_absent: list[str] = field(default_factory=list)
    source: str | None = None
    missing_days: list[str] = field(default_factory=list)
    missing_days_union: list[str] = field(default_factory=list)
    extra_non_trading_days: list[str] = field(default_factory=list)


def _csi300_bar_dates(cache_dir: Path) -> set[pd.Timestamp] | None:
    """读取 `.cache/bars/000300_index.parquet` 的日期集合，供日历交叉对照（可选）。"""
    pq = Path(cache_dir) / "000300_index.parquet"
    if not pq.is_file():
        return None
    try:
        frame = pd.read_parquet(pq)
    except Exception:
        return None
    if "date" not in frame.columns:
        return None
    ds = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    ds = ds.dropna()
    if ds.empty:
        return None
    return set(ds.tolist())


def _load_meta(meta_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not meta_path.exists():
        return None, "缺少同名的 .meta.json"
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"无法读取 meta: {exc}"


def run_bars_cache_qc(
    cache_dir: Path,
    *,
    stale_days: float = 7.0,
    parquet_paths: Iterable[Path] | None = None,
    reference_time: datetime | None = None,
    calendar_dir: Path | None = None,
) -> dict[str, Any]:
    """扫描缓存并返回结构化报告（不写磁盘）。

    若 ``calendar_dir`` 下存在本地交易日历缓存，则缺日相对官方日历（上证口径）校验；
    否则回退为扫描并集近似，并在汇总 ``info`` 中说明过渡期。
    """
    cache_dir = Path(cache_dir)
    cal_root = Path(calendar_dir) if calendar_dir is not None else default_calendar_dir()
    paths = list(parquet_paths) if parquet_paths is not None else discover_qfq_parquets(cache_dir)
    reports: list[FileReport] = []

    calendar_trade_dates: set[pd.Timestamp] | None = None
    calendar_meta: dict[str, Any] | None = None
    calendar_asof: date | None = None
    loaded_cal = load_trade_calendar_disk(cal_root)
    calendar_active = loaded_cal is not None
    if loaded_cal is not None:
        cal_frame, calendar_meta = loaded_cal
        calendar_trade_dates = set(cal_frame["trade_date"].tolist())
        end_d = _parse_iso_date(calendar_meta.get("end")) if calendar_meta else None
        now = reference_time or datetime.now(timezone.utc)
        today = now.date() if now.tzinfo is None else now.astimezone(timezone.utc).date()
        if end_d is not None:
            calendar_asof = min(today, end_d)
        else:
            calendar_asof = today

    csi300_dates = _csi300_bar_dates(cache_dir) if calendar_active else None

    # 第一遍：加载日期，构建全体并集（交易日代理）
    per_file_dates: dict[str, set[pd.Timestamp]] = {}
    prefetch_failed: dict[str, FileReport] = {}
    for pq in paths:
        key = str(pq.resolve())
        meta_path = pq.with_suffix("").with_suffix(".meta.json")
        parsed = _parse_filename_stem(pq)
        sym_guess = parsed[0] if parsed else ""
        adj_guess = parsed[1] if parsed else ""
        try:
            frame = pd.read_parquet(pq)
        except Exception as exc:
            fr = FileReport(
                parquet_path=key,
                meta_path=str(meta_path),
                symbol_from_filename=sym_guess,
                adjust_from_filename=adj_guess,
            )
            fr.errors.append(f"无法读取 parquet: {exc}")
            prefetch_failed[key] = fr
            continue
        if "date" not in frame.columns:
            fr = FileReport(
                parquet_path=key,
                meta_path=str(meta_path),
                symbol_from_filename=sym_guess,
                adjust_from_filename=adj_guess,
            )
            fr.errors.append("parquet 缺少 date 列")
            prefetch_failed[key] = fr
            continue
        ds = _dates_series(frame).dropna()
        per_file_dates[key] = set(ds.tolist())

    union_ts: set[pd.Timestamp] = set()
    for s in per_file_dates.values():
        union_ts |= s
    union_sorted = sorted(union_ts)

    calendar_csi300_info: list[str] = []
    calendar_csi300_warnings: list[str] = []
    if calendar_active and calendar_trade_dates is not None:
        if csi300_dates is not None and union_ts:
            min_u = min(union_ts)
            max_u = max(union_ts)
            c_idx = {d for d in csi300_dates if min_u <= d <= max_u}
            cal_idx = {d for d in calendar_trade_dates if min_u <= d <= max_u}
            only_cal = sorted(cal_idx - c_idx)
            only_idx = sorted(c_idx - cal_idx)
            if only_cal or only_idx:
                calendar_csi300_warnings.append(
                    "沪深300 指数缓存与上证日历在扫描并集窗口内有差异："
                    f"仅日历有 {len(only_cal)} 日，仅指数有 {len(only_idx)} 日"
                )
            sample_cal = ", ".join(ts.strftime("%Y-%m-%d") for ts in only_cal[:5])
            more_cal = f" …另有 {len(only_cal) - 5} 个" if len(only_cal) > 5 else ""
            sample_idx = ", ".join(ts.strftime("%Y-%m-%d") for ts in only_idx[:5])
            more_idx = f" …另有 {len(only_idx) - 5} 个" if len(only_idx) > 5 else ""
            calendar_csi300_info.append(
                f"CSI300 交叉：窗口 [{min_u.date()}, {max_u.date()}]；"
                f"仅日历: {sample_cal}{more_cal}；仅指数: {sample_idx}{more_idx}"
            )
        elif csi300_dates is None:
            calendar_csi300_info.append(
                "未找到 .cache/bars/000300_index.parquet，跳过 CSI300 与日历交叉对照"
            )

    # 第二遍：逐项检查
    for pq in paths:
        key = str(pq.resolve())
        if key in prefetch_failed:
            reports.append(prefetch_failed[key])
            continue

        parsed = _parse_filename_stem(pq)
        meta_path = pq.with_suffix("").with_suffix(".meta.json")
        fr = FileReport(
            parquet_path=key,
            meta_path=str(meta_path),
            symbol_from_filename=parsed[0] if parsed else "",
            adjust_from_filename=parsed[1] if parsed else "",
        )
        if parsed is None:
            fr.errors.append("文件名不符合 {symbol}_{adjust}.parquet")
            reports.append(fr)
            continue

        meta, meta_err = _load_meta(meta_path)
        if meta_err:
            fr.errors.append(meta_err)
        if meta is not None:
            fr.meta = dict(meta)
            fr.source = str(meta.get("source", "")) or None
            msym = meta.get("symbol")
            if msym is not None:
                try:
                    meta_sym = normalize_symbol(str(msym))
                except ValueError:
                    meta_sym = _cell_symbol(msym)
                if meta_sym != parsed[0]:
                    fr.errors.append(f"meta.symbol={msym!r} 与文件名代码 {parsed[0]} 不一致")
            madj = str(meta.get("adjust", "")).strip().lower()
            if madj and madj != "qfq":
                fr.warnings.append(f"meta.adjust={meta.get('adjust')!r}，文件名要求 qfq")

            fetched = _parse_iso_datetime(meta.get("fetched_at"))
            if fetched is not None:
                now = reference_time or datetime.now(timezone.utc)
                if now.tzinfo is None:
                    now = now.replace(tzinfo=timezone.utc)
                age_days = (now - fetched).total_seconds() / 86400.0
                if age_days > stale_days:
                    fr.warnings.append(
                        f"fetched_at 距今 {age_days:.1f} 天，超过阈值 {stale_days} 天"
                    )
            elif meta.get("fetched_at") not in (None, ""):
                fr.warnings.append(f"无法解析 fetched_at: {meta.get('fetched_at')!r}")

        frame = pd.read_parquet(pq)

        fr.rows_in_file = len(frame)

        if meta is not None and "rows" in meta:
            try:
                meta_rows = int(meta["rows"])
            except (TypeError, ValueError):
                fr.warnings.append(f"meta.rows 非整数: {meta.get('rows')!r}")
            else:
                if meta_rows != fr.rows_in_file:
                    fr.errors.append(f"meta.rows={meta_rows} 与 parquet 行数 {fr.rows_in_file} 不一致")

        missing_cols = [c for c in REQUIRED_BAR_COLUMNS if c not in frame.columns]
        if missing_cols:
            fr.errors.append(f"缺少必要列: {missing_cols}")
            reports.append(fr)
            continue

        for oc in OPTIONAL_BAR_COLUMNS:
            if oc in frame.columns:
                fr.optional_columns_present.append(oc)
            else:
                fr.optional_columns_absent.append(oc)

        sym_col = frame["symbol"].map(_cell_symbol)
        bad_sym_mask = sym_col != parsed[0]
        if bool(bad_sym_mask.any()):
            n = int(bad_sym_mask.sum())
            fr.errors.append(f"symbol 列与文件名代码不一致的行数: {n}")

        # 日期与缺失交易日（相对并集）
        dseries = _dates_series(frame)
        if dseries.isna().any():
            fr.errors.append(f"date 列有 {int(dseries.isna().sum())} 个无法解析的值")

        valid_dates = dseries.dropna()
        if valid_dates.duplicated().any():
            dup_n = int(valid_dates.duplicated().sum())
            fr.warnings.append(f"存在重复交易日: {dup_n} 行")

        file_dates = per_file_dates.get(key, set())
        if file_dates:
            first = min(file_dates)
            last = max(file_dates)
            missing_union = [
                ts for ts in union_sorted if first <= ts <= last and ts not in file_dates
            ]
            fr.missing_days_union = [ts.strftime("%Y-%m-%d") for ts in missing_union]

            if calendar_trade_dates is not None and calendar_asof is not None:
                asof_ts = pd.Timestamp(calendar_asof)
                window_end = last if last <= asof_ts else asof_ts
                if window_end < first:
                    expected_trade = []
                else:
                    expected_trade = sorted(
                        ts
                        for ts in calendar_trade_dates
                        if first <= ts <= window_end
                    )
                missing_cal = [ts for ts in expected_trade if ts not in file_dates]
                fr.missing_days = [ts.strftime("%Y-%m-%d") for ts in missing_cal]
                for d in fr.missing_days:
                    fr.warnings.append(f"相对交易日历缺失: {d}")

                extra_ts = sorted(ts for ts in file_dates if ts not in calendar_trade_dates)
                fr.extra_non_trading_days = [ts.strftime("%Y-%m-%d") for ts in extra_ts]
                for d in fr.extra_non_trading_days:
                    fr.errors.append(f"非交易日出现行情（相对上证日历）: {d}")
            else:
                fr.missing_days = list(fr.missing_days_union)

        # OHLC 与成交
        for col in ("open", "high", "low", "close", "volume"):
            if not pd.api.types.is_numeric_dtype(frame[col]):
                fr.errors.append(f"列 {col} 非数值类型")

        if not fr.errors:
            o = pd.to_numeric(frame["open"], errors="coerce")
            h = pd.to_numeric(frame["high"], errors="coerce")
            l = pd.to_numeric(frame["low"], errors="coerce")
            c = pd.to_numeric(frame["close"], errors="coerce")
            vol = pd.to_numeric(frame["volume"], errors="coerce")

            if o.isna().any() or h.isna().any() or l.isna().any() or c.isna().any() or vol.isna().any():
                fr.errors.append("OHLCV 中存在 NaN")

            if not fr.errors:
                mx = np.maximum.reduce([o.to_numpy(), c.to_numpy(), l.to_numpy()])
                mn = np.minimum.reduce([o.to_numpy(), c.to_numpy(), l.to_numpy()])
                eps = 1e-9
                bad_high = h.to_numpy() + eps < mx
                bad_low = l.to_numpy() - eps > mn
                if bool(bad_high.any()):
                    fr.errors.append(f"high < max(open,close,low) 的行数: {int(bad_high.sum())}")
                if bool(bad_low.any()):
                    fr.errors.append(f"low > min(open,close,low) 的行数: {int(bad_low.sum())}")

                if bool((c <= 0).any()):
                    fr.errors.append(f"close <= 0 的行数: {int((c <= 0).sum())}")
                non_positive_ohlc = (o <= 0) | (h <= 0) | (l <= 0)
                if bool(non_positive_ohlc.any()):
                    fr.warnings.append(f"存在 open/high/low <= 0 的行: {int(non_positive_ohlc.sum())}")

                if bool((vol < 0).any()):
                    fr.errors.append(f"volume < 0 的行数: {int((vol < 0).sum())}")
                if bool((vol == 0).any()):
                    fr.warnings.append(f"volume == 0 的交易日: {int((vol == 0).sum())}")

            if "amount" in frame.columns:
                amt = pd.to_numeric(frame["amount"], errors="coerce")
                if amt.isna().any():
                    fr.warnings.append("amount 列存在无法解析的数值")
                elif bool((amt < 0).any()):
                    fr.errors.append(f"amount < 0 的行数: {int((amt < 0).sum())}")

        src_note = fr.source or "(meta 未记录 source)"
        fr.info.append(
            f"数据源: {src_note}；可选列 present={fr.optional_columns_present or '无'}，"
            f"absent={fr.optional_columns_absent or '无'}"
        )

        reports.append(fr)

    error_count = sum(len(r.errors) for r in reports)
    warning_count = sum(len(r.warnings) for r in reports)
    missing_day_total = sum(len(r.missing_days) for r in reports)
    missing_union_total = sum(len(r.missing_days_union) for r in reports)
    extra_non_trade_total = sum(len(r.extra_non_trading_days) for r in reports)

    summary_info: list[str] = []
    if not calendar_active:
        summary_info.append(
            "未使用官方交易日历（.cache/calendar 无有效缓存），缺日判断回退为扫描并集近似"
        )
    if calendar_csi300_info:
        summary_info.extend(calendar_csi300_info)
    if calendar_csi300_warnings:
        for w in calendar_csi300_warnings:
            summary_info.append(f"[日历×CSI300] {w}")

    return {
        "cache_dir": str(cache_dir.resolve()),
        "calendar_dir": str(cal_root.resolve()),
        "stale_days": stale_days,
        "union_trading_dates": len(union_sorted),
        "calendar_active": calendar_active,
        "calendar_source": (calendar_meta or {}).get("source") if calendar_active else None,
        "calendar_rows": (calendar_meta or {}).get("rows") if calendar_active else None,
        "calendar_asof": calendar_asof.isoformat() if calendar_asof is not None else None,
        "calendar_id": CALENDAR_ID if calendar_active else None,
        "files_scanned": len(paths),
        "summary": {
            "errors": error_count,
            "warnings": warning_count,
            "files_with_missing_days": sum(1 for r in reports if r.missing_days),
            "missing_day_rows_total": missing_day_total,
            "missing_vs_union_rows_total": missing_union_total,
            "extra_non_trading_rows_total": extra_non_trade_total,
        },
        "info": summary_info,
        "files": [
            {
                "parquet": r.parquet_path,
                "meta": r.meta_path,
                "symbol_file": r.symbol_from_filename,
                "adjust_file": r.adjust_from_filename,
                "rows": r.rows_in_file,
                "source": r.source,
                "optional_present": r.optional_columns_present,
                "optional_absent": r.optional_columns_absent,
                "errors": r.errors,
                "warnings": r.warnings,
                "info": r.info,
                "missing_days": r.missing_days,
                "missing_vs_union": r.missing_days_union,
                "extra_non_trading_days": r.extra_non_trading_days,
            }
            for r in reports
        ],
    }


def format_report_text(report: dict[str, Any]) -> str:
    """人类可读的简短报告。"""
    lines: list[str] = []
    lines.append(f"缓存目录: {report['cache_dir']}")
    lines.append(f"日历目录: {report.get('calendar_dir', '')}")
    if report.get("calendar_active"):
        lines.append(
            f"交易日历: 已加载 ({report.get('calendar_id')})，来源 {report.get('calendar_source')}，"
            f"rows={report.get('calendar_rows')}，as_of={report.get('calendar_asof')}"
        )
    else:
        lines.append("交易日历: 未找到缓存，缺日判断使用并集近似（过渡期）")
    lines.append(
        f"扫描 qfq 文件数: {report['files_scanned']}，并集交易日数量: {report['union_trading_dates']}"
    )
    for msg in report.get("info") or []:
        lines.append(f"[信息] {msg}")
    s = report["summary"]
    lines.append(
        f"汇总: 错误 {s['errors']}，警告 {s['warnings']}，"
        f"存在日历缺失交易日的文件 {s.get('files_with_missing_days', 0)}"
        f"（日历缺失条目共 {s.get('missing_day_rows_total', 0)}；"
        f"相对并集缺失 {s.get('missing_vs_union_rows_total', 0)}；"
        f"非交易日行情 {s.get('extra_non_trading_rows_total', 0)}）"
    )
    for block in report["files"]:
        sym = block.get("symbol_file") or "?"
        lines.append("")
        lines.append(f"--- {sym} ({Path(block['parquet']).name}) ---")
        if block["errors"]:
            for msg in block["errors"]:
                lines.append(f"  [错误] {msg}")
        if block["warnings"]:
            for msg in block["warnings"]:
                lines.append(f"  [警告] {msg}")
        for msg in block.get("info", []):
            lines.append(f"  [信息] {msg}")
        md = block.get("missing_days") or []
        if md:
            show = md[:20]
            more = f" … 另有 {len(md) - len(show)} 个" if len(md) > 20 else ""
            lines.append(f"  相对日历缺失交易日（共 {len(md)}）: {', '.join(show)}{more}")
        mu = block.get("missing_vs_union") or []
        if mu:
            show = mu[:10]
            more = f" … 另有 {len(mu) - len(show)} 个" if len(mu) > 10 else ""
            lines.append(f"  相对并集缺失（对照，共 {len(mu)}）: {', '.join(show)}{more}")
        ex = block.get("extra_non_trading_days") or []
        if ex:
            show = ex[:10]
            more = f" … 另有 {len(ex) - len(show)} 个" if len(ex) > 10 else ""
            lines.append(f"  非交易日出现行情（共 {len(ex)}）: {', '.join(show)}{more}")
    return "\n".join(lines) + "\n"


def main_cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "只读检查 A 股日线缓存（*_qfq.parquet + .meta.json）。"
            "不要用合成/features 数据当作线上缓存的黄金标准。"
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="缓存目录（默认：ASHARE_CACHE_DIR 或仓库根 .cache/bars）",
    )
    parser.add_argument(
        "--calendar-dir",
        type=Path,
        default=None,
        help="交易日历目录（默认：ASHARE_CALENDAR_DIR 或仓库根 .cache/calendar）",
    )
    parser.add_argument(
        "--stale-days",
        type=float,
        default=7.0,
        help="meta.fetched_at 早于此天数则警告（默认 7）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="将完整 JSON 报告写入该路径",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="仅输出汇总一行（仍写入 --out）",
    )
    ns = parser.parse_args(argv)
    directory = ns.cache_dir if ns.cache_dir is not None else default_cache_dir()
    cal = ns.calendar_dir if ns.calendar_dir is not None else default_calendar_dir()
    report = run_bars_cache_qc(directory, stale_days=ns.stale_days, calendar_dir=cal)
    if ns.out is not None:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if ns.quiet:
        s = report["summary"]
        print(
            f"files={report['files_scanned']} errors={s['errors']} warnings={s['warnings']} "
            f"missing_day_entries={s['missing_day_rows_total']}"
        )
    else:
        print(format_report_text(report), end="")
    if report["summary"]["errors"] > 0:
        return 1
    return 0
