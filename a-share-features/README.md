# A 股 ETF 特征工程流水线

> **与主链路口径（S8 Step B 已对齐）**
>
> - 生产 / 半自动盯盘金标准在本仓库主链路（不是本目录）：**6 位代码 + 前复权 `qfq` + watch=`159338`/`513010`**。
> - **恒生科技以 `513010` 为准**；历史代码 `513180.SH` 是另一只 ETF，**禁止**当作 `513010` 别名（仅可显式 `--codes 513180.SH` 做一次性研究，会打日志警告）。
> - 旁路默认 **qfq**，特征文件名 `etf_features_qfq_*`；与主链路一致。
> - 默认从主链路缓存读入（`.cache/bars/{6位}_qfq.parquet`）；`--raw-dir` 仍可走旧 raw 路径。
> - `source` 含 `synthetic` 的 raw **默认拒绝**；仅 `--allow-synthetic` 可用于 fixtures/联调。合成样例在 `tests/fixtures/synthetic/`（及 `a-share-data/fixtures/synthetic/`），**禁止**当金标准。

清洗日 K → 计算技术指标 → 写出策略/回测用特征矩阵。

- **标的（默认）**：`513010.SH`（恒生科技）、`159338.SZ`（中证 A500）
- **复权（默认）**：前复权（`adj_mode=qfq`）

## 环境（Windows）

推荐使用已有虚拟环境：

```bat
C:\Users\sousenw\.venvs\a-share-quant\Scripts\activate
cd /d 本仓库\a-share-features
pip install -r requirements.txt
```

依赖：`pandas`、`numpy`、`pyarrow`。不升环境、不接 tushare。

## 默认路径（双入口）

缓存目录**不要写死**某一台机器。解析顺序：

1. 环境变量 `ASHARE_CACHE_BARS_DIR`，否则 `QUANT_DASHBOARD_CACHE_BARS`
2. 当前目录或本仓库下的 `./.cache/bars`（目录存在才用）
3. 少数本机路径只作 fallback（例如旧的 `/workspace/quant-dashboard/.cache/bars`）

请用环境变量或 `--cache-dir` 覆盖。未设置且仓库里已有 `.cache/bars` 时，默认就是这份主链路缓存。

| 用途 | 怎么指定 |
|------|----------|
| **主入口：主链路缓存** | `--cache-dir` 或上面的环境变量；否则 `./.cache/bars` |
| 旧入口：原始行情 | `--raw-dir`（Windows 常见 `C:\Users\sousenw\a-share-data\raw\`） |
| 特征输出 | `--features-dir` |

## 用法

```bat
:: 默认：缓存优先（主链路 qfq bars）
python pipeline.py --features-dir C:\Users\sousenw\a-share-data\features

:: 显式指定缓存目录
python pipeline.py --cache-dir C:\Users\sousenw\quant-dashboard\.cache\bars

:: 旧路径：显式 --raw-dir（真实交付 parquet）
python pipeline.py --raw-dir C:\Users\sousenw\a-share-data\raw --features-dir C:\Users\sousenw\a-share-data\features

:: 研究-only 代码（会警告；不是 513010 别名）
python pipeline.py --codes 513180.SH 159338.SZ --raw-dir ... --allow-synthetic

:: fixtures 合成样例（必须带门禁）
python pipeline.py --raw-dir tests\fixtures\synthetic --allow-synthetic --codes 513180.SH 159338.SZ
```

输出文件：

- `etf_features_qfq_{start}_{end}.parquet`
- `etf_features_qfq_latest.parquet`

## 适配器：主链路缓存 → 旁路 raw

模块：`adapters/from_ashare_cache.py`

| 主链路 | 旁路 raw |
|--------|----------|
| `symbol` 6 位 | `ts_code`（`513010`→`513010.SH`，`159338`→`159338.SZ`） |
| `date` | `trade_date` YYYYMMDD |
| `open/high/low/close` | 同名 |
| `volume` | `vol`（单位 **「手」**，常量 `VOLUME_UNIT = "hand"`；**不**换算为股） |
| `amount` | `amount` |
| `pct_change`（若有） | `pct_chg`；缺失则按 close/pre_close 推算 |
| — | `pre_close`：前日 close |
| meta.source | `source=ashare_cache:{meta.source}` |

## 原始字段（`--raw-dir` parquet）

必需：`ts_code`, `trade_date`(YYYYMMDD), `source`, `open`, `high`, `low`, `close`, `pre_close`, `pct_chg`, `vol`, `amount`

建议：`nav`, `nav_date`；可选：`accum_nav`, `iopv`, `premium_to_nav_est`, …

## 清洗要点

1. 代码别名：`513010` / `159338` → 规范 ts_code；`513180` 仅研究用
2. 两只 ETF 按交易日并集 **outer-align**（不对价格 ffill）
3. `nav` 按 `nav_date` asof 对齐；`nav_lag_days` 为日历日差
4. OHLC / 量额校验 → `is_valid`；`vol` 单位见上（「手」）
5. 指标：MA / MACD / RSI 等（窗口不足为 NaN）

## 特征列简表

| 列名 | 含义 |
|------|------|
| `ts_code` | 证券代码 |
| `trade_date` | 交易日 YYYYMMDD |
| `adj_mode` | 复权模式，固定 `qfq` |
| `open/high/low/close/pre_close` | OHLC 与昨收 |
| `pct_chg` | 涨跌幅（%） |
| `vol/amount` | 成交量（「手」）/ 成交额 |
| `nav/accum_nav/nav_date/nav_lag_days` | 净值相关 |
| `iopv` / `premium_to_nav_est` | IOPV / 溢价估计 |
| `ret_*` / `vol_*` / `ma*` / `macd_*` / `rsi_14` / `trend_dir` | 技术指标 |
| `is_valid` / `missing_flags` | 校验与缺失标记 |

## 测试

```bat
cd /d 本仓库\a-share-features
python -m tests.run_tests
:: 若已安装 pytest：
python -m pytest tests/ -q
```

## Fixtures

- `tests/fixtures/cache_bars/`：裁剪后的主链路 qfq 样例（离线）
- `tests/fixtures/synthetic/`：历史合成 raw（仅 `--allow-synthetic`）
