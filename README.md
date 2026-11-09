# A 股可视化量化工具（阶段 4）

自用的 A 股研究脚手架：拉日线、落到本地缓存、跑可插拔策略、看 K 线，做研究回测和人工清单，再把券商客户端里的成交抄回来和信号对照。阶段 4 把研究用费用扣进实际收益，给出相对买入持有和沪深 300 的超额，并用可配置日期标出 train / test。

**阶段 1 是工具和看图。** 大约每月 3% 从阶段 2 起就是策略升级的**方向**，不是示例承诺，也不是示例策略的验收。不要把短均线改密去造这个数字。图、清单和成交日志都**不会**连接券商或自动下单。

试用记录（产品预期，不是核定过的普遍结论，更不是交易指令）：

- 裸的 10/30 双均线，在 `000001`、`600519`、大约 5 年、收盘对收盘、不计费用时显得很弱（大约每月 -0.08% 和 -0.40%），假金叉很多。只调窗口不够。
- 过滤之后工具链能用，假交叉少一些，但示例的粗算月收益仍大约在 -0.09% 到 -0.26%，离大约 3%/月还有约 3 个百分点。跟单优先看 **next_open**，不要把收盘价当成已经成交的价格。

## 这一版相对裸交叉改了什么

默认策略名仍是 `dual_ma_10_30`，窗口仍是 10 和 30。额外规则：

- 上穿或下穿要连续 `confirm_days`（默认 2）个交易日才确认，减少单日假交叉。
- 最短持有 `min_hold_days`（默认 5）。确认后的死叉在此之前不离场；止损和止盈可以提前离场。
- 趋势过滤：收盘价要站上更长的均线（默认 60）。只改短均线窗口不算升级。
- ADX 近似（Wilder，默认 14 日，下限 20）。震荡里的金叉不开仓。缺高低价时这个过滤自动跳过。
- 建议仓位默认 `0.5`，不是写死的 0/1 满仓。需要满仓时把 `target_weight` 设为 1。
- 止损默认 8%（收盘价相对入场收盘价）。这不是盘中触发。盘中止损仍是以后的假设。回测可以另选 `next_open_stop`：触发仍看收盘，假定下一交易日开盘才卖出。止盈钩子存在，默认关闭（`take_profit=None`）。
- 回测并排表的第一行锁定为次日开盘成交，标「跟单看 next_open」。收盘成交只作对照。
- 信号和清单带上持有天数、当前腿浮动盈亏、连续亏损段数、距上次交叉的交易日、建议仓位。持有不超过 5 个交易日就完成的来回，标记为「短持翻脸」。

把确认、最短持有、趋势、ADX、止损都关掉并把仓位设回 1，可以退回裸的 0/1 交叉，只用于对照，不是日常示例。

## 环境

- Python 3.11 或更高（本仓库在 3.12 下开发）
- 能访问外网的第一次拉数。同一区间再次运行只读 `.cache/bars/`，可以断网
- 事件日历默认只读仓库里的 JSON。`FETCH_EVENTS` 另说，默认关闭，离线笔记本不依赖它

Debian / Ubuntu 上如果 `python3 -m venv` 报 ensurepip 不可用，先装虚拟环境包：

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip
```

若仍提示缺少 ensurepip，再装与解释器版本对应的包，例如 `python3.12-venv`。

## 安装

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

`pip install -e .` 让 `data`、`strategy`、`viz`、`backtest`、`orders`、`journal` 在任意工作目录都能导入。笔记本第一格也会把仓库根目录加进 `sys.path`。

## 运行

日常入口是笔记本。请从**仓库根目录**启动：

```bash
source .venv/bin/activate
jupyter lab notebooks/daily_entry.ipynb --ip=127.0.0.1 --port=8765
```

不需要单独注册内核。若你用的是系统里另一个 Jupyter，而不是这个虚拟环境，做一次：

```bash
python -m ipykernel install --user --name ashare-quant --display-name "Python (ashare-quant)"
```

无界面执行同一本笔记本（在仓库根目录）：

```bash
source .venv/bin/activate
jupyter nbconvert --execute notebooks/daily_entry.ipynb \
  --to notebook \
  --output /tmp/daily_entry.executed.ipynb \
  --ExecutePreprocessor.timeout=300
```

笔记本默认做这些事：

1. 按样本取日线。默认 `demo` 是平安银行 `000001`、贵州茅台 `600519`。`banks`、`consumer`、`liquid` 是更宽的静态名单，第一次要联网
2. 跑带过滤的 10/30 双均线。事件约束默认关闭，图上仍显示本地政策日历
3. 并排看成交假设。第一行锁定为次日开盘（标「跟单看 next_open」），然后才是收盘成交和次日开盘止损
4. 打印信号表。清单带诊断列和止损/交叉原因。连续亏损达到 3，或短持翻脸反复出现，写「建议暂停加仓」，并写明仍持有还是空仓
5. 导出 CSV 和 Markdown。禁投名单命中则整单拒绝、不写文件
6. 用次日开盘价登记演示成交，强制 `is_simulated=true`，按研究用费率扣佣金和印花税，再对照滑点、是否跟满、扣费后的实际收益、相对买入持有和沪深 300 的超额，以及 train / test
7. 换到均值回归再看一遍并排回测，并打印一组参数敏感度
8. 画出 K 线，并给出不拉行情的 `block_entries` / `reduce` 样本

不想开 Jupyter 时，用同一条路径的脚本。HTML 图和清单写到 `output/`：

```bash
source .venv/bin/activate
python scripts/plot_demo.py
python scripts/plot_demo.py --offline
python scripts/plot_demo.py --refresh
python scripts/plot_demo.py --sample liquid    # 银行、消费，以及保险、电力、券商各一只
python scripts/plot_demo.py --offline --split 2024-01-01
python scripts/plot_demo.py --event-mode block_entries
python scripts/plot_demo.py --fetch-events
python scripts/event_modes.py               # 不拉行情，只看三种事件开关
```

浏览器打开 `output/000001_dual_ma_10_30.html`。HTML 内嵌了 plotly.js，离线也能看。清单在 `output/orders.csv` 和 `output/orders.md`。演示成交在 `output/fills.csv` 和 `output/fills.json`。

日常看板是同一套模块的浅色桌面页面。在仓库根目录：

```bash
source .venv/bin/activate
streamlit run dashboard/app.py --server.port 8766 --server.address 127.0.0.1
```

浏览器打开 <http://127.0.0.1:8766>。默认只读本地行情缓存。顶栏先选频率：日频是双均线和均值回归，周频是周频均值回归。自选、K 线、右侧政策与事件、底部回测摘要和信号预览在同一页。周频时主图改成周 K（一根对应该周最后一个交易日），信号和清单按当周收盘信号、下一交易日开盘成交，并写明单票、行业、总仓上限和研究用费用。点「生成下单清单」打开预览，主按钮导出 CSV，旁边是 Markdown，文本框用来复制。黄条写明系统不会自动下单。跟单摘要锁定为次日开盘，并标「跟单看 next_open」。旁边的粗体是「回测未扣费；对账看成交日志」。

界面上仍开放过、这一版已经定下来的选择：

- K 线沿用 A 股习惯，红涨绿跌。图上的买入标记是红、卖出标记是绿，和涨跌同色；信号表里的方向用同样的颜色，并且始终带「买入 / 卖出」文字。
- 「事件日降仓提示」默认关闭，和 `EventGuard` 的 `off` 一致。打开后才在高影响事件日把目标仓位降到 0。
- 建议仓位继续是占总资产的比例。清单预览里的估算股数只针对买入，按参考资金 10 万元乘以这个比例，再向下取整到 100 股。不够一手时写「资金不足一手」，这一行建议仍在；卖出不代算数量。导出文件仍是原来的 CSV / Markdown 列。
- 本地事件日历没有利好、利空字段，卡片只显示影响等级。筛选「海外」「财报」在现有日历上可能是空的，把对应事件加进 `data/calendars/cn_events.json` 后才会出现。
- 不做深色模式。窄屏时各栏会上下叠放。
- 胜率按已平仓来回的信号价计算，两种成交假设共用这个数字，不含费用。它不是每月 3% 的验收。

不访问网络的单元测试：

```bash
source .venv/bin/activate
python -m pytest -q
```

## 缓存和离线

默认目录是仓库根下的 `.cache/bars/`。也可以指定：

```bash
export ASHARE_CACHE_DIR=/你的路径
```

**交易日历**单独落在 `.cache/calendar/`（上证开市日，来源 `akshare.tool_trade_date_hist_sina`）。目录可用环境变量覆盖：

```bash
export ASHARE_CALENDAR_DIR=/你的路径
```

文件名像 `000001_qfq.parquet`，旁边有同名 `.meta.json`，记录已经向数据源查询过的起止日期、复权方式和来源。查询过的区间再次请求不会联网；把区间加长才会补拉并整表覆盖。前复权价格在除权除息后会被追溯调整，需要对齐最新价时用 `refresh=True` 或 `--refresh`。

`offline=True`（脚本 `--offline`，笔记本 `OFFLINE = True`）只读缓存。缓存没盖住请求区间时直接报错，不会偷偷联网。

### 交易日历刷新（需联网）

QC **默认不联网**；先显式拉一次日历再跑 QC：

```bash
source .venv/bin/activate
python3 scripts/refresh_trade_calendar.py
# 或指定目录：python3 scripts/refresh_trade_calendar.py --calendar-dir /你的路径
```

写入 `cn_sse_trade_dates.parquet`（失败时回退为同名 `.csv`）与 `cn_sse_trade_dates.meta.json`。

### 缓存数据质量（只读）

对 `.cache/bars/` 里 `*_qfq.parquet` 与同名 `.meta.json` 做**只读**检查（不修改缓存、不在默认路径下偷偷拉数）。若 `.cache/calendar/`（或 `ASHARE_CALENDAR_DIR`）里已有上证交易日历缓存，则**缺日**按「日历 ∩ [该文件首根, 该文件末根]」再截断到 `min(今天 UTC 日期, meta.end)` 后的期望集合比对：相对日历缺日为**警告**（停牌与数据源漏行在此口径下无法区分）；文件中出现日历以外的日期为**错误**（更像脏数据）。仍保留「相对全部已扫描 qfq 日期**并集**」的副指标 `missing_vs_union`，并在汇总中保留 `union_trading_dates`。若日历缓存缺失，则自动回退为仅并集近似，并在报告 `info` 中说明过渡期。可选：若存在 `000300_index.parquet`，会与日历做对称差**信息/警告**对照，不修改日历。

缺列、OHLC 逻辑、成交指向、meta.rows 与行数一致性、文件名/列/meta 的股票代码、`fetched_at` 是否过旧等仍会写入报告；可选列（如振幅、换手）只记来源差异，不因缺列而报错。合成数据或 features 样例**不得**当作线上缓存的黄金标准，本工具面向真实落地文件。

```bash
source .venv/bin/activate
python3 scripts/qc_bars_cache.py
python3 scripts/qc_bars_cache.py --cache-dir /你的路径 --calendar-dir /你的日历目录 --stale-days 7 --out /tmp/bars_qc.json
python3 scripts/qc_bars_cache.py --quiet
```

有任一错误检查时进程退出码为 `1`（例如 OHLC 违反、`close <= 0`、meta 行数不匹配、非交易日误入行情）。退出码 `0` 仍可能含警告（如 `volume==0`、`fetched_at` 过旧、相对日历缺日）。

事件日历是另一条路径：默认读 `data/calendars/cn_events.json`，不访问网络。只有把 `fetch=True` 打开时才尝试 akshare 的 LPR，失败就退回本地文件。行情缓存命中，不等于事件外源也可用。

```python
from data import get_bars

bars = get_bars("000001", "2023-01-01", "2026-11-06", adjust="qfq")
```

`adjust` 可用 `qfq`（前复权，默认）、`hfq`（后复权）或 `""`（不复权）。代码可写成 `000001`、`sz000001` 或 `600519.SH`。

## 生产数据口径（S8 已拍板）

半自动盯盘、日/周回测与看板的**生产口径**固定为：

| 项 | 约定 |
|----|------|
| 代码 | **6 位**（如 `159338`、`513010`）；入参别名会归一成 6 位 |
| 复权 | 默认 **前复权 `qfq`**；同一回测任务内禁止与后复权混算收益 |
| 监控池 `watch` | `159338` 中证A500ETF + **`513010` 恒生科技ETF** |
| 恒生科技 | **仅 `513010`**。`513180` 是另一只产品，**禁止**当作 `513010` 的别名或静默替换 |
| 行情源 | 免费开源（akshare）；东财优先、失败可回退腾讯等；不把旁路/合成样例当金标准 |
| 基准 | 沪深300 走 `get_csi300` / `000300_index.*` 缓存；无缓存时超额退化为只比买入持有 |

旁路仓库 `a-share-features` / `a-share-data` 若仍保留研究用途，必须与主链路隔离；特征文件的 `adj_mode` 与文件名须与真实复权一致。合成 raw（`source=synthetic_hfq` 等）只允许出现在 fixtures / `--generate-sample`，**不得**作为生产或盯盘对照的金标准。

口径变更仍经产品经理澄清后再改代码。

旁路特征工程在本仓库子目录 `a-share-features/`，不是独立仓。生产盯盘的金标准仍在主链路（6 位 + `qfq` + `watch`）。旁路默认跟这套口径：`513010` / `159338`、前复权。它的缓存用 `ASHARE_CACHE_BARS_DIR`、`QUANT_DASHBOARD_CACHE_BARS` 或 `--cache-dir`；没设时优先本仓库 `./.cache/bars`。

## 回测

`compare_fills` 对同一条目标权重跑两种假设。笔记本和 `scripts/plot_demo.py` 把跟单视图**锁定为 next_open**，并排表的第一行就是它：

- `next_open`：成交价用下一交易日开盘。隔夜缺口算旧仓位，开盘后的日内涨跌算新仓位。**跟单看 next_open。**
- `close`：当天收盘才知道的权重，按收盘价成交，从下一根 K 线的收盘到收盘才计收益。只作对照。

另外可以单独跑 `fill="next_open_stop"`（次日开盘止损）：止损仍用收盘价触发，不是盘中。触发日收盘后仓位还在，下一交易日开盘才退出，隔夜缺口由旧仓位承担。其他买卖仍按收盘假设。最后一根 K 线没有下一根开盘，这一天和收盘假设相同。策略里的止损同样看收盘价相对入场收盘价。盘中止损没有实现。

报告总收益、最大回撤（正数，0.2 表示 20%）和换手。半仓进出一轮，换手是 1。`rough_monthly_return` 只是把整段收益按大约 21 个交易日折成单月，方便和「每月 3%」这个方向对照，**不是**稳定月收益，也不是示例已经兑现。

回测路径没有滑点、佣金、印花税，也不处理涨跌停和停牌。这里的成交只是价格假设。手工成交记在 `journal`，费用从实际收益里扣，和这两种假设并排看，但会计方法不同，差值不是审计结论。

## 信号表和清单

`signal_board` 的列至少包括：代码、方向、日期、策略、原因、价格、建议仓位、持有天数、当前腿浮动盈亏、连续亏损段数、距上次交叉的交易日、标记。

`export_order_list` 写成 CSV 和 Markdown。默认列包括建议仓位、持有天数、当前腿浮动盈亏、连续亏损、距上次交叉、短持翻脸，以及 `reason`（止损或均线确认的原文）。

连续亏损达到 3（`pause_after_losses`，可改），或短持翻脸的**卖出**至少 2 次，清单标「建议暂停加仓」，并写明「仍持有，不是新开仓」或「当前空仓」。汇总写在 Markdown 开头；行内只标该代码最新的一条，更早的买入不盖这句。这是研究提示，不会自动停手。持有状态默认看该代码在这份清单里最后一行的建议仓位；笔记本和脚本会用持仓快照覆盖。

浮动盈亏用信号价和最近收盘价估算，连续亏损用信号价的来回，都没有费用。持有不超过 5 个交易日就完成的来回，买卖两行都标「短持翻脸」。还没平掉的仓，浮动盈亏只出现在仍未平仓的那笔买入上。

### 先写禁投名单，再确认导出会拒绝

`orders/banned.txt` 一行一个 6 位代码。`#` 开头是注释，注释掉的代码**不会**生效。默认文件只有注释，所以日常导出能写文件。

验证拒绝时这样做：

1. 在 `orders/banned.txt` 里另起一行，写上清单中会出现的代码，例如 `000001`。不要写在 `#` 后面。
2. 再导出（笔记本里的导出格，或 `python scripts/plot_demo.py --offline`）。
3. 应当看到 `BannedSymbolError`。`output/orders.csv` 和 `output/orders.md` 都不会被这次导出创建或覆盖。脚本的退出码是 2。
4. 确认之后删掉那一行，或把 `#` 加回去，否则后面的导出会一直被拒绝。

不想改仓库里的默认文件时，可以用一段临时名单核对同一条规则：

```bash
source .venv/bin/activate
python - << 'PY'
from pathlib import Path
import pandas as pd
from orders import BannedSymbolError, export_order_list
from orders.export import load_banned

path = Path("/tmp/banned_check.txt")
path.write_text("000001\n", encoding="utf-8")
assert load_banned(path) == {"000001"}
board = pd.DataFrame({
    "symbol": ["000001"],
    "side": ["buy"],
    "date": ["2024-01-02"],
    "strategy": ["dual_ma_10_30"],
    "reason": ["示例"],
})
try:
    export_order_list(board, "/tmp/banned_check_orders", banned=load_banned(path))
except BannedSymbolError as exc:
    print("拒绝导出:", exc)
else:
    raise SystemExit("应当拒绝")
print("文件未写出:", not Path("/tmp/banned_check_orders.csv").exists())
PY
```

## 政策与事件

本地 JSON 字段：`date`、`title`、`region`、`impact`、`summary`、`sectors`、`symbols`。`symbols` 为空表示全市场。`sectors` 只是说明，没有行业成分表，不会自动展开成股票。

`EventGuard` 默认 `off`，不改仓位。`block_entries` 在事件日禁止从空仓新开；`reduce` 在事件日把目标仓位降到 0。只作用于 `impact` 达到门槛的日子，默认 `high`。图上的「政策与事件」面板不依赖这个开关。

不拉行情也能看这两种开关和默认 `off` 的差别：

```bash
python scripts/event_modes.py
```

笔记本最后一格调用同一个 `event_switch_samples()`。开仓日遇到事件时，`block_entries` 和 `reduce` 都会把当天的新开仓压住。已经持有时，`block_entries` 不砍仓，`reduce` 当天降到 0，下一天若策略仍给出持有会回来。

日历是研究用的起步名单，含一条按惯例写下的日期（见 JSON 里的说明）。发布日期若有调整，请改文件，不要当成官方日程。

## 样本

`data/universe.py` 里是静态名单，不是实时指数成分，也没有行业成分表。

| 名字 | 内容 |
| --- | --- |
| `demo` | `000001` 平安银行、`600519` 贵州茅台。离线路径用这个 |
| `banks` | 平安、招商、兴业、工商 |
| `consumer` | 茅台、五粮液、伊利、泸州老窖 |
| `extra` | `601318` 中国平安、`600900` 长江电力、`600030` 中信证券。按常见大盘成交挑选，不是打分选股 |
| `liquid` | 银行、消费，再加上 `extra`。静态名单，不是沪深 300 成分，也不是为了靠近每月 3% |
| `watch` | `159338` 中证A500ETF、`513010` 恒生科技ETF。盯盘名单，不并进上面的股票池 |

```bash
python scripts/plot_demo.py --sample banks
```

笔记本里改 `SAMPLE`。未缓存的代码第一次会访问 akshare。

## 手工成交

`journal.save_fills` / `append_fills` 把成交写成 CSV 和 JSON。每笔有：代码、方向、价格、数量（股，不是行情里的手）、成交时间、关联的信号日期。费用列是 `commission`、`stamp_tax`、`other_cost`（元）。可选 `order_id`、`account`。没有券商接口。

`save_fills(..., simulated=True)` 强制每行 `is_simulated=true`。演示文件带这个标记，不能当成实盘业绩。自己的真实成交不要传 `simulated=True`。

`apply_cost_schedule` 按研究用费率写入佣金和印花税。笔记本和脚本的默认是佣金万 2.5、单笔最低 5 元、卖出印花税万 5。这不是券商价目，改参数即可。

`compare_actual` 对照：

- 相对收盘假设、次日开盘假设的滑点。正数表示更差（买贵或卖便宜）。跟单对照锁定 next_open
- 有没有同方向的成交。给出参考资金时，还看股数是否接近 `资金 × 建议仓位 / 信号价`
- 按成交结转的逐笔盈亏，以及按收盘价盯市的逐日盈亏。买入费用进成本，卖出费用从已实现盈亏里扣，`actual_return` 是扣费后的
- 同一条权重在回测里的收盘收益和次日开盘收益
- 相对同一标的买入持有的超额。传入沪深 300 时，再给相对基准的超额。指数拉不到时，超额只和买入持有比
- `split_date`：之前标 `train`，当天及之后标 `test`。测试段实际收益从训练段结束时的权益接着算，不是从最初本金重算

`simulate_follow_fills` 只生成演示成交，价格用下一根开盘，每笔 `is_simulated=true`。换成自己的记录时直接改文件或调用 `save_fills`。卖出数量超过当时持仓会拒绝，不记成融券。

沪深 300 用 `data.get_csi300`，走指数日线（`index_zh_a_hist`，代码 `000300`），缓存为 `.cache/bars/000300_index.parquet`，不和个股 `000300` 混用。`--offline` 且缓存没有覆盖时，演示继续，只是没有指数超额。

## 第二种策略

`MeanReversion` 用收盘价相对一条更长的均线（默认 20 日）做均值回归：连续几天明显低于均线才买，回到均线附近或收盘止损才卖。建议仓位默认 50%。它不是双均线，也不把短窗改成 5 日去增加交易。`mean_reversion_sensitivity` 在次日开盘假设下扫几组窗口和偏离，只是敏感度，不是寻优，也不是每月 3% 的验收。`strategy.available_strategies()` 里能看到 `DualMovingAverage` 和 `MeanReversion`。

## 周频均值回归

`WeeklyMeanReversion` 登记在现有策略表里。它用自身收盘价的 z-score：`(收盘 − 过去 W 日均值) / (样本标准差 + 1e-8)`，默认 W=20。信号只在当周最后一个交易日收盘产生，下一交易日开盘换仓。做多 z ≤ −1 的超卖股票，最多 10 只；z ≥ −0.25，或连续持有满 8 个信号周，就调出。

约束默认单票 10%、行业 30%、总仓 80%。入选后先等权，再按 |z| 从大到小截断；总和超过总仓时，把还能交易的部分等比缩小。没有行业字段时只保留单票和总仓，并写进日志。北交所不进池。ST、停牌、上市不足 60 个交易日、信号日前 20 日日均成交额低于 2000 万、收盘价低于 2 元，都不新开仓。

规格稿写的是后复权。本仓库日线默认是前复权，周频研究沿用前复权，避免同一回测里混用两套价格。

费用沿用成交日志的研究费率（佣金万 2.5、卖出印花税万 5、最低佣金 5 元），并给出 0.5 倍、1 倍、2 倍三档。Sharpe 的无风险利率按 0 计算。训练段是 2018–2021，测试段从 2022 起；数据不够长时，脚本会按你给的区间照实报告。基准是股票池等权买入持有；本地已有沪深 300 缓存才加指数，不临时请求行情接口。

这不是每月约 3% 的验收，也不会自动下单。看板上用顶栏「频率」在日频策略和这套周频策略之间切换。

```bash
source .venv/bin/activate
python scripts/weekly_mr.py --demo
python scripts/weekly_mr.py --symbols 000001 600519
python scripts/weekly_mr.py --sample watch
```

`--sample` 用和看板相同的样本名，含 `watch`（159338、513010）。它和 `--symbols` 互斥，两个一起给会退出。两条实盘命令都只读 `.cache/bars` 里已经有的前复权文件。没有全 A、也没有从 2018 补齐的股票池时，它不会联网下载。

## 换策略

策略一次只吃一只标的的日线，返回三张表：

| 字段 | 含义 |
| --- | --- |
| `signals` | 仅在仓位变化当天有行。`side` 为 `buy` 或 `sell`，`weight` 是变化后的建议仓位 |
| `positions` | 每个交易日的目标权重，可以是 0 到 1 之间，不限于满仓 |
| `indicators` | 与行情对齐的指标。双均线提供 `ma_short`、`ma_long`，以及启用时的 `ma_trend`、`adx`。均值回归提供 `ma`、`distance` |

新策略继承 `strategy.Strategy`，实现 `run`，加上 `@register`。

```python
from strategy import Strategy, StrategyResult, register

@register
class MyStrategy(Strategy):
    name = "my_strategy"

    def run(self, bars):
        ...
        return StrategyResult(signals=..., positions=..., indicators=...)
```

`strategy.available_strategies()` 返回已登记的类名。

## 目录

```
data/         拉数、缓存、沪深 300、事件日历、静态样本
a-share-features/  旁路特征工程。默认 qfq / watch；不是盯盘金标准
strategy/     策略接口、双均线、均值回归、诊断、信号表、事件约束
viz/          K 线、均线、买卖点、短持翻脸、政策与事件（plotly）
backtest/     收益、回撤、换手；收盘、次日开盘、次日开盘止损
orders/       CSV / Markdown 清单和禁投名单。没有下单函数
journal/      手工成交日志，以及和信号、回测假设的对照
notebooks/    日常入口 daily_entry.ipynb
dashboard/    日常看板 app.py。streamlit run，端口 8766
scripts/      plot_demo.py、event_modes.py、refresh_trade_calendar.py、qc_bars_cache.py
tests/        不联网的单元测试
```

## 已知限制

开发时在 Python 3.12 上解析到的版本是 akshare 1.20.35、pandas 3.1.1、plotly 7.1.0。`requirements.txt` 只约束下限，以后安装到的版本可能不同。阶段 2 的默认行为和阶段 1 的裸 0/1 交叉不同，不要拿旧的信号条数来对。

- 行情来自 akshare。优先东方财富 `stock_zh_a_hist`，失败时改走腾讯 `stock_zh_a_hist_tx`。东财接口偶尔限流或改版，必要时 `pip install -U akshare` 后再试。
- 东财日线的成交量单位是**手**，成交额是元，涨跌幅和换手率是百分数。腾讯兜底不写入换手率。图上的成交量轴不标注单位。
- 默认前复权。前复权历史价会在除权后变化；长期分红的股票甚至可能出现负的前复权价。缓存因此可能过期，用 `--refresh`。回测要求价格为正，负的前复权价会被拒绝。
- 回测假设没有滑点、费用、涨跌停和停牌。手工成交的 `actual_return` 会扣记录里的佣金、印花税和其他费用；演示脚本里的费率只是研究假设。回测收益和成交盈亏不是同一套账。
- 沪深 300 需要至少一次成功的指数拉取才会进缓存。失败或离线未命中时，超额只和同一标的的买入持有比。买入持有用区间首尾收盘价，和回测、和按股数结转的实际收益都不是同一套账。
- `train` / `test` 只是按一个日期切开，不是滚动样本外，也没有自动调参。测试段收益从训练段结束权益接着算。
- 止损和止盈看收盘价相对入场收盘价，不是盘中触发。盘中止损没有实现。`next_open_stop` 只是把止损的离场假设改成下一根开盘。
- 双均线从空仓开始。区间一开始短均线已经在上、但确认条件还没满足时，目标仓位仍是 0。
- 横轴去掉了周末。法定节假日仍会留空。落在周末的事件留在表里，只有当天有 K 线才在图上打点。
-  bars QC 使用的上证交易日历来自新浪接口落地缓存；相对日历的「缺日」**警告**常见于正常停牌，不等价于数据源一定错误。日历与沪深300指数日期的差异只做对照提示。
- 事件日历是手工起步名单，不是完整宏观数据库。2026-03-05 按两会惯例写入，会期若变动请自行修改。
- 样本是静态名单，按常见大盘成交挑选，不是指数调样，不是打分选股，也没有行业成分映射。没有分钟线。没有券商接口，成交要自己抄进来。演示成交带 `is_simulated=true`，不是实盘业绩。
- 大约每月 3% 仍只是方向。示例双均线和均值回归的历史结果远没有达到它，不能把示例跑通当成这个目标已经验收。
- 港股、美股未实现，只在 `data/bars.py` 里留了注释。
