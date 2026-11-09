# journal

手工成交日志，以及和信号、两种回测假设的对照。

`save_fills` / `append_fills` 把成交写成 CSV 和 JSON。每笔至少有：代码、方向、价格、数量（股）、成交时间、关联的信号日期。费用列是 `commission`、`stamp_tax`、`other_cost`（元，缺省 0）。`apply_cost_schedule` 按研究用费率写入佣金和印花税，不是券商价目。可选列还有 `order_id` 和 `account`。

`compare_actual` 计算：

- 相对收盘价假设、次日开盘价假设的滑点。正数表示更差（买贵或卖便宜）。
- 信号有没有被同方向成交跟上；给出参考资金时，还比较目标股数。
- 按成交结转的逐笔盈亏，以及按收盘价盯市的逐日盈亏。买入费用进成本，卖出费用从已实现盈亏里扣，`actual_return` 是扣费后的。
- 同一条目标权重在回测里的收盘成交收益和次日开盘成交收益。跟单对照锁定 next_open。
- 相对同一标的买入持有的超额。传入沪深 300 日线时，再给相对基准的超额。
- `split_date` 之前标 `train`，当天及之后标 `test`。测试段实际收益从训练段结束时的权益接着算。

`simulate_follow_fills` 只生成演示数据，每笔 `is_simulated=true`。`save_fills(..., simulated=True)` 会再次强制这个标记，避免演示文件被看成实盘业绩。

本包没有下单函数。
