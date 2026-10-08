# OKX 永续数据来源核验

核验日期：2026-10-08。范围为 BTC-USDT-SWAP，2024-01-01 至 2026-10-01 UTC（右边界不含）。已完成全区间采集及结构检查，正式评估仍等待时序、切分和基线验收。

## 已核验来源

当前公开合约元数据返回线性合约、USDT 结算，`ctVal=0.01`、`ctMult=1`、`ctValCcy=BTC`。这是一份当前快照，不能自动代替历史合约规格。

K线使用官方 `history-candles`，请求 `bar=1H`。输入成交量取 `volCcy`（BTC），成交额取 `volCcyQuote`（USDT），另存 `vol`（合约张数）。只接受 `confirm=1`。已取得2024年边界的旧K线和近期K线样本。

资金费率 REST 历史接口提供最近约三个月；一年以前的探测返回空。较长历史使用官方 `market-data-history` 的月度资金费率文件。2024年第一季度和2026年第三季度均返回目录；已读取2024年1月及2026年9月小文件。

归档 CSV 字段为 `instrument_name / funding_rate / funding_time`。归档按 UTC+8 月份分组，事件时间统一转 UTC；末月尾部需与近期 REST 衔接。REST 结算记账取 `realizedRate`，归档同名含义必须通过重叠数据核对，不能把预测费率当结算值。

## 全区间采集结果

不可变快照 `20261008T011408835523Z_7bb06df6f064` 位于 `research/data/snapshots/`，摘要见 `data_snapshot_report.json`。交易 K 线 24,096 根，另有 256 根预热（2023-12-21 08:00 UTC 起）；原始标记价 K 线 24,096 根，两者缺口均为 0。

资金事件 3,012 个，从 2024-01-01 00:00 至 2026-09-30 16:00 UTC，观测到全部相邻间隔为 8h。所有月度档案齐全，最后 UTC+8 部分月份用 REST 桥接；280 个重叠事件实际费率一致，全部 3,012 个事件一对一匹配原始标记价。未补零、未替换成交价。所有非零交易 bar 的 volCcy/vol 均约为 0.01 BTC/合约，最大浮点偏差约 1.73e-18。

Freqtrade 文档提示 OKX 标记价仅约三个月；本次实际 API 可获得整个目标区间。以保存的全量响应和缺口检查为本次证据，不能将文档限制直接视作当前实际覆盖。真实成交额在原始 candles.csv 保留，因为 Freqtrade 原生 OHLCV 存储不会保存 amount。

## 尚需验收

归档事件与 8h 网格一致，不独立证明交易所历史结算时间表或历史合约公告。历史文件缺少事前发布日志；60秒可用延迟是研究假设，不能据此声称真实 point-in-time。后续还需 Freqtrade 原生格式读回、边界与标签核验、切分以及统一基线结果。

## 独立完整性验收

独立审计从全部保存的原始 JSON/ZIP 重构数据，再逐字段核对 CSV。33 个档案各自恰好覆盖 UTC+8 月首到下月月首的全部 8h 事件；研究区间内档案覆盖 3,011 次，REST 单独补齐 2026-09-30 16:00 UTC 的末端 1 次。文件清单与磁盘实际集合相符，全部记录哈希一致。

另从官方接口重新取得每月固定抽样及区间端点：72 条交易/标记价、预热首条、2024-01/2025-05/2026-09 三份资金档案和 281 个近期资金事件，内容均与保存快照一致。Freqtrade 三个原生文件哈希仍与读回验收时相同。证据见 data_integrity_audit.json 和 data_remote_recheck.json。

据此接受“已公布官方来源的数据覆盖及字段一致性”通过。旧快照里的 historical_schedule_completeness_proven=false 仍保留，其含义是缺少独立交易所内部时间表证据，不再要求用户替数据完整性作确认。正式评估仍由研究口径、标签、切分和基线验收控制。

采集时配置和版本清单只保存哈希，随后正常更新；未保存当时文件全文，因此无法仅凭快照还原其全文。数据范围、实际请求参数和原始响应均有完整记录；原快照未改写。

原始响应、小文件和校验记录在忽略 Git 的 `research/data/source_inventory/`。正式数据采用独立快照，不修改这些探测样本。

## 官方资料

- [OKX K线接口](https://app.okx.com/docs-v5/en/#order-book-trading-market-data-get-candlesticks-history)
- [OKX 资金费率历史接口](https://app.okx.com/docs-v5/en/#public-data-rest-api-get-funding-rate-history)
- [OKX 历史市场数据接口](https://www.okx.com/docs-v5/en/#public-data-rest-api-get-historical-market-data)
- [OKX 历史下载入口](https://www.okx.com/en-us/historical-data)
