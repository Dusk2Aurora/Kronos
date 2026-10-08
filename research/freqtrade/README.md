# Freqtrade 研究适配

官方仓库位于 D:\file\freqtrade，stable 2026.9，commit 1f394eaebc2f46a83d26971388628707802f8602。使用该目录独立 .venv；Kronos 环境负责数据与后续特征研究。回测成交、持仓、资金费用和净收益交给 Freqtrade。这里保存配置及策略，运行副本放在 Freqtrade user_data/kronos_research，不修改框架核心。

已通过 pip check、freqtrade --version、backtesting --help、环境隔离和 TA-Lib 计算检查。依赖锁及环境记录位于本目录 environment。复现先检出固定 commit，再安装依赖锁，最后 pip install --no-deps -e D:\file\freqtrade。基础 Python 来自 E:\conda\envs\kronos，该目录必须保留。

首轮：OKX isolated futures、BTC/USDT:USDT、1h、1 倍杠杆、4h 持有，每 8h 一次机会。用户授权代理选择 UTC 04/12/20（北京时间 12/20/次日04）起点：例如策略在04:00行只使用已完成256根K线，04:01决策后于05:00开盘入场，09:00退出，跨过08:00资金结算；用于回测研究。

config.base.json 的 fee=0.0007 是每边费用代理：5 bps 手续费加 2 bps 滑点假设；config.stress.json 覆盖为 0.0014。费用代理不模拟价格滑点、冲击或订单簿。资金费用另按实际结算率和标记价计算，不设置 futures_funding_rate 缺失兜底。

接入必须满足以下条件：

- 交易K线连续、完成，保留 volCcy（BTC）、vol（合约张数）和真实 volCcyQuote（USDT）。Freqtrade OHLCV 仅保存 date/OHLCV；真实 amount 与原始字段保留快照旁表，特征按 UTC 时间戳合并。
- 每个资金事件有唯一原始标记价。框架 inner join 会丢弃无法匹配的事件；原始预检查须拒绝缺口，不能允许框架补成可评估数据。
- 框架资金计算包含开仓和闭仓时刻的结算事件，标签沿用。正费率多头支付、空头收取，不另写独立资金账本。
- 右端排除 2026-10-01 00:00 UTC。框架裁剪包含 stopdt，导入和交易验收明确左闭右开，预热或区间外交易不计入研究样本。
- 当前合约单位不证明历史相同；档案、间隔审计和 REST 重叠匹配不独立证明资金结算时间表完整性，须保留证据边界。
- 应急止损或强平可能提前退出；验收需核实 exit_reason 与4h持有，不将此类交易混作严格4h标签。

同步副本后在 Freqtrade 目录检查配置和策略（不执行回测）：

```powershell
.\.venv\Scripts\python.exe -m freqtrade show-config --config user_data/kronos_research/config.base.json
.\.venv\Scripts\python.exe -m freqtrade list-strategies --strategy-path user_data/kronos_research/strategies --no-color
```

数据与时序验收通过后才运行基线，用 --cache none --export trades 保存结果，压力测试追加第二份 config。冻结表征由 M0 ready_to_encode 阻塞。

2026-10-08 已完成全快照导入和原生读回核验；新04/12/20工程回测21笔严格4h，每笔跨一次结算，资金费用逐笔与原始费率、标记价及BTC仓位精确匹配（8笔收取、13笔支付）。起点是按是否跨资金结算选择，没有用收益比较择优。原00/08/16工程报告另存为freqtrade_smoke_report_original_phase.json。原始快照未重采集。用于工程验证的短区间延伸到2024-01-08 02:00 UTC，让最后一笔于01:00完成4h持有；正式标签仍须剔除跨研究/切分右边界的交易，禁止将强制提前平仓视为4h样本。

本机网络通过环境代理访问；ccxt_async_config 的 aiohttp_trust_env=true 让异步市场元数据请求使用相同环境。首次启动还会获取并缓存 OKX 杠杆梯度。输出目录必须预先创建，否则本版本可能将路径按文件名处理。

完整标签适配器 build_reference_labels.py 消费两份官方导出（7/14 bps）并生成机会、事后标签及切分索引。config.reference.json 固定1000USDT参考仓位和1000万USDT虚拟余额，只用于独立参考标签，不用于投资组合业绩。完整导出范围20240101-20260930T2000，排除最后一次跨研究右边界的入场。

已验收3011条标签：严格4h、官方开盘价、1倍杠杆、实际资金费逐笔匹配，两种成本下仓位相同且压力口径净收益不高于基础口径。每折train/validation/test根据labelable_at（退出+60s假设）做purge；FINAL训练2192条、验证269条，2026-04至09最终holdout548条。完整来源配置和导出ZIP保存在标签bundle/provenance。

标签和funding/profit字段只保存在禁止入特征的侧表。最终holdout不参与E00是否值得加入冻结表征的晋级决定，开发WF折用于该决定；holdout留待所有对照方案锁定后统一评估。

2026-10-08：E00已完成4折×6基线×7/14bps共48份官方组合导出，每折10000USDT、1倍杠杆。普通13项特征、36次logistic训练使用训练折标准化/拟合、验证logloss选择C；阈值固定0.5，成本压力复用信号。KronosE00.py严格读取不带收益的decision sidecar，约束动量方向、仓位比例、原生信号移位和退出。买入持有独立使用首个允许入场至末日23点，属于长期市场对照。正式交易与原始价格/资金逐笔核验通过；看[E00结果](../initialization/E00_results.md)。

新增复现入口（需要新输出目录，常规继续不要重跑）：

```powershell
& '.\.venv\Scripts\python.exe' research\baselines\prepare_e00.py --output-dir research\runs\E00_new_id
& '.\.venv\Scripts\python.exe' research\baselines\run_e00.py --output-dir research\runs\E00_new_id
& '.\.venv\Scripts\python.exe' research\freqtrade\check_e00_causality.py --help
```

prepare/run使用Kronos解释器，run通过子进程调用固定的Freqtrade解释器；不安装额外跨环境依赖。初次验收曾将realized_rate误写成funding_rate，已保留失败记录；--audit-existing只重验现有导出，不重复回测。官方lookahead/recursive已在两个开发模式验收，范围与工具参数覆盖限制见[e00_causality_report.json](../initialization/e00_causality_report.json)。M0已验收ready_to_encode，当前仍禁止冻结表征提取，最终holdout未评估。
