# Kronos 研究初始化清单

M0 工程准备已验收，当前 `ready_to_encode=true`。冻结 tokenizer／骨干的推理、表征提取及训练仍禁用，最终 holdout 封存。阶段以 [配置](../configs/initial_experiment.yaml) 为准；目录入口见 [研究 README](../README.md)。

## 当前待办

- [ ] **冻结表征前：锁定共用协议。** 固定 E00R／E01R／E02R 相同目标、19项普通特征、Ridge 预测头及选择预算；保留13项消融、旧 logistic、固定规则与无信息参与对照。普通基线不要求先盈利，不把少亏或减少暴露当作信息增量。
- [ ] **协议锁定后：明确下一阶段范围。** 按路线图第5章准备历史隐藏表征，使用 `decode_s1()` 的最后有效历史 context；原生多路径生成属于另行登记的研究线。当前尚未开始冻结提取。

## 已验收步骤与证据

| 步骤 | 状态及主要证据 |
| --- | --- |
| 0：隔离环境 | 已完成。[研究环境](../environment/README.md)、[验证](../environment/verification.json)、[Freqtrade 环境](../freqtrade/environment/verification.json)。原上游4项模型回归通过。 |
| 1：研究范围 | 已完成。OKX BTC-USDT 永续，2024-01-01 至 2026-10-01 UTC 左闭右开；1h、256根历史、4h持有，每8h一次机会。UTC 04/12/20起点由用户授权代理选择；每边7/14bps费用代理，实际资金另计。详见配置。 |
| 2：版本与登记 | 已完成。[版本 manifest](version_manifest.json)、[登记模板](../registry/experiment_template.yaml)。Kronos-small／tokenizer revision与hash固定，Freqtrade stable 2026.9固定commit。 |
| 3–4：来源、覆盖及快照 | 已完成。[数据摘要](data_snapshot_report.json)、[独立全量审计](data_integrity_audit.json)、[官方远程复取](data_remote_recheck.json)。24,096研究小时＋256预热、3,012资金事件、真实成交额及原始来源完整。验收限于已公布官方数据的一致性与覆盖。 |
| 5：Freqtrade、时序和标签 | 已完成。[原生导入](freqtrade_import_manifest.json)、[工程回测](freqtrade_smoke_report.json)、[标签](reference_labels_report.json)。3,011条官方净收益标签均严格4h，原价格及资金逐笔核验；跨边界和标签可用时间执行purge。 |
| 6：切分与原晋级条件 | 已完成。[原 E00 协议](../baselines/E00_protocol.md)。开发测试为2025Q2/Q3/Q4、2026Q1；每折以前季度验证、更早历史训练。原经济筛选已锁定，但现金也可通过相对亏损动量的门槛，不能据此证明信息价值。 |
| 7：原 E00 | 已完成。[E00结果](E00_results.md)、[报告](e00_report.json)。六策略×四折×两成本共48份官方导出；普通13特征logistic参与8/1091次，未建立优于现金的信息价值。 |
| 8：必要因果复现 | 已完成。[因果报告](e00_causality_report.json)。未来扰动、窗口／缺口拒绝、训练隔离、官方lookahead及recursive检查通过；工具覆盖限制保留。 |
| 9：M0验收 | 已完成。[验收报告](M0_acceptance_report.json)、[阶段转换](M0_state_transition.json)。12组工程证据通过，止于冻结表征前。 |
| E00稀疏诊断与设计复核 | 已完成。[稀疏诊断](E00_sparse_gate_diagnosis.md)、[基线研究](baseline_revision_research.md)。8次放行与成交一致，没有引擎丢信号；普通分数尚无稳定筛选优势。历史研究建议的状态以当前登记为准。 |
| E00R登记、实现与开发评估 | 已完成。[独立配置](../configs/baseline_revision_v2.yaml)、[登记](../registry/E00R_20261008_phase4_v1.json)、[结果](E00R_results.md)、[基线角色](../baselines/E00_baseline_roles.md)。32次Ridge拟合、112份官方导出；19项主信息门控507笔、经济门控94笔，尚无稳定信息优势或优于现金的盈利证据。已见开发结果属于探索修订。 |

## 数据和评估边界

- 标签 bundle：`research/data/labels/reference_labels_20261008_phase4`；不可变快照：`20261008T011408835523Z_7bb06df6f064`。标签、未来资金及收益字段禁止入特征。
- 参考标签用固定1000USDT名义仓位及大额虚拟余额，不能当组合业绩。正式开发组合每折10000USDT独立重置。
- 最终2026-04至09 holdout含548条有效机会，未查看业绩或分类比例；不用于选模型、阈值或是否加入Kronos。
- checkpoint训练截止尚未证明，早期折属于历史研究，不宣称严格事前样本外。
- `7/14bps`是每边费用代理，并非成交价格滑点模拟。回测、成交、持仓与资金记账使用官方Freqtrade；[适配说明](../freqtrade/README.md)。

## 当前图表

用户偏好保留多项基线，优先按真实时间绘制折线图，已记录于 [AGENTS.md](../../AGENTS.md)。当前本机图表在 `research/runs/baseline_time_series_20261008_v3/`：

- [固定规则基础成本](../runs/baseline_time_series_20261008_v3/rule_baselines_equity_base.png)、[压力成本](../runs/baseline_time_series_20261008_v3/rule_baselines_equity_stress.png)。
- [普通模型及无信息对照基础成本](../runs/baseline_time_series_20261008_v3/model_baselines_equity_base.png)、[压力成本](../runs/baseline_time_series_20261008_v3/model_baselines_equity_stress.png)。
- [图表核验报告](../runs/baseline_time_series_20261008_v3/report.json)：160条官方闭仓权益曲线＋16条派生均值，配SVG和CSV底表；各季度独立重置，不代表小时盯市。未新增训练／回测或读取holdout。

旧绘图v1／v2已压缩归档到本机 `research/runs/archive/`。历史审计、旧基线与失败证据保留；依赖缓存清理不改变已验收结论。
