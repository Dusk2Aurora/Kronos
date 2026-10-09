# 第一项冻结表征实验：完成与验收清单

当前 `stage: frozen_development`，`status: first_frozen_experiment_completed_holdout_sealed`。路线图第5章／M1内的有限研究与完整综合结论已完成，**没有待运行的正式实验，当前等待用户验收**。停止当前BTC单资产、1h历史、4h固定动量参与任务上的预测头扩容和冻结表征路线扩展；这不是环境阻断，也不是证明Kronos在所有任务上无信息。

状态以[配置](../configs/initial_experiment.yaml)为准，结论与证据见[完整开发期结论](FIRST_EXPERIMENT_CONCLUSION.md)、[综合manifest与hash](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/manifest.json)、[最终验收与文档hash](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/final_acceptance.json)、[全部筛选JSON](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/all_screens.json)。综合报告仅修正文案中的资金符号译法和二值零规则；原版本保留，指标与结论不变。会话goal状态由会话记录管理；本清单记录已完成研究范围。[初始化TODO](../initialization/TODO.md)仅保留M0／E00／E00R历史，不继续追加。

## 完整有限范围

- [x] M0与普通基线前置验收：[M0报告](../initialization/M0_acceptance_report.json)、[E00R结果](../initialization/E00R_results.md)。
- [x] 首轮同Ridge预训练／三个随机骨干比较 `FROZEN_20261008_v1`：64次新拟合、128份新官方导出；[协议](../configs/frozen_comparison_v1.yaml)、[登记](../registry/FROZEN_20261008_v1.json)、[结果](../initialization/FROZEN_results.md)。
- [x] 有限小MLP读出 `FROZEN_MLP_20261008_v2`：180次新拟合、20个三头集成、160份新官方导出；[协议](../configs/frozen_mlp_v2.yaml)、[登记](../registry/FROZEN_MLP_20261008_v2.json)、[结果](../runs/FROZEN_MLP_20261008_v2/results.md)。
- [x] 同坐标量化前后诊断 `QUANT_20261008_v1`：普通19项加u20／q20均为39维，32次新拟合、8个选中头、64份新官方导出；[协议](../configs/frozen_quant_v1.yaml)、[登记](../registry/QUANT_20261008_v1.json)、[结果](../runs/QUANT_20261008_v1/results.md)。
- [x] 各轮工程编码／缓存checksum、训练折预处理、验证选择、模型与参数独立复算、sidecar动作和7／14bps官方回放核验完成；全部候选、种子、零交易和失败证据保留。
- [x] 损失与泛化诊断完成：[MLP损失诊断](../runs/FROZEN_MLP_20261008_v2/mlp_loss_diagnostics.json)、[MLP训练／验证／开发测试图](../runs/FROZEN_MLP_20261008_v2/mlp_rmse_by_fold.png)、[原Ridge损失诊断](../runs/ridge_loss_review_20261008_v1/README.md)。
- [x] 配对MSE、匹配参与数、完整UTC日历效用、3／7／14天块区间、成本压力、集中度与换手底表完成。
- [x] 真实时间季度曲线、PNG／SVG、CSV、来源hash、端点和视觉核验完成；三轮合计 **276次新增正式拟合、352份新增官方导出**，加原E00／E00R共 **512条官方曲线**。
- [x] 汇总预测、经济、拟合能力与量化证据，形成[完整结论](FIRST_EXPERIMENT_CONCLUSION.md)和停止／后续建议；各轮完成前后配置与TODO快照已保留于对应运行provenance。

## 当前可支持结论

首轮Ridge、有限MLP和两个量化变体的**信息筛选与两类经济筛选均未通过**，同坐标连续u相对二值q的**量化可读性筛选未通过**。工程完成不代表研究晋级；当前证据未建立稳定预训练信息优势、优于现金的盈利价值或量化瓶颈。

保留量化主rank50的局部正向信号：两个变体相对普通19项在至少3/4折少亏，匹配UTC入场月份×固定方向数量后的中位筛选增量及正向折数检查通过。它们未同时获得预测改善、关键配对区间及现金／压力盈利证据，不能把少亏、低暴露或匹配诊断解释成盈利或已证明的量化瓶颈。逐项结果以[量化分析](../runs/QUANT_20261008_v1/analysis_report.json)为准。

MLP能强烈拟合训练样本，但没有开发预测增量；不同时期的损失差不能唯一归因于过拟合／欠拟合。u/q比较只针对同20坐标、当前任务与Ridge预算的可读性，不代表一般信息损失。原512维骨干加19项为531维，与39维量化诊断只作维度不同的历史参照。

## 证据、图表与底表

| 运行 | 工程／回放验收 | 分析与图表 |
| --- | --- | --- |
| 首轮Ridge | [编码](../runs/FROZEN_20261008_v1/encoding_acceptance.json)、[缓存独立复算](../runs/FROZEN_20261008_v1/root_cache_recompute_review.json)、[头独立复算](../runs/FROZEN_20261008_v1/root_head_review.json)、[官方回放](../runs/FROZEN_20261008_v1/engine_report.json) | [分析](../runs/FROZEN_20261008_v1/analysis_report.json)、[图表hash／端点](../runs/FROZEN_20261008_v1/presentation/report.json)、[主rank50图](../runs/FROZEN_20261008_v1/presentation/rank50_equity_base.png) |
| 有限MLP | [头重载](../runs/FROZEN_MLP_20261008_v2/head_acceptance.json)、[官方回放](../runs/FROZEN_MLP_20261008_v2/engine_report.json) | [分析](../runs/FROZEN_MLP_20261008_v2/analysis_report.json)、[图表hash／端点](../runs/FROZEN_MLP_20261008_v2/presentation/report.json)、[主rank50图](../runs/FROZEN_MLP_20261008_v2/presentation/rank50_equity_base.png) |
| 同坐标量化 | [编码](../runs/QUANT_20261008_v1/encoding_acceptance.json)、[39维输入](../runs/QUANT_20261008_v1/encoding_head_input_audit.json)、[32头正规方程／预测复算](../runs/QUANT_20261008_v1/head_acceptance.json)、[官方回放](../runs/QUANT_20261008_v1/engine_report.json) | [分析](../runs/QUANT_20261008_v1/analysis_report.json)、[图表hash／端点](../runs/QUANT_20261008_v1/presentation/report.json)、[主rank50图](../runs/QUANT_20261008_v1/presentation/rank50_equity_base.png)、[经济门控图](../runs/QUANT_20261008_v1/presentation/economic_equity_base.png) |

完整[512条官方曲线CSV](../runs/QUANT_20261008_v1/presentation/official_curves.csv)与[综合季度预测损失图](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/prediction_loss_by_quarter.png)、[预测损失CSV](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/prediction_loss_by_quarter.csv)可查。量化的[配对预测](../runs/QUANT_20261008_v1/paired_prediction_metrics.csv)、[配对组合](../runs/QUANT_20261008_v1/paired_portfolio_metrics.csv)、[匹配随机](../runs/QUANT_20261008_v1/matched_random_summary.csv)、[时间块](../runs/QUANT_20261008_v1/paired_block_bootstrap.csv)保留原数值；各运行presentation保留两成本、四门控、全部随机骨干及历史基线。

## 历史失败与元数据说明

- MLP v1在正式拟合前的历史行索引整数契约预检失败，**正式拟合为0**：[失败登记](../registry/FROZEN_MLP_20261008_v1.json)。v2只修正CSV解析，原候选、种子、轮次与180次正式预算不变，原工件和源码快照保留。
- MLP图表的历史配置布局预检曾失败；修正为读取E00R／FROZEN的parent_configuration后生成，未新增拟合或回放。[预检登记快照](../runs/FROZEN_MLP_20261008_v2/provenance/failed_preflight_registry.json)及图表provenance保留。
- 首轮原文字节／行尾hash修复历史保留在原登记与provenance，不覆盖旧快照。
- Bootstrap沿用共用工作流：基础seed17＋从0开始的折序号×100，四折独立重采样后等权；主7天、3／14天敏感性、5,000次抽样不变。执行后的[种子说明](../runs/QUANT_20261008_v1/bootstrap_seed_clarification.json)补齐原分析计划未写明的派生规则，独立复核确认仅增加说明字段，计算完全相同；未改锁定协议hash或重跑。区间只覆盖独立参考效用和预测损失，不覆盖复合组合收益。

## 持续边界与交接

最终holdout `[2026-04-01, 2026-10-01)` 继续封存，不编码、不评估标签／分类比例／业绩，不用于加入Kronos、晋级或调参。骨干／tokenizer训练、微调／解冻、原生未来路径生成、git push、部署与实盘仍禁用。

开发季度此前已见且checkpoint训练截止未证明，结果属于探索性历史研究，不能宣称严格事前样本外。随机骨干保留预训练tokenizer，只隔离骨干预训练贡献。真实OHLCVA、官方净收益、带符号的资金费用、固定动量方向／仓位／时序与purge规则保持配置约束；完整性证据限于已公布官方数据覆盖与一致性。

未来重新探索须另行确定明确问题、独立数据和有限预算并登记，当前不自动启动新实验。复现入口见[冻结README](README.md)，命令用于重建原设计，不是每次工作的默认动作；本机ignored数据、模型、ZIP及缓存不能随clone取得，需原manifest与已保存来源字节。
