# M2-CI 未来独立验证协议草案

**DRAFT / UNAPPROVED / DO NOT EXECUTE。** 这不是正式封存协议，也不授权第四次正式测试、采集服务、部署或实盘。生成此草案不会改变现有CONSUMED终态。

## 候选问题与进入条件

开发状态INCONCLUSIVE：暂停当前 Kronos 条件融合的复杂度扩展，优先准备固定 B5 的独立复现；当前未达到实际增量门槛。 H1只研究既有R2正值RV输出给定B5的互补性，当前锁定开发alpha=0.1。若不满足开发晋级，不把H1安排为默认确认路线；H3优先候选是固定B5与固定R2的独立预测复现。不能通过扩大latent/Adapter/网络寻找新成功路径。

同构廉价对照固定B5、B5+R1/B2/R2，HAR/EWMA/persistence及FIT常数分母诊断保留；融合6权重的开发选择已经结束，未来不调alpha、阈值、校准或模型。未来候选可使用本轮原VALID推理所用的原完整TRAIN冻结模型及预处理，模型路径见validation_audit.json；各未来实际工件SHA必须在后续正式seal明确登记，不能只引用本草案。OOF各折模型用于历史诊断，不自动替代未来固定模型。

## 独立区间与数据契约

正式开始前需用户另行明确授权，并封存完整协议、模型、code、config、候选集、依赖及来源SHA和预运行审计。首个决策机会必须严格晚于**后续正式协议实际封存UTC时间**。具体日期现在不填。已观察2026Q2/Q3及开发TRAIN/VALID均禁止重用为确认holdout。若历史256根小时窗口的事前可用性证据充分，可作为未来输入；否则需先满足256完成小时的暖启动及可用时间。采集与固定前瞻预测系统仅可在后续授权后建设，记录预测时的input/source/model/clock哈希，标签可用后另表追加。

保持BTC-USDT-SWAP、UTC04/12/20、+60秒可用延迟、256×1hOHLCVA、4h RV49价格、标签floor1e-12与prediction[1e-12,1]。每个机会共享同ID/期限/标签；官方真实amount/raw字段/单位/URL/采集UTC/SHA和完整grid审计保留。缺数、未完成、无法验证时钟的机会不能兜底；缺失报告和处理规则必须预登记，不能事后按表现排除。

## 效应、样本量与功效：候选，尚待审批

H1候选主要门槛：相对B5平均QLIKE Regret降低≥5%，paired rawQLIKE Δ>0且主CI下限>0，R2mix相对R1mix/B2mix各pairedCI下限>0；时间子段方向一致性阈值及多重检验家族必须在正式seal明确批准。raw损失百分比不使用，raw/Regret配对差不是两份独立证据。

H3候选主要比较为L(R2)−L(B5)，正值支持B5；实际重要性门槛要结合开发变异、资源和未来基线审批，不把事后Q2/Q3效应倒选为门槛。下面统一的δ=0.0168673只是5%×开发B5Regret的规划量级，不等于未来H3相对R2的5%门槛。

融合规划使用所选 alpha 的开发期配对块方差，受同一 OOF 选择影响，只是条件性情景。

H1条件规划：

| months | expected_opportunities_approx | planning_SE_7day | MDE_80power_two_sided95 | power_at_5pct_base_Regret_approx | planning_status | future_stationarity_unproven |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 273.938 | 0.00381531 | 0.0106889 | 0.993073 | CONDITIONAL_SCENARIO | 是 |
| 6 | 547.875 | 0.00269783 | 0.0075582 | 0.999991 | CONDITIONAL_SCENARIO | 是 |
| 9 | 821.812 | 0.00220277 | 0.00617124 | 1 | CONDITIONAL_SCENARIO | 是 |
| 12 | 1095.75 | 0.00190765 | 0.00534445 | 1 | CONDITIONAL_SCENARIO | 是 |
| 18 | 1643.62 | 0.00155759 | 0.00436373 | 1 | CONDITIONAL_SCENARIO | 是 |
| 24 | 2191.5 | 0.00134891 | 0.0037791 | 1 | CONDITIONAL_SCENARIO | 是 |

假设真实增量达到5%时，融合情景的网格最短月数为3；这不表示实测约0.385%的微小增量用3个月就能确认。事后观测效应的额外诊断估得融合名义样本约18,585机会/204个月；效应经过OOF选择、区间跨零，平方根外推也未证明平稳，这不是等待建议或正式样本承诺。当前优先B5独立复现，不为追逐微小融合增量持续等待或扩大搜索。

事后观测效应的诊断补充，不属于正式门槛：

| contrast | months | N_reference | observed_point_gain | paired_bootstrap_SE_7day | expected_opportunities_approx | planning_SE_7day | power_at_observed_signed_effect_two_sided95 | nominal_N80_at_observed_effect | nominal_months80_at_observed_effect | planning_status | positive_contrast_means | selection_conditioned_optimistic | future_stationarity_unproven |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B5_minus_mix_R2 | 3 | 1267 | 0.00129769 | 0.00177405 | 273.938 | 0.00381531 | 0.0633553 | 18585.5 | 203.537 | POST_HOC_OBSERVED_EFFECT_SCENARIO | R2_mix_lower_loss_than_B5 | 是 | 是 |
| B5_minus_mix_R2 | 6 | 1267 | 0.00129769 | 0.00177405 | 547.875 | 0.00269783 | 0.0769006 | 18585.5 | 203.537 | POST_HOC_OBSERVED_EFFECT_SCENARIO | R2_mix_lower_loss_than_B5 | 是 | 是 |
| B5_minus_mix_R2 | 9 | 1267 | 0.00129769 | 0.00177405 | 821.812 | 0.00220277 | 0.090612 | 18585.5 | 203.537 | POST_HOC_OBSERVED_EFFECT_SCENARIO | R2_mix_lower_loss_than_B5 | 是 | 是 |
| B5_minus_mix_R2 | 12 | 1267 | 0.00129769 | 0.00177405 | 1095.75 | 0.00190765 | 0.104467 | 18585.5 | 203.537 | POST_HOC_OBSERVED_EFFECT_SCENARIO | R2_mix_lower_loss_than_B5 | 是 | 是 |
| B5_minus_mix_R2 | 18 | 1267 | 0.00129769 | 0.00177405 | 1643.62 | 0.00155759 | 0.132519 | 18585.5 | 203.537 | POST_HOC_OBSERVED_EFFECT_SCENARIO | R2_mix_lower_loss_than_B5 | 是 | 是 |
| B5_minus_mix_R2 | 24 | 1267 | 0.00129769 | 0.00177405 | 2191.5 | 0.00134891 | 0.160894 | 18585.5 | 203.537 | POST_HOC_OBSERVED_EFFECT_SCENARIO | R2_mix_lower_loss_than_B5 | 是 | 是 |
| R2_minus_B5 | 3 | 1267 | 0.104547 | 0.0189209 | 273.938 | 0.0406916 | 0.72884 | 325.718 | 3.56707 | POST_HOC_OBSERVED_EFFECT_SCENARIO | direct_R2_higher_loss_than_B5 | 是 | 是 |
| R2_minus_B5 | 6 | 1267 | 0.104547 | 0.0189209 | 547.875 | 0.0287733 | 0.952888 | 325.718 | 3.56707 | POST_HOC_OBSERVED_EFFECT_SCENARIO | direct_R2_higher_loss_than_B5 | 是 | 是 |
| R2_minus_B5 | 9 | 1267 | 0.104547 | 0.0189209 | 821.812 | 0.0234933 | 0.993615 | 325.718 | 3.56707 | POST_HOC_OBSERVED_EFFECT_SCENARIO | direct_R2_higher_loss_than_B5 | 是 | 是 |
| R2_minus_B5 | 12 | 1267 | 0.104547 | 0.0189209 | 1095.75 | 0.0203458 | 0.99926 | 325.718 | 3.56707 | POST_HOC_OBSERVED_EFFECT_SCENARIO | direct_R2_higher_loss_than_B5 | 是 | 是 |
| R2_minus_B5 | 18 | 1267 | 0.104547 | 0.0189209 | 1643.62 | 0.0166123 | 0.999993 | 325.718 | 3.56707 | POST_HOC_OBSERVED_EFFECT_SCENARIO | direct_R2_higher_loss_than_B5 | 是 | 是 |
| R2_minus_B5 | 24 | 1267 | 0.104547 | 0.0189209 | 2191.5 | 0.0143866 | 1 | 325.718 | 3.56707 | POST_HOC_OBSERVED_EFFECT_SCENARIO | direct_R2_higher_loss_than_B5 | 是 | 是 |

H3直接配对量级规划：

| months | expected_opportunities_approx | planning_SE_7day | MDE_80power_two_sided95 | power_at_5pct_base_Regret_approx | planning_status | future_stationarity_unproven |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 273.938 | 0.0406916 | 0.114001 | 0.0699059 | CONDITIONAL_SCENARIO | 是 |
| 6 | 547.875 | 0.0287733 | 0.0806108 | 0.0902054 | CONDITIONAL_SCENARIO | 是 |
| 9 | 821.812 | 0.0234933 | 0.0658184 | 0.110822 | CONDITIONAL_SCENARIO | 是 |
| 12 | 1095.75 | 0.0203458 | 0.0570004 | 0.131686 | CONDITIONAL_SCENARIO | 是 |
| 18 | 1643.62 | 0.0166123 | 0.0465407 | 0.173893 | CONDITIONAL_SCENARIO | 是 |
| 24 | 2191.5 | 0.0143866 | 0.0403054 | 0.216352 | CONDITIONAL_SCENARIO | 是 |

每天3机会与月平均30.4375日只是样本上限近似。用开发7日block bootstrap SE按sqrt(1267/n)外推；双侧5%近似power=Phi(δ/SE−z.975)+Phi(−δ/SE−z.975)，80% MDE近似(z.975+z.8)SE。没有证明不同季节/波动状态/未来分布稳定或观察独立性；不把1267机会当1267独立样本。网格内无80%月份则明确不足，不默认24月够。正式样本截止、最小完整日数/机会数/事件数、流失规则与最大等待期须经后续审批后冻结；不可观察结果后延长。

## 统计、一次性与停止规则

拟保留共享完整UTC日循环stationary bootstrap，7日主块及3/14敏感性，至少5000已登记draw及明确seed，各模型同权重；主统计是机会加权配对loss差，不等权平均任意段。确认阶段只分析固定模型/alpha，不能用未来样本选择。正式的检验家族、区间/显著性/多重性、有效抽样及样本不足INCONCLUSIVE规则需随批准协议封存。

持续采集仅允许盲核时钟、来源、缺失和预测有限性，不打开模型效果/未来标签统计用于中途调参。达预登记终点且完整审计通过后，exclusive一次性claim，记录SEALED→CLAIMED→CONSUMED生命周期；失败/中断保留，重启规则另行审批，不能自动反复试验。正式报告成功、失败、无增量及不足全部留档。

若H1实际效应低于获批门槛、CI不支持、或者廉价同构组合覆盖改善，停止当前Kronos条件融合路线；未通过不扩大容量补救。若H3不复现B5优势，不把3.1事后结果升级，按证据评估是否继续简单风险模型。极端风险低估、排序和经济收益为不同结论；未来平均预测确认通过也只安排另行经济价值实验，不自动交易。

## 尚待正式seal的工件

后续批准才填写：实际开始/结束UTC及sample-stop规则、checkpoint与fixedheads/三seedTCN/Scaler/median/fusion的逐项SHA、完整代码/依赖/native库/配置SHA、新数据manifest与不可变原始来源、fresh prediction-only接口审查、point-in-time及未来扰动、single/batch一致性、共同机会/labelable purge、资源测量、预运行PASS及正式claim。任何条目缺失时不得执行。

当前开发配置SHA `0a21ccd388c9bcc5aa02e2fe2a9818f9f5b847d8513aeb2998ca753ebfa94bed`；开发源与结果由本轮delivery_manifest追溯。本草案状态始终UNAPPROVED，配置future_formal_test_approved=false，保留原holdout=CONSUMED。本轮到开发报告和草案交付为止。
