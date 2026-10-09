# 冻结实验 3：独立风险排序与非线性基线审计

Phase A / Stage1-5 已验收并保留原交付。用户随后明确完全同意推进第三轮，现已授权按封存协议执行 Phase B；真实原文存于 authorization/user_approval.json。

- [x] 中央 `research/researchstate.json` 与不可变状态历史已建立并验证；前两轮工件保留。
- [x] 阅读新指令、Roadmap、前两轮证据与第二轮接口，识别原接口固定DEV截止的复用边界。
- [x] 锁定第三轮科学协议及28候选/阶段预算，保存注册、授权屏障与来源证据。
- [x] 实现边界感知输入/表征/标签；只复用DEV缓存与固定公式，不修改旧源码。
- [x] 实现同33特征LightGBM4.6.0 Gamma四结构、统一QLIKE选择与完整早停历史。
- [x] 实现日块季度分层paired AUROC/AP统计、两共同终点、无效draw规则及辅助风险诊断。
- [x] 执行DEV/合成预验收，全部候选与确定性/因果/精度保护检查、独立统计复核。
- [x] 冻结可执行源码/模型/依赖/输入bundle及hash，保证无批准的正式入口必拒绝。
- [x] 交付完整中文解封前Markdown/PDF，完成数值与视觉QA，researchstate进入待用户审批。

Phase B：**已完成，唯一正式测试已 CONSUMED**。TRAIN到2026-03-01；VALID到2026-04-01；HOLDOUT [2026-04-01,2026-10-01)。完成正式模型重放与封存结果逐位一致后，按锁定范围一次读取、标签、编码与评估；不做测试调参，不push。

验收完成：63项最终synthetic、2501项DEV工件、82+9项独立源码/数值复核通过；18页最终PDF v2及全页视觉QA通过。封存bundle已建立。Phase A已验收；随后用户完整授权Phase B。Stage6-8现已完成，未push。

## Phase B 正式执行

- [x] 保存随后明确用户授权原文，绑定协议及封存bundle SHA。
- [x] 创建唯一正式执行claim。
- [x] 完成28正式候选及2规定重放，所有所选模型与Phase A逐位一致。
- [x] 官方5m独立区间全覆盖与raw/canonical核验。
- [x] 两完整季度标签、冻结编码、九基线预测及共同主要统计。
- [x] 全标签、点指标、全部主要bootstrap draw与CI独立复核。
- [x] 终态封存、最终报告PDF数值和视觉QA、researchstate与交付manifest。

正式结束：Phase B / Stage6-8 已完成；holdout CONSUMED；工程及独立审查 PASS。主要结论：CONFIRMED_RANKING_INCREMENT。全部前两轮与Phase A工件原样保留。最终完整中文PDF v2、全页QA与数值底表已完成；不自动第四轮，不push。
