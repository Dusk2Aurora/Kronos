# Kronos M2-CI 条件增量开发研究

项目：`M2_CONDITIONAL_INCREMENT_01`。本目录保存隔离的实现、协议与进度；运行工件位于 `research/runs/M2_CONDITIONAL_INCREMENT_01/`。

本研究检验已有 B5 预测条件下 R2 预测输出的互补性。先完成既有证据审计、开发期 walk-forward OOF、有限一参数融合、统计及项目决策，再交付未获批准的未来独立验证草案。2026年Q2/Q3 已 CONSUMED，不用于本研究训练、融合选择或晋级；2025年季度仍属于原TRAIN开发范围。原实验及其全部封存工件保持不变。

用户已确认开发筛选：OOF 平均 QLIKE Regret 改善至少 5%，共同 7 日块配对区间下限大于零，至少 4/5 折改善、原 VALID 方向为正，并优于同预算廉价融合。该筛选不是新的独立确认，正式门槛仍须后续审批。

配置与登记已在新拟合前封存，科学源及输入绑定212项；95次拟合完成后另封存新模型和OOF预测。此处的文件存在不代表研究已经验收，最终以 delivery_manifest、独立数值/图表/文档验收及researchstate为准。当前进度见 [TODO](TODO.md)。

交付入口（完成后可读）：

- [开发报告](../runs/M2_CONDITIONAL_INCREMENT_01/reports/M2_Conditional_Increment_Development_Report.md)
- [一页项目决策](../runs/M2_CONDITIONAL_INCREMENT_01/reports/M2_Conditional_Increment_Project_Decision.md)
- [未来独立协议草案：未批准](../runs/M2_CONDITIONAL_INCREMENT_01/reports/M2_Conditional_Increment_Future_Holdout_Protocol_DRAFT.md)
- [实验工件](../runs/M2_CONDITIONAL_INCREMENT_01/)

复现应先读取封存的科学配置与来源，不在既有输出目录重跑。后续VALID、benchmark、独立review必须使用 execute_locked 入口，恢复训练时TF32关闭、确定性开启及线程1；直接fresh process默认flags与训练不同。协议草案不会自动授权新的正式测试。
