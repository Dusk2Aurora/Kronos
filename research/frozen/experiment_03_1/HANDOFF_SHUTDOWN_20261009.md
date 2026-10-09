# 冻结风险实验 3.1：关机交接

状态：**PAUSED_FOR_USER_SHUTDOWN**。用户在北京时间2026-10-09明确要求“留档，我要关电脑明天继续”。本交接只留档和停止进程，不开展新训练、评估或PDF生成；不push。

## 已落盘

- 实验：`FROZEN_RISK_03_1_B5_v1`，路径：`research/runs/FROZEN_RISK_03_1_B5_v1`。
- 配置：`research/configs/frozen_risk_03_1_b5_v1.yaml`；SHA256：`4636e4f7cc5db9ab3a9e425b518c5f2674322a8d67293bed1ab358e8cd96741c`。
- 因果窗口、TRAIN标准化、复用标签、预训练前source seal和合成模型检查已完成。
- 五个候选完整保存了`.pt`、`history_*.csv`和`audit_*.json`：`w16_wd0.0001_s17`、`w16_wd0.0001_s29`、`w16_wd0.0001_s43`、`w16_wd0.001_s17`、`w16_wd0.001_s29`。各自重载预测精确一致，审计PASS。
- 原始`fit_events.jsonl`共11条：5组STARTED/COMPLETED，加第6候选的STARTED。必须保留原始记录。
- 磁盘留档及哈希清单位于`research/runs/FROZEN_RISK_03_1_B5_v1/provenance/shutdown_20261009/`；权重、曲线、审计、日志、登记及TODO更新前版本均有副本。
- 第三轮原报告、协议、561项交付封存和确认性结论保持不变；既有holdout仍为CONSUMED。

## 明确中断点

第6候选`w16_wd0.001_s43`此前以内存挂起。其原进程PID 52944及launcher PID 46720已因用户关机要求终止，并核验不存在。**没有第6候选的磁盘检查点、epoch历史或最优权重，不能声称能够从其上次epoch继续。**

完整训练循环未结束，因此尚无`selected.json`、全12候选VALID预测聚合、完整`training_resources.json`或`candidate_audit.json`。尚未生成B5的Q2/Q3预测、补充指标和3.1 PDF。原始`training_claim.json`仍在，不能删除它以重跑全部训练。

## 用户恢复后执行

1. 先核验本留档manifest、原pretrain review seal、5个已完成候选和第三轮原561项哈希。读取本交接和3.1 TODO，以新配置为研究范围；不要仅用初始化TODO的旧阶段快照推断。
2. 独立记录并封存**关机导致的操作性恢复补充说明**：固定结构、种子、损失、数据和VALID-only选择全部保持；只允许重做被关机中断的第6候选，以及原先未开始的6个width32候选。前5个不重训，不增加科学搜索。原协议“无retry”与原始预算保留，不得悄悄改写；诚实登记1次不完整尝试和重做原因，资源中单列已消耗的中断/暂停时间。
3. 实现并审查有界恢复入口，复用原`b5_model.py`。从5个权重恢复VALID预测和已保存metadata/resources，恢复与原循环一致的12候选汇总及选择规则；不得直接调用原`run train`、删除claim或覆写旧权重/历史/seal/journal。新的恢复入口、操作性说明和来源哈希须在任何重做前封存。
4. 原候选顺序：先重做`w16_wd0.001_s43`，再`w32_wd0.0001_s17/s29/s43`及`w32_wd0.001_s17/s29/s43`。完整完成目标仍为12个候选；总尝试须额外披露关机中断的一次。
5. 所选结构和三seed锁定后，才统一生成补充Q2/Q3预测，再顺序运行资源测量、独立数值复核、PDF出版与逐页视觉验收。数值复核及finalizer对操作性尝试记录的校验须相应独立审查；不得把含中断的journal伪装成从未中断。
6. 3.1仍为**事后探索**，不添加新的确认性成功路径。报告保持全部13家族、7/3/14日原配对multiplicities、全部尝试及真实资源，保留原3.0 PDF历史附录；无push。

## 已准备但未执行的交付入口

- 科学执行：`research/frozen/experiment_03_1/run.py`（原训练入口不可直接重跑；evaluate/benchmark需完整候选验收后使用）。
- 独立数值：`research/delivery/frozen_risk_03_1_review.py`。
- 出版：`research/delivery/frozen_risk_03_1_report.py`。
- PDF文本和字节复核：`research/delivery/frozen_risk_03_1_pdf_qa.py`；其PASS不替代视觉验收。
- 最终封存：`research/delivery/frozen_risk_03_1_finalize.py`。
- 项目状态：`research/researchstate_03_1.py refresh/verify`，不要调用旧phase_b refresh覆盖3.1阶段。

研究解释器：`D:\file\Kronos-master\.venv\Scripts\python.exe`。所有恢复均从项目根目录执行。等待用户明确恢复后再开展研究。
