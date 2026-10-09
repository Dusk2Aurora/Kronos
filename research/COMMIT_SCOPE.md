# 冻结研究提交与本地证据边界

截至 2026-10-09，首轮冻结实验、风险实验 02、风险实验 03、03.1 的 B5 补充及 M2-CI 开发研究均已完成。用户随后要求停止开发、交付 PDF，再清理工作区并提交；本次整理没有运行新实验。

## 当前结果入口

- 首轮：见 [完整结论](frozen/FIRST_EXPERIMENT_CONCLUSION.md)。Ridge、有限 MLP 和同坐标量化未通过预定筛选。
- 风险 02：开发证据仍需独立验证；见 [登记摘要](registry/summaries/FROZEN_RISK_02_v1.json)。
- 风险 03：原共主要风险排序比较通过；见 [登记摘要](registry/summaries/FROZEN_RISK_03_v1.json)。这不证明交易盈利，也未证明 checkpoint 的训练截止。
- 风险 03.1：事后 B5 补充中，R2 相对 B5 的平均 AUROC 差为 -0.077168；见 [登记摘要](registry/summaries/FROZEN_RISK_03_1_B5_v1.json)。不改写风险 03 原预登记结论。
- M2-CI：开发筛选结论为 `INCONCLUSIVE`。R2 融合的 OOF Regret 平均改善 0.3847%，3/5 折方向为正，配对 7 日 ΔQLIKE 区间跨零。见 [开发报告](reports/m2_ci/M2_Conditional_Increment_Development_Report.md)、[项目决策](reports/m2_ci/M2_Conditional_Increment_Project_Decision.md)和[未来协议草案](reports/m2_ci/M2_Conditional_Increment_Future_Holdout_Protocol_DRAFT.md)。草案未获正式独立测试批准。

中央研究状态为本地 `research/researchstate.json` revision 17，content SHA256 为 `cd943b96c957cf3a34e54be0b6c66e07ec16109a2d55d82d166f059073f70c79`，旧 holdout 为 `CONSUMED`。初始化及首轮文档中的 `SEALED` 是当时的历史状态，不能用于判断当前实验阶段。当前没有待运行研究任务。

## Git 中保留什么

提交研究源码、锁定配置、阶段 TODO、关键因果与统计测试，以及八份 [实验登记摘要](registry/summaries/)。摘要记录原完整登记的 SHA256，包含失败试验，不替代完整登记或原始证据。

三份 M2-CI Markdown 报告按原字节复制，来源及 SHA256 见 [来源索引](reports/m2_ci/source_index.json)。报告所引用的数值工件和图表位于下述本地证据目录。

## 本地保留什么

以下内容由 `.gitignore` 排除，未因本次整理改写或删除：

- `research/data/`、`research/cache/`、`research/runs/`：原始数据、模型、标签、预测、失败尝试和封存 manifest。
- `research/researchstate.json`、`research/state_history/`、八份完整实验登记：包含本机附件来源及完整溯源链。
- `research/delivery/` 下的生成目录、`output/`：PDF、审阅记录、渲染检查及交付 manifest。
- `tmp/pdfs/` 与便携 Poppler runtime：被报告清单引用的渲染证据和复现依赖。
- `research/frozen/experiment_02/lock.py`、`research/frozen/experiment_03/lock.py`、`research/m2_ci/initialize.py`、`research/researchstate.py`：历史初始化及 Phase A 索引脚本包含个人附件路径且源字节已封存，因此保留本地原件，排除 Git 发布。这些脚本不属于当前待运行入口；复现原研究需恢复本地证据包，新研究应另行登记。

最新 PDF 为本地 `output/pdf/Kronos_M2_CI_Development_Report_v1.pdf`，24 页，SHA256 为 `27647d615ac288e03840ca1643d7eae2aaa56ee5abcb4de6761699d3253c1b51`。

仅清理未被封存清单引用的 Python/pytest 缓存、重复临时脚本和已解压的 Poppler 下载包及下载元数据。明细保存在本地 `tmp/cleanup_20261009/`。新的关键测试文件保留，因为它们验证因果时序、purge、数据和统计边界。

Git clone 包含设计与实现；完整数值复现还需上述本地证据包。核验当前索引使用 `research/researchstate_m2_ci.py verify`，不使用旧阶段 helper 的 `refresh` 覆盖 M2 状态。
