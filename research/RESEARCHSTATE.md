# 中央研究进度索引维护

`researchstate.py` 维护 `researchstate.json` 和 `state_history/`，只读取元数据并核验文件字节 SHA256，不运行实验、不解析 PDF 内容、不读取市场数值或 holdout 标签、预测、评估结果。

配置、当前阶段 TODO、实验登记以及对应验收／交付 manifest 是各自范围内的权威证据；中央记录提供跨实验索引，历史初始化配置不能代表第三轮进展。工程 PASS 与科学结论分开记录；第三轮 Phase A 的 `NOT_EVALUATED_HOLDOUT_SEALED` 不表示研究已完成或晋级。

从项目根目录使用显式研究解释器：

```powershell
& '.\.venv\Scripts\python.exe' research\researchstate.py verify
& '.\.venv\Scripts\python.exe' research\researchstate.py refresh
& '.\.venv\Scripts\python.exe' research\researchstate.py set-phase PhaseA_awaiting_approval --status awaiting_user_approval --note 'Phase A preflight delivered; holdout remains sealed.'
```

- `verify` 只读核验当前内容、证据字节 hash、历史文件及前序 hash 链；出现错误时先核查权威来源，不覆盖历史来消除错误。
- `refresh` 索引现有权威证据，第三轮保留登记状态、工程预验收、科学结论、holdout 状态及报告路径。交付 manifest 尚未产生时不因此阻断；发布后按其中预期 hash 核验协议、sealed bundle、PDF／Markdown 和来源工件，缺失、冲突或不完整元数据进入 blockers。
- `set-phase` 仅显式记录 Phase A 进度；`refresh` 不从工程 PASS 或交付状态自动改变阶段，也不推断批准。重复刷新且元数据未变化时不新增 revision。

每次内容变化追加一个不可变 revision，保留完整历史及前序 hash，当前文件通过临时文件替换。已有第一、第二轮验收与失败证据不得覆盖；来源演进以新的对应 manifest 和版本记录解释。

此工具仅支持 Phase A。Phase B 需用户另行明确批准并按第三轮锁定协议的独立授权及一次性执行机制操作；交付报告、工程通过、中央进度修改均不构成 holdout 授权。中央 holdout 始终保持 `SEALED`，授权字段为空。
