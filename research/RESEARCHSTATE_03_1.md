# 3.1补充实验状态索引

`researchstate_03_1.py`仅解析元数据并逐项核对文件字节SHA256，不执行训练、市场数据读取、报告内容读取或研究授权。原第三轮正式结论、全部实验条目、holdout `CONSUMED`、限制与历史证据保持不变；B5仅作为独立` supplementary_experiments`条目。

固定历史锚为revision 11，content SHA256为`d7df52beb741590e349daa3a4d590133794053e2b15cb4d8c42e7104eb97ce13`。验证涵盖完整不可变history、前向previous hash链、当前payload hash及history一致性，逐一核对原673条证据（含正式交付绑定）的既有SHA和expected SHA。旧helper只导入纯哈希/链函数，不调用其Phase B refresh、不重建旧snapshot。

补充协议、registry、TODO均绑定字节SHA；registry必须记录`experiment_id=FROZEN_RISK_03_1_B5_v1`与`protocol_sha256`。刷新仅新增revision，不覆写历史；同一payload幂等。

用户关机暂停使用registry `status=paused_for_user_shutdown`。必须填写`shutdown_handoff`对象：`path`固定为`research/frozen/experiment_03_1/HANDOFF_SHUTDOWN_20261009.md`、`sha256`、位于补充run目录内的`shutdown_manifest_path`及`shutdown_manifest_sha256`。两文件须实际存在且字节SHA匹配；未准备完成时不得运行refresh。索引显示补充`status=PAUSED`与当前阶段`phase=Supplement03_1_paused`、`status=paused`，原673证据、研究历史与holdout保持不变；暂停不等于补充实验完成。

本次关机交接为已完成5/12候选；第6候选`w16_wd0.001_s43`未完成，RAM状态不能称为已保存checkpoint。明天等待用户恢复后，先独立封存operational amendment并核验已有5个候选，5个不重训，未完成候选重做。原train有排他training_claim且尚未产出完整聚合结果，不得直接重跑。helper仅绑定交接元数据，不实现候选恢复、终止进程或新研究授权。

完成manifest路径为`research/runs/FROZEN_RISK_03_1_B5_v1/delivery_manifest.json`，必须有`experiment_id`、`protocol_sha256`、`status=COMPLETE`、`primary_result`、`pdf_path/pdf_sha256`、`md_path/md_sha256`与非空`source_and_artifact_sha256`映射。PDF路径固定`output/pdf/Kronos_Frozen_Risk_03_1_Report_v1.pdf`；MD固定run目录`reports/Kronos_Frozen_Risk_03_1_Report_v1.md`。所有映射仅核对字节，不解析CSV/PDF研究值。文件变化导致verify失败；preview会派生当前元数据，refresh显式提交新revision。旧证据发生变化直接拒绝刷新。

```powershell
& '.\.venv\Scripts\python.exe' research\researchstate_03_1.py preview
& '.\.venv\Scripts\python.exe' research\researchstate_03_1.py refresh
& '.\.venv\Scripts\python.exe' research\researchstate_03_1.py verify
```

旧`researchstate_phase_b.py verify`针对原Phase B currentstage；3.1以后使用新helper验证。旧Phase B `refresh`会重新派生第三轮阶段，不用于3.1状态。
