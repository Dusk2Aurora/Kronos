# 冻结实验 2：未来条件风险信息增量

本目录是用户新授权的第二项冻结研究空间，配置以 [frozen_risk_02_v1.yaml](../../configs/frozen_risk_02_v1.yaml) 为准，阶段/范围以本目录 [TODO](TODO.md) 为准。初始化与第一轮文件是保留的历史阶段，不覆写其验收工件。

正式运行：`research/runs/FROZEN_RISK_02_v1/`；注册表：[FROZEN_RISK_02_v1.json](../../registry/FROZEN_RISK_02_v1.json)。完整 PDF 位于 `output/pdf/Kronos_Frozen_Risk_02_Report_20261008_v1.pdf`。工程 PASS，主预测增量 INCONCLUSIVE，预训练骨干支持，是否保留风险路线尚需独立验证；当前有限开发预算已结束。

只使用历史 1h OHLCVA 作输入。标签独立采集官方 5min 成交价，以入场 open 和48个已完成 close 构建未来4h RV；逐记录先拒绝 holdout 时间再解释数值。R0/R1/R2/R3统一信息、机会、切分和有限选择；无微调、收益优化、Freqtrade变更或 push。

核验入口（在项目根目录，用显式研究解释器）：

```powershell
& '.\.venv\Scripts\python.exe' research\frozen\experiment_02\data.py audit
& '.\.venv\Scripts\python.exe' research\frozen\experiment_02\evaluate.py --help
& '.\.venv\Scripts\python.exe' research\frozen\experiment_02\charts.py --help
```

完整实现顺序见 `lock.py → data.py collect → run.py prepare → audit.py pre → run.py fit → audit.py post → evaluate.py → charts.py → finalize.py → report.py`。已发表 run 与报告不可覆写；完整重新采集/拟合应复制实现并另登记复现版本与输出目录。统计/图重算建议复制已封存预测到新验证目录后执行，防止改动原结果。

最终 `final_acceptance.json` 绑定科学验收与结果hash；`delivery_manifest.json` 绑定最新版报告、源码和独立PDF数值/视觉核验。首次正式源码快照与事后纯验收精度修正同时保留，详见 [ENGINEERING_NOTES](ENGINEERING_NOTES.md)。
