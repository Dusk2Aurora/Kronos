# Kronos Crypto 研究工作区

研究 OKX BTC-USDT 永续上，Kronos 是否为固定动量策略的参与／跳过提供增量信息。回测使用独立 Freqtrade 官方引擎。路线图见 [Word](Kronos_Crypto_Research_Roadmap.docx)；当前阶段及下一步以 [TODO](initialization/TODO.md) 和 [配置](configs/initial_experiment.yaml) 为准。

## 当前入口

M0 工程准备已验收，E00 和 E00R 开发期基线已完成。下一步是锁定 E00R／E01R／E02R 共用比较协议；冻结表征提取仍禁用，最终 holdout 未评估。普通基线尚未建立稳定信息优势或优于现金的盈利价值。

| 内容 | 入口 |
| --- | --- |
| 下一步与验收索引 | [初始化 TODO](initialization/TODO.md) |
| 市场、时序、成本、切分、阶段 | [首轮配置](configs/initial_experiment.yaml) |
| Ridge、19项特征及13项消融、门控设计 | [独立 E00R 配置](configs/baseline_revision_v2.yaml) |
| 多项基线的定位 | [基线角色](baselines/E00_baseline_roles.md) |
| E00R 开发结果 | [E00R 结果](initialization/E00R_results.md) |
| 原 E00 结果与稀疏诊断 | [E00 结果](initialization/E00_results.md)、[诊断](initialization/E00_sparse_gate_diagnosis.md) |
| 独立环境与回测复现 | [研究环境](environment/README.md)、[Freqtrade 适配](freqtrade/README.md) |
| 数据字段与单位 | [数据契约](data_contracts/okx_usdt_perp.yaml) |
| 实验版本与尝试记录 | [登记目录](registry/) |

`initial_experiment.yaml` 中的 `first_round_head` 记录原 E00 logistic；E00R 的 Ridge 设计在独立配置中。当前协议尚未锁定，不把原 E00 参数自动沿用到 E01R／E02R。

## 表征与生成的研究边界

主线按路线图第5章，用历史256根 OHLCVA，经冻结 tokenizer 和 Kronos 的 `decode_s1()` 提取最后有效历史位置的 context，拼接普通特征后训练同类小读出器。E01R 与普通特征、随机冻结骨干对照；随机骨干保留原 tokenizer。这检验预训练表征对当前任务的增量，不等于原生未来路径生成评估。

路线图第7章的多路径生成属于单独研究线：需保留每条路径，先计算逐路径收益／风险，再统计均值和分位数；官方预测接口会先对路径取均值，不能从平均 OHLC 恢复预测分布。采样比例须校准后才能解释为市场概率。本轮未启用生成评估。

## 文件保留与清理

- `baselines/`、`scripts/`、`freqtrade/` 保存研究实现与有效核验入口。`verify_*` 是因果、交易账本或验收检查，按需要执行，不作为每次工作的默认全量测试。
- 上游 `tests/test_kronos_regression.py` 的4项模型回归保留；已有环境验收记录其通过，未因工作区整理重跑模型。
- `initialization/` 中的审计、旧 E00、原决策起点报告及复核 JSON 是历史证据，可能已被 registry 的来源 hash 引用，保留原路径和内容。当前状态读 TODO／配置，历史报告不代表当前阶段。
- `.gitattributes` 保留研究文件原始字节，包括 LF／CRLF，避免 Git 自动转换换行使已记录的 SHA256 校验失效；跨机器读取时仍使用 UTF-8（个别历史源码带 BOM）。
- `data/`、`runs/`、`.venv`、权重缓存与外部 Freqtrade 运行副本不提交。仅克隆此仓库不能取得本机数据、模型和完整运行工件；复现需按 manifest 与依赖锁准备对应来源。
- 当前时间折线图见本机 `runs/baseline_time_series_20261008_v3/`，含 PNG／SVG、CSV 数值底表和核验报告；各季度独立重置、只反映已平仓权益。
- 被替代的绘图 v1（失败记录）与 v2（无端点标记版）压缩保存在本机 `runs/archive/`，目录树与原文件字节保留。需要历史版本时先解压到 `runs/`；当前 v3 不依赖这两个目录。

清理仅移除可重建缓存，并归档被替代的绘图输出；数据快照、标签、原始回测、实验尝试与失败证据保留。
