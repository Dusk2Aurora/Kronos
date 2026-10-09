# 冻结历史表征开发实验

当前进度、验收索引、结果评审与下一步维护在 [冻结 TODO](TODO.md)；[初始化 TODO](../initialization/TODO.md) 保留 M0 与交接前基线记录。

路线图第5章／M1的第一项冻结表征有限研究已完成，当前状态为 `first_frozen_experiment_completed_holdout_sealed`，没有待运行的正式实验，等待用户验收。[完整开发期结论](FIRST_EXPERIMENT_CONCLUSION.md)、[综合manifest与hash](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/manifest.json)和[全部筛选](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/all_screens.json)为当前交付入口。

首轮Ridge、[有限MLP结果](../runs/FROZEN_MLP_20261008_v2/results.md)和[同坐标量化结果](../runs/QUANT_20261008_v1/results.md)合计276次新正式拟合、352份新官方导出；[综合官方曲线CSV](../runs/QUANT_20261008_v1/presentation/official_curves.csv)保留512条曲线。信息、经济及量化可读性筛选均未通过。量化主rank50仍保留至少3/4折相对普通19项少亏和匹配参与数诊断的正向信号，但缺少完整预测、配对区间与现金／压力盈利证据，不能解释为盈利或量化瓶颈。

停止当前BTC单资产、1h历史、4h固定动量参与任务上的扩头及冻结表征路线扩展；不训练骨干／tokenizer、不打开holdout、不push。MLP v1拟合前预检失败为0次正式拟合，图表配置布局预检及bootstrap seed派生说明保留于[冻结TODO](TODO.md)。旧工件保持原路径，以下首轮命令作为复现入口保留，不是新增研究或覆盖已有运行的授权。

首轮Ridge按路线图第5章提取历史隐藏向量；固定动量只决定参与／跳过。主比较为 E00R 普通19项、E01R 普通19项＋预训练历史context、E02R 普通19项＋随机骨干context。三个随机种子全部保留，tokenizer相同；这个对照仅检验骨干预训练贡献。

规范参数见 [共用协议](../configs/frozen_comparison_v1.yaml)，阶段见 [初始化配置](../configs/initial_experiment.yaml)。目标、普通特征、训练成员、Ridge预算、验证选择和门控与已验收E00R完全一致后，复用其原始工件。工程验收、信息筛选和盈利证据分别判定，不因普通模型未盈利而阻止本轮研究。

## 执行与工件

研究解释器为 `D:/file/Kronos-master/.venv/Scripts/python.exe`。官方回放由适配脚本调用 `D:/file/freqtrade/.venv/Scripts/python.exe -m freqtrade backtesting`，不修改框架核心。每条命令均从项目根目录执行。

执行顺序为 `lock_protocol.py` → `encoder.py` → `heads.py` → `replay.py` → `analyze.py` → `charts.py`。所有CLI以 `--help` 查看精确参数，示例：

```powershell
& '.\.venv\Scripts\python.exe' research/frozen/encoder.py --protocol research/configs/frozen_comparison_v1.yaml --output research/runs/FROZEN_20261008_v1/cache/pretrained --variant pretrained --device cuda
& '.\.venv\Scripts\python.exe' research/frozen/encoder.py --protocol research/configs/frozen_comparison_v1.yaml --output research/runs/FROZEN_20261008_v1/cache/random_s17 --variant random --seed 17 --device cuda
& '.\.venv\Scripts\python.exe' research/frozen/heads.py --output-dir research/runs/FROZEN_20261008_v1
& '.\.venv\Scripts\python.exe' research/frozen/replay.py --output-dir research/runs/FROZEN_20261008_v1 --workers 3
& '.\.venv\Scripts\python.exe' research/frozen/analyze.py --output-dir research/runs/FROZEN_20261008_v1
& '.\.venv\Scripts\python.exe' research/frozen/charts.py --output-dir research/runs/FROZEN_20261008_v1
```

随机骨干另有种子29、43两条同型命令。上述是原始执行顺序，不是每次工作的默认动作。协议只锁一次；头、分析和图表拒绝覆盖已有产物；编码严格校验源码、配置、输入、精度和分块hash。`replay.py --resume`重用已有且hash不变的官方ZIP，仅运行未触碰任务；没有ZIP的部分失败输出保留并拒绝覆写。新试验需要新的ID、协议版本和登记，不按旧结果调参后覆盖运行。

每个缓存保存256根历史窗口的归一化、tokenizer与骨干hash、时间字段、机会ID、边界及源码契约。评估模式、冻结参数和推理模式中只使用 `tokenizer.encode(half=True)` 与 `decode_s1()`，末端context为512维；不调用采样forward或未来路径生成。预训练和随机骨干使用同样的float32、独立窗口标准化及最后有效位置。模型、tokenizer及数据快照仅从固定本地来源加载。

首轮Ridge运行目录包含 `provenance/`、`cache/`、64次拟合尝试、16个选中头、开发预测与64份无结果字段sidecar、128份官方回放、配对统计底表及 `presentation/` 图表。工程核验包括批量与单条token一致、重复计算、未来扰动、缺口/可用时间/真实amount拒绝、分块checksum及抽样缓存独立重算。标准化与下游头只用训练折拟合，选择与门槛只来自前一验证季度。

## 解释边界

开发季度此前已查看，本轮是探索性历史研究；checkpoint训练截止未知。最终holdout不参与编码、拟合、模型选择、收益或分类比例统计。参考机会净效用和预测损失的时间块区间不是复合投资组合收益置信区间；匹配随机抽样区间也不是策略业绩置信区间。

组合曲线为官方已平仓权益，各季度独立重置10,000 USDT，含实际资金费用；7/14 bps为每边费用代理。闭仓平线不代表持仓无风险，三个随机骨干政策的均值只是描述性汇总，不是另一个已执行组合；MLP每组的三个预测头均值则是该组实际执行分数。所有敏感性、随机种子、零交易、负结果都保留。

完成结果入口：[完整开发期结论](FIRST_EXPERIMENT_CONCLUSION.md)、[首轮Ridge报告](../initialization/FROZEN_results.md)、[MLP协议](../configs/frozen_mlp_v2.yaml)、[量化协议](../configs/frozen_quant_v1.yaml)。本机数据、模型、缓存和导出为ignored工件，clone不会自动取得；完整复现需遵循manifest与已保存源码/配置字节，而非直接重跑示例。三轮的完整预算、验收与失败索引见[冻结TODO](TODO.md)，当前无待运行正式实验。
