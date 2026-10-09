"""Publish Chinese review reports from verified generated artifacts, never fit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import html
import numpy as np
import pandas as pd
from research.frozen.experiment_03 import guard
from research.frozen import build_overall_pdf as pdfbase
from research.frozen.experiment_03.run import immutable_json,immutable_csv

ROOT,RUN=guard.ROOT,guard.RUN
TMP=ROOT/'tmp/pdfs/frozen_risk03_v2'; TMP.mkdir(parents=True,exist_ok=True)
PDF=ROOT/'output/pdf/Kronos_Frozen_Risk_03_Preflight_Report_v2.pdf'

class Document(pdfbase.Document):
    def __init__(self,path,phase='A'):
        super().__init__(path);self.phase=phase
        self.c.setTitle('Kronos 第三轮冻结风险实验 — '+('解封前预验收' if phase=='A' else '独立测试报告'))
        self.c.setSubject('FROZEN_RISK_03_v1 | Independent Risk Ranking Validation & Nonlinear Baseline Audit')
    def start(self,title,subtitle='',wide=False):
        super().start(title,subtitle,wide)
        self.c.setFillColor(pdfbase.colors.white);self.c.rect(0,0,self.w,34,fill=1,stroke=0)
        self.c.setFillColor(pdfbase.GRAY);self.c.setFont('CN',8)
        footer='Phase A 预验收 | 独立 holdout 封存 | 时间均为 UTC' if self.phase=='A' else 'Phase B 一次正式评估 | 不进行测试反馈选择 | 时间均为 UTC'
        self.c.drawString(42,22,footer);self.c.drawRightString(self.w-42,22,str(self.number))

def jread(rel):return json.loads((RUN/rel).read_text(encoding='utf-8'))
def number(value,d=8):return f'{float(value):.{d}f}'

def early_stop_plot(events):
    plt=pdfbase.plt;fig,axes=plt.subplots(2,2,figsize=(10,5.6))
    for ax,event in zip(axes.flat,[e for e in events if e['family']=='B2' and e['status']=='completed']):
        name=f"B2_leaves_{event['num_leaves']}_minleaf_{event['min_data_in_leaf']}"
        history=pd.read_csv(RUN/'preflight/preview'/f'{name}_history.csv',float_precision='round_trip')
        ax.plot(history.iteration,history.train_QLIKE,label='TRAIN',lw=1.2)
        ax.plot(history.iteration,history.validation_QLIKE,label='VALID Mar 2026',lw=1.2)
        ax.axvline(event['best_iteration'],color='#777777',ls='--',lw=.8)
        ax.set_title(f"leaves={event['num_leaves']}, minleaf={event['min_data_in_leaf']} | best={event['best_iteration']}")
        ax.set_xlabel('Boosting iteration');ax.set_ylabel('Raw QLIKE (lower better)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('DEV engineering preview — 4 predeclared B2 structures; no holdout',fontsize=12)
    fig.tight_layout();path=TMP/'b2_history.png';fig.savefig(path,dpi=180);fig.savefig(TMP/'b2_history.svg');plt.close(fig)
    return path

def write_markdown(selected,events,tests,cache,thresholds,review):
    lines=['# Kronos 第三轮冻结风险实验：解封前预验收报告',
      '\n实验：`FROZEN_RISK_03_v1`；当前阶段：Phase A / Stage 5 审批交付。',
      '\n工程预验收：PASS。独立研究结论：`NOT_EVALUATED_HOLDOUT_SEALED`。未下载、读取数值、生成标签、编码、预测或评价本轮 holdout；未 push。',
      '\n## 1. 项目进度与本轮目的',
      '\n项目中央状态为 `research/researchstate.json`，配套 `research/researchstate.py` 和不可变 hash 链历史。初始化已验收；第一轮返回预测及有限MLP/量化路线未证明稳定经济增量；第二轮条件风险主结论INCONCLUSIVE，预训练对照有支持。本轮检验保留独立区间中的风险排序，以及普通非线性模型是否覆盖增量。',
      '\n## 2. 固定协议',f'\n协议SHA256：`{guard.CONFIG_SHA}`。',
      '\nTRAIN [2024-01-01,2026-03-01)，VALID [2026-03-01,2026-04-01)，HOLDOUT [2026-04-01,2026-10-01)。测试季度为Q2/Q3，按季度内差值后等权汇总。训练2,369机会、验证92机会；每8小时UTC04/12/20决策，60秒延迟，下一小时入场，持有4小时；label_end及labelable_at严格purge。',
      '\n## 3. 数据、标签与冻结特征',
      '\n小时数据来源为既有官方快照；混合CSV先过滤时间再解析OHLCVA，整文件hash仅为元数据。DEV 5m数据236,448行、789页，复用已验收raw/CSV证据；本轮重新建机会钟，不复用旧折purge。标签为入场5m开价加48个已完成收价构成49价格、48个log收益平方之和，非年化；epsilon=1e-12、原值保留，不插值、不回退1h。真实volume/amount分别为volCcy/volCcyQuote。',
      '\nR1普通19+风险14=33特征；HAR仅3个历史logRV；R2为R1加512维预训练表征；三个随机冻结骨干保留原tokenizer。5m只用于标签。256小时历史窗口float64均值/标准差(ddof0,eps1e-5)、clip5后float32；decode_s1最后上下文，无微调、TF32/autocast关闭、batch8。',
      '\n每变体复用2,459窗口、补2个DEV窗口。缺失ID为2025-09-30T20:00:00Z、2025-12-31T20:00:00Z。所有复用窗口key及cache chunks SHA校验，首/中/尾独立重编码；冻结权重前后不变。',
      '\n## 4. 模型、选择及全预算',
      '\n线性6家族×4lambda=24，B2四结构=4，共28候选；另有规定R2与B2各一次确定性重放，完全一致。所有候选、事件、模型、验证预测和B2每轮历史保留。lambda=.001/.01/.1/1，验证原始QLIKE选择；并列取较大lambda。B2仅同33列，LightGBM4.6.0 Gamma、leaves7/15×minleaf30/60、lr.05、最多300轮、early-stop30、CPU单线程、所有参数固定；结构并列取较少叶、较大minleaf。',
      '\n两类模型均按训练effectiveRV中位数缩放目标并恢复原单位，预测clip[1e-12,1]。线性优化float64；LightGBM原生Dataset训练标签实际float32，转换误差已记录，验证QLIKE仍使用原始float64标签。树早停有额外选择自由度，参数L2不等于线性容量匹配。QLIKE可负，不计算百分比改善。',
      '\n### 验证期选择工件（DEV工程预演，不是独立结论）\n',
      '| 模型 | 维度 | 选择 | 验证原始QLIKE |\n|---|---:|---|---:|']
    for s in selected:
        choice=f"leaves={s['num_leaves']}, minleaf={s['min_data_in_leaf']}, iteration={s['best_iteration']}" if s['family']=='B2' else f"lambda={s['lambda']}"
        lines.append(f"| {s['family']} | {s['dimension']} | {choice} | {number(s['validation_qlike'],12)} |")
    lines += ['\n## 5. 主要终点与统计',
      '\n事件：log((RV_raw+eps)/(EWMA+eps))>log2；评分：log((模型预测+eps)/(同一EWMA+eps))，不是概率，不在测试期归一化。共同主要终点R2−R1与R2−B2均须两季度正向、季度等权ΔAUROC≥.03、7日块配对95%CI下限>0，每季度正/负事件至少各10；辅助不能替代失败主要终点。',
      '\n季度分层 circular stationary bootstrap，全UTC决策日为单位保留空日，7日主分析、3/14日敏感性，各5000次，seed20261008+blockdays×1000+quarter_index。模型与两比较共用相同日multiplicities。单类draw记NaN不替代不补抽；joint-valid百分位CI；任何主要季度/等权联合无效比例>1%→INCONCLUSIVE。保存全部draw CSV及日期multiplicities NPZ。',
      '\nINVALIDATED优先；事件/有效draw不足→INCONCLUSIVE；共同终点通过且R2 pooled Regret>1.10×R1→RANKING_ONLY，阻止绝对RV仓位缩放建议；共同终点通过无红旗→CONFIRMED_RANKING_INCREMENT，但不证明风险管理安全。仅胜R1未胜B2→NONLINEAR_BASELINE_COMPETITIVE，并非统计等价。明确负向CI或有正CI但幅度不足按锁定NOT_SUPPORTED，否则INCONCLUSIVE。',
      '\n## 6. 校准与图表',
      f"\n训练q90={thresholds['train_q90']:.16g}，q99={thresholds['train_q99']:.16g}，单位为4h平方log收益。模型预测decile、surprise score quartile、EWMA历史quartile均由TRAIN封存。观察RV/预测RV校准、高/极端风险低估、事件发生率、Regret/logMSE及7日sum(loss)/sum(count)诊断均已实现；同epsilon surpriseMSE与additive-epsilon logMSE是恒等式，非独立证据。",
      '\n验证月真实时间曲线与数值底表在 `diagnostics/development_validation/`：RV_timeline、surprise_score_timeline、rolling_QLIKE_Regret（PNG/SVG/CSV）。没有绘制或统计真实holdout。',
      '\n## 7. 预验收、独立复核与失败记录',
      f"\n最终synthetic测试日志为preflight/tests/risk03_final_tests.txt；退出码{tests['exit_code']}。实际DEV审计2501项PASS，全2461标签独立逐段复算、所有模型重载、验证候选选择及训练标准化复核。独立审查状态{review['status']}；模型对照、bootstrap、守卫和正式数据路径审查留档。",
      '\n原9个离线/环境suite保留，8项直接PASS；E00R在当前完成阶段的stage断言失败，使用归档阶段配置原断言重验PASS。M0引用第二轮已验收历史重验，保留原输出路径断言。',
      '\n保留的工程失败：首轮机会表RangeIndex被误判role overlap（任何fit前修复）；旧M0 verifier拒绝新run输出路径（9suite执行结果不丢失，改为引用已验收证据）；缓存ID object dtype在allow_pickle=False读取时拒绝（任何fit前，原NPZ保留，v2仅改Unicode metadata，四矩阵逐位未变）。不改科学协议，不弱化断言，不扩候选。',
      '\n## 8. Checkpoint证据与限制',
      '\n官方HF历史commit将当前相同LFS weight hash绑定至2025-06-30公开日期；这支持权重在本轮holdout前已公开，不能充当checkpoint-specific训练截止声明。早期DEV不能全部称严格事前OOS。官方raw API/card/source字节、URL、UTC采集时间与SHA保存在provenance/official_sources。',
      '\n## 9. 交付与审批',
      '\n审批绑定完整 `preflight/sealed_bundle.json` 与协议/代码/依赖/输入hash；源码封存副本、dirty文件hash、checkpoint/tokenizer revision、binary/wheel及全部模型均保留。最终交付清单另保存PDF/MD/QA与bundle SHA。',
      '\n正式入口：`.venv/Scripts/python.exe -m research.frozen.experiment_03.formal --execute-once`。当前调用必须拒绝。收到真实后续用户批准记录后，一次claim→同28正式拟合并与preview逐位比对→才准test读/下载→完整两季测试→全标签/点指标/每bootstrap draw及CI独立复核→CONSUMED并出版正式报告；任何研究执行错误INVALIDATED，不能反馈重试或开第四轮。出版错误保留PUBLICATION_PENDING，仍为CONSUMED，不重跑研究。',
      '\n用户附件Stage5要求：提交预验收结果和最终协议SHA256后，停止运行，等待用户明确授权开启holdout。附件本身不是授权。',
      '\n需用户随后批准的准确文字：**授权开启 FROZEN_RISK_03_v1 holdout，按已锁定协议执行正式测试。**',
      '\n目前研究结果：**NOT_EVALUATED_HOLDOUT_SEALED**。Stage6-8待批准。']
    path=RUN/'reports/Kronos_Frozen_Risk_03_Preflight_Report_v2.md'
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')

def phase_a():
    cfg=guard.config();pdfbase.init_fonts()
    selected=jread('preflight/preview/selected_models.json');cache=jread('features/cache_audit_v2.json');thresholds=jread('calibration/training_thresholds.json')
    tests=jread('preflight/tests/final_suites.json');review=jread('independent_review/preflight_review.json');prep=jread('preflight/preparation.json')
    if any(v['status']!='PASS' for v in (tests,review,prep,jread('preflight/actual_audit.json'))):raise ValueError('Preflight not accepted')
    if PDF.exists():raise FileExistsError('Preserve accepted PDFs; publish a new version')
    events=[json.loads(v) for v in (RUN/'preflight/preview/fit_events.jsonl').read_text().splitlines()]
    write_markdown(selected,events,tests,cache,thresholds,review)
    d=Document(PDF)
    d.start('第三轮冻结风险实验','FROZEN_RISK_03_v1 — 独立风险排序验证与非线性基线审计\nPhase A / Stage 5 可审批交付 | 2026-10-08')
    d.heading('当前交付结论')
    d.para('工程预验收 PASS。固定28候选与两次规定重放全部完成，实际DEV审计2,501项通过。独立测试区间尚未打开，研究结论为 NOT_EVALUATED_HOLDOUT_SEALED。',13,bold=True)
    d.para('本报告的所有具体模型数值来自TRAIN/VALID工程预演。第三轮的共同主要终点、测试事件数、置信区间、风险幅度红旗及路线停止决定均待另行授权后评价。')
    d.table([['对象','现状'],['项目researchstate','中央JSON、CLI与不可变hash链历史已建立'],['第三轮科学协议','已锁定，28候选/阶段，无扩展搜索'],['独立holdout','[2026-04-01,2026-10-01)，SEALED'],['操作边界','不微调、不修改Freqtrade策略/用户目录、不push']])
    d.heading('协议 SHA256');d.para(guard.CONFIG_SHA,9)
    d.para('附件明确要求完成Stage1-4后提交审批，随后单独批准Stage6。本报告是解封前交付，不将DEV预演写成独立确认。')

    d.start('项目进度与本轮问题','中央进度索引：research/researchstate.json；历史来源按各轮正式manifest追溯')
    d.table([['阶段','验收/结论','本轮用途'],['初始化M0 / E00 / E00R','已完成相应验收','沿用官方时序与来源约束'],['第一轮冻结预测','未证明稳定预测/经济增量','停止当前收益预测扩模'],['第二轮条件风险','工程PASS；主INCONCLUSIVE；预训练对照支持','需要保留区间独立验证'],['第三轮 Phase A','工程实现与DEV预验收完成','提交具体审批包'],['第三轮 Phase B','尚未授权、未执行','检验风险排序与绝对幅度']],[100,180,231])
    d.para('H1：R2在独立区间的surprise-event风险排序优于普通线性R1。H2：该增量也优于只用同33普通特征的非线性B2。H3：即便排序有效，绝对RV幅度与低估仍需单独诊断。')
    d.para('普通非线性模型的作用是审核表征增量是否已被低成本模型覆盖；树早停带有额外选择自由度，不能声称容量或L2完全匹配。三个随机骨干是辅助预训练信息对照，不能替代共同主要终点。')
    d.para('第二轮开发季度R2−R1季度等权QLIKE改善均值0.0155909，7日块95%CI约[-0.024650,0.053614]，跨零。第三轮不利用该已见范围重新优化协议。')

    d.start('来源审计与checkpoint时间证据','保存官方响应原始字节、URL、UTC采集时间与SHA256；没有请求holdout行情')
    d.table([['来源','核验结果'],['第二轮交付','protocol / final_acceptance / delivery_manifest / 最新源码hash可追溯'],['预训练权重','Kronos-small revision 901c26c1332695a2a8f243eb2f37243a37bea320'],['原tokenizer','revision 0e0117387f39004a9016484a186a908917e22426'],['公开权重时间','同一small权重2025-06-30 16:44:37Z；tokenizer 17:04:10Z'],['训练截止证明','checkpoint-specific截止仍未证明；不能用公开时间替代'],['LightGBM官方证据','v4.6.0文档与Gamma objective源文件留存']],[110,401])
    d.para('历史HF add-model revision所指LFS SHA与当前本地权重完全相同；支持权重在保留测试区间之前已公开。Roadmap论文截止描述未形成到本checkpoint的可验证映射，早期DEV仍按历史研究解读。')
    d.para('权重SHA：b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020\ntokenizer SHA：59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee',8.5)
    d.para('官方来源底表：provenance/official_sources/*.source.json；依赖底表：provenance/dependencies.json。公开数据的完整性证据不等于交易所内部账本或历史首次发布时点证明。')

    d.start('区间、决策时序与读取屏障','所有区间按[start,end_exclusive)，purge同时考虑持有终点与labelable_at')
    d.table([['用途','开始UTC','右边界UTC','机会'],['TRAIN','2024-01-01','2026-03-01','2,369'],['VALID','2026-03-01','2026-04-01','92'],['TEST_Q2','2026-04-01','2026-07-01','尚未读取/计数'],['TEST_Q3','2026-07-01','2026-10-01','尚未读取/计数']])
    d.para('每8h一次、UTC04/12/20决策起点。例如04:01决策，只看当时已完成可用的小时历史；05:00入场，09:00结束4小时标签，09:01可标注。标准化和目标缩放仅用TRAIN，选择仅用VALID。')
    d.para('1h既有快照是混合范围。读取器逐记录检查bar_open_at后，才解析数值；Phase A拒绝>=2026-04-01的数值。旧整文件的byte SHA只用于元数据，不借此分析测试范围。5m DEV已有236,448行，来源789页，不新增下载。')
    d.para('新的完整机会钟恢复旧折内部边界剔除的两项DEV机会，并剔除本轮TRAIN/VALID各自跨右边界的机会。不能把旧walk-forward included IDs直接当第三轮样本。')

    d.start('标签与普通风险特征','4h平方log收益，不年化；5m只用于标签，特征仅用可用1h OHLCVA')
    d.para('P0是入场5m bar开价；P1…P48为[entry,entry+4h)内48个已完成5m bar收价。RV_raw = Σ(log(Pi/Pi−1))²。保留RV_raw，并以max(RV_raw,1e-12)进行拟合与既有损失计算。全2461 DEV标签另以逐段价格比值独立重构。')
    d.table([['部分','数量/定义'],['普通特征','19项：历史收益、波动、量额比例与固定动量交互'],['风险特征','14项：HAR/RV/绝对收益/极差方差/比率/尾部/量额比'],['HAR','3个log(4×历史4/24/168h平均r²)'],['R0 persistence / EWMA','末4小时r²之和 / decay .97历史255收益归一权重'],['R1','普通19+风险14=33；R0预测不进入拟合特征'],['R2 / 三随机','R1+512冻结表征=545']],[120,391])
    d.para('真实volume=volCcy（BTC），amount=volCcyQuote（USDT）；不以价格×量替代成交额。缺失、错位、未确认或来源冲突阻断全量评价，不插值，不回退小时标签，不按样本剔除制造可评估性。')
    d.para('OHLC成交价首末点不是精确边界tick、标记价或订单簿；本实验不评价收益，不计算交易成本或资金收益。')

    d.start('冻结编码与缓存核验','256历史小时 / 每窗口归一化 / float32推理 / 最后有效上下文512维')
    d.para('六通道OHLCVA以float64历史均值和总体标准差(ddof=0,epsilon=1e-5)归一、clip±5后转float32。原tokenizer.encode(half=True)，decode_s1最后上下文；eval、无梯度、无微调，TF32与autocast关闭，批大小8。')
    d.table([['变体','共用/新DEV','重编码最大绝对差'],*[ [a['variant'],f"{a['reused_rows']} / {len(a['new_DEV_ids'])}",f"{a['max_reencode_abs']:.3e}"] for a in cache]])
    d.para('每个共用窗口以原contract+history values/timestamps身份重建key；全chunks及文件SHA、归一/池化/revision、原tokenizer与冻结权重state SHA核对。首/中/尾重新编码，单样本/批量/重复/future perturb均按登记atol=rtol=1e-4。')
    d.para('补入DEV ID：2025-09-30T20:00:00Z、2025-12-31T20:00:00Z。它们的历史均在DEV，没有读取未来保留区间。')
    d.para('初版NPZ标识存为object导致pickle禁用读取拒绝。原文件保留，v2只把ID改成Unicode U20；四个float32矩阵逐位不变。读取先验发布hash再加载，不启用pickle。')

    d.start('固定模型矩阵与优化精度','28候选/阶段；另两次预登记确定性重放，不作为新候选')
    d.table([['模型','输入','候选与选择'],['HAR / R1 / R2 / 三随机','3 / 33 / 545维','每家族lambda .001/.01/.1/1，共24'],['B2 LightGBM4.6.0 Gamma','同R1的33列','leaves7/15×minleaf30/60，共4'],['persistence / EWMA','固定历史统计','无需拟合，无额外选择'],['统一验证目标','原始绝对单位QLIKE','lower better；不按AUROC/测试选'],['线性并列 / B2并列','exact tie','较大lambda / 较少leaves再较大minleaf']],[132,133,246])
    d.para('线性L-BFGS-B解析梯度，float64，训练StandardScaler，截距不惩罚，目标TRAIN中位RV缩放。求解success且max|gradient|≤1e-5；训练loss不裁剪，预测统一clip[1e-12,1]并留裁剪计数。')
    d.para('B2内置Gamma log-link与QLIKE数据项一致；feval得到正均值，不重复exp。原生训练label为float32，转换最大绝对误差留档；恢复原单位的train/validation metric使用原始float64目标。树L2和线性L2不宣称语义相同。')
    d.para('B2 learning_rate .05，最多300轮，early-stop30、min_delta0、CPU单线程、deterministic/force_col_wise，固定全部随机种子17，无行列抽样，无missing、bundling；完整参数写入模型text。')

    d.start('DEV验证期选择工件','2026-03：92机会。只用于预演固定选择过程，不判定独立研究增量',wide=True)
    rows=[['模型','维度','所选参数','原始QLIKE']]
    for s in selected:
        choice=f"leaves{s['num_leaves']}/minleaf{s['min_data_in_leaf']}/iter{s['best_iteration']}" if s['family']=='B2' else 'lambda '+str(s['lambda'])
        rows.append([s['family'],s['dimension'],choice,number(s['validation_qlike'],12)])
    d.table(rows,[160,65,255,d.width-480],9)
    d.para('这张表不是第三轮主要终点。不能用负QLIKE的百分比改善描述结果，也不因本表重新选择是否加入Kronos或修改模型网格。全部28候选预测与每候选模型均保留。',9)
    d.para('正式阶段将重新执行同28 train/validation拟合，并逐位比较所有所选模型/选择结果，比较通过之前公共holdout读取入口仍拒绝。',9)

    d.start('B2全部早停轨迹','DEV工程预演：每轮TRAIN/VALID原始QLIKE；虚线为验证最优轮',wide=True)
    d.image(early_stop_plot(events),height=365)
    d.para('四结构均保存模型、全参数与逐轮CSV。所选结构7/30，最优20轮；其训练运行50轮。其他7/60最优24运行54；15/30最优10运行40；15/60最优21运行51。',8.8)

    d.start('共同主要终点与成功门槛','仅独立Q2/Q3有资格用于该结论；目前所有主要指标待运行')
    d.para('事件Ysurprise = 1{log((RV_raw+epsilon)/(同一EWMA预测+epsilon)) > log2}。模型评分S = log((模型RV预测+epsilon)/(同一EWMA预测+epsilon))；是连续风险排序分数，不是概率，不进行测试归一化或重新校准。')
    d.table([['共同主要要求','R2−R1','R2−B2'],['统计量','两季度内ΔAUROC等权','同左'],['最低绝对增量','等权均值≥0.03','等权均值≥0.03'],['配对95%CI','7日主块下限>0','7日主块下限>0'],['方向一致','Q2和Q3各>0','Q2和Q3各>0'],['事件充分性','每季度正/负各≥10','同一事件同一样本'],['最终成功','所有要求共同通过','不可由辅助替换']])
    d.para('不使用6个月pooled AUROC作为共同主要统计。辅助AP、q90高RV排序、三随机、方差loss与校准只帮助解释，不新增成功路径。当前事件比例、效应及置信区间未读取。')

    d.start('配对bootstrap与无效draw处理','季度分层 / 完整UTC决策日 / circular stationary / 固定5000尝试')
    d.table([['设置','锁定值'],['主块 / 敏感性','平均7日 / 3日与14日'],['随机流','20261008 + block_days×1000 + quarter_index'],['配对','所有模型与两比较完全共享日multiplicities'],['重采样单位','完整UTC日，保留全部机会与空日'],['每draw统计','季度内AUROC差，随后Q2/Q3等权'],['单类季度draw','NaN，不用0/0.5，不补抽'],['置信区间','joint-valid draws百分位95%CI'],['无效阈值','任何主要季度或联合无效比例>1%→INCONCLUSIVE']],[130,381])
    d.para('精确日期重复权重AUROC含并列平均；synthetic用sklearn、独立正负pairwise rank与展开日期复核。正式时另用正负比较矩阵复算每个主要模型/季度/块的全部5000draw及所有CI，不只复核一个最优结果。')
    d.para('保存各block draws.csv和日期multiplicities.npz，包含无效尝试。3/14日敏感性不用于重新挑选主块。')

    d.start('结论、风险幅度与停止规则','预注册优先级决定结果；当前不填任何正式状态')
    d.table([['状态','含义'],['INVALIDATED','时间/数据/版本/工程失效或暴露后重试；保留错误，禁止正式重跑'],['INCONCLUSIVE','事件/有效draw不足，CI跨零/宽或季度方向不一致等'],['RANKING_ONLY','两共同终点通过，但R2 pooled Regret>1.10×R1'],['CONFIRMED_RANKING_INCREMENT','两共同终点通过且无该红旗；不证明风险管理安全'],['NONLINEAR_BASELINE_COMPETITIVE','R2胜R1，未证明胜B2；不等于统计等价'],['NOT_SUPPORTED','足够事件时明确非正向CI，或正CI但低于登记增量'],['本轮当前','NOT_EVALUATED_HOLDOUT_SEALED']],[205,306],8.8)
    d.para('绝对RV的风险管理用途与排序分开。红旗阻止逆方差仓位缩放建议；无红旗也不等于可安全投入交易。第三轮结束后停止当前有限实验，不自动启动第四轮、tokenizer改造、微调或新风险头。')

    d.start('训练期阈值与固定校准','所有阈值/分箱由2,369 TRAIN机会及TRAIN预测生成；测试不refit')
    d.table([['封存项','数值/规则'],['TRAIN RV q90',f"{thresholds['train_q90']:.16g}"],['TRAIN RV q99',f"{thresholds['train_q99']:.16g}"],['EWMA历史quartile',', '.join(f'{v:.10g}' for v in thresholds['historical_ewma_quartiles'])],['模型预测 / surprise评分','预测decile / 评分quartile，重复quantile折叠'],['分箱边界','searchsorted right，等号进入右侧；全部样本保留'],['高/极端RV诊断','effectiveRV>TRAIN q90/q99，预测/实际<.5为低估']],[135,376])
    d.para('校准表同时保存观察RV/预测RV比值，以及surprise-score分组的真实事件数/率；RV可靠性图不是事件概率校准。历史EWMA分箱只做描述，不能按见到测试结果筛选状态。')
    d.para('rawQLIKE、正QLIKE Regret、既有floor-epsilon logRV_MSE均保留。另报告additive-epsilon logMSE；同epsilon surprise_MSE与它数值恒等，不能当作另一独立确认。7日滚动采用损失sum/机会count，保留空日，不均分每日均值。')

    diagram=RUN/'diagnostics/development_validation'
    for title,name,note in [
      ('DEV验证月：真实RV与预测','RV_timeline.png','92个验证机会，UTC真实决策时间；单位4h平方log收益，不年化。'),
      ('DEV验证月：surprise风险评分','surprise_score_timeline.png','共同EWMA分母的log风险比；评分不是概率，未进行测试期标准化。'),
      ('DEV验证月：7日QLIKE Regret','rolling_QLIKE_Regret.png','7个UTC日累计loss sum / 机会count；包含所有保留家族。')]:
        d.start(title,'DEV工程诊断，非独立holdout结论；数值底表逐机会可查',wide=True)
        d.image(diagram/name,height=370);d.para(note,9)
        d.para('来源：diagnostics/development_validation/row_metrics_and_timeline.csv 与 calendar_rolling_7day.csv；已按TRAIN固定分箱生成所有表。',8.5)

    d.start('预验收、独立审查与失败保留','原断言和科学规则保留；工程失败不隐去')
    log=(RUN/'preflight/tests/risk03_final_tests.txt').read_text(encoding='utf-8');passed=next((line for line in log.splitlines() if ' passed' in line),'详见最终测试日志')
    d.table([['验收','证据'],['第三轮合成检查',passed.split(',')[0]],['实际DEV核验','2501项PASS；2461标签独立复算，所有所选预测重载'],['预算与确定性','28候选+2重放，60事件，无候选失败'],['原9suite','8直接PASS；E00R当前stage FAIL原样保留，历史配置原断言PASS'],['M0','引用已验收第二轮历史重验，原输出路径保护不改'],['独立审查','guard/data/formal/model/statistics/precision与DEV数值PASS']],[130,381],8.8)
    d.para('保留3类拟合前工程异常：机会RangeIndex误判role overlap；M0 verifier拒绝新run输出路径；NPZ object ID被pickle禁用拒绝。分别以ID索引、引用既有验收证据、仅Unicode metadata v2修复。没有改变科学协议、样本目标、模型网格或数值断言。')
    d.para('正式一旦claim，任何不可修复工程失败或暴露后重试均INVALIDATED。异常终止落盘独立于成功hash校验，使版本冲突本身也能留下不可重试记录。')

    d.start('具体审批包与下一步','Stage 5：等待随后单独批准；本次附件提交不是解封授权')
    d.para('协议：research/configs/frozen_risk_03_v1.yaml\n科学协议SHA256：'+guard.CONFIG_SHA,10,bold=True)
    d.para('最终封存bundle：research/runs/FROZEN_RISK_03_v1/preflight/sealed_bundle.json。审批将绑定该文件SHA与协议/源码/输入/依赖hash；PDF、Markdown、逐页QA及bundle SHA由delivery_manifest记录。')
    d.para('源副本preflight/sealed_source；28模型及全部候选预测preflight/preview；公开来源与wheel/native DLL provenance；训练校准calibration；实际与合成检查preflight；独立审查independent_review。researchstate记录当前PhaseA_awaiting_approval及SEALED。')
    d.heading('收到批准后的唯一执行路径')
    d.para('用户消息记录与协议/bundle绑定→唯一claim→同28正式拟合与已封存preview逐位比对→才准打开test值/下载5m→两个完整季度→全量标签/点指标/全部主要draw和CI独立复核→CONSUMED并出版报告。研究执行错误保留INVALIDATED，不重新尝试；仅出版错误记录PUBLICATION_PENDING，仍保留CONSUMED，不重跑研究。')
    d.heading('需用户随后明确批准的文字')
    d.para(cfg['authority']['approval_phrase'],12,bold=True)
    d.para('依据附件Stage5：“停止运行，等待用户明确授权开启holdout。”本轮已完成可自主完成的解封前工作，Stage6-8未授权、未执行。未push。')
    d.end()
    immutable_json(RUN/'reports/pdf_generation_v2.json',{'status':'PASS','pdf_path':str(PDF.relative_to(ROOT)),'pdf_sha256':guard.sha(PDF),'pages':d.number,
      'protocol_sha256':guard.CONFIG_SHA,'primary_result':'NOT_EVALUATED_HOLDOUT_SEALED','selected_models':selected,
      'render_method':'PyMuPDF fallback; Poppler unavailable in this Windows environment'})
    render(PDF)
    print(json.dumps({'PDF':str(PDF),'pages':d.number,'holdout':'SEALED'},ensure_ascii=False),flush=True)

def render(path):
    fitz=pdfbase.fitz;directory=TMP/'rendered';directory.mkdir(exist_ok=True);doc=fitz.open(path)
    images=[]
    for i,page in enumerate(doc):
        target=directory/f'page_{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(target);images.append(target)
    from PIL import Image,ImageOps,ImageDraw
    width=360;height=500;sheet=Image.new('RGB',(width*4,height*((len(images)+3)//4)),'#eeeeee')
    for i,path in enumerate(images):
        im=Image.open(path);im.thumbnail((width-12,height-28));x=(i%4)*width;y=(i//4)*height
        sheet.paste(im,(x+(width-im.width)//2,y+20));ImageDraw.Draw(sheet).text((x+8,y+4),str(i+1),fill='black')
    sheet.save(TMP/'contact_sheet.png')

def phase_b():
    """Read only already-generated consumed results; never open market sources."""
    guard.verify_authorization()
    terminal=jread('authorization/formal_terminal.json')
    if terminal['state']!='CONSUMED' or jread('independent_review/formal_numeric.json')['status']!='PASS':
        raise ValueError('No accepted consumed formal result for publication')
    result=jread('metrics/formal_result.json');pdfbase.init_fonts()
    path=ROOT/'output/pdf/Kronos_Frozen_Risk_03_Report_v1.pdf'
    if path.exists():raise FileExistsError('Formal PDF already published')
    d=Document(path,'B');d.start('第三轮独立风险实验报告',f"FROZEN_RISK_03_v1 | {result['status']} | holdout CONSUMED")
    d.para('两个完整保留季度按已锁定协议一次执行，无测试反馈调参。主要结论：'+result['status'],13,bold=True)
    rows=[['共同主要比较','Q2 Δ','Q3 Δ','等权Δ','7日块95%CI']]
    primary=result['bootstrap']['7']
    for name,c in primary['comparisons'].items():
        fmt=lambda v:'未定义' if v is None else f'{v:+.6f}'
        rows.append([name,*[fmt(v) for v in c['quarter_deltas']],fmt(c['mean_delta']),f"[{fmt(c['ci_lower'])},{fmt(c['ci_upper'])}]"])
    d.table(rows,size=8)
    d.para('风险幅度红旗：'+str(result['risk_magnitude_red_flag'])+'。排序结论不自动支持收益或安全仓位管理。协议 SHA256：'+guard.CONFIG_SHA)
    d.start('季度事件与完整主指标','没有把pooled六个月AUROC当共同主要统计')
    for q in result['quarter_metrics']:
        d.heading(q['id']+f" | N={q['opportunities']} | positive={q['positive']} | negative={q['negative']}")
        d.table([['模型','AUROC','AP'],*[[f,'未定义' if m['AUROC'] is None else f"{m['AUROC']:.8f}",'未定义' if m['AP'] is None else f"{m['AP']:.8f}"] for f,m in q['models'].items()]],size=8)
    d.start('方差幅度与辅助风险诊断','同epsilon surpriseMSE与additive logMSE不是独立证据')
    d.table([['模型','原始QLIKE','Regret','logRV MSE'],*[[f,*[f'{m[k]:.8f}' for k in ('raw_QLIKE','QLIKE_Regret','logRV_MSE')]] for f,m in result['variance_losses'].items()]],size=8)
    d.para('高RV阈值及所有分箱来自封存TRAIN。完整校准、评分分组实际事件率、低估/极端RV及历史EWMA分箱表在diagnostics/holdout/。')
    d.start('配对统计稳健性与无效draw','固定5000尝试；无类draw记NaN，无补抽')
    d.table([['块长','有效联合draw','无效比例','状态'],*[[b,v['joint_valid_draws'],max(v['invalid_fractions'].values()),v['status']] for b,v in result['bootstrap'].items()]])
    d.para('3/14日为预登记敏感性，7日主规则不变。原始5000 draw表、完整calendar date multiplicities全部保留；独立正负pairwise比较矩阵复核全部主要draw与CI。')
    for title,name in [('独立测试：RV与预测','RV_timeline.png'),('独立测试：surprise评分','surprise_score_timeline.png'),('独立测试：7日Regret','rolling_QLIKE_Regret.png')]:
        d.start(title,'真实UTC时间；Q2/Q3分面；全部模型保留',wide=True);d.image(RUN/'diagnostics/holdout'/name,height=380)
        d.para('数值底表：diagnostics/holdout/row_metrics_and_timeline.csv及calendar_rolling_7day.csv。',8.5)
    d.start('审计、版本与停止决定','全部原始来源、失败尝试与不可重复生命周期保留')
    d.para('正式数据raw/canonical全量一致；全标签49价格/48时间戳与独立逐段收益复算；点指标、每个主要bootstrap draw与CI独立核验；正式train/validation模型与PhaseA封存结果逐位一致。状态CONSUMED，禁止再次运行。')
    d.para('当前checkpoint-specific训练截止仍未证明；相同权重在保留区间之前已公开。普通B2与线性头容量/正则语义不声称完全匹配。')
    d.para('停止本轮有限研究。是否开始新的研究阶段需用户读完正式结果另行决定，不自动进行第四轮、微调或tokenizer改造；未push。')
    d.end();render(path)
    text='# Kronos 第三轮冻结风险实验正式结果\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nHoldout CONSUMED；不自动开始第四轮。\n'
    with (RUN/'reports/Kronos_Frozen_Risk_03_Report_v1.md').open('x',encoding='utf-8') as f:f.write(text)
    immutable_json(RUN/'reports/formal_pdf_generation.json',{'status':'PASS','pdf':str(path.relative_to(ROOT)),'sha256':guard.sha(path),'pages':d.number,'visual_QA_required':True})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['A','B'],default='A');a=p.parse_args()
    (phase_a if a.phase=='A' else phase_b)()
if __name__=='__main__':main()
