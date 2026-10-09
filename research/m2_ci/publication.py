"""Assemble completed, independently reviewed M2 development documents."""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'research/runs/M2_CONDITIONAL_INCREMENT_01'
NAMES={'NO_DEVELOPMENT_INCREMENT':'停止当前 R2 输出融合路线，优先准备 B5 的独立复现。',
       'NON_SPECIFIC_ENSEMBLE_GAIN':'优先选择廉价集成；当前证据不足以把 Kronos 作为必需组件。',
       'CANDIDATE_INCREMENT_NOT_CONFIRMED':'保留有限候选，仅准备全新前瞻验证；尚未独立确认。',
       'INCONCLUSIVE':'暂停当前 Kronos 条件融合的复杂度扩展，优先准备固定 B5 的独立复现；当前未达到实际增量门槛。'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(p,float_precision='round_trip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v):
    if v is None:return 'NA'
    if isinstance(v,(bool,np.bool_)):return '是' if v else '否'
    if isinstance(v,(float,np.floating)):
        if not np.isfinite(v):return 'NA'
        return f'{v:.6g}'
    return str(v).replace('|','/').replace('\n',' ')
def table(df):
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def write(p,text):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:f.write(text.strip()+'\n')

def publish():
    s=read(RUN/'metrics/summary.json'); outcome=s['status']; r=s['screens']['R2']
    audit=read(RUN/'independent_review/numeric_review.json')
    if audit['status']!='PASS':raise ValueError('Independent numerical PASS required')
    consumed=['summary.json','pooled_metrics.csv','point_metrics.csv','paired_intervals.csv','tail_metrics.csv',
        'state_metrics.csv','gain_concentration.csv','resource_benchmark.csv','future_sample_scenarios.csv',
        'future_direct_R2_B5_scenarios.csv','future_observed_gain_scenarios.csv','future_observed_gain_scenarios.json','FIT_fixed_thresholds.json']
    protected=[RUN/'metrics'/name for name in consumed]+[RUN/'models/candidate_audit.json',
        RUN/'fusion/all_weight_candidates.csv',ROOT/'research/m2_ci/config.yaml']
    protected.extend(p for p in (RUN/'models').rglob('*') if p.is_file())
    for path in protected:
        if audit['source_and_artifact_sha256'].get(path.relative_to(ROOT).as_posix())!=sha(path):
            raise ValueError('Publication consumer changed after independent review: '+str(path))
    figure=read(RUN/'figures/v2/manifest.json')
    if figure['status']!='PASS':raise ValueError('Completed real figures required')
    for path,digest in {**figure['sources_sha256'],**figure['files_sha256']}.items():
        if sha(ROOT/path)!=digest:raise ValueError('Figure/source changed after chart generation: '+path)
    config=yaml.safe_load((ROOT/'research/m2_ci/config.yaml').read_text(encoding='utf-8'))
    pool=csv(RUN/'metrics/pooled_metrics.csv'); points=csv(RUN/'metrics/point_metrics.csv')
    intervals=csv(RUN/'metrics/paired_intervals.csv'); candidates=csv(RUN/'fusion/all_weight_candidates.csv')
    tail=csv(RUN/'metrics/tail_metrics.csv'); states=csv(RUN/'metrics/state_metrics.csv')
    concentration=csv(RUN/'metrics/gain_concentration.csv'); bench=csv(RUN/'metrics/resource_benchmark.csv')
    fits=read(RUN/'models/candidate_audit.json'); resources=pd.DataFrame(fits['resources'])
    planning=csv(RUN/'metrics/future_sample_scenarios.csv'); directplanning=csv(RUN/'metrics/future_direct_R2_B5_scenarios.csv')
    observedplanning=csv(RUN/'metrics/future_observed_gain_scenarios.csv')
    observednote='假设真实增量达到5%时，融合情景的网格最短月数为3；这不表示实测约0.385%的微小增量用3个月就能确认。事后观测效应的额外诊断估得融合名义样本约18,585机会/204个月；效应经过OOF选择、区间跨零，平方根外推也未证明平稳，这不是等待建议或正式样本承诺。当前优先B5独立复现，不为追逐微小融合增量持续等待或扩大搜索。'
    thresholds=read(RUN/'metrics/FIT_fixed_thresholds.json'); direct=intervals[(intervals.period=='OOF')&(intervals.block_days==7)&(intervals.contrast=='R2_minus_B5')].iloc[0]
    rec=NAMES[outcome]
    screens=pd.DataFrame([{'组合':'B5+'+k,'alpha':v['alpha'],'OOF ΔQLIKE':v['mean_gain'],
        'Regret改善%':100*v['relative_Regret_improvement'],'改善折数':v['positive_folds'],
        '7日CI下限':v['ci_lower_7d'],'7日CI上限':v['ci_upper_7d'],'原VALID Δ':v['VALID_gain'],
        '主筛选':v['primary_screen_pass']} for k,v in s['screens'].items()])
    folds=pd.DataFrame([{'折':f['id'],'FIT起点':config['roles']['train_start'][:10],
       'FIT右边界':f['fit_end'][:10],'内层VALID右边界/OOF起点':f['evaluation_start'][:10],
       'OOF右边界':f['evaluation_end'][:10],'FIT N':f['expected_counts'][0],
       '内层VALID N':f['expected_counts'][1],'OOF N':f['expected_counts'][2]} for f in config['roles']['folds']])
    chosen=[]
    for f in config['roles']['folds']:
        selected=read(RUN/'models'/f['id']/'selected_models.json')['selected']
        heads={name:read(selected[name]) for name in ['har','R1','R2','B2']}
        chosen.append({'折':f['id'],'HAR lambda':heads['har']['lambda'],'R1 lambda':heads['R1']['lambda'],
            'R2 lambda':heads['R2']['lambda'],'B2 leaves':heads['B2']['num_leaves'],
            'B2 minleaf':heads['B2']['min_data_in_leaf'],'B2 best_iteration':heads['B2']['best_iteration'],
            **{'B5 epoch'+str(seed):read(RUN/'models'/f['id']/f'B5_s{seed}.json')['metadata']['best_epoch'] for seed in [17,29,43]}})
    foldgain=pd.DataFrame([{'时期':f,'B5+R2 Δ':s['screens']['R2']['fold_gains'][i] if i<5 else s['screens']['R2']['VALID_gain'],
        'B5+R1 Δ':s['screens']['R1']['fold_gains'][i] if i<5 else s['screens']['R1']['VALID_gain'],
        'B5+B2 Δ':s['screens']['B2']['fold_gains'][i] if i<5 else s['screens']['B2']['VALID_gain']} for i,f in enumerate([x['id'] for x in config['roles']['folds']]+['VALID'])])
    key=['B5','R2','R1','B2','har','ewma','persistence','constant_RV','mix_R2','mix_R1','mix_B2']
    metriccols=['fold_id','family','N','positive','raw_QLIKE','QLIKE_Regret','logRV_MSE','surprise_AUROC','surprise_AP','absolute_q90_AUROC','absolute_q90_AP']
    pooledcols=['period','family','N','raw_QLIKE','QLIKE_Regret','logRV_MSE','surprise_AUROC','surprise_AP']
    ci=intervals[intervals.period.isin(['OOF','VALID'])][['period','contrast','block_days','point_gain','ci_lower','ci_upper','valid_draws']]
    tailkey=tail[tail.family.isin(['B5','R2','mix_R2','mix_R1','mix_B2','constant_RV'])][['fold_id','family','subset','FIT_threshold','N','under_count','under_fraction','mean_prediction','mean_actual']]
    statekey=states[states.family.isin(['B5','R2','mix_R2','mix_R1','mix_B2'])][['fold_id','family','EWMA_bin','N','QLIKE_Regret','logRV_MSE']]
    costcols=['name','batch_windows','batch_mean_seconds','per_window_mean_seconds','trainable_parameters_at_fit','frozen_parameters','B2_nodes','B2_leaves','checkpoint_bytes','cuda_peak_allocated_bytes','rss_sampled_max_bytes']
    costcols=[x for x in costcols if x in bench.columns]
    fsum=resources.groupby('family',sort=False).agg(fits=('candidate','count'),fit_seconds=('elapsed_seconds','sum'),checkpoint_bytes=('checkpoint_bytes','sum')).reset_index()
    theta=.05*r['base_OOF_Regret']
    power_note=('所选 alpha=0，融合与 B5 完全相同，零方差不能给出有意义的 MDE 或所需样本；该融合规划标为不适用。' if r['alpha']==0 else
        '融合规划使用所选 alpha 的开发期配对块方差，受同一 OOF 选择影响，只是条件性情景。')
    report=f'''# Kronos M2-CI 条件增量开发研究报告

实验：`M2_CONDITIONAL_INCREMENT_01`。判定：**`{outcome}`**。

{rec} 现有 R2 输出按 OOF 选择的融合权重为 **{fmt(r['alpha'])}**，相对 B5 的机会加权平均 QLIKE 增量为 **{fmt(r['mean_gain'])}**，对应 Regret 改善 **{fmt(100*r['relative_Regret_improvement'])}%**；主 7 日条件区间 **[{fmt(r['ci_lower_7d'])}, {fmt(r['ci_upper_7d'])}]**，改善 **{r['positive_folds']}/5** 折，原 VALID 增量 **{fmt(r['VALID_gain'])}**。

这是开发研究结果。历史 B5 架构曾由 2026 年 3 月 VALID 选择，Kronos checkpoint 训练截止尚未证明；本轮 OOF 不能解释为完整嵌套、严格事前独立验证。已消费的 **2026年Q2/Q3** 没有进入本轮数值分析、权重选择或晋级；本轮2025年季度评估仍属于原TRAIN开发范围。没有新的正式独立测试，也没有交易收益结论。

## 1. 问题与原证据边界

H1：给定 B5，现有 R2 的正值原单位 RV 输出是否互补？H2：这种改善是否超过同预算的 B5+R1/B2？H3：B5 在 3.1 的事后优势是否能在今后真正独立数据复现？本轮只回答 H1/H2 的开发筛选；H3 直接对照保留为开发诊断及未来草案。

3.0 的正式结论针对 R2 相对 R1/B2 的 surprise 排序；3.1 加入 B5 后为事后探索，不能升级为独立确认。排序、平均风险预测、绝对高 RV 识别、极端低估保护、可交易经济价值分别判断。融合失败也只约束现有输出，不能证明所有 Kronos latent 都没有潜在信息。

来源审计原 3.0 最终清单561项、3.1清单226项全部SHA一致。DEV输入通过原3.0正式manifest中sealed_bundle_sha256绑定的嵌套bundle追溯，8项源链核验通过。原工件、旧TODO和CONSUMED终态保留。

## 2. 因果时序、分折与实现

原开发数据2461条：TRAIN2369条（2024-01-01至2026-03-01右不含），原VALID92条（3月至4月右不含）。五个OOF段共1267条。所有范围为[start,end)，FIT、内层验证及评估分别要求label_end和labelable_at严格小于角色右边界；不只检查决策时间。每折选择之后不在FIT+内层验证上重拟合。

{table(folds)}

每天UTC04/12/20机会；04:01决策、05:00入场边界、09:00标签结束，labelable_at再加60秒。256根已完成且当时可用1h OHLCVA；真实成交额保留。4h RV由入场5m open及随后48根5m close共49价格的48自然对数收益平方和构成，单位为4h squared natural-log return，不年化。全部2461标签逐一复算一致；新标签没有另造。

可用时间使用冻结契约的60秒延迟假设。官方历史数据覆盖与一致性审计不证明真实历史发布延迟、事前checkpoint可用性或交易所内部账本。

预测输入NPZ仅有ID、窗口、33项普通特征、512冻结hidden、HAR3项、R0基线2项；未来RV保存在独立labels/development.csv。metadata里的label_end/labelable_at仅做时钟筛选，未传入预测器。标签、profit、funding及事件没有进入推理函数。

R1/B2为33普通特征，R2为同33+512；R0 persistence/EWMA不进入拟合特征。Kronos每窗口6项OHLCVA以float64均值和总体标准差+1e-5规范化、clip±5后float32；tokenizer.encode(half=True)，冻结decode_s1最后context为512维。复用原已核验DEV-only缓存，没有重读混合TEST数组。B5输入256×11：同6市场通道及5个已知日历通道，日历固定除以59/23/6/31/12；width32、7个因果残差卷积层，dilation1至64，感受野257；最后hidden与FIT标准化33特征拼接后标量头。

HAR/R1/R2各4个lambda(.001/.01/.1/1)，B2四结构(7/15 leaves ×30/60 minleaf)，仅按内层原单位QLIKE选择。TCN固定width32/wd.001，种子17/29/43，每种子最多120epoch、patience15、batch64、lr.001，内层QLIKE选择最早最佳epoch。原单位预测截断[1e-12,1]，标签有效值max(RV,1e-12)。所有标准化及标签median缩放按本折FIT重建；三种子聚合为原单位RV算术均值，不能对score或指标先平均。

HAR/R1/R2读出沿用带L2惩罚的log-link QLIKE标量模型：p=FIT_median×exp(标准化特征的线性score)，L-BFGS-B拟合；不是首轮收益实验的MSE Ridge。B2使用原Gamma LightGBM，B5使用原QLIKE目标的监督TCN。本轮没有更换损失寻找有利口径。

实际拟合 **{fits['fits']}** 次，TCN **{fits['actual_B5_epochs']}** epoch，B2 **{fits['actual_B2_iterations']}** iteration。没有扩结构、重新拟合、解冻或自动重试。各候选保存模型、内层预测、训练历史、选择和实际资源；OOF/VALID完整重载预测通过独立复核。原VALID复用既有完整TRAIN冻结工件，新增拟合0次。

{table(pd.DataFrame(chosen))}

具体模型路径及SHA见[selected_fold_map.json](../models/selected_fold_map.json)，原VALID模型身份见[validation_audit.json](../predictions/validation_audit.json)。预测表逐条包含机会ID、决策/输入窗口/逻辑可用模型时点和fold_id；模型路径由fold映射解引用。历史logical_ready_at不冒充现实训练墙钟日期。

## 3. 有限融合、主效应和判定

所有组合均为 `(1-alpha)*B5_RV + alpha*other_RV`，alpha固定[0,.1,.25,.5,.75,1]，每组合6项。按1267条OOF机会加权rawQLIKE最低项选择，精确同分选较小alpha；三组总18项全部保留。selected_weights.json先冻结，再生成/检查原VALID，不以VALID重新选择。

主增量Δ=L(B5)−L(mix)，正值有利融合。rawQLIKE=log(p)+y/p；Regret=y/p−log(y/p)−1。因此同标签/epsilon下配对差相同，不能作为两项独立成功证据。rawQLIKE可为负，不报告其百分比改善；这里百分比只用平均Regret作分母。

开发筛选需同时满足Regret改善≥5%、7日配对区间下限>0、至少4/5折正向、原VALID正向，并对选定R1mix和B2mix分别有正7日区间下限。未证明特异性不等于证明模型等效。

{table(screens)}

各时期配对点值：

{table(foldgain)}

全部预登记权重：

{table(candidates)}

{rec} 不扩大模型搜索以寻求正结果。

INCONCLUSIVE表示正负方向尚不确定，不表示接近晋级。R2mix的7日区间上限约为B5平均Regret的1.52%，仍低于5%研究门槛；该区间本身又是选择条件下的开发区间。保留预定分类，不根据已见结果另改判定规则。

## 4. 不确定性、时间分布与集中性

每个OOF折独立循环stationary UTC日块，原VALID独立；完整日历包括空日。7日主块、3/14日敏感性各5000次，每折所有32模型及所有候选使用完全相同日权重。OOF合并每抽样总损失分子/总机会数，未等权平均五折。seed固定20261009+block×1000+fold_index。有效抽样5000，没有重抽无效结果。

所选alpha不在抽样中重选，同一OOF曾用于选择权重；这些区间是选择条件下的开发区间，可能乐观，不能当作正式确认。折/VALID完整区间及全部抽样保存在paired_intervals.csv、bootstrap/。

{table(ci)}

![融合每日配对损失](../figures/v2/gain_sum.png)

![每折独立累计配对损失](../figures/v2/cumulative_gain_sum.png)

以上为损失差，不是资金或权益。每折单独从0累计；不用伪连续全年曲线。集中性指标中的max_day_signed_share用有符号总增量作分母，负总量或接近0时不能解释为通常的百分比占比；删最佳正日仅诊断，不重训或重选。

{table(concentration[concentration.contrast.isin(['B5_minus_mix_R2','B5_minus_mix_R1','B5_minus_mix_B2'])])}

R2mix在WF04和WF05的最佳正日分别贡献该折有符号总增量约66.0%和57.1%；删除各自最佳正日后平均差仍为正，但原VALID删除最佳正日后转负。时间局部信号不能替代跨折与区间门槛。

## 5. 完整模型对照与分母诊断

下面包含11个基模型/种子、3个选定融合及18个全部候选；折内的完整指标底表为point_metrics.csv。OOF pooled ranking混合了各折不同模型和FIT常数，仅作描述，主要解释折内排序。所有损失用同机会/标签。

{table(pool[pooledcols])}

B5的OOF Regret约0.337347，R2约0.441894；加入10% R2后为0.336049，仅小幅下降。logRV MSE反而由0.568310升至0.572627，没有同时改善全部预测指标。B5在各OOF折的surprise AUROC约0.841–0.917；FIT常数经EWMA分母后也有约0.572–0.658的AUROC，但其绝对q90 AUROC均为0.5。这支持保留分母诊断，也说明B5排序不能完全归为常数分母效应。

![主模型风险误差时间线](../figures/v2/rolling_Regret_primary.png)

![历史廉价基线](../figures/v2/rolling_Regret_historical_baselines.png)

![三种子模型](../figures/v2/rolling_Regret_seed_models.png)

全部32族7日机会加权误差曲线底表rolling_Regret_all32.csv，全部alpha时间图alpha_R1/R2/B2保留；每条曲线只在自己的评估期间显示，不展示拟合内成绩。

Surprise事件为log((RV_raw+epsilon)/(sameEWMA+epsilon))>log2，score为log((predRV+epsilon)/(sameEWMA+epsilon))。诊断常数为各折FIT有效RV算术均值，原VALID使用原TRAIN均值。常数本身不提供绝对高RV排序（双类别时AUROC=.5），但除以EWMA后的score与负log(EWMA+epsilon)同序，能产生surprise排序。因此较高surprise AUROC不单独证明复杂网络能精细预测绝对RV；分母诊断不改变事件、不替代QLIKE。

![常数分母诊断](../figures/v2/ranking_denominator_diagnostic.png)

{table(points[points.family.isin(key)][metriccols])}

## 6. 风险状态与极端低估

EWMA四分位边界、q90/q99都只来自对应FIT，原VALID来自原TRAIN；严格大于分位数为尾部。低估定义pred/effectiveRV<.5，N小或0明确保留，空尾部NA，不把稳定平均误差或surprise排序当作极端保护。

![FIT固定尾部低估](../figures/v2/FIT_fixed_tail_underprediction.png)

{table(tailkey)}

各折固定q90尾部中，B5低于实际RV一半的比例约30.0%–100%，R2约50.0%–100%。混合在WF05仅从14/34降为13/34，其他OOF折低估计数不变；q99每折只有0–5个样本，多折全部低估，不能宣称可靠的极端风险保护。

风险状态四分位是历史EWMA状态，非未来RV分组；下面列各组N、Regret及logMSE，完整32族state_metrics.csv保留。

{table(statekey)}

## 7. 资源与工程代价

95个拟合候选实际资源汇总：

{table(fsum)}

共同8个DEV窗口，选定WF05工件，3次预热20次实际推理，CPU线程1及CUDA前后同步。每个完整融合实跑全部路径，不把组件时延相加估算；R2完整路径重新tokenizer+encoder，不用缓存hidden。alpha=0/1仍测完整双路径以揭示潜在组件代价，真正alpha0运行可以直接删去R2路径。B2节点/叶子数不当作神经网络可训练参数；Kronos预训练资源没有在本轮重测或预算配平。

{table(bench[costcols])}

共同8窗口下，B5三种子集成平均批次时延约49.44ms，完整B5+R2约68.38ms，增加约38.3%；对应工件约0.292MB与115.158MB，后者包含28,699,418个冻结参数。当前微小且不稳的增量尚不足以支持额外组件代价。测量只代表本机固定批次推理，不能换算为总训练资源配平。

CUDA峰值为同一基准进程全部常驻B5和Kronos模型下的绝对allocator峰值，不能当作各模型独立驻留峰值比较。RSS是调用前后采样最大WorkingSet，不是连续内存峰值。输入预处理和模型加载不计入时延；完整资源定义与20次原始时延保存在resource_benchmark.json。推理速度不等于全生命周期工程成本。

## 8. 下一阶段与未批准草案

{power_note} 功效规划固定量级δ=5%×开发B5平均Regret={fmt(theta)}。用开发7日paired bootstrap SE、样本数平方根缩放、每天3机会与365.25/12日/月计算双侧5%近似power及80% MDE；这不证明未来平稳、有效独立样本数或实际功效。估计MDE= (z.975+z.8)×SE。无需把Q2/Q3重新当holdout。

融合条件规划：

{table(planning)}

{observednote}

{table(observedplanning)}

独立B5对R2直接比较的同量级情景（δ是开发B5参照量级，未批准为正式H3门槛）：

{table(directplanning)}

本轮直接R2−B5开发QLIKE差={fmt(direct.point_gain)}，7日条件区间[{fmt(direct.ci_lower)},{fmt(direct.ci_upper)}]，正值支持B5。这也没有确认H3。若3–24月网格都达不到80%近似power，报告“网格内不足”，不能视为24月必然够；实需更长区间或重新审批目标效应。新正式起点必须严格晚于将来获批准的协议封存时刻，真实历史仅用于因果256窗口；不得将任何已观测区间包装为新holdout。

草案只提出审查事项与候选规则；未封存正式未来协议、未采集/部署前瞻系统、未自动开始第四次正式冻结实验。经济价值实验仅在未来预测确认后另立方案，不在同一确认测试中搜索交易规则。

## 9. 工程失败、复核与复现

训练前code_review初版及v2 FAIL、第一次prepare读取数值前拒绝均保留。修复涉及目录统一、正式manifest→nested bundle来源绑定、隔离LightGBM版本取证、Windows RSS API及候选生命周期，科学配置和候选集合未更改。data_review_v2过早PASS的遗漏由v3纠正；v3 addendum模板句由独立correction纠正，实际核心源码及全部8审查文件SHA无漂移。未删除失败记录。科学训练完成的95次STARTED/COMPLETED一一对应，无失败科学拟合或自动补训。

训练已经通过既有b5.configure关闭TF32并启用确定性；fresh进程默认设置不同，后续VALID、资源与复核由新增受限execute_locked入口先恢复同样运行设置再调用冻结实现。该入口不提供训练/选权重动作，记录独占claim/success/failure、运行flags和核心封存/权重/journal前后SHA。没有修改212项冻结实现、权重或科学候选；此运行入口与独立审查一并由最终交付绑定。

独立numeric_review状态PASS，检查{audit['check_count']}项；重新载入模型和预测，独立计算损失、排序、角色、候选选择、时间块权重和决策，而非重新训练。15项关键合成测试及未来扰动、repeat/single-batch一致性记录保留。首版图表因图例与脚注重叠验收FAIL并完整保留；本报告展示figures/v2修正版，数值与CSV不变。科学source seal绑定212项；最终delivery_manifest绑定报告、草案、决策、所有模型/数据/图表/失败记录及后处理代码。原researchstate revision15的900条旧证据保持，M2由专属helper追加索引，CONSUMED原终态不变。

复现必须在**新空目录/新登记编号**执行，已有模型、claim、预测和报告均独占创建；以下为保存的原执行顺序，不能在已封存目录重复写入：

```powershell
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.runner prepare
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.preflight tests
# 独立代码审查PASS后才source seal
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.preflight seal
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.runner train-oof
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.seal_outputs
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.statistics select
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.execute_locked validation
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.statistics evaluate
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.execute_locked benchmark
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.planning_supplement
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.execute_locked review
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.charts --output-tag v2
& '.\\.venv\\Scripts\\python.exe' -B -m research.m2_ci.publication publish
```

协议全文[config_preregistered.yaml](../provenance/config_preregistered.yaml)，SHA `{s['protocol_sha256']}`；选权重、训练源封存、模型身份、真实数据和原始来源、依赖锁及实际环境版本均可从[实验目录](../)追溯。原3.0 manifest SHA `e17ed339eb054ea8bd620a52600746f9d3474c0e57266c0db8442e17378b139e`；3.1 SHA `23f4eba2a470ddb484019b64fbe546b34316a62b2439dfe28938c66f82f26b7c`。无push、部署或实盘。
'''
    decision=f'''# M2-CI 一页项目决策

**开发判定：{outcome}。** {rec}

**1. Kronos对B5是否有实际条件增量？** 当前R2输出融合alpha={fmt(r['alpha'])}，OOF平均ΔQLIKE={fmt(r['mean_gain'])}、Regret改善{fmt(100*r['relative_Regret_improvement'])}%；7日条件CI[{fmt(r['ci_lower_7d'])},{fmt(r['ci_upper_7d'])}]，{r['positive_folds']}/5折正向，原VALID Δ={fmt(r['VALID_gain'])}。必须同时通过5%、CI、4/5折、VALID及廉价特异性。判定按预登记规则得到上述状态。开发架构曾用原VALID选择，checkpoint训练截止未证，所选权重CI可能乐观；没有独立确认。

{table(screens)}

**2. 是否足以支持额外复杂度？** {rec} 比较只针对现有RV输出互补性，不证明所有latent缺少信息。实际同8窗口测得完整路径时延与存储见资源表，R2完整路径包含冻结编码器；不存在总预训练预算配平。排序、平均预测、极端保护、经济收益分别判断，当前没有交易alpha结论。

**3. 下一次独立测试验证什么、需要多少数据、何时停止？** 若保留H1候选，则只验证固定B5+R2能否相对B5降低至少候选5%Regret、配对CI下限>0并胜过廉价同构融合；否则优先独立复现固定B5相对R2的预测优势。H3开发直接差={fmt(direct.point_gain)}，CI[{fmt(direct.ci_lower)},{fmt(direct.ci_upper)}]，不能充当确认。

{power_note} 3/6/9/12/18/24月样本情景已保存；**假设真实增量达到5%**时融合80%近似power网格最短月数={fmt(s['planning']['minimum_months_80power_grid'])}（NA表示不足或不适用）。{observednote} H3独立量级规划另见direct_R2_B5情景；依赖未证明的未来平稳，正式阈值和时长仍待审批。测试机会起点必须晚于将来新协议封存，绝不重用CONSUMED Q2/Q3。

若未来融合未优于B5、实际效应小于获批门槛或廉价方案覆盖增量，则停止Kronos条件融合。平均预测通过后才另立经济实验，不扩大结构搜索。未来协议DRAFT未获批准；当前不执行新正式测试、不部署、不实盘、不push。

[完整报告](M2_Conditional_Increment_Development_Report.md) · [未批准未来草案](M2_Conditional_Increment_Future_Holdout_Protocol_DRAFT.md)
'''
    draft=f'''# M2-CI 未来独立验证协议草案

**DRAFT / UNAPPROVED / DO NOT EXECUTE。** 这不是正式封存协议，也不授权第四次正式测试、采集服务、部署或实盘。生成此草案不会改变现有CONSUMED终态。

## 候选问题与进入条件

开发状态{outcome}：{rec} H1只研究既有R2正值RV输出给定B5的互补性，当前锁定开发alpha={fmt(r['alpha'])}。若不满足开发晋级，不把H1安排为默认确认路线；H3优先候选是固定B5与固定R2的独立预测复现。不能通过扩大latent/Adapter/网络寻找新成功路径。

同构廉价对照固定B5、B5+R1/B2/R2，HAR/EWMA/persistence及FIT常数分母诊断保留；融合6权重的开发选择已经结束，未来不调alpha、阈值、校准或模型。未来候选可使用本轮原VALID推理所用的原完整TRAIN冻结模型及预处理，模型路径见validation_audit.json；各未来实际工件SHA必须在后续正式seal明确登记，不能只引用本草案。OOF各折模型用于历史诊断，不自动替代未来固定模型。

## 独立区间与数据契约

正式开始前需用户另行明确授权，并封存完整协议、模型、code、config、候选集、依赖及来源SHA和预运行审计。首个决策机会必须严格晚于**后续正式协议实际封存UTC时间**。具体日期现在不填。已观察2026Q2/Q3及开发TRAIN/VALID均禁止重用为确认holdout。若历史256根小时窗口的事前可用性证据充分，可作为未来输入；否则需先满足256完成小时的暖启动及可用时间。采集与固定前瞻预测系统仅可在后续授权后建设，记录预测时的input/source/model/clock哈希，标签可用后另表追加。

保持BTC-USDT-SWAP、UTC04/12/20、+60秒可用延迟、256×1hOHLCVA、4h RV49价格、标签floor1e-12与prediction[1e-12,1]。每个机会共享同ID/期限/标签；官方真实amount/raw字段/单位/URL/采集UTC/SHA和完整grid审计保留。缺数、未完成、无法验证时钟的机会不能兜底；缺失报告和处理规则必须预登记，不能事后按表现排除。

## 效应、样本量与功效：候选，尚待审批

H1候选主要门槛：相对B5平均QLIKE Regret降低≥5%，paired rawQLIKE Δ>0且主CI下限>0，R2mix相对R1mix/B2mix各pairedCI下限>0；时间子段方向一致性阈值及多重检验家族必须在正式seal明确批准。raw损失百分比不使用，raw/Regret配对差不是两份独立证据。

H3候选主要比较为L(R2)−L(B5)，正值支持B5；实际重要性门槛要结合开发变异、资源和未来基线审批，不把事后Q2/Q3效应倒选为门槛。下面统一的δ={fmt(theta)}只是5%×开发B5Regret的规划量级，不等于未来H3相对R2的5%门槛。

{power_note}

H1条件规划：

{table(planning)}

{observednote}

事后观测效应的诊断补充，不属于正式门槛：

{table(observedplanning)}

H3直接配对量级规划：

{table(directplanning)}

每天3机会与月平均30.4375日只是样本上限近似。用开发7日block bootstrap SE按sqrt(1267/n)外推；双侧5%近似power=Phi(δ/SE−z.975)+Phi(−δ/SE−z.975)，80% MDE近似(z.975+z.8)SE。没有证明不同季节/波动状态/未来分布稳定或观察独立性；不把1267机会当1267独立样本。网格内无80%月份则明确不足，不默认24月够。正式样本截止、最小完整日数/机会数/事件数、流失规则与最大等待期须经后续审批后冻结；不可观察结果后延长。

## 统计、一次性与停止规则

拟保留共享完整UTC日循环stationary bootstrap，7日主块及3/14敏感性，至少5000已登记draw及明确seed，各模型同权重；主统计是机会加权配对loss差，不等权平均任意段。确认阶段只分析固定模型/alpha，不能用未来样本选择。正式的检验家族、区间/显著性/多重性、有效抽样及样本不足INCONCLUSIVE规则需随批准协议封存。

持续采集仅允许盲核时钟、来源、缺失和预测有限性，不打开模型效果/未来标签统计用于中途调参。达预登记终点且完整审计通过后，exclusive一次性claim，记录SEALED→CLAIMED→CONSUMED生命周期；失败/中断保留，重启规则另行审批，不能自动反复试验。正式报告成功、失败、无增量及不足全部留档。

若H1实际效应低于获批门槛、CI不支持、或者廉价同构组合覆盖改善，停止当前Kronos条件融合路线；未通过不扩大容量补救。若H3不复现B5优势，不把3.1事后结果升级，按证据评估是否继续简单风险模型。极端风险低估、排序和经济收益为不同结论；未来平均预测确认通过也只安排另行经济价值实验，不自动交易。

## 尚待正式seal的工件

后续批准才填写：实际开始/结束UTC及sample-stop规则、checkpoint与fixedheads/三seedTCN/Scaler/median/fusion的逐项SHA、完整代码/依赖/native库/配置SHA、新数据manifest与不可变原始来源、fresh prediction-only接口审查、point-in-time及未来扰动、single/batch一致性、共同机会/labelable purge、资源测量、预运行PASS及正式claim。任何条目缺失时不得执行。

当前开发配置SHA `{s['protocol_sha256']}`；开发源与结果由本轮delivery_manifest追溯。本草案状态始终UNAPPROVED，配置future_formal_test_approved=false，保留原holdout=CONSUMED。本轮到开发报告和草案交付为止。
'''
    write(RUN/'reports/M2_Conditional_Increment_Development_Report.md',report)
    write(RUN/'reports/M2_Conditional_Increment_Project_Decision.md',decision)
    write(RUN/'reports/M2_Conditional_Increment_Future_Holdout_Protocol_DRAFT.md',draft)
    print(json.dumps({'status':'REPORTS_WRITTEN','development_result':outcome}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['publish']);a=p.parse_args();publish()
