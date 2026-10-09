"""Publish a reviewed frozen development result without changing immutable trial inputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baselines'))
from prepare_e00 import ROOT,require,sha,write_json,git
from datetime import datetime, timezone
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import yaml

PERIODS={'WF01':'2025Q2','WF02':'2025Q3','WF03':'2025Q4','WF04':'2026Q1'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])

def finalize(output):
    output=output.resolve(); regpath=ROOT/'research/registry'/f'{output.name}.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    require(reg['status']=='engine_verified_pending_analysis','Wrong completion status')
    analysis=json.loads((output/'analysis_report.json').read_text(encoding='utf-8'))
    graphs=json.loads((output/'presentation/report.json').read_text(encoding='utf-8'))
    require(analysis['status']=='completed_exploratory_development_analysis' and graphs['status']=='passed','Analysis/charts incomplete')
    require((output/'visual_review.json').exists(),'Main-agent visual review required')
    qa=json.loads((output/'visual_review.json').read_text(encoding='utf-8'))
    require(qa['status']=='passed','Visual review failed')
    for name in ['root_head_review.json','root_cache_recompute_review.json','encoding_acceptance.json']:
        require(json.loads((output/name).read_text(encoding='utf-8'))['status']=='passed','Missing engineering acceptance')
    for category in ['preparation_hashes','engine_hashes']:
        for rel,digest in reg['artifacts'][category].items():require(sha(output/rel)==digest,'Immutable artifact changed '+rel)
    require(len(reg['engine_attempts'])==128 and all(a['status']=='verified' for a in reg['engine_attempts']),'Incomplete official exports')
    report_path=ROOT/'research/initialization/FROZEN_results.md'
    require(not report_path.exists(),'Do not overwrite result document')
    paired=pd.read_csv(output/'paired_portfolio_metrics.csv')
    metrics=pd.read_csv(output/'all_portfolio_metrics.csv')
    prediction=pd.read_csv(output/'paired_prediction_metrics.csv')
    matched=pd.read_csv(output/'matched_random_summary.csv')
    reference=pd.read_csv(output/'reference_gate_metrics.csv')
    boot=pd.read_csv(output/'paired_block_bootstrap.csv')
    heads=json.loads((output/'selected_heads.json').read_text(encoding='utf-8'))
    pre_heads=[h for h in heads if h['feature_set']=='ridge_pretrained']
    info_pass=analysis['information_screen']['passed']
    text=['# 第一项冻结历史表征实验：开发期结果',
          f'实验 `{output.name}`，2026-10-08。E01R 预训练冻结历史表征与 E00R 普通19项、E02R 三个随机冻结骨干同类 Ridge 的开发期比較已完成。',
          f'工程验收通过；信息筛选 {"通过" if info_pass else "未通过"}，50%信息门控经济筛选 {"通过" if analysis["economic_screens"]["rank50"]["passed"] else "未通过"}，经济门控筛选 {"通过" if analysis["economic_screens"]["economic"]["passed"] else "未通过"}。登记结论为 `{analysis["conclusion"]}`。',
          '本次完成不等于盈利或实盘许可。开发季度此前已见，checkpoint训练截止未知；最终holdout的标签、分类比例及业绩未评估。',
          '## 共用设计与工程验收',
          '用户在冻结结果产生前接受最低增量口径。目标为官方7bps基础成本参考净收益×10000（bps），训练折StandardScaler，Ridge alpha=n_train×lambda；四候选[0.001,0.01,0.1,1]、SVD、验证MSE选择，同分取更大lambda。50%验证分位数为主门控，25%/75%为预定敏感性；经济门控严格预测净收益>0。压力成本复用同一预测与动作。',
          '每机会单独使用256根已完成历史OHLCVA，真实amount与BTC单位volume。按窗口mean/std(ddof=0)+1e-5标准化并clip至±5，固定tokenizer encode及decode_s1最后context。骨干和tokenizer未训练或微调，不生成未来路径。',
          '四组缓存各2,460×512 float32，组合输入531项。64次新Ridge拟合、16个选中头、4,364条开发测试预测、64份无未来结果字段sidecar、128份官方Freqtrade导出均验收。旧E00/E00R及13项消融和无信息参与对照复用已核验工件，没有重跑旧基线或按收益删减。',
          '批量与单条token完全一致，重复计算差0；四组缓存首/中/末独立重算最大差均<4×10⁻⁶（注册容差1e-4）。缺口、重复、未确认、可用时间、真实amount缺失与ID/边界错误均拒绝。四骨干状态hash不同、tokenizer相同，随机初始化17/29/43全部保存；续跑复用20块且manifest不变。',
          '## 真实时间组合曲线',
          '每季度独立以10,000 USDT重置；实际资金已计入，7/14 bps是每边费用代理，非成交价格冲击模拟。曲线仅反映官方已平仓权益，不认证小时持仓盯市风险。',
          f'![50%主信息门控基础成本](../runs/{output.name}/presentation/rank50_equity_base.png)',
          f'![经济门控基础成本](../runs/{output.name}/presentation/economic_equity_base.png)']
    for gate,title in [('rank50','50%信息门控'),('economic','经济门控')]:
        rows=[]
        for fold,period in PERIODS.items():
            base=paired.loc[paired.fold_id.eq(fold)&paired.cost.eq('base')&paired.gate.eq(gate)].iloc[0]
            stress=paired.loc[paired.fold_id.eq(fold)&paired.cost.eq('stress')&paired.gate.eq(gate)].iloc[0]
            n=reference.loc[reference.fold_id.eq(fold)&reference.cost.eq('base')&reference['mode'].eq('ridge_pretrained_'+gate),'opportunities'].iloc[0]
            rows.append([period,f'{int(base.E01R_trades)}/{int(n)}',f'{base.E01R_return_pct:+.2f}%',f'{base.E00R_return_pct:+.2f}%',
                         f'{base.increment_vs_ordinary_pp:+.2f}',f'{base.E02R_seed_mean_return_pct:+.2f}%',f'{stress.E01R_return_pct:+.2f}%'])
        text += ['### '+title,table(['测试季度','预训练笔数/机会','预训练7bps','普通19项7bps','增量(百分点)','随机骨干均值7bps','预训练14bps'],rows)]
        b=paired.loc[paired.cost.eq('base')&paired.gate.eq(gate)]
        s=paired.loc[paired.cost.eq('stress')&paired.gate.eq(gate)]
        text += [f'相对普通19项的基础季度收益中位增量为 {b.increment_vs_ordinary_pp.median():+.2f} 个百分点，{int((b.increment_vs_ordinary_pp>0).sum())}/4折为正；{int((b.E01R_return_pct>0).sum())}/4季度优于现金。压力成本季度收益中位数 {s.E01R_return_pct.median():+.2f}%。主信息门控和经济门控分别判断，不以较低暴露或较少亏损直接宣称预训练信息。']
    text += ['## 预测与匹配参与数量后的信息',
             table(['季度','预训练MSE(bps²)','普通19项MSE','随机骨干种子均值MSE','相对普通改善(%)','相对随机改善(%)'],
                   [[PERIODS[r.fold_id],f'{r.E01R_MSE_bps_squared:.2f}',f'{r.E00R_MSE_bps_squared:.2f}',f'{r.E02R_seed_mean_MSE_bps_squared:.2f}',
                     f'{100*r.relative_MSE_improvement_vs_ordinary:+.2f}',f'{100*r.relative_MSE_improvement_vs_seed_mean:+.2f}'] for r in prediction.itertuples()]),
             f'MSE相对普通19项的中位改善为 {100*prediction.relative_MSE_improvement_vs_ordinary.median():+.2f}%，相对三个随机骨干MSE均值为 {100*prediction.relative_MSE_improvement_vs_seed_mean.median():+.2f}%。随机汇总是全部种子的损失均值，没有挑选最优种子，也没有把平均分数作为另一个可执行组合。']
    sel=matched.loc[matched.cost.eq('base')&matched['mode'].eq('ridge_pretrained_rank50')].sort_values('fold_id')
    text += [table(['季度','预训练所选均值净bps','匹配随机精确期望bps','筛选增量bps','随机抽样范围(非CI)'],
                   [[PERIODS[r.fold_id],f'{r.selected_mean_bps:+.2f}',f'{r.exact_null_mean_bps:+.2f}',f'{r.exact_selection_lift_bps:+.2f}',f'[{r.random_p025_bps:+.2f},{r.random_p975_bps:+.2f}]'] for r in sel.itertuples()]),
             f'在UTC入场月份×固定方向内匹配相同参与数量，所选参考交易净收益增量中位数为 {sel.exact_selection_lift_bps.median():+.2f} bps，{int((sel.exact_selection_lift_bps>0).sum())}/4折为正。随机抽样范围不是组合收益置信区间；精确分层期望用于点估计。']
    agg=boot.loc[boot.fold_id.eq('EQUAL_FOLD_MEAN')&boot.cost.eq('base')&boot.mean_block_days.eq(7)&boot.comparison.isin(['rank50_vs_ordinary','rank50_vs_mean_random_backbone','rank50_vs_matched_expected_counts'])]
    text += ['### 配对时间块不确定性',table(['50%主门控比较','四折等权日均参考差bps','95%区间'],
                  [[r.comparison,f'{r.point_difference:+.3f}',f'[{r.lower_95:+.3f},{r.upper_95:+.3f}]'] for r in agg.itertuples()]),
             '按完整UTC日历汇总配对独立参考交易效用，折内stationary bootstrap 5,000次，主平均块长7天，3/14天敏感性全部保存；四折独立重采样后等权。约每折13个有效7天块，依赖折内近似平稳及折间依赖可忽略。以上是独立参考机会效用区间，不是复合Freqtrade组合收益的bootstrap。预测损失的配对区间另存底表。',
             '### 预先声明筛选结果',table(['信息检查','结果'],[[k,'通过' if v else '未通过'] for k,v in analysis['information_screen']['checks'].items()]),
             '经济筛选还需至少3/4季度优于现金、相对普通模型中位增量≥1个百分点、压力成本中位收益与增量均为正、闭仓回撤中位变化不更差。两类门控完整检查存于analysis_report，不沿用“现金也能通过”的旧相对动量筛选作为充分成功条件。']
    pm=pd.read_csv(output/'prediction_metrics.csv')
    tail=pm.loc[pm.feature_variant.eq('ridge_pretrained')].sort_values('fold_id')
    text += ['## 全部对照、集中度与边界',
             '各随机初始化的每折预测、rank25/50/75及经济门控、两种成本全部保留。当前只有一个共同读出器和预登记候选；没有因开发结果追加特征、种子、阈值或更大预测头。',
             '预训练预测误差中，最差5%机会占平方误差的比例为 '+ '／'.join(f'{100*x:.1f}%' for x in tail.top_5pct_squared_error_share) +'（依次2025Q2/Q3/Q4、2026Q1）。逐组合最大盈亏、正盈利集中度、官方成交名义换手与实际资金另有CSV；不把高维表征带来的拟合能力或收益集中归因于预训练。',
             '开发结果已见且checkpoint训练截止未知，不能称严格事前样本外。E02R保留预训练tokenizer，不能声称去除了全部预训练信息。数据证据仅证明已公布官方数据的一致性与覆盖；历史事前可用性仍是保守研究假设。',
             '原始响应格式核查中曾显示最新原始OHLCV页面；该显示未进入特征、拟合、门控或结果选择，也未查看holdout净收益标签、分类比例或业绩。本次holdout未编码或评估。',
             '## 可追溯入口与下一步',
             f'- [共用锁定协议](../configs/frozen_comparison_v1.yaml)、[阶段转换](frozen_state_transition_v1.json)、[实验登记](../registry/{output.name}.json)。',
             f'- [编码验收](../runs/{output.name}/encoding_acceptance.json)、[缓存独立重算](../runs/{output.name}/root_cache_recompute_review.json)、[下游头独立复算](../runs/{output.name}/root_head_review.json)、[官方回放验收](../runs/{output.name}/engine_report.json)。',
             f'- [配对预测](../runs/{output.name}/paired_prediction_metrics.csv)、[配对组合](../runs/{output.name}/paired_portfolio_metrics.csv)、[匹配随机](../runs/{output.name}/matched_random_summary.csv)、[时间块](../runs/{output.name}/paired_block_bootstrap.csv)、[集中度与换手](../runs/{output.name}/portfolio_concentration_turnover.csv)。',
             f'- [图表核验](../runs/{output.name}/presentation/report.json)、[全部官方曲线CSV](../runs/{output.name}/presentation/official_curves.csv)、[分析报告](../runs/{output.name}/analysis_report.json)。PNG/SVG包含两种成本及全部预定敏感性。',
             '本次授权范围已完成。下一步先审阅已锁定的负结果或证据不足原因，再另行确定研究范围；不自动解封最终holdout、扩头、解冻或进入实盘。']
    report_path.write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    shutil.copy2(report_path,output/'results.md')
    reg['status']='completed_exploratory_development'
    reg['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    reg['result']=analysis
    reg['artifacts']['summary_document_sha256']=sha(report_path)
    reg['artifacts']['summary_document']=str(report_path)
    reg['provenance']['final_source_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'research/frozen').glob('*.py')}
    reg['artifacts']['final_hashes']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.suffix!='.log'}
    before=ROOT/'research/configs/initial_experiment.yaml'; postdir=output/'provenance/completion_state'; postdir.mkdir()
    shutil.copy2(before,postdir/'before_config.yaml')
    config=yaml.safe_load(before.read_text(encoding='utf-8'))
    config['status']='frozen_development_completed_holdout_sealed'
    config['evaluation_readiness'].update(next_research_action='review_frozen_development_result_before_new_scope',
        frozen_development_results='research/initialization/FROZEN_results.md',
        frozen_information_screen_passed=bool(info_pass),
        frozen_economic_information_gate_passed=bool(analysis['economic_screens']['rank50']['passed']),
        frozen_economic_gate_passed=bool(analysis['economic_screens']['economic']['passed']),
        current_task_stop='frozen_development_complete_before_holdout_or_backbone_training')
    for key in ['E01R','E02R']:
        config['experiments'].append({'id':key,'status':'completed_exploratory_development','registry':str(regpath.relative_to(ROOT)).replace('\\','/'),
                                     'results':'research/initialization/FROZEN_results.md','head':'Ridge','final_holdout_evaluated':False})
    for entry in config['experiments']:
        if entry['id'] in ['E01','E02']:entry['status']='superseded_by_same_head_Ridge_frozen_development'
    before.write_text('# Development frozen representation experiment completed; final holdout sealed.\n'+yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    shutil.copy2(before,postdir/'after_config.yaml')
    p=ROOT/'research/initialization/TODO.md'
    t=p.read_text(encoding='utf-8').replace('开发期冻结编码、64次新Ridge拟合与128份官方回放待完成。','开发期冻结编码、64次新Ridge拟合与128份官方回放已验收，结果见 [冻结结果](FROZEN_results.md)。')
    t=t.replace('- [ ] **E01R／E02R开发实验**','- [x] **E01R／E02R开发实验**').replace('缓存、因果验收、读出器、官方回放与结果待完成。','缓存、因果验收、读出器、官方回放、配对结果与图表均完成，信息与两类经济筛选状态见 [结果](FROZEN_results.md)。')
    t+='\n下一步：审阅冻结开发结果后再确定新增研究范围。本轮不自动开始新试验、解冻训练或最终holdout评估。\n'
    p.write_text(t,encoding='utf-8')
    p=ROOT/'research/README.md'; t=p.read_text(encoding='utf-8')
    t=t.replace('当前授权开发期冻结历史编码、下游 Ridge 与官方回放，最终 holdout 仍封存。','开发期冻结历史编码、下游 Ridge 与官方回放已完成，结果见 [冻结实验报告](initialization/FROZEN_results.md)，最终 holdout 仍封存。')
    p.write_text(t,encoding='utf-8')
    transition={'at_utc':reg['completed_at_utc'],'note':'Post-result status update; immutable trial configurations remain in provenance',
                'before_config_sha256':sha(postdir/'before_config.yaml'),'after_config_sha256':sha(before),
                'final_holdout_evaluated':False,'results_sha256':sha(report_path)}
    write_json(postdir/'transition.json',transition)
    reg['completion_state_transition']=transition
    # Include the completion-state evidence, retaining the previously saved trial inputs.
    reg['artifacts']['final_hashes'].update({p.relative_to(output).as_posix():sha(p) for p in postdir.iterdir()})
    write_json(regpath,reg)
    print(json.dumps({'status':reg['status'],'conclusion':analysis['conclusion'],'report':str(report_path),'holdout_evaluated':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True,type=Path)
    finalize(parser.parse_args().output_dir)
