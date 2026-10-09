"""Publish bounded MLP development evidence and completion state, preserving prior bytes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
import argparse
from datetime import datetime, timezone
import json
import shutil
import pandas as pd
import yaml

def table(frame):
    def value(x):
        if isinstance(x, (int, float)):
            return f'{x:.2f}'
        return str(x)
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join('---' for _ in frame.columns) + ' |\n' + '\n'.join('| ' + ' | '.join(value(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None))

def main(output):
    output = output.resolve(); regpath = ROOT / 'research/registry' / f'{output.name}.json'
    reg = json.loads(regpath.read_text(encoding='utf-8'))
    require(reg['status'] == 'engine_verified_pending_analysis', 'Official replay not accepted')
    report = output / 'results.md'; require(not report.exists(), 'Do not overwrite completed report')
    analysis = json.loads((output / 'analysis_report.json').read_text(encoding='utf-8'))
    chart = json.loads((output / 'presentation/report.json').read_text(encoding='utf-8'))
    heads = json.loads((output / 'head_acceptance.json').read_text(encoding='utf-8'))
    require(heads['status'] == 'passed' and heads['reloaded_models'] == 180, 'Head acceptance incomplete')
    require(chart['status'] == 'passed' and not analysis['final_holdout_inspected'], 'Presentation / scope not accepted')
    for category in ('preparation_hashes','engine_hashes'):
        for rel, digest in reg['artifacts'][category].items():
            require(sha(output / rel) == digest, 'Changed accepted input: ' + rel)
    pair = pd.read_csv(output / 'paired_portfolio_metrics.csv')
    prediction = pd.read_csv(output / 'paired_prediction_metrics.csv')
    matched = pd.read_csv(output / 'matched_random_summary.csv')
    matched = matched.loc[matched.cost.eq('base') & matched['mode'].eq('mlp_pretrained_rank50')]
    info = analysis['information_screen']['passed']
    economic = {k:v['passed'] for k,v in analysis['economic_screens'].items()}
    sections = [f'# 有限小型 MLP 冻结读出：开发期结果\n\n实验 `{output.name}`，2026-10-08。',
        f'工程、官方回放与图表核验完成。信息筛选：**{"通过" if info else "未通过"}**；50%信息门控经济筛选：**{"通过" if economic["rank50"] else "未通过"}**；经济门控筛选：**{"通过" if economic["economic"] else "未通过"}**。登记结论 `{analysis["conclusion"]}`。',
        '## 锁定设计与证据边界\n\n五组为普通19项、普通19项＋预训练context、普通19项＋三个随机骨干context。复用首轮的真实历史窗口缓存，不重编码、训练或微调骨干／tokenizer。',
        '单隐层16个Tanh单元；CPU float32、训练折输入StandardScaler及目标均值／标准差标准化、全批次AdamW固定300轮，学习率0.003。正则化候选0.001／0.01／0.1；每候选头种子17／29／43取预测均值。按验证集集成MSE选择，完全同分取较大正则化，不选择种子或训练轮次。目标仍为官方基础成本净收益×10000 bps。',
        '共180次固定预算拟合、20个选中三头集成、5,455条开发测试集成预测、80份无未来结果sidecar、160份官方导出。19项输入头有337个参数，531项输入头有8,529个参数；同隐层宽度与选择预算不等于参数数目相同。',
        '主信息门控用验证50%分位数，25%／75%为预定敏感性；经济门控预测基础净收益严格>0。14bps复用基础预测与动作，无再次拟合或重复扣成本。三头预测均值是实际执行策略；三个随机骨干政策的损失／收益均值仅是描述性对照，不是另一个执行策略。',
        '开发季度此前已见，本轮为探索性历史研究。checkpoint训练截止未知，不能宣称严格事前样本外。最终holdout未编码、拟合或评估标签、分类比例与业绩。原Ridge/logistic、13项消融、固定规则与无信息参与对照保留，未按开发收益换主比较。',
        '## 真实时间组合曲线\n\n各季度独立重置10,000 USDT，含实际资金；7／14bps为每边费用代理。仅反映官方已平仓权益，不是小时盯市风险。',
        '![50%信息门控基础成本](presentation/rank50_equity_base.png)\n\n![经济门控基础成本](presentation/economic_equity_base.png)\n\n![MLP与原Ridge对照](presentation/head_comparison_equity_base.png)']
    loss = json.loads((output/'mlp_loss_diagnostics.json').read_text(encoding='utf-8'))
    require(loss['new_fits'] == 0 and not loss['parameters_changed'] and not loss['holdout_inspected'], 'Loss diagnostic changes scope')
    for rel, digest in loss['output_sha256'].items(): require(sha(output/rel) == digest, 'Changed loss diagnostic')
    losses = pd.DataFrame(loss['rows'])
    pre_losses = losses.loc[losses.family.eq('mlp_pretrained')]
    frame = pre_losses.pivot(index='fold_id',columns='role',values='RMSE_bps')[['train','validation','test']].reset_index()
    frame.columns = ['季度折','训练RMSE(bps)','验证RMSE(bps)','开发测试RMSE(bps)']
    sections += ['## 拟合能力与泛化诊断\n\n![训练／验证／测试RMSE](mlp_rmse_by_fold.png)\n\n' + table(frame),
        '训练预测取实际三头均值，不是成员损失均值。预训练表征小MLP在训练集能强烈降低误差，但训练损失下降本身不证明开发期可用信息；三集合覆盖不同时期和噪声分布，不能仅凭差距唯一归因于过拟合或欠拟合。诊断未重新拟合、改参数或选种子；[完整损失底表](mlp_loss_chart_values.csv)、[审计与来源hash](mlp_loss_diagnostics.json)。']
    for gate, title in [('rank50','50%主信息门控'),('economic','独立经济门控')]:
        base = pair.loc[pair.cost.eq('base') & pair.gate.eq(gate)].sort_values('fold_id')
        stress = pair.loc[pair.cost.eq('stress') & pair.gate.eq(gate)].set_index('fold_id')
        frame = base[['fold_id','MLP_pretrained_trades','MLP_pretrained_return_pct','MLP_ordinary19_return_pct','increment_vs_ordinary_pp','MLP_random_backbone_seed_mean_return_pct']].copy()
        frame['预训练14bps收益%'] = frame.fold_id.map(stress.MLP_pretrained_return_pct)
        frame.columns = ['季度折','预训练笔数','预训练7bps收益%','普通MLP收益%','相对普通增量pp','随机骨干均值收益%','预训练14bps收益%']
        sections.append('### ' + title + '\n\n' + table(frame))
        sections.append(f'基础季度收益相对普通MLP的中位增量 {base.increment_vs_ordinary_pp.median():+.2f} 个百分点，{int((base.increment_vs_ordinary_pp>0).sum())}/4折为正；{int((base.MLP_pretrained_return_pct>0).sum())}/4季度优于现金。压力成本季度收益中位数 {stress.MLP_pretrained_return_pct.median():+.2f}%。')
    p = prediction.copy()
    p['relative_MSE_improvement_vs_ordinary'] *= 100; p['relative_MSE_improvement_vs_seed_mean'] *= 100
    p.columns = ['季度折','预训练MSE(bps²)','普通MLP MSE','随机骨干损失均值MSE','相对普通改善%','相对随机改善%']
    sections.append('## 预测、匹配参与数与配对证据\n\n' + table(p))
    sections.append(f'相对普通MLP的MSE跨折中位改善 {prediction.relative_MSE_improvement_vs_ordinary.median()*100:+.2f}%；相对三个随机骨干损失均值 {prediction.relative_MSE_improvement_vs_seed_mean.median()*100:+.2f}%。')
    m = matched[['fold_id','selected_mean_net_bps','exact_null_mean_bps','exact_selection_lift_bps']].copy() if 'selected_mean_net_bps' in matched else matched[['fold_id','exact_null_mean_bps','exact_selection_lift_bps']].copy()
    sections.append('UTC入场月份×固定方向内匹配参与数量的参考净收益诊断：\n\n' + table(m))
    bootstrap = pd.read_csv(output / 'paired_block_bootstrap.csv')
    b = bootstrap.loc[bootstrap.fold_id.eq('EQUAL_FOLD_MEAN') & bootstrap.cost.eq('base') & bootstrap.mean_block_days.eq(7),['comparison','point_difference','lower_95','upper_95','unit']]
    sections.append('四折等权、主7天时间块配对结果：\n\n' + table(b))
    sections.append('配对区间仅用于独立参考效用或预测损失，不能当组合复利收益置信区间；匹配随机抽样分位数也不是组合业绩区间。保存3／14天敏感性；每季度有效7天块约13个，近似折内平稳与跨折独立假设构成限制。')
    recommendation = ('存在进一步研究信号，但本次不自动晋级；须结合全部筛选项和风险／成本证据另行确定范围。' if info or any(economic.values()) else '本轮有限非线性读出仍未达到预登记筛选标准。按停止规则，不在当前任务上继续增加预测头容量或自动解冻；量化前后诊断仅在明确可检验瓶颈问题后另行登记。不能由一次失败证明表征完全无信息。')
    sections += ['## 结论与下一步\n\n' + recommendation,
        '最终holdout继续封存；不利用它决定是否加入Kronos、选择预测头、阈值或推进下一阶段。',
        '## 可追溯入口\n\n[锁定协议](../../configs/' + Path(reg['configuration']['path']).name + ')、[实验登记](../../registry/' + output.name + '.json)、[冻结TODO](../../frozen/TODO.md)、[预测头审核](head_acceptance.json)、[官方回放验收](engine_report.json)、[分析报告](analysis_report.json)、[图表核验](presentation/report.json)、[官方曲线CSV](presentation/official_curves.csv)。全部候选、三种子模型、预测、训练诊断、成本敏感性和旧基线均保留。']
    if reg.get('failure_history'):
        sections.append('首个MLP v1在拟合前的历史行索引类型预检中失败，正式拟合次数为0；失败登记与工件保留。v2仅修正字段解析，预测头结构、优化器、候选、种子、轮数和筛选标准相同，总正式拟合仍为180次。')
    report.write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    completed = output / 'provenance/completion_state'; completed.mkdir()
    shutil.copy2(config_path,completed/'before_config.yaml')
    shutil.copy2(ROOT/'research/frozen/TODO.md',completed/'before_TODO.md')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['status'] = 'frozen_mlp_development_completed_holdout_sealed'
    config['evaluation_readiness'].update(next_research_action='review_bounded_MLP_result_and_apply_stop_rule',
        current_task_stop='bounded_MLP_development_complete_before_holdout_or_backbone_training',
        frozen_mlp_results=report.relative_to(ROOT).as_posix(),frozen_mlp_information_screen_passed=bool(info),
        frozen_mlp_economic_information_gate_passed=bool(economic['rank50']),frozen_mlp_economic_gate_passed=bool(economic['economic']))
    config['experiments'].append({'id':'FROZEN_MLP','status':'completed_exploratory_development',
        'registry':regpath.relative_to(ROOT).as_posix(),'results':report.relative_to(ROOT).as_posix(),
        'head':'MLP_16_Tanh_three_head_ensemble','final_holdout_evaluated':False})
    config_path.write_text('# Bounded MLP development completed; final holdout remains sealed.\n'+yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    shutil.copy2(config_path,completed/'after_config.yaml')
    transition = {'at_utc':datetime.now(timezone.utc).isoformat(),'before_config_sha256':sha(completed/'before_config.yaml'),
        'after_config_sha256':sha(config_path),'report_sha256':sha(report),'final_holdout_evaluated':False}
    write_json(completed/'transition.json',transition)
    reg.update(status='completed_exploratory_development',completed_at_utc=transition['at_utc'],result=analysis)
    reg['result']['next_step'] = recommendation
    reg['completion_state_transition'] = transition
    reg['provenance']['final_source_sha256'] = {p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'research/frozen').glob('mlp*.py')}
    reg['artifacts']['summary_document'] = str(report); reg['artifacts']['summary_document_sha256'] = sha(report)
    reg['artifacts']['final_hashes'] = {p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.suffix!='.log'}
    write_json(regpath,reg)
    print(json.dumps({'status':reg['status'],'conclusion':analysis['conclusion'],'report':str(report),'holdout_evaluated':False}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir)
