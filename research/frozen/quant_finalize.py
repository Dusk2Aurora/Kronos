"""Publish reviewed same-coordinate quantization evidence without overwriting trials."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
from quant_charts import verify as verify_charts
import numpy as np
import pandas as pd
import yaml

FAMILIES = ('ridge_quant_continuous', 'ridge_quant_binary')
NAMES = {'ridge_quant_continuous': '连续u20', 'ridge_quant_binary': '二值q20'}
PERIODS = {'WF01': '2025Q2', 'WF02': '2025Q3', 'WF03': '2025Q4', 'WF04': '2026Q1'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(frame):
    def cell(value):
        if pd.isna(value):
            return '不适用'
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
            return f'{value:.3f}'
        return str(value)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join('---' for _ in frame.columns) + ' |',
                      *['| ' + ' | '.join(cell(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)]])


def checks_table(checks):
    return table(pd.DataFrame([{'预登记检查': k, '结果': '通过' if v else '未通过'} for k, v in checks.items()]))


def main(output):
    output = output.resolve()
    require(output.is_relative_to((ROOT / 'research/runs').resolve()) and output != (ROOT / 'research/runs').resolve(), 'Output outside research runs')
    regpath = ROOT / 'research/registry' / f'{output.name}.json'
    reg = read_json(regpath)
    require(reg['status'] == 'engine_verified_pending_analysis', 'Official replay not accepted')
    report = output / 'results.md'
    completed = output / 'provenance/completion_state'
    require(not report.exists() and not completed.exists(), 'Refuse to overwrite published result or completion state')
    analysis = read_json(output / 'analysis_report.json')
    chart = read_json(output / 'presentation/report.json')
    heads = read_json(output / 'head_acceptance.json')
    encoding = read_json(output / 'encoding_acceptance.json')
    require(analysis['status'] == 'completed_exploratory_development_analysis' and analysis['experiment'] == output.name,
            'Analysis incomplete or belongs to another experiment')
    require(chart['status'] == 'passed' and not chart['holdout_accessed'] and not analysis['final_holdout_inspected'], 'Presentation or scope failed')
    require(heads['status'] == 'passed' and heads['independently_recomputed_heads'] == 32
            and heads['selected_heads'] == 8 and heads['test_predictions'] == 2182 and heads['signals'] == 32
            and heads['formal_refits'] == 0 and not heads['holdout_inspected'], 'Head acceptance incomplete')
    require(encoding['status'] == 'passed' and encoding['rows'] == 2460 and encoding['dimension'] == 20
            and encoding['same_coordinates'] and not encoding['new_weight_training'] and not encoding['holdout_inspected'], 'Quant encoding acceptance incomplete')
    for group in ('encoding_hashes', 'preparation_hashes', 'engine_hashes'):
        require(reg['artifacts'].get(group), 'Missing accepted hashes: ' + group)
        for relative, digest in reg['artifacts'][group].items():
            path = (output / relative).resolve()
            require(path.is_relative_to(output) and sha(path) == digest, 'Changed accepted artifact: ' + relative)
    verify_charts(output)
    require(chart['official_curves'] == 512 and chart['new_official_curves'] == 64
            and chart['historical_official_curves'] == 448, 'Chart curve scope differs')
    if (output / 'visual_review.json').exists():
        require(read_json(output / 'visual_review.json')['status'] == 'passed', 'Recorded visual review failed')
    attempts = read_json(output / 'training_attempts.json')
    chosen = read_json(output / 'selected_heads.json')
    require(len(attempts) == 32 and all(a['status'] == 'completed' for a in attempts) and len(chosen) == 8, 'Fit budget incomplete')
    require(len(reg['engine_attempts']) == 64 and all(a['status'] == 'verified' for a in reg['engine_attempts']), 'Official export budget incomplete')
    common = yaml.safe_load((output / 'provenance/common_config.yaml').read_text(encoding='utf-8'))
    require(sha(output / 'provenance/common_config.yaml') == reg['configuration']['sha256']
            and analysis['protocol_sha256'] == reg['configuration']['sha256'], 'Protocol changed')
    require(common['fit_budget']['new_Ridge_fits'] == 32 and common['fit_budget']['new_official_exports'] == 64, 'Registered budget differs')
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    live_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(live_config['stage'] == 'frozen_development'
            and live_config['evaluation_readiness']['final_holdout_evaluation_allowed'] is False, 'Completion scope cannot unseal holdout')
    prediction = pd.read_csv(output / 'paired_prediction_metrics.csv')
    portfolio = pd.read_csv(output / 'paired_portfolio_metrics.csv')
    matched = pd.read_csv(output / 'matched_random_summary.csv')
    bootstrap = pd.read_csv(output / 'paired_block_bootstrap.csv')
    require(len(prediction) == 4 and set(prediction.fold_id) == set(PERIODS) and len(portfolio) == 32, 'Paired result scope differs')
    q_pass = analysis['quantization_screen']['passed']
    information = analysis['information_screens']
    economic = analysis['economic_screens']
    require(set(information) == set(FAMILIES) and set(economic) == set(FAMILIES), 'Missing family screen')
    outcome = '通过' if q_pass else '未通过'
    summary = pd.DataFrame([{'变体': NAMES[f], '信息筛选': '通过' if information[f]['passed'] else '未通过',
              '50%门控经济筛选': '通过' if economic[f]['rank50']['passed'] else '未通过',
              '经济门控经济筛选': '通过' if economic[f]['economic']['passed'] else '未通过'} for f in FAMILIES])
    sections = [f'# 同坐标量化前后诊断：开发期结果\n\n实验 `{output.name}`。工程编码、32个预测头、64份官方回放及图表端点与来源hash核验完成。',
        f'连续u相对二值q的任务可读性筛选：**{outcome}**。登记结论为 `{analysis["conclusion"]}`。两个变体的信息与经济判断分别如下，不用其中表现较好的版本代替预登记比较。\n\n' + table(summary),
        '## 锁定问题、设计与预算\n\n本轮检验在同一个预训练冻结tokenizer、同一历史窗口、同一20个坐标和同一Ridge读出规则下，符号量化后是否出现可观测的任务预测损失。没有训练骨干、tokenizer或投影，也没有生成未来路径。',
        '每个机会仅使用已完成且当时可用的256根历史真实OHLCVA，包含正式成交额。沿用窗口mean/std（ddof=0）加1e-5并clip至±5。冻结tokenizer的quant_embed输出经L2归一化得到连续u20；同位置BSQuantizer产生q20，正坐标取+1/√20、零及负坐标取−1/√20。取最后有效历史位置，u与q共享坐标、样本和时间边界。缓存共2,460个唯一开发机会，各20维float32；原token IDs、批量／单条、重复计算、未来扰动、缺口／未确认／可用时间及holdout拒绝均保留工程证据。',
        '两组均拼接相同普通19项为39维，使用训练折StandardScaler和带截距Ridge，SVD求解，alpha=n_train×lambda。固定lambda候选0.001／0.01／0.1／1，按时间顺序验证集净收益MSE选择，完全相等取更大的lambda；不选择种子、特征、投影、训练轮数或测试阈值。目标为独立官方参考交易7bps基础成本已实现净收益×10000，单位bps；实际资金费用已计入。',
        '四折×两变体×四候选共32次新拟合、8个选中头、2,182条测试预测、32份无未来结果sidecar、64份官方导出。保存全部候选的系数、截距、训练scaler、训练／验证ID、验证预测及选择记录；独立审核Ridge normal equations、保存预测、验证选择和sidecar动作，不增加拟合。普通19／13项Ridge及原Ridge531维、MLP、三个随机骨干、logistic、固定规则与无信息参与对照复用已验收工件，不重跑或按收益删减。',
        '主信息门控为验证50%覆盖分位门槛，25%／75%仅为预定敏感性；测试分数等于门槛时全部纳入并报告实际参与数。经济门控严格预测基础净收益>0，不重复扣成本或强迫交易。14bps压力成本沿用相同预测和动作。7／14bps为每边费用代理，非订单簿、冲击或成交价格滑点模拟；实际资金另计。',
        '## 真实时间官方组合曲线\n\n横轴为真实UTC时间，各季度独立从10,000 USDT重置，曲线仅在已平仓事件更新。未平仓浮盈亏和小时盯市风险不在闭仓权益中，不能将四折当作连续年度复利。',
        '![50%主信息门控基础成本](presentation/rank50_equity_base.png)\n\n![独立经济门控基础成本](presentation/economic_equity_base.png)\n\n![50%主信息门控压力成本](presentation/rank50_equity_stress.png)',
        '![历史骨干参考](presentation/backbone_reference_rank50_equity_base.png)\n\n历史512维骨干context加普通19项为531维，与量化20维加普通19项的39维不同，路径和读出器也可能不同；该面板只提供解释参照，不能作为量化或骨干贡献的因果对照。全部随机骨干种子与两成本、四门控保存在独立面板和CSV。']
    primary = prediction[['fold_id', 'continuous_MSE_bps_squared', 'binary_MSE_bps_squared', 'ordinary19_MSE_bps_squared',
                          'relative_MSE_improvement_continuous_vs_binary']].copy()
    primary.fold_id = primary.fold_id.map(PERIODS)
    primary.relative_MSE_improvement_continuous_vs_binary *= 100
    primary.columns = ['测试季度', 'u39维MSE(bps²)', 'q39维MSE(bps²)', '普通19 MSE(bps²)', 'u相对q改善%']
    sections.append('## 同坐标量化可读性\n\n' + table(primary))
    sections.append(f'u相对q的跨折中位相对MSE改善为 {prediction.relative_MSE_improvement_continuous_vs_binary.median()*100:+.3f}%，'
                    f'{int((prediction.relative_MSE_improvement_continuous_vs_binary>0).sum())}/4折为正。最低条件为中位改善≥1%、至少3/4折为正，'
                    '以及四折等权主7天块的配对(q平方误差−u平方误差)95%下界>0。三项必须同时成立。\n\n' + checks_table(analysis['quantization_screen']['checks']))
    reference = prediction[['fold_id', 'ordinary13_MSE_bps_squared', 'historical_pretrained531_MSE_bps_squared',
                            'historical_random531_seed_mean_MSE_bps_squared']].copy()
    reference.fold_id = reference.fold_id.map(PERIODS)
    reference.columns = ['测试季度', '普通13 MSE(bps²)', '历史预训练531 MSE(bps²)', '历史随机531损失均值MSE(bps²)']
    sections.append('维度不同的历史预测损失仅作参照；随机汇总是全部三个骨干的损失均值，未挑种子，也未平均分数形成新执行策略。\n\n' + table(reference))
    for family, short in zip(FAMILIES, ('continuous', 'binary')):
        column = 'relative_MSE_improvement_' + short + '_vs_ordinary'
        sections.append(f'## {NAMES[family]}：相对普通19项的信息与经济证据\n\nMSE相对普通19项的中位改善为 '
                        f'{prediction[column].median()*100:+.3f}%，{int((prediction[column]>0).sum())}/4折为正。'
                        '信息筛选同时要求≥1%的中位MSE改善、≥3/4折正向改善、匹配参与数后的中位净收益增量≥1bps且≥3/4折为正，'
                        '以及相对普通19项与匹配无信息对照的四折等权7天块95%下界均>0、两比较各≥3/4折参考效用为正。\n\n'
                        + checks_table(information[family]['checks']))
        lift = matched.loc[matched.cost.eq('base') & matched['mode'].eq(family + '_rank50')].sort_values('fold_id')
        columns = ['fold_id', 'selected_count', 'selected_mean_bps', 'exact_null_mean_bps', 'exact_selection_lift_bps', 'random_p025_bps', 'random_p975_bps']
        frame = lift[columns].copy()
        frame.fold_id = frame.fold_id.map(PERIODS)
        frame.columns = ['测试季度', '参与数', '所选净bps', '匹配随机期望bps', '筛选增量bps', '抽样2.5%分位bps', '抽样97.5%分位bps']
        sections.append('在UTC入场月份×固定动量方向内匹配相同参与数量。精确分层期望用于点估计；3,000次随机抽样范围是事后诊断分位数，不是组合收益置信区间或可执行政策。零参与保留不适用，不降低门槛。\n\n' + table(frame))
        for gate, title in [('rank50', '50%主信息门控'), ('economic', '独立经济门控')]:
            b = portfolio.loc[portfolio.family.eq(family) & portfolio.gate.eq(gate) & portfolio.cost.eq('base')].sort_values('fold_id')
            s = portfolio.loc[portfolio.family.eq(family) & portfolio.gate.eq(gate) & portfolio.cost.eq('stress')].set_index('fold_id')
            frame = b[['fold_id', 'quant_trades', 'quant_return_pct', 'ordinary_return_pct', 'increment_vs_ordinary_pp', 'closed_drawdown_change_vs_ordinary_pp']].copy()
            frame['stress_return_pct'] = frame.fold_id.map(s.quant_return_pct)
            frame.fold_id = frame.fold_id.map(PERIODS)
            frame.columns = ['测试季度', '成交笔数', '7bps净收益%', '普通19净收益%', '增量pp', '闭仓回撤变化pp', '14bps净收益%']
            sections.append('### ' + title + '\n\n' + table(frame))
            sections.append(f'相对普通19项的基础季度收益中位增量为 {b.increment_vs_ordinary_pp.median():+.3f} 个百分点；'
                            f'{int((b.increment_vs_ordinary_pp>0).sum())}/4季度增量为正，{int((b.quant_return_pct>0).sum())}/4季度优于现金。'
                            f'压力成本季度净收益中位数 {s.quant_return_pct.median():+.3f}%，增量中位数 {s.increment_vs_ordinary_pp.median():+.3f} 个百分点。\n\n'
                            + checks_table(economic[family][gate]['checks']))
    sections.append('经济筛选预登记要求相对普通模型季度中位增量≥1个百分点，至少3/4折普通模型增量为正且至少3/4季度优于现金，'
                    '压力成本中位净收益及增量均>0，闭仓最大回撤中位变化不更差。该口径分别应用于主信息门控和经济门控；相对亏损动量少亏或低暴露不等同于盈利或信息优势。')
    intervals = bootstrap.loc[bootstrap.fold_id.eq('EQUAL_FOLD_MEAN') & bootstrap.cost.eq('base') & bootstrap.mean_block_days.eq(7),
                             ['comparison', 'point_difference', 'lower_95', 'upper_95', 'unit']]
    sections.append('## 配对时间块不确定性\n\n' + table(intervals))
    sections.append('配对参考交易效用按完整UTC决策日历汇总，零机会日保留；预测误差差按完整日历汇总后缩放回每机会均方误差(bps²)。'
                    '折内stationary bootstrap 5,000次，主平均块长7天，3／14天敏感性全部保存；种子17＋从0开始的折序号×100，沿用先前共用工作流且未改变；四折独立重采样后等权。'
                    '约每季度13个有效7天块，依赖折内近似平稳及折间依赖可忽略。效用单位为独立参考净bps／日，不能当复合Freqtrade组合收益或其置信区间。')
    recommendation = ('该轮预登记比较出现进一步研究信号；需要结合具体通过项、效应大小和独立数据复现，不能由此自动推广一般量化信息损失或晋级。'
                      if q_pass or any(v['passed'] for v in information.values()) or any(v['passed'] for family in economic.values() for v in family.values())
                      else '该轮没有达到预登记量化、信息或经济筛选的充分证据。保留负结果与不确定性，不在本任务上继续增加容量、候选、种子或改测试门槛。'
                           '一次任务诊断未建立量化瓶颈，不能证明表示完全无信息或tokenizer没有损失。')
    sections += ['## 结论、边界与下一步\n\n' + recommendation,
        '本结果是同坐标u/q在固定净收益参与任务、既定Ridge和开发样本下的任务可读性，不能称一般信息损失、tokenizer因果效果或预训练普遍有效性。'
        '开发季度此前已见，checkpoint训练截止未知，不能宣称严格事前样本外。数据完整性证据只覆盖已公布官方数据的一致性，不能扩大为历史事前可用性的证明。',
        '最终holdout继续封存，未编码、评估标签／分类比例或收益，不用于决定是否加入Kronos、筛选预测头或阈值。复现需要另行登记独立数据和问题；'
        '本轮完成不授予骨干／tokenizer训练、微调、原生未来路径生成、实盘或holdout许可。',
        '长期goal的下一步是汇总首轮Ridge、有限MLP的拟合／泛化诊断及本量化诊断，形成第一项冻结实验的完整可复核结论、瓶颈证据和停止／后续建议。'
        '本脚本只完成本轮登记，不将长期goal自动标记完成。',
        '## 可追溯入口\n\n[锁定协议](../../configs/' + Path(reg['configuration']['path']).name + ')、[实验登记](../../registry/' + output.name + '.json)、'
        '[冻结TODO](../../frozen/TODO.md)、[编码验收](encoding_acceptance.json)、[输入审核](encoding_head_input_audit.json)、'
        '[预测头独立审核](head_acceptance.json)、[官方回放](engine_report.json)、[分析报告](analysis_report.json)、'
        '[图表核验](presentation/report.json)、[全部官方曲线CSV](presentation/official_curves.csv)、'
        '[配对预测](paired_prediction_metrics.csv)、[配对组合](paired_portfolio_metrics.csv)、[匹配随机](matched_random_summary.csv)、'
        '[时间块](paired_block_bootstrap.csv)、[每日配对底表](paired_daily_series.csv)、[集中度与换手](portfolio_concentration_turnover.csv)。'
        '全部候选、缓存、工程失败记录、预定敏感性和历史工件保留原字节。']
    # All accepted-byte, output hash and scope checks above precede publication.
    completed.mkdir()
    shutil.copy2(config_path, completed / 'before_config.yaml')
    shutil.copy2(ROOT / 'research/frozen/TODO.md', completed / 'before_TODO.md')
    shutil.copy2(Path(__file__), completed / 'quant_finalize.py')
    report.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    config = live_config
    config['status'] = 'frozen_quant_development_completed_holdout_sealed'
    config['evaluation_readiness'].update(next_research_action='synthesize_first_frozen_experiment_Ridge_MLP_quantization_evidence',
        current_task_stop='quant_development_complete_goal_synthesis_pending_holdout_and_weights_sealed',
        frozen_quant_results=report.relative_to(ROOT).as_posix(), frozen_quant_screen_passed=bool(q_pass),
        frozen_quant_information_screens={f: bool(information[f]['passed']) for f in FAMILIES},
        frozen_quant_economic_screens={f: {g: bool(v['passed']) for g, v in economic[f].items()} for f in FAMILIES})
    config['experiments'].append({'id': 'QUANT', 'run_id': output.name, 'status': 'completed_exploratory_development',
        'registry': regpath.relative_to(ROOT).as_posix(), 'results': report.relative_to(ROOT).as_posix(),
        'head': 'Ridge_same_coordinate_u20_vs_q20_plus_ordinary19', 'final_holdout_evaluated': False})
    config_path.write_text('# Same-coordinate quantization development completed; final holdout and weights remain sealed.\n'
                           + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    shutil.copy2(config_path, completed / 'after_config.yaml')
    transition = {'at_utc': datetime.now(timezone.utc).isoformat(), 'before_config_sha256': sha(completed / 'before_config.yaml'),
        'after_config_sha256': sha(config_path), 'report_sha256': sha(report), 'final_holdout_evaluated': False,
        'backbone_or_tokenizer_trained': False, 'long_term_goal_synthesis_pending': True}
    write_json(completed / 'transition.json', transition)
    reg.update(status='completed_exploratory_development', completed_at_utc=transition['at_utc'], result=analysis)
    reg['result']['next_step'] = recommendation + ' 汇总首轮Ridge、有限MLP及量化诊断，完成长期goal的最终综合结论。'
    reg['completion_state_transition'] = transition
    reg['provenance']['final_source_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'research/frozen').glob('quant*.py')}
    reg['artifacts']['summary_document'] = str(report)
    reg['artifacts']['summary_document_sha256'] = sha(report)
    reg['artifacts']['final_hashes'] = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file() and p.suffix != '.log'}
    write_json(regpath, reg)
    print(json.dumps({'status': reg['status'], 'conclusion': analysis['conclusion'], 'report': str(report), 'holdout_evaluated': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    main(parser.parse_args().output_dir)
