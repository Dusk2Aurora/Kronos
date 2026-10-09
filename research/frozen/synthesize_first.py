"""Close the finite first frozen experiment from accepted DEV artifacts only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
import json
import shutil
from datetime import datetime, timezone
import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

RUNS = ('FROZEN_20261008_v1', 'FROZEN_MLP_20261008_v2', 'QUANT_20261008_v1')

def table(frame):
    def fmt(x):
        return f'{x:.2f}' if isinstance(x, (float, np.floating)) else str(x)
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join('---' for _ in frame.columns) + ' |\n' + '\n'.join('| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None))

def main():
    destination = ROOT / 'research/runs/FIRST_FROZEN_SYNTHESIS_20261008_v1'
    report = ROOT / 'research/frozen/FIRST_EXPERIMENT_CONCLUSION.md'
    require(not destination.exists() and not report.exists(), 'Refuse synthesis overwrite')
    inputs, analyses, registers = {}, {}, {}
    for run_id in RUNS:
        path = ROOT / 'research/registry' / (run_id + '.json')
        reg = json.loads(path.read_text(encoding='utf-8'))
        require(reg['status'] == 'completed_exploratory_development', 'Incomplete run: ' + run_id)
        for rel, digest in reg['artifacts']['final_hashes'].items():
            artifact = ROOT / 'research/runs' / run_id / rel
            require(sha(artifact) == digest, 'Changed final artifact: ' + str(artifact))
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        a = ROOT / 'research/runs' / run_id / 'analysis_report.json'
        analyses[run_id] = json.loads(a.read_text(encoding='utf-8'))
        require(not analyses[run_id]['final_holdout_inspected'], 'Holdout scope violation')
        registers[run_id] = reg
    ridge, mlp, quant = [analyses[x] for x in RUNS]
    qpass = quant['quantization_screen']['passed']
    information = [ridge['information_screen']['passed'], mlp['information_screen']['passed'],
                   *(v['passed'] for v in quant['information_screens'].values())]
    economics = [*(v['passed'] for v in ridge['economic_screens'].values()),
                 *(v['passed'] for v in mlp['economic_screens'].values()),
                 *(v['passed'] for f in quant['economic_screens'].values() for v in f.values())]
    recommendation = ('发现需要独立数据复核的局部信号，保留候选并停止当前开发数据上的搜索。该信号不自动授权tokenizer重建、骨干解冻或最终holdout。'
                      if any(information) or any(economics) or qpass else
                      '停止当前BTC单资产、1h历史、4h固定动量参与任务上的冻结表征扩展；不继续加大预测头、不启动tokenizer重建或骨干微调。')
    destination.mkdir()
    for run_id in RUNS:
        shutil.copy2(ROOT / 'research/registry' / (run_id + '.json'), destination / (run_id + '.json'))
    rows = []
    r = pd.read_csv(ROOT / 'research/runs' / RUNS[0] / 'paired_prediction_metrics.csv')
    m = pd.read_csv(ROOT / 'research/runs' / RUNS[1] / 'paired_prediction_metrics.csv')
    q = pd.read_csv(ROOT / 'research/runs' / RUNS[2] / 'paired_prediction_metrics.csv')
    for frame, names in ((r, {'E00R_MSE_bps_squared':'普通19 Ridge', 'E01R_MSE_bps_squared':'预训练531 Ridge', 'E02R_seed_mean_MSE_bps_squared':'随机531 Ridge损失均值'}),
                         (m, {'MLP_ordinary19_MSE_bps_squared':'普通19 MLP','MLP_pretrained_MSE_bps_squared':'预训练531 MLP','MLP_random_backbone_seed_mean_MSE_bps_squared':'随机531 MLP损失均值'}),
                         (q, {'continuous_MSE_bps_squared':'连续u39 Ridge', 'binary_MSE_bps_squared':'二值q39 Ridge'})):
        for row in frame.to_dict('records'):
            for key, label in names.items():
                rows.append({'fold_id':row['fold_id'], 'variant':label, 'MSE_bps_squared':row[key], 'RMSE_bps':np.sqrt(row[key])})
    values = pd.DataFrame(rows)
    dates = {'WF01':'2025-06-30','WF02':'2025-09-30','WF03':'2025-12-31','WF04':'2026-03-31'}
    values['quarter_end_UTC'] = values.fold_id.map(dates)
    values.to_csv(destination / 'prediction_loss_by_quarter.csv', index=False)
    font = Path('C:/Windows/Fonts/msyh.ttc'); font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(), 'axes.unicode_minus':False,'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, part in values.groupby('variant', sort=False):
        ax.plot(pd.to_datetime(part.quarter_end_UTC, utc=True), part.RMSE_bps, marker='o', label=label,
                linestyle='--' if '随机' in label else '-')
    ax.set(title='第一项冻结实验：四个开发测试季度的预测损失', xlabel='开发测试季度末（UTC）', ylabel='官方7bps净收益预测RMSE（bps）')
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=10)
    fig.text(.08,.015,'每点为独立季度整体损失；随机均值为sqrt(三个骨干MSE均值)，不是可执行集成；此前已见DEV，非严格事前样本外。',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    for suffix in ('png','svg'): fig.savefig(destination / ('prediction_loss_by_quarter.'+suffix),dpi=150)
    plt.close(fig)
    loss_table = values.pivot(index='fold_id', columns='variant',values='RMSE_bps').reset_index()
    qci = pd.read_csv(ROOT / 'research/runs' / RUNS[2] / 'paired_block_bootstrap.csv')
    qci = qci.loc[qci.fold_id.eq('EQUAL_FOLD_MEAN') & qci.cost.eq('base') & qci.mean_block_days.eq(7) & qci.comparison.eq('continuous_vs_binary_MSE')]
    sections = ['# 第一项冻结表征实验：完整开发期结论\n\n2026-10-08；用户长期goal的有限研究验收。',
        '**当前证据未建立稳定的预训练信息优势或经济价值。**' if not any(information) and not any(economics) else '**存在达到部分预登记标准的候选信号，详见逐项筛选；尚未证明可部署经济价值。**',
        recommendation,
        '## 完成范围与停止依据\n\n完成路线图第5章／M1要求的冻结历史编码、普通特征增量对照、预训练／随机骨干同头比较、一次有限小MLP读出和同坐标量化前后诊断。三轮分别64、180、32次新增正式拟合，总计276次；新增官方回放128＋160＋64＝352份。加上原E00的48份、E00R的112份，综合图表保留512条官方曲线。拟合重构和诊断复算不计新正式拟合，MLP失败v1正式拟合为0。',
        '任务保持BTC-USDT-SWAP、1h、256根已完成历史、24h固定动量方向、每8h UTC04/12/20决策、60秒可用延迟、下一开盘入场、严格4h持有。只决定参与／跳过，未改方向、仓位、任务或成本。真实OHLCVA与成交额、实际签名资金费用、官方Freqtrade账本及purge规则均保留。每边7／14bps是费用代理；各季度组合独立重置10,000USDT。',
        '停止是基于已完成有限设计的研究决定，不是环境阻断，也不是证明Kronos在所有任务上无信息。尚未检验多资产、多周期、其他标签、池化、稳健损失或原生未来路径；这些属于另行登记的新研究，不能以当前已见季度反复调参。',
        '## 预测损失：按真实季度呈现\n\n![开发季度预测RMSE](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/prediction_loss_by_quarter.png)\n\n' + table(loss_table),
        f'同Ridge预训练531项相对普通19项的跨折中位MSE改善为 {r.relative_MSE_improvement_vs_ordinary.median()*100:+.2f}%；相对三个随机骨干损失均值 {r.relative_MSE_improvement_vs_seed_mean.median()*100:+.2f}%。普通特征增量与随机骨干贡献必须同时判断，单独优于随机骨干不足以晋级。',
        f'小MLP预训练531项相对普通MLP的中位MSE改善为 {m.relative_MSE_improvement_vs_ordinary.median()*100:+.2f}%。训练RMSE仅7.94／9.06／11.24／12.45bps，开发测试达121.23／115.25／144.22／147.08bps；强化样本内拟合没有形成开发增量。Ridge用直接SVD求解，非优化器未收敛。不能仅凭不同时期的损失差唯一归因于过拟合／欠拟合，但当前证据不支持继续扩头。',
        '## 同坐标量化瓶颈\n\n连续u是quant_embed输出的L2归一化20坐标，q是在完全相同坐标上以sign(u)/sqrt(20)量化；都加同19项普通特征，维度39、同Ridge预算。比较末端单一历史位置，不比较六通道重构或误把coarse decoder输出当量化前latent。',
        f'连续u相对二值q的跨折中位MSE改善 {q.relative_MSE_improvement_continuous_vs_binary.median()*100:+.2f}%，正向 {int((q.relative_MSE_improvement_continuous_vs_binary>0).sum())}/4折；量化筛选 **{"通过" if qpass else "未通过"}**。主7日块四折等权配对MSE差（q误差−u误差，bps²）：\n\n' + table(qci[['point_difference','lower_95','upper_95','unit']]),
        '该结果只说明此任务、此坐标和Ridge预算下的可读性差异，不能证明一般信息损失或用20维对512维差异归因骨干。即使连续值略好，也需要普通特征增量、时间一致性和配对区间；不能自动重建tokenizer。',
        '## 经济结果：保留真实时间曲线\n\n![同坐标主50%门控](../runs/QUANT_20261008_v1/presentation/rank50_equity_base.png)\n\n![同坐标经济门控](../runs/QUANT_20261008_v1/presentation/economic_equity_base.png)\n\n![原预训练／随机Ridge](../runs/FROZEN_20261008_v1/presentation/rank50_equity_base.png)\n\n![有限MLP](../runs/FROZEN_MLP_20261008_v2/presentation/rank50_equity_base.png)',
        '这些是官方已平仓权益，闭仓之间平线不代表持仓没有浮动风险，不拼成年连续复利。14bps及25%／75%敏感性图保留各运行presentation目录，压力成本复用相同预测和动作。匹配无信息参与只用于事后诊断，不是可执行政策；参考效用区间不是复利组合业绩置信区间。']
    for family in ('ridge_quant_continuous','ridge_quant_binary'):
        pair = pd.read_csv(ROOT/'research/runs'/RUNS[2]/'paired_portfolio_metrics.csv')
        base = pair.loc[pair.family.eq(family) & pair.cost.eq('base') & pair.gate.eq('rank50')]
        sections.append(f'{family}：50%基础季度收益%为 '+ '／'.join(f'{v:+.2f}' for v in base.quant_return_pct)+f'；相对普通中位增量{base.increment_vs_ordinary_pp.median():+.2f}个百分点，{int((base.quant_return_pct>0).sum())}/4季度优于现金。')
    screens = {'Ridge_information':ridge['information_screen'], 'Ridge_economic':ridge['economic_screens'],
               'MLP_information':mlp['information_screen'],'MLP_economic':mlp['economic_screens'],
               'quantization':quant['quantization_screen'],'quant_information':quant['information_screens'],'quant_economic':quant['economic_screens']}
    write_json(destination/'all_screens.json',screens)
    sections += ['## 验收口径与证据限制\n\n使用用户采用的1%预测MSE中位改善、≥3/4折正向、匹配参与数中位增量≥1bps、主7日时间块95%下界>0；经济另要求季度增量≥1个百分点、≥3/4季度优于现金、压力成本与回撤检验。没有因负结果降低阈值。全部逐项通过／失败见[汇总JSON](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/all_screens.json)。',
        '开发测试为2025Q2／Q3／Q4、2026Q1，已见历史季度上的探索性比较；checkpoint训练截止仍未证明，不宣称严格事前样本外。最终holdout [2026-04-01,2026-10-01) 未编码或评估业绩、标签统计、分类比例。早期源字段核验曾显示原始holdout OHLCV页面，已记录；这不等于揭开标签业绩，也不以其选择模型。',
        'tokenizer保持预训练，随机骨干只隔离骨干预训练贡献。encode(half=True)返回显式s1与s2两组ID；decode_s1消费两组ID及时间信息。当前适配一直显式传二组ID，未用整型拆分接口，不修改上游核心。缓存按每机会独立标准化，禁止全历史一次编码后切hidden。',
        '故障记录完整保留：MLP v1在拟合前索引类型检查失败；v2修正CSV字段解析后按原180预算执行。图表预检历史配置布局故障修复后生成，未新增拟合／回放。行尾原文字节hash问题已在首轮登记修复历史中记录。',
        '## 交付与下一步\n\n'+recommendation+' 当前有限目标已完成，没有待运行的正式实验。若未来重新探索，优先使用独立期间／资产建立新登记并明确任务信号、成本及数据可用性；不直接消耗当前封存holdout去选择是否加入Kronos。',
        '[独立冻结TODO](TODO.md)；[首轮Ridge报告](../initialization/FROZEN_results.md)；[MLP报告](../runs/FROZEN_MLP_20261008_v2/results.md)；[量化报告](../runs/QUANT_20261008_v1/results.md)；[512条官方曲线CSV](../runs/QUANT_20261008_v1/presentation/official_curves.csv)；[预测损失底表](../runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/prediction_loss_by_quarter.csv)。本机ignored数据／模型／ZIP不能随clone取得；复现需原manifest和已保存源字节。不push、不部署、不实盘。']
    report.write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    config_path = ROOT/'research/configs/initial_experiment.yaml'
    shutil.copy2(config_path,destination/'before_config.yaml')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['status']='first_frozen_experiment_completed_holdout_sealed'
    config['evaluation_readiness'].update(first_frozen_experiment_completed=True,first_frozen_experiment_conclusion=report.relative_to(ROOT).as_posix(),
        next_research_action='stop_current_frozen_task_review_complete_synthesis',current_task_stop='first_frozen_experiment_complete_holdout_and_weight_training_forbidden')
    config_path.write_text('# First frozen experiment completed; final holdout remains sealed.\n'+yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    shutil.copy2(config_path,destination/'after_config.yaml')
    write_json(destination/'manifest.json',{'status':'completed_bounded_first_frozen_experiment','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'runs':list(RUNS),'new_formal_fits':276,'new_official_exports':352,'all_official_curves':512,
        'information_signal_established':any(information),'economic_signal_established':any(economics),'quantization_screen_passed':qpass,
        'recommendation':recommendation,'holdout_evaluated':False,'weight_training_performed':False,'pushed':False,
        'input_registries_sha256':inputs,'source_sha256':{Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))},
        'report_sha256':sha(report),'outputs_sha256':{p.name:sha(p) for p in destination.iterdir() if p.is_file()}})
    print(json.dumps({'status':'complete','report':str(report),'recommendation':recommendation},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args(); main()
