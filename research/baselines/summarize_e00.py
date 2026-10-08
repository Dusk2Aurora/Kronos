"""Present verified development results without rerunning any model or engine."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from prepare_e00 import ROOT, sha, write_json, require


def summarize(output):
    output = output.resolve()
    registry = json.loads((ROOT / 'research/registry' / (output.name + '.json')).read_text(encoding='utf-8'))
    require(registry['status'] == 'completed', 'E00 not verified')
    require(sha(output / 'metrics.csv') == registry['artifacts']['final_hashes']['metrics.csv'], 'Metric table changed')
    metrics = pd.read_csv(output / 'metrics.csv')
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    base = metrics.loc[metrics.cost.eq('base')]
    rows = []
    for fold, group in base.groupby('fold_id'):
        group = group.set_index('mode')
        rows.append({'fold_id': fold, 'opportunities': int(group.loc['fixed_momentum', 'trades']),
                     'fixed_momentum_return_pct': group.loc['fixed_momentum', 'net_return_pct'],
                     'ordinary_gate_return_pct': group.loc['ordinary_features_gate', 'net_return_pct'],
                     'cash_return_pct': group.loc['cash', 'net_return_pct'],
                     'gate_trades': int(group.loc['ordinary_features_gate', 'trades'])})
    table = pd.DataFrame(rows)
    presentation = output / 'presentation'
    require(not presentation.exists(), 'Use a new presentation directory')
    presentation.mkdir()
    table.to_csv(presentation / 'participation_vs_cash.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), constrained_layout=True)
    x = range(4)
    for values, shift, title, color in ((table.fixed_momentum_return_pct, -.18, 'Fixed momentum', '#d95f02'),
                                       (table.ordinary_gate_return_pct, .18, 'Ordinary gate', '#1b9e77')):
        bars = axes[0].bar([i + shift for i in x], values, .35, label=title, color=color)
        axes[0].bar_label(bars, fmt='%+.2f%%', padding=3)
    axes[0].axhline(0, color='black', linewidth=1, label='Cash: 0% in all folds')
    axes[0].set_ylim(min(table.fixed_momentum_return_pct) * 1.2, 7)
    axes[0].set_ylabel('Quarter net return (%)')
    axes[0].legend(loc='lower left')
    bars = axes[1].bar(x, table.gate_trades, color='#1b9e77', width=.5)
    axes[1].bar_label(bars, labels=[f'{n}/{total} ({n/total:.2%})'
                                   for n, total in zip(table.gate_trades, table.opportunities)], padding=4)
    axes[1].set_ylim(0, max(table.gate_trades) + 1.5)
    axes[1].set_ylabel('Ordinary gate executed trades (count)')
    for ax in axes:
        ax.set_xticks(list(x), ['2025 Q2', '2025 Q3', '2025 Q4', '2026 Q1'])
    fig.suptitle('E00: skipping losses is not evidence of profitable prediction\n7 bps per side + realized funding | 10,000 USDT reset per fold', fontsize=12)
    fig.supxlabel('Source: E00_20261008_phase4_v1/metrics.csv (official Freqtrade exports)\nDevelopment only: 1,091 eligible opportunities; final holdout sealed', fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(presentation / ('participation_vs_cash.' + ext), dpi=170)
    plt.close(fig)
    diagnostics = {'selected_gate_trades': int(table.gate_trades.sum()),
                   'development_eligible_opportunities': int(table.opportunities.sum()),
                   'participation_fraction': float(table.gate_trades.sum() / table.opportunities.sum()),
                   'gate_beats_cash_folds': int(table.ordinary_gate_return_pct.gt(0).sum()),
                   'gate_equals_cash_folds': int(table.ordinary_gate_return_pct.eq(0).sum()),
                   'gate_loses_to_cash_folds': int(table.ordinary_gate_return_pct.lt(0).sum()),
                   'interpretation': 'Economic screen relative to losing momentum passes; sparse gate does not establish positive information value over cash'}
    write_json(presentation / 'diagnostics.json', diagnostics)
    compact = {**report, 'diagnostics': diagnostics,
               'source_registry': str(ROOT / 'research/registry' / (output.name + '.json')),
               'metrics_sha256': sha(output / 'metrics.csv'),
               'presentation_artifact_sha256': {p.name: sha(p) for p in presentation.iterdir()}}
    target = ROOT / 'research/initialization/e00_report.json'
    require(not target.exists(), 'Preserve prior compact report')
    write_json(target, compact)
    lines = ['# E00 开发期基线结果', '',
             '已完成36次普通特征训练和48份官方Freqtrade组合回测；逐笔开/平仓价、方向、7/14bps费用、1倍杠杆、持有期及原始资金事件核验通过。所有测试均为2025Q2/Q3/Q4、2026Q1；最终2026Q2/Q3 holdout未揭开。', '',
             '每折初始10000USDT，99%可交易余额；下表是每边7bps费用代理加实际资金费，收益为季度组合净收益百分比。', '',
             '| 测试季度 | 有效机会 | 固定动量 | 普通门控 | 门控交易数 | 现金 |',
             '|---|---:|---:|---:|---:|---:|']
    for quarter, row in zip(['2025Q2', '2025Q3', '2025Q4', '2026Q1'], table.to_dict('records')):
        lines.append(f"| {quarter} | {row['opportunities']} | {row['fixed_momentum_return_pct']:+.2f}% | {row['ordinary_gate_return_pct']:+.2f}% | {row['gate_trades']} | 0.00% |")
    lines += ['',
              f"门控相对固定动量四折全部改善，季度增量中位数{report['median_base_increment_pp']:.2f}个百分点；14bps下增量中位数{report['median_stress_increment_pp']:.2f}个百分点。官方已平仓回撤差值中位数{report['median_closed_trade_drawdown_change_pp']:.2f}个百分点。预定经济筛选通过。", '',
              '关键解释：普通门控仅参与8/1091次（0.73%），1折优于现金、2折不如现金、1折持平。改善主要来自跳过亏损策略，尚不足以证明普通特征有稳定正收益或额外信息价值。不能把相对亏损动量的39.94个百分点改善解释为模型赚取39.94%收益。保留已固定方案，不据此调整阈值或重选特征。', '',
              '买入持有是长期多头永续市场对照；其持有期不同于其他4h策略。回撤是官方已平仓资金曲线口径，不能代表持仓期间盯市回撤。每边费用代理也不模拟成交价滑点或冲击。早期折因checkpoint训练截止未知，仍属历史研究。', '',
              '核验脚本曾误用资金字段funding_rate而非快照realized_rate；失败登记已保留，修正后只重新验收现有48份ZIP，未重跑训练/引擎。所有训练尝试、完整配置/策略、原始官方导出和hash均存实验目录及registry。', '',
              f"图表：[六基线及成本对照](../runs/{output.name}/returns_comparison.png)、[门控与现金及参与次数](../runs/{output.name}/presentation/participation_vs_cash.png)。", '',
              f"底表：[metrics.csv](../runs/{output.name}/metrics.csv)、[参与次数底表](../runs/{output.name}/presentation/participation_vs_cash.csv)。", '',
              f"登记：[实验registry](../registry/{output.name}.json)。验收：[e00_report.json](e00_report.json)。", '',
              '下一步完成第8步官方因果核验及第9步M0验收；经济筛选与工程准备程度分别记录。当前未提取冻结表征。', '']
    (ROOT / 'research/initialization/E00_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(diagnostics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    summarize(parser.parse_args().output_dir)
