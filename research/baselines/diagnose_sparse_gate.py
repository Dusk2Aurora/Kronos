"""Diagnose retained E00 models without refitting, changing gates or using holdout."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from prepare_e00 import ROOT, sha, require, write_json


def diagnose(output):
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs') and not output.exists(), 'Use new research output directory')
    run = ROOT / 'research/runs/E00_20261008_phase4_v1'
    bundle = ROOT / 'research/data/labels/reference_labels_20261008_phase4'
    registry = json.loads((ROOT / 'research/registry/E00_20261008_phase4_v1.json').read_text(encoding='utf-8'))
    require(registry['status'] == 'completed', 'Original E00 not verified')
    for relative in ('training_attempts.json', 'selected_heads.json', 'development_features.csv',
                     'development_test_predictions.csv', 'metrics.csv', 'gate_increments.csv'):
        require(sha(run / relative) == registry['artifacts']['final_hashes'][relative], 'Original artifact changed')
    output.mkdir(parents=True)
    source_paths = [Path(__file__), run / 'training_attempts.json', run / 'selected_heads.json',
                    run / 'development_features.csv', run / 'development_test_predictions.csv',
                    bundle / 'split_index.csv', bundle / 'labels_forbidden_as_features.csv']
    # This is a read-only diagnostic of an existing trial, not a model-selection attempt.
    write_json(output / 'diagnostic_plan.json', {'scope': 'Existing development models only; no fit/threshold/feature selection',
                                                'sources': {str(p): sha(p) for p in source_paths},
                                                'final_holdout_used': False})
    splits = pd.read_csv(bundle / 'split_index.csv')
    splits = splits.loc[splits.fold_id.isin(['WF01', 'WF02', 'WF03', 'WF04']) & splits.included.eq(True)]
    allowed = set(splits.opportunity_id)
    require(all(pd.Timestamp(x) < pd.Timestamp('2026-04-01T00:00:00Z') for x in allowed), 'Holdout in development membership')
    targets = {}
    with (bundle / 'labels_forbidden_as_features.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['opportunity_id'] in allowed:
                target = int(row['base_target'])
                require(target == int(float(row['base_profit_ratio']) > 0), 'Target semantic mismatch')
                targets[row['opportunity_id']] = target
    labels = pd.Series(targets)
    features = pd.read_csv(run / 'development_features.csv').set_index('opportunity_id')
    predictions = pd.read_csv(run / 'development_test_predictions.csv').set_index('opportunity_id')
    trials = json.loads((run / 'training_attempts.json').read_text(encoding='utf-8'))
    selected = json.loads((run / 'selected_heads.json').read_text(encoding='utf-8'))
    rows, plot_rows = [], []
    for selected_head in selected:
        fold = selected_head['fold_id']
        model = next(a for a in trials if a['fold_id'] == fold and a['C'] == selected_head['C'] and a['seed'] == selected_head['seed'])
        train_ids = splits.loc[splits.fold_id.eq(fold) & splits.role.eq('train'), 'opportunity_id']
        prior = float(labels.loc[train_ids].mean())
        for role in ('train', 'validation', 'test'):
            ids = splits.loc[splits.fold_id.eq(fold) & splits.role.eq(role), 'opportunity_id']
            x = (features.loc[ids].to_numpy() - np.array(model['scaler_mean'])) / np.array(model['scaler_scale'])
            logits = x @ np.array(model['coef'][0]) + model['intercept'][0]
            p = 1 / (1 + np.exp(-logits))
            y = labels.loc[ids].to_numpy()
            if role == 'test':
                require(np.allclose(p, predictions.loc[ids, 'probability'], rtol=0, atol=1e-12), 'Saved probability mismatch')
                require(np.array_equal(p >= .5, predictions.loc[ids, 'gate'].to_numpy()), 'Saved gate mismatch')
                plot_rows.append((fold, prior, p))
            quantiles = np.quantile(p, [0, .1, .5, .9, .99, 1])
            rows.append({'fold_id': fold, 'role': role, 'samples': len(ids), 'profitable_labels': int(y.sum()),
                         'positive_label_fraction': float(y.mean()), 'train_positive_prior': prior,
                         'probability_min': float(quantiles[0]), 'probability_p10': float(quantiles[1]),
                         'probability_median': float(quantiles[2]), 'probability_p90': float(quantiles[3]),
                         'probability_p99': float(quantiles[4]), 'probability_max': float(quantiles[5]),
                         'gate_count_at_fixed_0_5': int((p >= .5).sum()), 'auc': float(roc_auc_score(y, p)),
                         'log_loss': float(log_loss(y, p)),
                         'train_prior_constant_log_loss': float(log_loss(y, np.full(len(y), prior)))})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'probability_and_labels.csv', index=False)
    test = table.loc[table.role.eq('test')]
    increments = pd.read_csv(run / 'gate_increments.csv')
    metrics = pd.read_csv(run / 'metrics.csv')
    cash_comparison = []
    for cost in ('base', 'stress'):
        fixed = metrics.loc[metrics.cost.eq(cost) & metrics['mode'].eq('fixed_momentum')].set_index('fold_id')
        diff = -fixed.net_return_pct
        dd_change = -fixed.closed_trade_max_drawdown_pct
        cash_comparison.append({'cost': cost, 'cash_minus_momentum_median_pp': float(diff.median()),
                                'cash_positive_increment_folds': int(diff.gt(0).sum()),
                                'cash_drawdown_change_median_pp': float(dd_change.median())})
    write_json(output / 'cash_screen_counterexample.json', cash_comparison)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (fold, prior, p), quarter in zip(axes.flat, plot_rows, ['2025 Q2', '2025 Q3', '2025 Q4', '2026 Q1']):
        row = test.loc[test.fold_id.eq(fold)].iloc[0]
        ax.hist(p, bins=np.linspace(.2, .6, 33), color='#4575b4', edgecolor='white')
        ax.axvline(.5, color='#d73027', linewidth=2, linestyle='--', label='Entry threshold: 0.50')
        ax.axvline(prior, color='#e6ab02', linewidth=2, label=f'Train positive prior: {prior:.3f}')
        ax.set_title(f'{quarter} | n={len(p)} | entries={int((p >= .5).sum())} | AUC={row.auc:.3f}')
        ax.text(.98, .72, f'Median={np.median(p):.3f}\nMax={np.max(p):.3f}', transform=ax.transAxes, ha='right', fontsize=10)
        ax.set_xlim(.2, .6)
        ax.set_xlabel('Predicted P(reference trade net return > 0)')
        ax.set_ylabel('Eligible opportunities (count)')
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Sparse E00 gate: most predictions stay near the ~38% training prior\nFixed 0.50 threshold | 7 bps per side + funding labels | 1,091 development tests', fontsize=12)
    fig.supxlabel('Source: retained E00 models, development features and labels; no refitting or threshold changes; final holdout sealed', fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(output / ('probability_distribution.' + ext), dpi=170)
    plt.close(fig)
    result = {'status': 'diagnosed_sparse_gate_no_engine_loss', 'selected_entries': int(test.gate_count_at_fixed_0_5.sum()),
              'eligible_opportunities': int(test.samples.sum()),
              'test_auc': test.auc.tolist(), 'test_logloss_better_than_train_prior_folds': int((test.log_loss < test.train_prior_constant_log_loss).sum()),
              'cause_supported_by_evidence': 'Low base positive frequency and weak discriminating model keep almost all probabilities below fixed 0.5 threshold',
              'research_limitation': 'The economic screen versus losing momentum also accepts cash; it does not establish a useful predictive gate',
              'win_probability_is_not_expected_return': True, 'final_holdout_used': False,
              'models_refitted': False, 'threshold_changed': False,
              'artifacts_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / 'report.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    diagnose(parser.parse_args().output_dir)
