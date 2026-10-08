"""Post-E00 exploratory fixed-coverage diagnostics; never fit or unseal holdout."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FOLDS = ('WF01', 'WF02', 'WF03', 'WF04')
QUANTILES = (.5, .25, .75)
SEEDS = (17, 29, 43)
DRAWS = 1000
LIMIT = pd.Timestamp('2026-04-01T00:00:00Z')
SCOPE = 'Post-E00 exploratory existing development data; no new clean OOS evidence'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, encoding='utf-8').strip()


def filtered_csv(path, key, allowed):
    # Outcome conversion happens only AFTER development membership filtering.
    with path.open(encoding='utf-8', newline='') as handle:
        return pd.DataFrame([row for row in csv.DictReader(handle) if row[key] in allowed])


def analyze(output):
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs') and not output.exists(), 'Use a new research/runs directory')
    run = ROOT / 'research/runs/E00_20261008_phase4_v1'
    bundle = ROOT / 'research/data/labels/reference_labels_20261008_phase4'
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    registry_path = ROOT / 'research/registry/E00_20261008_phase4_v1.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    require(registry['status'] == 'completed', 'E00 not completed')
    for name in ('training_attempts.json', 'selected_heads.json', 'development_features.csv', 'development_test_predictions.csv'):
        require(sha(run / name) == registry['artifacts']['final_hashes'][name], 'Changed E00 artifact: ' + name)
    for name in ('split_index.csv', 'labels_forbidden_as_features.csv'):
        require(sha(bundle / name) == manifest['artifact_sha256'][name], 'Changed label artifact: ' + name)
    snapshot = Path(manifest['sources']['snapshot'])
    require(sha(snapshot / 'candles.csv') == manifest['sources']['snapshot_csv_sha256']['candles.csv'], 'Changed candles')
    paths = [Path(__file__), registry_path, ROOT / 'research/configs/initial_experiment.yaml',
             ROOT / 'research/initialization/TODO.md', ROOT / 'research/initialization/E00_sparse_gate_diagnosis.md',
             ROOT / 'research/environment/requirements.lock.txt', ROOT / 'research/initialization/version_manifest.json',
             bundle / 'manifest.json', bundle / 'split_index.csv', bundle / 'labels_forbidden_as_features.csv',
             snapshot / 'manifest.json', snapshot / 'candles.csv']
    paths.extend(run / name for name in ('training_attempts.json', 'selected_heads.json', 'development_features.csv', 'development_test_predictions.csv'))
    output.mkdir(parents=True)
    provenance = output / 'provenance'
    provenance.mkdir()
    # Freeze executable, configuration and manifests before computing any diagnostics.
    for path in paths:
        if path.suffix != '.csv':
            dest = provenance / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    dirty = []
    for line in git('-c', 'core.quotePath=false', 'status', '--porcelain', '--untracked-files=all').splitlines():
        path = ROOT / line[3:]
        if path.is_file():
            dirty.append({'path': line[3:], 'status': line[:2], 'sha256': sha(path)})
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': SCOPE,
            'hypothesis': 'Existing probability ranking may contain information despite sparse fixed-0.5 gates',
            'primary_target_coverage': .5, 'sensitivity_coverage': [.25, .75], 'no_best_coverage_selection': True,
            'threshold': 'numpy.quantile(validation probability, 1-q, method=linear); score >= cutoff including ties',
            'null': 'Posthoc equal selected count within calendar UTC month AND fixed momentum direction; without replacement',
            'random_attempts': [{'fold_id': f, 'target_coverage': q, 'seed': s, 'draw_count': DRAWS,
                                 'rng': 'numpy.random.default_rng', 'stream_order': 'sorted month/direction'}
                                for f in FOLDS for q in QUANTILES for s in SEEDS],
            'interval': '2.5/97.5 percentiles of 3000 matched-count draws, NOT confidence interval for strategy performance',
            'statistic': 'Mean independent fixed nominal reference-trade net return in bps; NOT portfolio return',
            'funding': 'funding_fees / actual reference stake is approximate diagnostic only; not exact fee decomposition',
            'final_holdout_used': False, 'refit': False, 'backtest': False, 'frozen_extraction': False,
            'source_commit': git('rev-parse', 'HEAD'), 'dirty_files': dirty,
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
            'config_full': (ROOT / 'research/configs/initial_experiment.yaml').read_text(encoding='utf-8')}
    write_json(output / 'diagnostic_plan.json', plan)
    splits = pd.read_csv(bundle / 'split_index.csv')
    splits = splits.loc[splits.fold_id.isin(FOLDS) & splits.included.eq(True)].copy()
    allowed = set(splits.opportunity_id)
    require(all(pd.Timestamp(i) < LIMIT for i in allowed), 'Holdout in development set')
    labels = filtered_csv(bundle / 'labels_forbidden_as_features.csv', 'opportunity_id', allowed).set_index('opportunity_id')
    require(set(labels.index) == allowed and labels.index.is_unique, 'Development label membership mismatch')
    require(pd.to_datetime(labels.labelable_at, utc=True).lt(LIMIT).all(), 'Holdout boundary in labels')
    numeric = ['direction'] + [f'{c}_{n}' for c in ('base', 'stress') for n in ('profit_ratio', 'funding_fees', 'stake_amount', 'target')]
    labels[numeric] = labels[numeric].apply(pd.to_numeric)
    for cost in ('base', 'stress'):
        require(np.array_equal(labels[f'{cost}_target'], labels[f'{cost}_profit_ratio'].gt(0).astype(int)), 'Target mismatch')
    labels['month'] = labels.entry_at.str[:7]
    features = pd.read_csv(run / 'development_features.csv').set_index('opportunity_id').loc[sorted(allowed)]
    require(features.index.is_unique and np.isfinite(features.to_numpy()).all(), 'Invalid features')
    predictions = pd.read_csv(run / 'development_test_predictions.csv').set_index('opportunity_id')
    attempts = json.loads((run / 'training_attempts.json').read_text(encoding='utf-8'))
    heads = json.loads((run / 'selected_heads.json').read_text(encoding='utf-8'))
    score_rows, rows, monthly, null_rows, strata_rows = [], [], [], [], []
    max_error = 0.
    for head in heads:
        fold = head['fold_id']
        require(fold in FOLDS, 'Unexpected selected fold')
        model = next(a for a in attempts if a['fold_id'] == fold and a['C'] == head['C'] and a['seed'] == head['seed'])
        scores = {}
        ids_by_role = {}
        for role in ('validation', 'test'):
            ids = splits.loc[splits.fold_id.eq(fold) & splits.role.eq(role), 'opportunity_id'].tolist()
            ids_by_role[role] = ids
            x = (features.loc[ids].to_numpy() - np.array(model['scaler_mean'])) / np.array(model['scaler_scale'])
            scores[role] = 1 / (1 + np.exp(-(x @ np.array(model['coef'][0]) + model['intercept'][0])))
            score_rows.extend({'fold_id': fold, 'role': role, 'opportunity_id': i, 'probability': float(p)} for i, p in zip(ids, scores[role]))
        max_error = max(max_error, float(np.max(np.abs(scores['test'] - predictions.loc[ids_by_role['test'], 'probability'].to_numpy()))))
        require(max_error < 1e-12, 'Retained score reconstruction mismatch')
        require(np.array_equal(scores['test'] >= .5, predictions.loc[ids_by_role['test'], 'gate'].to_numpy()), 'Historical gate mismatch')
        test = labels.loc[ids_by_role['test']].copy().reset_index()
        for q in QUANTILES:
            cutoff = float(np.quantile(scores['validation'], 1 - q, method='linear'))
            selected = scores['test'] >= cutoff
            n = int(selected.sum())
            require(n > 0, 'No selected opportunities')
            groups = []
            for (month, direction), group in test.groupby(['month', 'direction'], sort=True):
                indices = group.index.to_numpy()
                count = int(selected[indices].sum())
                require(0 <= count <= len(indices), 'Invalid stratum count')
                groups.append((indices, count))
                strata_rows.append({'fold_id': fold, 'target_coverage': q, 'month': month, 'direction': int(direction),
                                    'eligible_count': len(indices), 'selected_count': count})
            draws = {c: [] for c in ('base', 'stress')}
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                for draw in range(DRAWS):
                    indices = np.concatenate([rng.choice(group, size=count, replace=False) for group, count in groups])
                    require(len(indices) == n and len(set(indices)) == n, 'Matched-count null invalid')
                    record = {'fold_id': fold, 'target_coverage': q, 'seed': seed, 'draw': draw, 'selected_count': n}
                    for cost in draws:
                        value = float(test.loc[indices, f'{cost}_profit_ratio'].mean() * 10000)
                        draws[cost].append(value)
                        record[f'{cost}_mean_net_bps'] = value
                    null_rows.append(record)
            for cost in ('base', 'stress'):
                net = test[f'{cost}_profit_ratio'].to_numpy() * 10000
                mean = float(net[selected].mean())
                random = np.array(draws[cost])
                rows.append({'fold_id': fold, 'target_coverage': q, 'cost': cost, 'per_side_fee_bps': 7 if cost == 'base' else 14,
                             'validation_cutoff': cutoff, 'validation_n': len(scores['validation']),
                             'validation_actual_coverage': float((scores['validation'] >= cutoff).mean()),
                             'test_n': len(test), 'selected_count': n, 'actual_coverage': float(selected.mean()),
                             'selected_reference_mean_net_bps': mean, 'all_reference_mean_net_bps': float(net.mean()),
                             'selection_lift_vs_all_bps': mean - float(net.mean()), 'profit_probability': float((net[selected] > 0).mean()),
                             'random_mean_net_bps': float(random.mean()), 'random_p025_bps': float(np.quantile(random, .025)),
                             'random_p975_bps': float(np.quantile(random, .975)), 'model_random_percentile': float((np.sum(random < mean) + .5 * np.sum(random == mean)) / len(random)),
                             'scope': SCOPE})
                for month, group in test.groupby('month', sort=True):
                    month_selected = selected[group.index.to_numpy()]
                    values = group[f'{cost}_profit_ratio'].to_numpy() * 10000
                    month_mean = float(values[month_selected].mean()) if month_selected.any() else None
                    monthly.append({'fold_id': fold, 'target_coverage': q, 'cost': cost, 'month': month,
                                    'eligible_count': len(group), 'selected_count': int(month_selected.sum()),
                                    'selected_mean_net_bps': month_mean, 'all_mean_net_bps': float(values.mean()),
                                    'lift_bps': month_mean - float(values.mean()) if month_mean is not None else None})
    timestamps = set(labels.entry_at) | set(labels.exit_at)
    candles = filtered_csv(snapshot / 'candles.csv', 'bar_open_at', timestamps).set_index('bar_open_at')
    require(candles.index.is_unique and set(candles.index) == timestamps, 'Missing/ambiguous candle opens')
    opens = pd.to_numeric(candles.open)
    labels['gross_directional_open_to_open_bps'] = labels.direction.to_numpy() * (opens.loc[labels.exit_at].to_numpy() / opens.loc[labels.entry_at].to_numpy() - 1) * 10000
    labels['base_minus_stress_net_bps'] = (labels.base_profit_ratio - labels.stress_profit_ratio) * 10000
    labels['funding_contribution_approx_bps'] = labels.base_funding_fees / labels.base_stake_amount * 10000
    cost_rows = []
    test_ids = set(splits.loc[splits.role.eq('test'), 'opportunity_id'])
    for name, ids in [('all_unique_development_members', sorted(allowed)), ('all_unique_development_test', sorted(test_ids))]:
        data = labels.loc[ids]
        for cost in ('base', 'stress'):
            net = data[f'{cost}_profit_ratio'] * 10000
            cost_rows.append({'sample_scope': name, 'cost': cost, 'n': len(data), 'mean_gross_price_bps': float(data.gross_directional_open_to_open_bps.mean()),
                              'mean_net_bps': float(net.mean()), 'profit_probability': float(net.gt(0).mean()),
                              'win_mean_bps': float(net.loc[net.gt(0)].mean()), 'loss_mean_bps': float(net.loc[net.le(0)].mean()),
                              'mean_base_minus_stress_net_bps': float(data.base_minus_stress_net_bps.mean()),
                              'mean_funding_approx_bps': float(data.funding_contribution_approx_bps.mean()),
                              'mean_gross_minus_net_bps': float(data.gross_directional_open_to_open_bps.mean() - net.mean())})
    table = pd.DataFrame(rows)
    for name, frame in [('fixed_coverage_summary.csv', table), ('monthly_lift.csv', pd.DataFrame(monthly)),
                        ('matched_random_draws.csv', pd.DataFrame(null_rows)), ('matched_strata.csv', pd.DataFrame(strata_rows)),
                        ('reconstructed_scores.csv', pd.DataFrame(score_rows)), ('cost_diagnostics.csv', pd.DataFrame(cost_rows))]:
        frame.to_csv(output / name, index=False)
    labels.to_csv(output / 'development_only_cost_rows.csv')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True, sharex=True)
    for j, fold in enumerate(FOLDS):
        for i, cost in enumerate(('base', 'stress')):
            data = table.loc[table.fold_id.eq(fold) & table.cost.eq(cost)].sort_values('target_coverage')
            ax = axes[i, j]
            x = data.target_coverage.to_numpy() * 100
            ax.fill_between(x, data.random_p025_bps, data.random_p975_bps, alpha=.22, color='#607d8b', label='Matched random 95% draw interval')
            ax.plot(x, data.random_mean_net_bps, '--', color='#607d8b', label='Matched random mean')
            ax.plot(x, data.selected_reference_mean_net_bps, 'o-', color='#245da8', label='Existing model rank')
            ax.axhline(data.all_reference_mean_net_bps.iloc[0], color='#999999', linestyle=':', label='All reference opportunities')
            ax.axhline(0, color='black', linewidth=.6)
            ax.set_title(f'{fold} | {7 if cost == "base" else 14} bps/side | n={int(data.test_n.iloc[0])}')
            ax.set_xlabel('Validation target coverage (%)')
            ax.set_ylabel('Mean independent reference net (bps)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Post-E00 exploratory ranking diagnosis | existing development tests 2025Q2-2026Q1\nReference label means, NOT portfolio returns; random matched within UTC month and direction')
    fig.supxlabel('Source: retained E00 models + verified reference labels; no refit or new OOS evidence; holdout sealed', fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(output / f'coverage_vs_matched_random.{ext}', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for fold in FOLDS:
        data = table.loc[table.fold_id.eq(fold) & table.cost.eq('base')].sort_values('target_coverage')
        ax.plot(data.target_coverage * 100, data.actual_coverage * 100, 'o-', label=f'{fold}: n={int(data.test_n.iloc[0])}')
    ax.plot([25, 75], [25, 75], 'k--', linewidth=.8, label='Equal validation/test coverage')
    ax.set(xlabel='Validation target coverage (%)', ylabel='Actual development test coverage (%)', title='Fixed validation cutoff transfers imperfectly to development tests\nPost-E00 exploratory; 50% primary hypothesis, 25/75% sensitivity only')
    ax.legend(fontsize=8)
    for ext in ('png', 'svg'):
        fig.savefig(output / f'actual_coverage.{ext}', dpi=170)
    plt.close(fig)
    result = {'status': 'completed_exploratory_diagnostic', 'scope': SCOPE, 'final_holdout_used': False,
              'new_clean_oos_evidence': False, 'models_refitted': False, 'old_gates_modified': False,
              'max_saved_probability_error': max_error, 'random_draws': len(null_rows),
              'primary_coverage': .5, 'no_coverage_selected_by_results': True, 'cost_summary': cost_rows,
              'primary_coverage_results': table.loc[table.target_coverage.eq(.5)].to_dict('records'),
              'interpretation_limits': ['Label means are independent reference trades, not Freqtrade portfolio performance',
                                        'Matched random is posthoc null, not a deployable historical decision policy',
                                        '95% intervals describe random draws, not statistical confidence in economic value',
                                        'Funding/stake is approximate; gross-minus-net includes expense and denominator effects',
                                        'No constant win-probability threshold inferred as conditional profitability'],
              'artifact_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / 'report.json', result)
    print(json.dumps({'status': result['status'], 'random_draws': len(null_rows), 'max_probability_error': max_error}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    analyze(parser.parse_args().output_dir)
