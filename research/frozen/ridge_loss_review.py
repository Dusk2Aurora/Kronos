"""Read-only Ridge train/validation/development-test loss review; no fitting."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(output):
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs') and not output.exists(), 'Use a new review directory')
    old = ROOT / 'research/runs/FROZEN_20261008_v1'
    registry_path = ROOT / 'research/registry/FROZEN_20261008_v1.json'
    registry = read_json(registry_path)
    require(registry['status'] == 'completed_exploratory_development', 'Prior Ridge run not accepted')
    sources = {str(registry_path): sha(registry_path), str(Path(__file__)): sha(Path(__file__))}
    def verified(relative):
        expected = None
        for group in ['preparation_hashes', 'analysis_hashes', 'final_hashes', 'engine_hashes']:
            expected = registry['artifacts'].get(group, {}).get(relative)
            if expected:
                break
        p = old / relative
        require(expected is not None and sha(p) == expected, 'Prior raw artifact changed: ' + relative)
        sources[str(p)] = expected
        return p
    config = yaml.safe_load(verified('provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    metrics = pd.read_csv(verified('prediction_metrics.csv'))
    ordinary_attempts = read_json(verified('baseline_reuse/training_attempts.json'))
    ordinary_heads = read_json(verified('baseline_reuse/selected_heads.json'))
    frozen_attempts = read_json(verified('training_attempts.json'))
    frozen_heads = read_json(verified('selected_heads.json'))
    ordinary_predictions = pd.read_csv(verified('baseline_reuse/development_test_predictions.csv'))
    frozen_predictions = pd.read_csv(verified('development_test_predictions.csv'))
    matrices = {'ridge19': pd.read_csv(verified('baseline_reuse/development_features.csv')).set_index('opportunity_id')}
    families = ['ridge19', 'ridge_pretrained', 'ridge_random_s17', 'ridge_random_s29', 'ridge_random_s43']
    for family in families[1:]:
        matrices[family] = pd.read_csv(verified(f'features_{family}.csv')).set_index('opportunity_id')
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    manifest = read_json(bundle / 'manifest.json')
    require(sha(bundle / 'manifest.json') == registry['provenance']['label_manifest_sha256'], 'Label manifest changed')
    sources[str(bundle / 'manifest.json')] = sha(bundle / 'manifest.json')
    for name in ['split_index.csv', 'labels_forbidden_as_features.csv']:
        require(sha(bundle / name) == manifest['artifact_sha256'][name], 'Bundle changed: ' + name)
        sources[str(bundle / name)] = sha(bundle / name)
    split = pd.read_csv(bundle / 'split_index.csv', dtype=str)
    split = split.loc[split.fold_id.isin([f['id'] for f in config['splits']['walk_forward_dates']]) & split.included.eq('True')]
    allowed = set(split.opportunity_id)
    cutoff = pd.Timestamp('2026-04-01T00:00Z')
    require(all(pd.Timestamp(x) < cutoff for x in allowed), 'Holdout ID encountered')
    labels = {}
    with (bundle / 'labels_forbidden_as_features.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['opportunity_id'] in allowed:
                require(row['opportunity_id'] not in labels and pd.Timestamp(row['labelable_at']) < cutoff, 'Invalid label membership')
                labels[row['opportunity_id']] = float(row['base_profit_ratio']) * 10000
    require(set(labels) == allowed and np.isfinite(list(labels.values())).all(), 'Missing development labels')
    rows, heads = [], []
    for fold in config['splits']['walk_forward_dates']:
        groups = {r: sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq(r), 'opportunity_id']) for r in ['train', 'validation', 'test']}
        ty = np.array([labels[x] for x in groups['train']])
        train_mean = float(ty.mean())
        for family in families:
            attempts = ordinary_attempts if family == 'ridge19' else frozen_attempts
            chosen = ordinary_heads if family == 'ridge19' else frozen_heads
            h = next(h for h in chosen if h['fold_id'] == fold['id'] and h['feature_set'] == family)
            candidates = [a for a in attempts if a['fold_id'] == fold['id'] and a['feature_set'] == family]
            winner = min(candidates, key=lambda a: (a['validation_mse_bps_squared'], -a['lambda']))
            require(winner['status'] == 'completed' and winner['lambda'] == h['lambda'] and
                    winner['validation_mse_bps_squared'] == h['validation_mse_bps_squared'], 'Validation selected head differs')
            require(winner['train_ids'] == groups['train'] and winner['validation_ids'] == groups['validation'], 'Fit membership differs')
            matrix = matrices[family]
            require(matrix.index.is_unique and set(matrix.index) == allowed and np.isfinite(matrix).all().all(), 'Matrix membership/finite')
            tx = matrix.loc[groups['train']].to_numpy(float)
            require(np.allclose(tx.mean(0), winner['scaler_mean']) and np.allclose(tx.var(0), winner['scaler_var']), 'Training scaler differs')
            require(np.isclose(train_mean, h['training_mean_bps']), 'Training constant differs')
            head_pred = {}
            for role in ['train', 'validation', 'test']:
                x = matrix.loc[groups[role]].to_numpy(float)
                pred = ((x - np.asarray(winner['scaler_mean'])) / np.asarray(winner['scaler_scale'])) @ np.asarray(winner['coef']) + winner['intercept']
                y = np.array([labels[x] for x in groups[role]])
                mse = float(np.mean((pred - y) ** 2))
                null = float(np.mean((train_mean - y) ** 2))
                if role == 'validation':
                    require(np.allclose(pred, winner['validation_predictions_bps']) and np.isclose(mse, h['validation_mse_bps_squared']), 'Validation reconstruction differs')
                if role == 'test':
                    stored = ordinary_predictions if family == 'ridge19' else frozen_predictions
                    stored = stored.loc[stored.fold_id.eq(fold['id']) & stored.feature_set.eq(family)].sort_values('opportunity_id')
                    require(stored.opportunity_id.tolist() == groups[role] and np.allclose(pred, stored.score_bps), 'Test reconstruction differs')
                    metric = metrics.loc[metrics.fold_id.eq(fold['id']) & metrics.feature_variant.eq(family)]
                    require(len(metric) == 1 and np.isclose(mse, metric.MSE_bps_squared.iloc[0]) and
                            np.isclose(null, metric.constant_MSE_bps_squared.iloc[0]), 'Registered test loss differs')
                rows.append({'fold_id': fold['id'], 'test_start_utc': fold['test'][0], 'test_end_exclusive_utc': fold['test'][1],
                             'family': family, 'role': role, 'samples': len(y), 'feature_count': matrix.shape[1],
                             'selected_lambda': h['lambda'], 'train_samples': len(ty), 'training_mean_bps': train_mean,
                             'MSE_bps_squared': mse, 'RMSE_bps': float(np.sqrt(mse)),
                             'constant_train_mean_MSE_bps_squared': null, 'constant_train_mean_RMSE_bps': float(np.sqrt(null)),
                             'relative_MSE_improvement': 1 - mse / null,
                             'role_start_utc': fold[role][0], 'role_end_exclusive_utc': fold[role][1]})
            heads.append({'fold_id': fold['id'], 'family': family, 'feature_count': matrix.shape[1],
                          'selected_lambda': h['lambda'], 'train_samples': len(ty),
                          'validation_samples': len(groups['validation']), 'test_samples': len(groups['test'])})
    output.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'loss_per_family_fold_role.csv', index=False)
    pd.DataFrame(heads).to_csv(output / 'selected_lambda_and_samples.csv', index=False)
    summarized = []
    for (fold_id, role), group in table.groupby(['fold_id', 'role'], sort=False):
        for family in ['ridge19', 'ridge_pretrained', 'mean_three_random_Ridge_losses']:
            subset = group.loc[group.family.eq(family)] if family != 'mean_three_random_Ridge_losses' else group.loc[group.family.str.startswith('ridge_random')]
            require(len(subset) == (3 if family == 'mean_three_random_Ridge_losses' else 1), 'Missing model group')
            row = subset.iloc[0]
            mse = float(subset.MSE_bps_squared.mean())
            summarized.append({'fold_id': fold_id, 'test_start_utc': row.test_start_utc, 'test_end_exclusive_utc': row.test_end_exclusive_utc,
                               'role': role, 'family': family, 'samples': int(row.samples), 'MSE_bps_squared': mse,
                               'RMSE_bps': float(np.sqrt(mse)), 'constant_MSE_bps_squared': float(row.constant_train_mean_MSE_bps_squared),
                               'constant_RMSE_bps': float(row.constant_train_mean_RMSE_bps),
                               'relative_MSE_improvement': 1 - mse / row.constant_train_mean_MSE_bps_squared,
                               'random_summary': 'mean_of_three_model_losses_not_ensemble_prediction_loss' if family.startswith('mean') else None})
    summary = pd.DataFrame(summarized)
    summary.to_csv(output / 'loss_chart_values.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), constrained_layout=True)
    names = {'ridge19': 'Ordinary 19 Ridge', 'ridge_pretrained': '19 + pretrained 512 Ridge',
             'mean_three_random_Ridge_losses': '19 + random 512: mean 3 losses'}
    colors = {'ridge19': '#3366AA', 'ridge_pretrained': '#CC5500', 'mean_three_random_Ridge_losses': '#008866'}
    for ax, role in zip(axes, ['train', 'validation', 'test']):
        g = summary.loc[summary.role.eq(role)].copy()
        for family in names:
            selected = g.loc[g.family.eq(family)].sort_values('test_start_utc')
            dates = pd.to_datetime(selected.test_start_utc, utc=True)
            ax.plot(dates, selected.RMSE_bps, marker='o', lw=2, color=colors[family], label=names[family])
        constant = g.loc[g.family.eq('ridge19')].sort_values('test_start_utc')
        ax.plot(pd.to_datetime(constant.test_start_utc, utc=True), constant.constant_RMSE_bps, ls='--', marker='s', color='#444444', label='Training-mean constant')
        ax.set_title({'train': 'Train (expanding earlier history)', 'validation': 'Validation (preceding quarter)', 'test': 'Development test (named quarter)'}[role])
        ax.set_ylabel('RMSE (net return bps)')
        ax.set_xlabel('Corresponding development test quarter start (UTC)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=30)
        ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.07))
    fig.suptitle('Saved validation-selected Ridge heads | no refit | official independent net-return labels\nTrain, validation and test cover different periods/distributions; 7 bps per side + realized funding', fontsize=12)
    for ext in ['png', 'svg']:
        fig.savefig(output / ('ridge_rmse_by_fold.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)
    note = '''# Ridge loss review

This diagnostic reconstructs saved validation-selected Ridge heads; it does not fit or select any new model.
Targets are official independent net-return labels in bps, net of the base 7 bps per-side cost proxy and realized funding.
All four folds retain ordinary19, pretrained512, and every random backbone seed (17, 29, 43).
Random summary MSE averages the three models' squared-error losses; plotted RMSE is the square root of that mean MSE, not a prediction ensemble and not the mean of three RMSEs.
The null benchmark predicts the training-fold mean on every role. Its train MSE equals the population variance of training outcomes; this is a no-conditional-feature-information benchmark.

The UTC x-axis names the corresponding development test quarter. Train expands over earlier history and validation uses the preceding quarter. Train/validation/test curves are not three evaluations on the same outcome distribution, so their gaps alone cannot prove underfitting, overfitting, or nonstationarity.
A lower training loss establishes in-sample fit only. Poor test performance can reflect noise, insufficient usable features, regularization, model limitations or time-varying relations; this diagnostic cannot distinguish these causes by itself.
531 inputs pass through a regularized linear Ridge readout; the feature count alone does not imply optimization failure. Ridge coefficients were already fitted by a direct SVD solver. A weak readout can fail to extract a nonlinear relation even when its linear optimization succeeds, but these losses do not establish that nonlinear relations exist.

Final holdout outcomes were filtered before numeric conversion and were not inspected. No old run, registry, prediction or selection artifact was modified.
'''
    (output / 'README.md').write_text(note, encoding='utf-8')
    write_json(output / 'manifest.json', {'status': 'read_only_derived_diagnostic_complete', 'new_fits': 0,
               'holdout_inspected': False, 'development_scope': 'previously_seen_exploratory_development',
               'source_sha256': sources, 'loss_reconstruction_verified': True,
               'outputs_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
               'random_summary': 'sqrt_mean_MSE_of_all_three_random_backbones_not_seed_selection_or_prediction_ensemble'})
    print(summary.to_json(orient='records'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/runs/ridge_loss_review_20261008_v1')
    run(parser.parse_args().output_dir)
