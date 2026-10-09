"""Read-only diagnostics of fixed validation-selected three-head MLP ensembles."""
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

FAMILIES = ['mlp19', 'mlp_pretrained', 'mlp_random_s17', 'mlp_random_s29', 'mlp_random_s43']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(output):
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs'), 'Run outside research/runs')
    names = ['mlp_loss_diagnostics.json', 'selected_loss_diagnostics.csv', 'mlp_loss_chart_values.csv',
             'mlp_rmse_by_fold.png', 'mlp_rmse_by_fold.svg']
    require(not any((output / n).exists() for n in names), 'Refuse to overwrite diagnostic artifacts')
    registry_path = ROOT / 'research/registry' / (output.name + '.json')
    registry = read_json(registry_path)
    sources = {str(Path(__file__)): sha(Path(__file__))}
    def verified(relative):
        p = output / relative
        expected = registry['artifacts']['preparation_hashes'].get(relative)
        require(expected is not None and sha(p) == expected, 'Registered preparation changed: ' + relative)
        sources[str(p)] = expected
        return p
    config = yaml.safe_load(verified('provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    verified('provenance/common_config.yaml')
    require(config['stage'] == 'frozen_development' and not config['evaluation_readiness']['final_holdout_evaluation_allowed'], 'Scope changed')
    for entry, filename in [('configuration', 'common_config.yaml'), ('parent_configuration', 'experiment_config.yaml')]:
        require(sha(output / 'provenance' / filename) == registry[entry]['sha256'], 'Raw configuration hash changed')
    source_relative = 'provenance/heads_source/research/frozen/mlp_heads.py'
    verified(source_relative)
    require(sha(output / source_relative) == registry['provenance']['canonical_source_sha256']['research/frozen/mlp_heads.py'], 'Fitted head source mismatch')
    acceptance = read_json(verified('head_acceptance.json'))
    require(acceptance['status'] == 'passed' and acceptance['reloaded_models'] == 180, 'Model reload acceptance missing')
    heads = read_json(verified('selected_heads.json'))
    attempts = read_json(verified('training_attempts.json'))
    predictions = pd.read_csv(verified('development_test_predictions.csv'))
    constituents = pd.read_csv(verified('constituent_test_predictions.csv'))
    require(len(heads) == 20 and len(attempts) == 180 and len(predictions) == 5455 and len(constituents) == 16365, 'Registered budget differs')
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    manifest = read_json(bundle / 'manifest.json')
    require(sha(bundle / 'manifest.json') == registry['provenance']['label_manifest_sha256'], 'Label manifest changed')
    sources[str(bundle / 'manifest.json')] = sha(bundle / 'manifest.json')
    for name in ['split_index.csv', 'labels_forbidden_as_features.csv']:
        require(sha(bundle / name) == manifest['artifact_sha256'][name], 'Label bundle changed: ' + name)
        sources[str(bundle / name)] = sha(bundle / name)
    split = pd.read_csv(bundle / 'split_index.csv', dtype=str)
    split = split.loc[split.fold_id.isin([f['id'] for f in config['splits']['walk_forward_dates']]) & split.included.eq('True')]
    allowed = set(split.opportunity_id)
    cutoff = pd.Timestamp('2026-04-01T00:00Z')
    require(allowed and all(pd.Timestamp(x) < cutoff for x in allowed), 'Holdout membership rejected')
    labels = {}
    with (bundle / 'labels_forbidden_as_features.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['opportunity_id'] in allowed:
                require(row['opportunity_id'] not in labels and pd.Timestamp(row['labelable_at']) < cutoff, 'Invalid development label')
                labels[row['opportunity_id']] = float(row['base_profit_ratio']) * 10000
    require(set(labels) == allowed and np.isfinite(list(labels.values())).all(), 'Missing development labels')
    ridge_dir = ROOT / 'research/runs/ridge_loss_review_20261008_v1'
    ridge_manifest = read_json(ridge_dir / 'manifest.json')
    ridge_name = 'loss_per_family_fold_role.csv'
    require(sha(ridge_dir / ridge_name) == ridge_manifest['outputs_sha256'][ridge_name], 'Derived Ridge diagnostic changed')
    sources[str(ridge_dir / ridge_name)] = sha(ridge_dir / ridge_name)
    sources[str(ridge_dir / 'manifest.json')] = sha(ridge_dir / 'manifest.json')
    ridge = pd.read_csv(ridge_dir / ridge_name)
    matrices = {family: pd.read_csv(verified(f'features_{family}.csv')).set_index('opportunity_id') for family in FAMILIES}
    rows = []
    for fold in config['splits']['walk_forward_dates']:
        groups = {r: sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq(r), 'opportunity_id']) for r in ['train', 'validation', 'test']}
        ytrain = np.array([labels[x] for x in groups['train']])
        target_mean, target_std = float(ytrain.mean()), float(ytrain.std(ddof=0))
        for family in FAMILIES:
            h = next(h for h in heads if (h['fold_id'], h['feature_set']) == (fold['id'], family))
            require(h['selection_recorded_before_test_predictions'] and h['test_labels_loaded'] is False and h['seeds'] == [17, 29, 43], 'Selection not locked before test')
            members = [a for a in attempts if (a['fold_id'], a['feature_set'], a['weight_decay']) == (fold['id'], family, h['weight_decay'])]
            require(len(members) == 3 and {a['seed'] for a in members} == {17, 29, 43}, 'Seed membership differs')
            members.sort(key=lambda a: a['seed'])
            xtrain = matrices[family].loc[groups['train']].to_numpy(float)
            for a in members:
                require(a['status'] == 'completed' and a['train_ids'] == groups['train'] and a['validation_ids'] == groups['validation'], 'Selected fit membership differs')
                require(a['target_std_ddof'] == 0 and a['target_mean_bps'] == target_mean and a['target_std_bps'] == target_std, 'Train-only target scaling differs')
                require(np.allclose(xtrain.mean(0), a['scaler_mean']) and np.allclose(xtrain.var(0), a['scaler_var']), 'Train-only scaler differs')
                require(sha(verified(a['model_path'])) == a['model_file_sha256'], 'Fixed model state file differs')
            ensemble = {role: np.mean([a[role + '_predictions_bps'] for a in members], axis=0) for role in ['train', 'validation']}
            test = predictions.loc[predictions.fold_id.eq(fold['id']) & predictions.feature_set.eq(family)].sort_values('opportunity_id')
            require(test.opportunity_id.tolist() == groups['test'], 'Test prediction IDs differ')
            ensemble['test'] = test.score_bps.to_numpy(float)
            member_test = []
            for seed in [17, 29, 43]:
                p = constituents.loc[constituents.fold_id.eq(fold['id']) & constituents.feature_set.eq(family) & constituents.seed.eq(seed)].sort_values('opportunity_id')
                require(p.opportunity_id.tolist() == groups['test'], 'Constituent test IDs differ')
                member_test.append(p.score_bps.to_numpy(float))
            require(np.allclose(np.mean(member_test, axis=0), ensemble['test'], rtol=1e-10, atol=1e-10), 'Test arithmetic ensemble differs')
            for role, pred in ensemble.items():
                y = np.array([labels[x] for x in groups[role]])
                require(pred.shape == y.shape and np.isfinite(pred).all(), 'Prediction dimensions/finite')
                mse = float(np.mean((pred - y) ** 2))
                constant = float(np.mean((target_mean - y) ** 2))
                if role == 'validation':
                    require(np.allclose(pred, h['validation_predictions_bps']) and np.isclose(mse, h['validation_mse_bps_squared']), 'Selected validation ensemble loss differs')
                ridge_family = family.replace('mlp', 'ridge')
                r = ridge.loc[ridge.fold_id.eq(fold['id']) & ridge.family.eq(ridge_family) & ridge.role.eq(role)]
                require(len(r) == 1 and int(r.samples.iloc[0]) == len(y), 'Ridge comparison membership differs')
                ridge_mse = float(r.MSE_bps_squared.iloc[0])
                rows.append({'fold_id': fold['id'], 'family': family, 'role': role, 'test_start_utc': fold['test'][0],
                             'test_end_exclusive_utc': fold['test'][1], 'role_start_utc': fold[role][0], 'role_end_exclusive_utc': fold[role][1],
                             'samples': len(y), 'train_samples': len(ytrain), 'selected_weight_decay': h['weight_decay'],
                             'head_seeds': '17|29|43', 'ensemble_MSE_bps_squared': mse, 'ensemble_RMSE_bps': float(np.sqrt(mse)),
                             'constant_MSE_bps_squared': constant, 'constant_RMSE_bps': float(np.sqrt(constant)),
                             'relative_MSE_improvement_vs_constant': 1 - mse / constant,
                             'Ridge_MSE_bps_squared': ridge_mse, 'Ridge_RMSE_bps': float(np.sqrt(ridge_mse)),
                             'relative_MSE_improvement_vs_Ridge': 1 - mse / ridge_mse})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'selected_loss_diagnostics.csv', index=False)
    summary = []
    for (fold, role), group in table.groupby(['fold_id', 'role'], sort=False):
        for family in ['mlp19', 'mlp_pretrained', 'mean_three_random_MLP_losses']:
            g = group.loc[group.family.eq(family)] if not family.startswith('mean') else group.loc[group.family.str.startswith('mlp_random')]
            require(len(g) == (3 if family.startswith('mean') else 1), 'Random loss group incomplete')
            r = g.iloc[0]
            mse, ridge_mse = float(g.ensemble_MSE_bps_squared.mean()), float(g.Ridge_MSE_bps_squared.mean())
            summary.append({'fold_id': fold, 'role': role, 'family': family, 'test_start_utc': r.test_start_utc,
                            'MSE_bps_squared': mse, 'RMSE_bps': float(np.sqrt(mse)),
                            'constant_MSE_bps_squared': r.constant_MSE_bps_squared, 'constant_RMSE_bps': r.constant_RMSE_bps,
                            'Ridge_MSE_bps_squared': ridge_mse, 'Ridge_RMSE_bps': float(np.sqrt(ridge_mse)),
                            'relative_MSE_improvement_vs_constant': 1 - mse / r.constant_MSE_bps_squared,
                            'relative_MSE_improvement_vs_Ridge': 1 - mse / ridge_mse})
    summary = pd.DataFrame(summary)
    summary.to_csv(output / 'mlp_loss_chart_values.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), constrained_layout=True)
    labels = {'mlp19': 'Ordinary 19 MLP ensemble', 'mlp_pretrained': '19 + pretrained 512 MLP ensemble',
              'mean_three_random_MLP_losses': '19 + random 512: mean 3 ensemble losses'}
    colors = ['#3366AA', '#CC5500', '#008866']
    for ax, role in zip(axes, ['train', 'validation', 'test']):
        group = summary.loc[summary.role.eq(role)]
        for (family, label), color in zip(labels.items(), colors):
            g = group.loc[group.family.eq(family)].sort_values('test_start_utc')
            ax.plot(pd.to_datetime(g.test_start_utc, utc=True), g.RMSE_bps, lw=2, marker='o', label=label, color=color)
        g = group.loc[group.family.eq('mlp19')].sort_values('test_start_utc')
        ax.plot(pd.to_datetime(g.test_start_utc, utc=True), g.constant_RMSE_bps, ls='--', color='#444444', marker='s', label='Training-mean constant')
        ax.set_title({'train': 'Train (expanding earlier history)', 'validation': 'Validation (preceding quarter)', 'test': 'Development test (named quarter)'}[role])
        ax.set_ylabel('RMSE (net return bps)')
        ax.set_xlabel('Corresponding development test quarter start (UTC)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=30)
        ax.grid(alpha=.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.07))
    fig.suptitle('Fixed validation-selected 3-head MLP ensembles | no refit | official independent net-return labels\nTrain, validation and test cover different periods/distributions; 7 bps per side + realized funding', fontsize=12)
    for ext in ['png', 'svg']:
        fig.savefig(output / ('mlp_rmse_by_fold.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)
    write_json(output / 'mlp_loss_diagnostics.json', {'status': 'completed_read_only_loss_diagnostics', 'new_fits': 0,
               'holdout_inspected': False, 'parameters_changed': False, 'seed_selection': False,
               'source_sha256': sources, 'selected_ensembles': 20, 'selected_constituents': 60,
               'ensemble_train_loss': 'MSE_of_arithmetic_mean_of_all_three_saved_train_predictions_not_mean_member_MSE',
               'random_summary': 'mean_MSE_of_three_random_backbone_ensembles_then_square_root_for_RMSE',
               'interpretation': 'Lower_train_loss_does_not_establish_usable_information; chronological_roles_have_different_noise_and_periods; dev_results_exploratory_not_new_clean_OOS',
               'Ridge_reference': str(ridge_dir), 'rows': summary.to_dict(orient='records'),
               'output_sha256': {name: sha(output / name) for name in names if name != 'mlp_loss_diagnostics.json'}})
    print(summary[['fold_id', 'role', 'family', 'RMSE_bps', 'relative_MSE_improvement_vs_constant', 'relative_MSE_improvement_vs_Ridge']].round(4).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/runs/FROZEN_MLP_20261008_v2')
    run(parser.parse_args().output_dir)
