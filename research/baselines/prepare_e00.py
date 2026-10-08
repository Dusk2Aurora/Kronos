"""Pre-register E00, fit ordinary heads on development data, emit decisions.

This module never computes final-holdout predictions or performance. Trading,
funding, wallet accounting and portfolio returns remain in official Freqtrade.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import yaml

from ordinary_features import FEATURE_NAMES, FEATURE_UNITS, features_for_opportunity

ROOT = Path(__file__).resolve().parents[2]
MODES = ('cash', 'buy_hold', 'fixed_momentum', 'constant_half_exposure',
         'vol_target', 'ordinary_features_gate')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, encoding='utf-8').rstrip('\r\n')


def iso(value):
    return pd.Timestamp(value).isoformat().replace('+00:00', 'Z')


def prepare(output):
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs'), 'Output must be in research/runs')
    require(not output.exists(), 'Use a new experiment directory')
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(config['task_id'] == 7 and config['first_round_head']['training_in_current_stage']
            == 'allowed_for_E00_ordinary_features_only', 'Ordinary training not authorized at this stage')
    require(not config['frozen_representation']['enabled_in_current_stage'], 'Frozen stage must remain disabled')
    require(config['labels_and_costs']['promotion_protocol']['status']
            == 'locked_before_development_performance_inspection', 'Lock promotion protocol first')
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('opportunities.csv', 'split_index.csv', 'labels_forbidden_as_features.csv'):
        require(sha(bundle / name) == manifest['artifact_sha256'][name], 'Label artifact hash changed: ' + name)
    snapshot = Path(manifest['sources']['snapshot'])
    require(sha(snapshot / 'manifest.json') == manifest['sources']['snapshot_manifest_sha256'], 'Snapshot manifest changed')
    require(sha(snapshot / 'candles.csv') == manifest['sources']['snapshot_csv_sha256']['candles.csv'], 'OHLCVA hash changed')
    engine = Path(config['backtest_engine']['repository_path'])
    require(git('rev-parse', 'HEAD', cwd=engine) == config['backtest_engine']['repository_commit'], 'Wrong engine commit')
    require(not git('status', '--porcelain', '--untracked-files=no', cwd=engine), 'Engine tracked files changed')
    output.mkdir(parents=True)
    provenance = output / 'provenance'
    provenance.mkdir()
    dirty = []
    for line in git('-c', 'core.quotePath=false', 'status', '--porcelain', '--untracked-files=all').splitlines():
        relative = line[3:]
        source = ROOT / relative
        if source.is_file():
            target = provenance / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            dirty.append({'path': relative, 'status': line[:2], 'sha256': sha(source)})
    for name, path in {'experiment_config.yaml': config_path,
                       'dataset_manifest.json': snapshot / 'manifest.json',
                       'labels_manifest.json': bundle / 'manifest.json',
                       'split_index.csv': bundle / 'split_index.csv',
                       'version_manifest.json': ROOT / 'research/initialization/version_manifest.json',
                       'kronos_requirements.lock.txt': ROOT / 'research/environment/requirements.lock.txt',
                       'freqtrade_requirements.lock.txt': ROOT / 'research/freqtrade/environment/requirements.lock.txt'}.items():
        shutil.copy2(path, provenance / name)
    canonical_paths = [ROOT / 'research/baselines/prepare_e00.py', ROOT / 'research/baselines/ordinary_features.py',
                       ROOT / 'research/baselines/run_e00.py', ROOT / 'research/freqtrade/strategies/KronosE00.py']
    for source in canonical_paths:
        target = provenance / 'canonical_source' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    registry = {'schema_version': 1, 'experiment_id': output.name,
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'E00', 'status': 'planned',
                'hypothesis': 'Causal ordinary OHLCVA features improve fixed momentum participation net of cost',
                'baseline_id': 'fixed_momentum', 'only_primary_change': 'ordinary_features_participation_gate',
                'configuration': {'path': str(config_path), 'sha256': sha(config_path), 'full_text': config_path.read_text(encoding='utf-8')},
                'provenance': {'repository_commit': git('rev-parse', 'HEAD'), 'dirty_files_and_sha256': dirty,
                               'engine_commit': config['backtest_engine']['repository_commit'],
                               'canonical_source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in canonical_paths},
                               'dataset_manifest_sha256': sha(snapshot / 'manifest.json'),
                               'label_manifest_sha256': sha(bundle / 'manifest.json'),
                               'environment_locks': {p.name: sha(p) for p in provenance.glob('*lock.txt')},
                               'version_manifest_sha256': sha(provenance / 'version_manifest.json')},
                'selection': {'test_used_for_model_or_threshold_selection': False,
                              'hyperparameter_selection': 'validation_log_loss',
                              'all_attempts': 'training_attempts.json', 'seeds': config['first_round_head']['seed_set']},
                'feature_contract': FEATURE_UNITS, 'final_holdout_inspected': False,
                'artifacts': {'root': str(output)}, 'result': {'conclusion': None}}
    registry_path = ROOT / 'research/registry' / (output.name + '.json')
    require(not registry_path.exists(), 'Experiment ID already registered')
    write_json(registry_path, registry)  # MUST precede fitting or performance inspection.
    split = pd.read_csv(bundle / 'split_index.csv')
    split = split.loc[split.fold_id.isin([f['id'] for f in config['splits']['walk_forward_dates']])
                      & split.included.eq(True)].copy()
    allowed = set(split.opportunity_id)
    holdout_start = pd.Timestamp(config['splits']['final_holdout'][0])
    require(all(pd.Timestamp(x) < holdout_start for x in allowed), 'Development includes holdout')
    opportunities = pd.read_csv(bundle / 'opportunities.csv')
    opportunities = opportunities.loc[opportunities.opportunity_id.isin(allowed)].copy()
    require(set(opportunities.opportunity_id) == allowed, 'Missing development opportunities')
    target_rows = []
    with (bundle / 'labels_forbidden_as_features.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['opportunity_id'] in allowed:
                target_rows.append((row['opportunity_id'], int(row['base_target'])))
    targets = pd.Series(dict(target_rows), name='base_target')
    require(targets.index.is_unique and set(targets.index) == allowed, 'Development targets incomplete')
    candles = pd.read_csv(snapshot / 'candles.csv', usecols=['bar_open_at', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    feature_rows = []
    for row in opportunities.to_dict('records'):
        values = features_for_opportunity(candles, row)
        require(values['momentum_direction_24h'] == row['direction'], 'Feature direction mismatch')
        feature_rows.append({'opportunity_id': row['opportunity_id'], **values})
    features = pd.DataFrame(feature_rows).set_index('opportunity_id')
    features.to_csv(output / 'development_features.csv')
    trials, predictions, selected = [], [], []
    opportunities = opportunities.set_index('opportunity_id')
    decisions_dir = output / 'decisions'
    decisions_dir.mkdir()
    try:
        for fold in config['splits']['walk_forward_dates']:
            groups = {}
            for role in ('train', 'validation', 'test'):
                members = split.loc[split.fold_id.eq(fold['id']) & split.role.eq(role)].sort_values('opportunity_id')
                ids = members.opportunity_id.tolist()
                require(ids and all(pd.Timestamp(opportunities.loc[x, 'labelable_at']) < pd.Timestamp(fold[role][1]) for x in ids), 'Unpurged label')
                groups[role] = ids
            require(not (set(groups['train']) & set(groups['validation']) or set(groups['train']) & set(groups['test'])
                         or set(groups['validation']) & set(groups['test'])), 'Fold role overlap')
            train_x = features.loc[groups['train'], list(FEATURE_NAMES)].to_numpy()
            scaler = StandardScaler().fit(train_x)
            require(scaler.n_samples_seen_ == len(groups['train']) and np.allclose(scaler.mean_, train_x.mean(axis=0)), 'Scaler not train-only')
            x_train = scaler.transform(train_x)
            x_valid = scaler.transform(features.loc[groups['validation'], list(FEATURE_NAMES)].to_numpy())
            y_train = targets.loc[groups['train']].to_numpy()
            require(set(y_train) == {0, 1}, 'Training fold needs both classes')
            candidates = []
            for C in sorted(config['first_round_head']['C']):
                for seed in config['first_round_head']['seed_set']:
                    pending = {'fold_id': fold['id'], 'C': C, 'seed': seed, 'status': 'planned'}
                    trials.append(pending)
                    write_json(output / 'training_attempts.json', trials)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', ConvergenceWarning)
                        model = LogisticRegression(C=C, random_state=seed, solver='lbfgs',
                                                   class_weight=None, max_iter=2000).fit(x_train, y_train)
                    valid_probability = model.predict_proba(x_valid)[:, 1]
                    score = float(log_loss(targets.loc[groups['validation']].to_numpy(), valid_probability, labels=[0, 1]))
                    record = {'fold_id': fold['id'], 'C': C, 'seed': seed, 'validation_log_loss': score,
                              'train_samples': len(groups['train']), 'validation_samples': len(groups['validation']),
                              'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
                              'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                              'validation_ids': groups['validation'], 'validation_probability': valid_probability.tolist(),
                              'iterations': model.n_iter_.tolist(), 'status': 'completed'}
                    pending.update(record)
                    write_json(output / 'training_attempts.json', trials)
                    candidates.append((score, C, seed, model))
            score, C, seed, model = min(candidates, key=lambda v: v[:3])
            probability = model.predict_proba(scaler.transform(features.loc[groups['test'], list(FEATURE_NAMES)].to_numpy()))[:, 1]
            gate = probability >= config['first_round_head']['decision_threshold']
            selected.append({'fold_id': fold['id'], 'C': C, 'seed': seed, 'validation_log_loss': score,
                             'test_samples': len(gate), 'participation_count': int(gate.sum())})
            for ident, prob, action in zip(groups['test'], probability, gate):
                predictions.append({'fold_id': fold['id'], 'opportunity_id': ident,
                                    'probability': float(prob), 'gate': int(action)})
            for mode in MODES:
                rows = []
                for ident, action in zip(groups['test'], gate):
                    direction = int(opportunities.loc[ident, 'direction'])
                    fraction = {'fixed_momentum': 1.0, 'constant_half_exposure': 0.5,
                                'ordinary_features_gate': float(action)}.get(mode)
                    if mode == 'vol_target':
                        annual_vol = features.loc[ident, 'hourly_log_return_std_72h'] * np.sqrt(config['baselines']['annualization_hours'])
                        fraction = min(1.0, config['baselines']['vol_target_annual'] / annual_vol) if annual_vol > 0 else 1.0
                    if mode == 'buy_hold':
                        rows = [{'signal_at': ident, 'direction': 1, 'exposure_fraction': 1.0}]
                        break
                    if mode != 'cash':
                        rows.append({'signal_at': ident, 'direction': direction, 'exposure_fraction': fraction})
                payload = {'schema_version': 1, 'mode': mode, 'decisions': rows}
                if mode == 'buy_hold':
                    payload['hold_until'] = iso(pd.Timestamp(fold['test'][1]) - pd.Timedelta(hours=1))
                write_json(decisions_dir / (fold['id'] + '_' + mode + '.json'), payload)
        pd.DataFrame(predictions).to_csv(output / 'development_test_predictions.csv', index=False)
        write_json(output / 'selected_heads.json', selected)
        registry['status'] = 'heads_fitted_pending_engine_backtests'
        registry['artifacts']['selected_heads'] = selected
        registry['artifacts']['preparation_hashes'] = {str(p.relative_to(output)): sha(p)
                                                      for p in output.rglob('*') if p.is_file()}
        write_json(registry_path, registry)
        print(json.dumps({'status': registry['status'], 'experiment': output.name, 'attempts': len(trials),
                          'development_feature_rows': len(features), 'selected': selected, 'holdout_inspected': False}))
    except Exception as error:
        if trials and trials[-1]['status'] == 'planned':
            trials[-1].update(status='failed', error=repr(error))
            write_json(output / 'training_attempts.json', trials)
        registry['status'] = 'failed_preparation'
        registry['result'] = {'conclusion': 'invalid_due_to_audit', 'failure_reason': repr(error)}
        write_json(registry_path, registry)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    prepare(parser.parse_args().output_dir)
