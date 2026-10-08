"""Pre-register E00R and fit development-only Ridge heads; no holdout analysis."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yaml

from ordinary_features_v2 import FEATURE_NAMES, FEATURE_SETS, FEATURE_UNITS, features_for_opportunity
from prepare_e00 import ROOT, git, require, sha, write_json

QUANTILES = (0.5, 0.25, 0.75)
ANCHOR = pd.Timestamp('2024-01-01T04:00:00Z')


def random_score(seed, opportunity_id):
    digest = hashlib.sha256(f'{seed}|{opportunity_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64


def periodic_action(opportunity_id, offset):
    stamp = pd.Timestamp(opportunity_id)
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0,
            'Periodic grid requires explicit UTC')
    elapsed = stamp.value - ANCHOR.value
    step = pd.Timedelta(hours=8).value
    require(elapsed % step == 0 and offset in (0, 1), 'Invalid periodic grid/offset')
    return int((elapsed // step) % 2 == offset)


def validation_cutoffs(scores):
    require(np.isfinite(scores).all(), 'Nonfinite validation predictions')
    return {f'rank{int(q * 100)}': float(np.quantile(scores, 1 - q, method='linear'))
            for q in QUANTILES}


def choose_candidate(candidates):
    # Exact ties only: never let test performance or fuzzy tolerances choose lambda.
    return min(candidates, key=lambda c: (c['validation_mse_bps_squared'], -c['lambda']))


def fit_candidates(train_x, train_y, valid_x, valid_y, lambdas, record):
    """Only train inputs fit the scaler/model; validation chooses regularization."""
    train_x, train_y = np.asarray(train_x), np.asarray(train_y)
    require(np.isfinite(train_x).all() and np.isfinite(train_y).all(), 'Nonfinite train inputs')
    scaler = StandardScaler().fit(train_x)
    scaled_train, scaled_valid = scaler.transform(train_x), scaler.transform(valid_x)
    candidates = []
    for lam in lambdas:
        alpha = len(train_y) * float(lam)
        record({'lambda': float(lam), 'alpha': alpha, 'status': 'running'})
        try:
            model = Ridge(alpha=alpha, fit_intercept=True, solver='svd').fit(scaled_train, train_y)
            prediction = model.predict(scaled_valid)
        except Exception as error:
            record({'lambda': float(lam), 'alpha': alpha, 'status': 'failed', 'error': repr(error)})
            raise
        candidate = {'lambda': float(lam), 'alpha': alpha,
                     'validation_mse_bps_squared': float(np.mean((prediction - valid_y) ** 2)),
                     'coef': model.coef_.tolist(), 'intercept': float(model.intercept_),
                     'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                     'scaler_var': scaler.var_.tolist(), 'scaler_n_samples_seen': int(scaler.n_samples_seen_),
                     'validation_predictions_bps': prediction.tolist(), 'status': 'completed'}
        record(candidate)
        candidates.append({**candidate, 'model': model})
    return scaler, choose_candidate(candidates)


def validate_design(config, revision):
    require(config['stage'] == 'ready_to_encode' and config['status'] == 'M0_accepted_ready_to_encode',
            'Expected accepted M0 stage')
    require(not config['frozen_representation']['enabled_in_current_stage'], 'Frozen extraction disabled')
    require(revision['experiment_family'] == 'E00R' and revision['final_holdout_access'] == 'forbidden'
            and revision['frozen_representation_extraction'] == 'forbidden', 'Wrong scope')
    h = revision['head']
    require(h['type'] == 'Ridge' and h['solver'] == 'svd' and h['fit_intercept'] is True
            and h['lambda_candidates'] == [.001, .01, .1, 1.0]
            and h['target_clipping'] == 'none' and h['sample_weight'] is None
            and h['tie_break'] == 'larger_lambda', 'Unsupported head design')
    require(revision['ordinary_features']['direction_aligned_horizons_hours'] == [1, 4, 12, 24, 72, 168]
            and revision['ordinary_features']['primary_count'] == 19, 'Wrong feature contract')
    require(revision['actions']['information_gate']['primary_validation_target_coverage'] == .5
            and revision['actions']['information_gate']['sensitivity_validation_target_coverage'] == [.25, .75]
            and revision['actions']['economic_gate']['costs_subtracted_again'] is False, 'Wrong gates')
    require(revision['target']['transformation'] == 'multiply_by_10000', 'Wrong target unit')
    folds = config['splits']['walk_forward_dates']
    require([f['id'] for f in folds] == ['WF01', 'WF02', 'WF03', 'WF04'], 'Only four development folds')


def prepare(output):
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs') and not output.exists(), 'Use new research/runs directory')
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    revision_path = ROOT / 'research/configs/baseline_revision_v2.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    revision = yaml.safe_load(revision_path.read_text(encoding='utf-8'))
    validate_design(config, revision)
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    require((ROOT / revision['target']['label_table']).resolve() == (bundle / 'labels_forbidden_as_features.csv').resolve(),
            'Revision must use verified labels')
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    for name in ('opportunities.csv', 'split_index.csv', 'labels_forbidden_as_features.csv'):
        require(sha(bundle / name) == manifest['artifact_sha256'][name], 'Changed label input ' + name)
    snapshot = Path(manifest['sources']['snapshot'])
    require(sha(snapshot / 'manifest.json') == manifest['sources']['snapshot_manifest_sha256'], 'Changed snapshot')
    require(sha(snapshot / 'candles.csv') == manifest['sources']['snapshot_csv_sha256']['candles.csv'], 'Changed OHLCVA')
    engine = Path(config['backtest_engine']['repository_path'])
    require(git('rev-parse', 'HEAD', cwd=engine) == config['backtest_engine']['repository_commit'], 'Wrong engine commit')
    require(not git('status', '--porcelain', '--untracked-files=no', cwd=engine), 'Modified engine core')
    registry_path = ROOT / 'research/registry' / (output.name + '.json')
    require(not registry_path.exists(), 'Already registered')
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
    copies = {'experiment_config.yaml': config_path, 'revision_config.yaml': revision_path,
              'dataset_manifest.json': snapshot / 'manifest.json', 'labels_manifest.json': bundle / 'manifest.json',
              'split_index.csv': bundle / 'split_index.csv',
              'version_manifest.json': ROOT / 'research/initialization/version_manifest.json',
              'kronos_requirements.lock.txt': ROOT / 'research/environment/requirements.lock.txt',
              'freqtrade_requirements.lock.txt': ROOT / 'research/freqtrade/environment/requirements.lock.txt'}
    for name, path in copies.items():
        shutil.copy2(path, provenance / name)
    paths = ['research/baselines/prepare_e00r.py', 'research/baselines/ordinary_features_v2.py',
             'research/baselines/ordinary_features.py', 'research/baselines/prepare_e00.py',
             'research/baselines/verify_e00r.py', 'research/freqtrade/strategies/KronosE00.py']
    for optional in ('run_e00r.py', 'summarize_e00r.py'):
        if (ROOT / 'research/baselines' / optional).is_file():
            paths.append('research/baselines/' + optional)
    for relative in paths:
        target = provenance / 'canonical_source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    planned = [{'fold_id': f['id'], 'feature_set': name, 'lambda': float(lam), 'status': 'planned'}
               for f in config['splits']['walk_forward_dates'] for name in FEATURE_SETS
               for lam in revision['head']['lambda_candidates']]
    registry = {'schema_version': 1, 'experiment_id': output.name, 'parent_experiment_id': 'E00_20261008_phase4_v1',
                'stage': 'E00R', 'status': 'planned',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'hypothesis': 'Net-return ordinary heads test selection information and positive expected net return separately',
                'baseline_id': 'fixed_momentum_and_noninformative_controls',
                'only_primary_change': 'predeclared_19_feature_Ridge_with_original13_ablation',
                'configuration': {'path': str(revision_path), 'sha256': sha(revision_path), 'full_text': revision_path.read_text(encoding='utf-8')},
                'parent_configuration': {'path': str(config_path), 'sha256': sha(config_path), 'full_text': config_path.read_text(encoding='utf-8')},
                'provenance': {'repository_commit': git('rev-parse', 'HEAD'), 'dirty_files_and_sha256': dirty,
                               'engine_commit': config['backtest_engine']['repository_commit'],
                               'canonical_source_sha256': {p: sha(ROOT / p) for p in paths},
                               'input_artifact_sha256': {name: sha(bundle / name) for name in manifest['artifact_sha256']},
                               'dataset_manifest_sha256': sha(snapshot / 'manifest.json'), 'label_manifest_sha256': sha(bundle / 'manifest.json'),
                               'environment_locks': {p.name: sha(p) for p in provenance.glob('*lock.txt')},
                               'version_manifest_sha256': sha(provenance / 'version_manifest.json')},
                'selection': {'test_used_for_model_or_threshold_selection': False, 'hyperparameter_selection': 'validation_MSE_bps_squared',
                              'tie_break': 'larger_lambda_exact_ties', 'all_attempts': planned},
                'feature_contract': FEATURE_UNITS, 'development_status': revision['evaluation']['development_status'],
                'final_holdout_inspected': False, 'artifacts': {'root': str(output)}, 'result': {'conclusion': None}}
    write_json(registry_path, registry)
    write_json(output / 'training_attempts.json', planned)
    try:
        split = pd.read_csv(bundle / 'split_index.csv')
        split = split.loc[split.fold_id.isin([f['id'] for f in config['splits']['walk_forward_dates']]) & split.included.eq(True)].copy()
        allowed = set(split.opportunity_id)
        holdout = pd.Timestamp(config['splits']['final_holdout'][0])
        require(allowed and all(pd.Timestamp(x) < holdout for x in allowed), 'Development includes holdout')
        opportunities = pd.read_csv(bundle / 'opportunities.csv')
        opportunities = opportunities.loc[opportunities.opportunity_id.isin(allowed)].set_index('opportunity_id')
        require(opportunities.index.is_unique and set(opportunities.index) == allowed, 'Missing opportunities')
        targets = {}
        with (bundle / 'labels_forbidden_as_features.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['opportunity_id'] in allowed:
                    require(row['opportunity_id'] not in targets, 'Duplicate development label')
                    targets[row['opportunity_id']] = float(row['base_profit_ratio']) * 10000
        targets = pd.Series(targets)
        require(set(targets.index) == allowed and np.isfinite(targets).all(), 'Incomplete targets')
        # No holdout OHLCVA values are parsed, transformed or validated.
        candles = pd.read_csv(snapshot / 'candles.csv', usecols=['bar_open_at', 'open', 'high', 'low', 'close', 'volume', 'amount'], dtype=str)
        candles = candles.loc[pd.to_datetime(candles.bar_open_at, utc=True) < holdout].copy()
        feature_rows = []
        for ident, row in opportunities.iterrows():
            values = features_for_opportunity(candles, row)
            require(values['momentum_direction_24h'] == row.direction and row.direction in (-1, 1), 'Direction mismatch')
            feature_rows.append({'opportunity_id': ident, **values})
        features = pd.DataFrame(feature_rows).set_index('opportunity_id')
        features.to_csv(output / 'development_features.csv')
        decisions_dir = output / 'decisions'
        decisions_dir.mkdir()
        predictions, selected, signals = [], [], []
        for fold in config['splits']['walk_forward_dates']:
            groups = {}
            for role in ('train', 'validation', 'test'):
                ids = sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq(role), 'opportunity_id'])
                start, end = map(pd.Timestamp, fold[role])
                require(ids and all(start <= pd.Timestamp(x) < end
                                    and pd.Timestamp(opportunities.loc[x, 'exit_at']) < end
                                    and pd.Timestamp(opportunities.loc[x, 'labelable_at']) < end for x in ids), 'Unpurged role')
                groups[role] = ids
            require(not any(set(groups[a]) & set(groups[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]), 'Role overlap')
            actions = {}
            train_mean = float(targets.loc[groups['train']].mean())
            for family, names in FEATURE_SETS.items():
                x_train = features.loc[groups['train'], list(names)].to_numpy(dtype=float)
                x_valid = features.loc[groups['validation'], list(names)].to_numpy(dtype=float)
                def record(candidate):
                    pending = next(p for p in planned if p['fold_id'] == fold['id'] and p['feature_set'] == family and p['lambda'] == candidate['lambda'])
                    pending.update(candidate, feature_names=list(names), train_ids=groups['train'], validation_ids=groups['validation'],
                                   train_samples=len(groups['train']), validation_samples=len(groups['validation']))
                    write_json(output / 'training_attempts.json', planned)
                scaler, winner = fit_candidates(x_train, targets.loc[groups['train']].to_numpy(), x_valid,
                                                targets.loc[groups['validation']].to_numpy(), revision['head']['lambda_candidates'], record)
                cutoffs = validation_cutoffs(winner['validation_predictions_bps'])
                scores = winner['model'].predict(scaler.transform(features.loc[groups['test'], list(names)].to_numpy(dtype=float)))
                gates = {gate: scores >= cutoff for gate, cutoff in cutoffs.items()}
                gates['economic'] = scores > 0
                selected.append({'fold_id': fold['id'], 'feature_set': family, 'lambda': winner['lambda'], 'alpha': winner['alpha'],
                                 'validation_mse_bps_squared': winner['validation_mse_bps_squared'], 'validation_cutoffs_bps': cutoffs,
                                 'training_mean_bps': train_mean, 'test_samples': len(scores), 'primary': family == 'ridge19'})
                for ident, score, idx in zip(groups['test'], scores, range(len(scores))):
                    predictions.append({'fold_id': fold['id'], 'feature_set': family, 'opportunity_id': ident, 'score_bps': float(score),
                                        'gate50': int(gates['rank50'][idx]), 'gate25': int(gates['rank25'][idx]),
                                        'gate75': int(gates['rank75'][idx]), 'economic': int(gates['economic'][idx])})
                for gate, values in gates.items():
                    actions[family + '_' + gate] = (values, family, gate, family == 'ridge19' and gate in ('rank50', 'economic'))
            for seed in revision['controls']['executable_random_gate']['seeds']:
                actions[f'random_q50_s{seed}'] = ([random_score(seed, x) < .5 for x in groups['test']], 'random', 'rank50', True)
            for offset in revision['controls']['periodic_half_gate']['offsets']:
                actions[f'periodic_offset{offset}'] = ([periodic_action(x, offset) for x in groups['test']], 'periodic', 'rank50', True)
            actions['constant_economic'] = ([train_mean > 0] * len(groups['test']), 'training_mean', 'economic', True)
            for mode, (values, family, gate, primary) in actions.items():
                path = Path('decisions') / (fold['id'] + '_' + mode + '.json')
                rows = [{'signal_at': ident, 'direction': int(opportunities.loc[ident, 'direction']), 'exposure_fraction': float(action)}
                        for ident, action in zip(groups['test'], values)]
                write_json(output / path, {'schema_version': 1, 'mode': 'ordinary_features_gate', 'decisions': rows})
                signals.append({'fold_id': fold['id'], 'mode': mode, 'path': path.as_posix(), 'selected_count': int(np.sum(values)),
                                'opportunities': len(values), 'family': family, 'gate': gate, 'primary': primary,
                                'sha256': sha(output / path), 'stress_reuses_base_signal': True})
        pd.DataFrame(predictions).to_csv(output / 'development_test_predictions.csv', index=False)
        write_json(output / 'selected_heads.json', selected)
        write_json(output / 'signals_manifest.json', {'schema_version': 1, 'signals': signals, 'final_holdout_inspected': False})
        registry['status'] = 'heads_fitted_pending_engine_backtests'
        registry['artifacts']['preparation_hashes'] = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
        write_json(registry_path, registry)
        print(json.dumps({'status': registry['status'], 'experiment': output.name, 'attempts': len(planned), 'holdout_inspected': False}))
    except Exception as error:
        registry['status'] = 'failed_preparation'
        registry['result'] = {'conclusion': 'invalid_due_to_audit', 'failure_reason': repr(error)}
        write_json(registry_path, registry)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    prepare(parser.parse_args().output_dir)
