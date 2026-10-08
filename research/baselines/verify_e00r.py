"""Targeted synthetic E00R causal checks; optionally audit a prepared run read-only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ordinary_features import FEATURE_NAMES as ORIGINAL, RETURN_HOURS, ordinary_features
from ordinary_features_v2 import FEATURE_NAMES, features_for_opportunity
from prepare_e00r import (ROOT, choose_candidate, fit_candidates, periodic_action,
                          random_score, require, sha, validation_cutoffs, validate_design)
import yaml


def synthetic_checks():
    dates = pd.date_range('2024-01-01T00:00:00Z', periods=280, freq='h')
    close = 100 * np.exp(np.arange(280) * .001 + .003 * np.sin(np.arange(280)))
    candles = pd.DataFrame({'bar_open_at': dates, 'open': close, 'high': close * 1.01,
                            'low': close * .99, 'close': close, 'volume': np.ones(280),
                            'amount': np.full(280, 100.)})
    opp = {'decision_boundary_at': dates[256], 'history_start_row': 0, 'history_end_row_exclusive': 256,
           'base_profit_ratio': 999, 'stress_profit_ratio': -999, 'funding': 999}
    values = features_for_opportunity(candles, opp)
    original = ordinary_features(candles, decision_boundary=dates[256], history_start_row=0, history_end_row_exclusive=256)
    require(tuple(values) == FEATURE_NAMES and len(FEATURE_NAMES) == 19, 'Feature contract')
    require(all(values[k] == original[k] for k in ORIGINAL), 'Changed original features')
    for horizon in RETURN_HOURS:
        require(values[f'direction_aligned_log_return_{horizon}h'] == values['momentum_direction_24h'] * values[f'log_return_{horizon}h'], 'Interaction mismatch')
    changed = candles.copy()
    changed.loc[256:, ['open', 'high', 'low', 'close', 'volume', 'amount']] = np.nan
    opp.update(base_profit_ratio=-888, stress_profit_ratio=888, funding=-888)
    require(values == features_for_opportunity(changed, opp), 'Future/label leakage')
    falling = candles.copy()
    for name in ['open', 'high', 'low', 'close']:
        falling[name] = candles[name].to_numpy()[::-1]
    negative = features_for_opportunity(falling, opp)
    require(negative['momentum_direction_24h'] == -1 and negative['direction_aligned_log_return_24h'] > 0, 'Short alignment')
    bad = candles.copy()
    bad['base_profit_ratio'] = 1
    try:
        features_for_opportunity(bad, opp)
    except ValueError:
        pass
    else:
        raise AssertionError('Outcome column accepted as candle feature')
    rng = np.random.default_rng(17)
    train = rng.normal(size=(40, 19))
    valid = rng.normal(size=(10, 19))
    ytrain = rng.normal(size=40) * 100
    yvalid = rng.normal(size=10) * 100
    recorded = []
    scaler, winner = fit_candidates(train, ytrain, valid, yvalid, [.001, .01, .1, 1], recorded.append)
    require(np.allclose(scaler.mean_, train.mean(axis=0)) and scaler.n_samples_seen_ == len(train), 'Scaler not train-only')
    changed_records = []
    fit_candidates(train, ytrain, valid * 1000, yvalid - 999, [.001, .01, .1, 1], changed_records.append)
    completed = [x for x in recorded if x['status'] == 'completed']
    changed_completed = [x for x in changed_records if x['status'] == 'completed']
    for before, after in zip(completed, changed_completed):
        require(before['coef'] == after['coef'] and before['intercept'] == after['intercept']
                and before['scaler_mean'] == after['scaler_mean'] and before['alpha'] == len(train) * before['lambda'], 'Validation contaminates fitting')
    require(winner['lambda'] == min(completed, key=lambda x: (x['validation_mse_bps_squared'], -x['lambda']))['lambda'], 'MSE choice')
    tied = [{'lambda': lam, 'validation_mse_bps_squared': 1.} for lam in [.001, .01, .1, 1]]
    require(choose_candidate(tied)['lambda'] == 1, 'Exact tie-break')
    cutoff = validation_cutoffs(np.array([-2., 0., 0., 2.]))
    require(cutoff == {'rank50': 0., 'rank25': .5, 'rank75': -.5}, 'Linear quantile')
    test = np.array([0., 1., 500.])
    require((test >= cutoff['rank50']).all() and not (test > 0)[0], 'Rank ties / strict economic gate')
    ids = ['2024-01-01T04:00:00Z', '2024-01-01T12:00:00Z', '2024-01-01T20:00:00Z']
    for ident in ids:
        expected = int.from_bytes(hashlib.sha256(f'17|{ident}'.encode()).digest()[:8], 'big') / 2**64
        require(random_score(17, ident) == expected and periodic_action(ident, 0) + periodic_action(ident, 1) == 1, 'Control contract')
    require([periodic_action(x, 0) for x in ids] == [1, 0, 1]
            and [periodic_action(x, 0) for x in ids[1:]] == [0, 1], 'Purge changes periodic phase')
    initial = yaml.safe_load((ROOT / 'research/configs/initial_experiment.yaml').read_text(encoding='utf-8'))
    revision = yaml.safe_load((ROOT / 'research/configs/baseline_revision_v2.yaml').read_text(encoding='utf-8'))
    validate_design(initial, revision)
    return ['direction_interactions_and_original13', 'future_perturbation_and_label_rejection',
            'train_only_scaler_and_ridge', 'alpha_and_validation_MSE_selection_exact_tie',
            'fixed_validation_quantiles_and_strict_economic_zero', 'stable_hash_and_absolute_periodic_phase',
            'ordinary_only_development_scope']


def audit_run(run):
    run = run.resolve()
    registry = json.loads((ROOT / 'research/registry' / (run.name + '.json')).read_text(encoding='utf-8'))
    for relative, digest in registry['artifacts']['preparation_hashes'].items():
        require(sha(run / relative) == digest, 'Preparation changed: ' + relative)
    config = yaml.safe_load((run / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    holdout = pd.Timestamp(config['splits']['final_holdout'][0])
    features = pd.read_csv(run / 'development_features.csv').set_index('opportunity_id')
    require(tuple(features.columns) == FEATURE_NAMES and all(pd.Timestamp(x) < holdout for x in features.index), 'Feature scope')
    attempts = json.loads((run / 'training_attempts.json').read_text(encoding='utf-8'))
    require(len(attempts) == 32 and all(x['status'] == 'completed' and x['fold_id'] != 'FINAL' for x in attempts), 'Attempt scope')
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    targets = {}
    with (bundle / 'labels_forbidden_as_features.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['opportunity_id'] in features.index:
                targets[row['opportunity_id']] = float(row['base_profit_ratio']) * 10000
    targets = pd.Series(targets)
    recomputed = {}
    for attempt in attempts:
        x = features.loc[attempt['train_ids'], attempt['feature_names']].to_numpy()
        require(np.allclose(x.mean(axis=0), attempt['scaler_mean']) and len(x) == attempt['scaler_n_samples_seen'], 'Persisted scaler isolation')
        require(all(pd.Timestamp(x) < holdout for x in attempt['train_ids'] + attempt['validation_ids']), 'Holdout fitting')
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=len(x) * attempt['lambda'], solver='svd', fit_intercept=True).fit(scaler.transform(x), targets.loc[attempt['train_ids']].to_numpy())
        pred = model.predict(scaler.transform(features.loc[attempt['validation_ids'], attempt['feature_names']].to_numpy()))
        mse = float(np.mean((pred - targets.loc[attempt['validation_ids']].to_numpy()) ** 2))
        require(np.allclose(scaler.scale_, attempt['scaler_scale']) and np.allclose(model.coef_, attempt['coef'])
                and np.isclose(model.intercept_, attempt['intercept']) and np.allclose(pred, attempt['validation_predictions_bps'])
                and np.isclose(mse, attempt['validation_mse_bps_squared']), 'Fit/prediction mismatch')
        recomputed.setdefault((attempt['fold_id'], attempt['feature_set']), []).append({**attempt, 'model': model, 'scaler': scaler})
    heads = json.loads((run / 'selected_heads.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(run / 'development_test_predictions.csv')
    for head in heads:
        winner = choose_candidate(recomputed[(head['fold_id'], head['feature_set'])])
        require(winner['lambda'] == head['lambda'], 'Wrong selected lambda')
        cutoffs = validation_cutoffs(winner['validation_predictions_bps'])
        require(all(np.isclose(cutoffs[k], head['validation_cutoffs_bps'][k]) for k in cutoffs), 'Threshold mismatch')
        rows = predictions.loc[predictions.fold_id.eq(head['fold_id']) & predictions.feature_set.eq(head['feature_set'])]
        require(all(pd.Timestamp(x) < holdout for x in rows.opportunity_id), 'Holdout predictions')
        pred = winner['model'].predict(winner['scaler'].transform(features.loc[rows.opportunity_id, winner['feature_names']].to_numpy()))
        require(np.allclose(pred, rows.score_bps), 'Test score mismatch')
        for gate, cutoff in cutoffs.items():
            require(np.array_equal((pred >= cutoff).astype(int), rows['gate' + gate[4:]]), 'Test threshold recalculated')
        require(np.array_equal((pred > 0).astype(int), rows.economic), 'Economic gate mismatch')
    # Recompute one causal ordinary window to establish that saved original13
    # are the unchanged upstream contract; inspect no holdout price or outcome.
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    candles = pd.read_csv(Path(manifest['sources']['snapshot']) / 'candles.csv', dtype=str,
                          usecols=['bar_open_at', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    candles = candles.loc[pd.to_datetime(candles.bar_open_at, utc=True) < holdout]
    opportunities = pd.read_csv(bundle / 'opportunities.csv').set_index('opportunity_id')
    ident = features.index[0]
    unchanged = features_for_opportunity(candles, opportunities.loc[ident])
    require(np.allclose(features.loc[ident, list(ORIGINAL)], [unchanged[k] for k in ORIGINAL]), 'Saved original13 mismatch')
    signals = json.loads((run / 'signals_manifest.json').read_text(encoding='utf-8'))['signals']
    for signal in signals:
        payload = json.loads((run / signal['path']).read_text(encoding='utf-8'))
        require(set(payload) == {'schema_version', 'mode', 'decisions'} and payload['mode'] == 'ordinary_features_gate', 'Strategy payload')
        require(all(set(row) == {'signal_at', 'direction', 'exposure_fraction'} and pd.Timestamp(row['signal_at']) < holdout for row in payload['decisions']), 'Sidecar leak')
        rows = payload['decisions']
        ids = [row['signal_at'] for row in rows]
        require(ids == sorted(set(ids)) and len(rows) == signal['opportunities']
                and sum(row['exposure_fraction'] for row in rows) == signal['selected_count'], 'Sidecar order/count')
        require(sha(run / signal['path']) == signal['sha256'], 'Sidecar manifest hash')
        if signal['family'].startswith('ridge'):
            saved = predictions.loc[predictions.fold_id.eq(signal['fold_id']) & predictions.feature_set.eq(signal['family'])].set_index('opportunity_id')
            column = 'economic' if signal['gate'] == 'economic' else 'gate' + signal['gate'][4:]
            expected = saved.loc[ids, column].to_numpy()
        elif signal['family'] == 'random':
            seed = int(signal['mode'].rsplit('s', 1)[1])
            expected = [random_score(seed, x) < .5 for x in ids]
        elif signal['family'] == 'periodic':
            expected = [periodic_action(x, int(signal['mode'][-1])) for x in ids]
        else:
            head = next(h for h in heads if h['fold_id'] == signal['fold_id'])
            expected = [head['training_mean_bps'] > 0] * len(ids)
        require(np.array_equal(expected, [row['exposure_fraction'] for row in rows]), 'Sidecar differs from producer gates')
        require(all(row['direction'] == opportunities.loc[row['signal_at'], 'direction'] for row in rows), 'Sidecar reverses direction')
    require(not registry['final_holdout_inspected'], 'Holdout inspected')
    return ['prepared_hashes_all_fits_thresholds_and_predictions', 'saved_original13_contract',
            'development_scope_and_outcome_free_sidecar_schema']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    checks = synthetic_checks()
    if args.run_dir:
        checks += audit_run(args.run_dir)
    print(json.dumps({'status': 'passed', 'checks': checks, 'formal_preparation_executed': False,
                      'holdout_inspected': False}))
