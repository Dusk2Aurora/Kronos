"""Bounded same-coordinate continuous/binary Ridge diagnostic; development only.

All 32 candidates are checkpointed. Saved coefficients and train scalers are
independently replayed without refitting; no extra candidate or cutoff search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
from prepare_e00r import fit_candidates, validation_cutoffs, choose_candidate
from ordinary_features_v2 import FEATURE_NAMES
from summarize_e00r import development_labels
from heads import reuse_baseline
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

FAMILIES = ('ridge_quant_continuous', 'ridge_quant_binary')
LAMBDAS = [0.001, 0.01, 0.1, 1.0]
FOLDS = ('WF01', 'WF02', 'WF03', 'WF04')
CUTOFF = pd.Timestamp('2026-04-01T00:00:00Z')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_design(config, common):
    h = common['head']
    require(h['type'] == 'Ridge' and h['lambda_candidates'] == LAMBDAS and h['solver'] == 'svd'
            and h['fit_intercept'] is True and h['tie_break'] == 'larger_lambda'
            and h['target_clipping'] == 'none' and h['sample_weight'] is None,
            'Unsupported same-head Ridge design')
    require(common['status'] == 'locked' and common['final_holdout_access'] == 'forbidden'
            and config['stage'] == 'frozen_development'
            and config['evaluation_readiness']['final_holdout_evaluation_allowed'] is False,
            'Stage or holdout authorization differs')
    require(common['backbone_training'] == common['tokenizer_training'] == 'forbidden', 'Frozen weights required')
    require([f['id'] for f in config['splits']['walk_forward_dates']] == list(FOLDS), 'Only registered development folds')
    require(all(pd.Timestamp(f['test'][1]) <= CUTOFF for f in config['splits']['walk_forward_dates']), 'Holdout boundary')
    require(common['actions']['information_gate']['primary_validation_target_coverage'] == .5
            and common['actions']['information_gate']['sensitivity_validation_target_coverage'] == [.25, .75]
            and common['actions']['economic_gate']['costs_subtracted_again'] is False,
            'Gate design differs')


def configs(output, registry):
    for key, name in [('configuration', 'common_config.yaml'), ('parent_configuration', 'experiment_config.yaml')]:
        require(sha(output / 'provenance' / name) == registry[key]['sha256'], 'Saved configuration changed')
    config = yaml.safe_load((output / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    common = yaml.safe_load((output / 'provenance/common_config.yaml').read_text(encoding='utf-8'))
    validate_design(config, common)
    return config, common


def verified_bundle(config, registry):
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    require(sha(bundle / 'manifest.json') == registry['provenance']['label_manifest_sha256'], 'Label manifest changed')
    manifest = read_json(bundle / 'manifest.json')
    for name in ['labels_forbidden_as_features.csv', 'opportunities.csv', 'split_index.csv']:
        require(sha(bundle / name) == manifest['artifact_sha256'][name], 'Changed label bundle: ' + name)
    snapshot = Path(manifest['sources']['snapshot'])
    if not snapshot.is_absolute():
        snapshot = ROOT / snapshot
    require(sha(snapshot / 'manifest.json') == manifest['sources']['snapshot_manifest_sha256']
            and sha(snapshot / 'candles.csv') == manifest['sources']['snapshot_csv_sha256']['candles.csv'], 'Snapshot changed')
    return bundle, snapshot


def quant_inputs(output, features, bundle, snapshot, registry):
    # This module is imported lazily so synthetic checks never load the encoder
    # or open data, and the loader remains the authority for cache byte hashes.
    from quant_encoder import load_cache
    cache = output / 'cache/quantization'
    loaded = load_cache(cache)
    continuous, binary = loaded['continuous'].copy(), loaded['binary'].copy()
    contract = read_json(cache / 'contract.json')
    require(contract['protocol_sha256'] == registry['configuration']['sha256']
            and contract['initial_config_sha256'] == registry['parent_configuration']['sha256'],
            'Quant cache protocol/stage differs')
    for key, path in {'snapshot_manifest_sha256': snapshot / 'manifest.json', 'candles_sha256': snapshot / 'candles.csv',
                      'opportunities_sha256': bundle / 'opportunities.csv', 'split_index_sha256': bundle / 'split_index.csv'}.items():
        require(contract['sources'][key] == sha(path), 'Quant cache source changed: ' + key)
    require(len(features) == 2460 and features.index.is_unique, 'Expected 2460 ordinary IDs')
    require(tuple(continuous.columns) == tuple(f'continuous_{i:02d}' for i in range(20))
            and tuple(binary.columns) == tuple(f'binary_{i:02d}' for i in range(20))
            and contract['coordinates'] == 'same_quant_embed_20' and contract['quantization']['dimension'] == 20,
            'Continuous/binary coordinates differ')
    continuous.columns = binary.columns = [f'quant_{i:02d}' for i in range(20)]
    require(all(pd.Timestamp(x) < CUTOFF for x in features.index), 'Holdout features')
    matrices = {}
    for family, frame in zip(FAMILIES, (continuous, binary)):
        require(frame.index.is_unique and set(frame.index) == set(features.index) and frame.shape == (2460, 20), 'Quant ID/shape mismatch')
        require(np.isfinite(frame.to_numpy(float)).all(), 'Nonfinite quant features')
        matrices[family] = features.join(frame, validate='one_to_one').astype(float)
        require(matrices[family].shape == (2460, 39), 'Combined feature dimension differs')
    expected = np.where(continuous.to_numpy() > 0, 1., -1.) / np.sqrt(20.)
    require(np.allclose(binary.to_numpy(), expected, rtol=1e-4, atol=1e-4), 'Binary values must be normalized tokenizer signs')
    return matrices, contract, {p.relative_to(output).as_posix(): sha(p) for p in cache.rglob('*') if p.is_file()}


def fold_groups(fold, split, opportunities):
    groups = {r: sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq(r), 'opportunity_id'])
              for r in ('train', 'validation', 'test')}
    for role, ids in groups.items():
        start, end = map(pd.Timestamp, fold[role])
        require(ids and len(ids) == len(set(ids)) and all(start <= pd.Timestamp(x) < end
                and pd.Timestamp(opportunities.loc[x, 'exit_at']) < end
                and pd.Timestamp(opportunities.loc[x, 'labelable_at']) < end for x in ids), 'Unpurged role membership')
    require(not set(groups['train']) & set(groups['validation']) and not set(groups['test']) &
            (set(groups['train']) | set(groups['validation'])), 'Role overlap')
    return groups


def prepare(output):
    output = output.resolve()
    require(output.is_relative_to((ROOT / 'research/runs').resolve()) and output != (ROOT / 'research/runs').resolve(), 'Output outside run scope')
    registry_path = ROOT / 'research/registry' / f'{output.name}.json'
    registry = read_json(registry_path)
    require(registry['status'] == 'quant_encoding_accepted_pending_heads', 'Quant encoding not accepted')
    require(not (output / 'training_attempts.json').exists(), 'Refuse to overwrite attempts')
    config, common = configs(output, registry)
    for key in ('configuration', 'parent_configuration'):
        require(sha(Path(registry[key]['path'])) == registry[key]['sha256'], 'Registered configuration changed')
    require(read_json(output / 'encoding_acceptance.json')['status'] == 'passed', 'Quant encoding acceptance missing')
    encoding_hashes = registry['artifacts'].get('encoding_hashes', {})
    require(encoding_hashes and 'encoding_acceptance.json' in encoding_hashes, 'Registered encoding hashes missing')
    for relative, value in encoding_hashes.items():
        path = (output / relative).resolve()
        require(path.is_relative_to(output) and sha(path) == value, 'Quant encoding artifact changed')
    sources = [Path(__file__), Path(__file__).with_name('heads.py'), Path(__file__).with_name('quant_encoder.py'),
               Path(__file__).with_name('encoder.py'), ROOT / 'model/kronos.py', ROOT / 'model/module.py',
               ROOT / 'research/baselines/prepare_e00.py', ROOT / 'research/baselines/prepare_e00r.py',
               ROOT / 'research/baselines/ordinary_features.py', ROOT / 'research/baselines/ordinary_features_v2.py',
               ROOT / 'research/baselines/summarize_e00r.py', ROOT / 'research/freqtrade/strategies/KronosE00.py']
    source_dir = output / 'provenance/heads_source'
    source_dir.mkdir()
    for p in sources:
        relative = p.relative_to(ROOT).as_posix()
        registry['provenance']['canonical_source_sha256'][relative] = sha(p)
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    trials = [{'fold_id': f, 'feature_set': family, 'lambda': lam, 'status': 'planned'}
              for f in FOLDS for family in FAMILIES for lam in LAMBDAS]
    registry['selection']['all_attempts'] = trials
    registry['status'] = 'quant_encoding_accepted_heads_running'
    write_json(registry_path, registry)
    write_json(output / 'training_attempts.json', trials)
    try:
        bundle, snapshot = verified_bundle(config, registry)
        labels, split, _ = development_labels(config)
        features = reuse_baseline(output, config, common, labels, split)
        matrices, contract, cache_hashes = quant_inputs(output, features, bundle, snapshot, registry)
        for family, matrix in matrices.items():
            matrix.to_csv(output / f'features_{family}.csv')
        write_json(output / 'encoding_head_input_audit.json', {'status': 'passed', 'rows': 2460,
                   'coordinates': list(matrices[FAMILIES[0]].columns[19:]), 'quant_dimension': 20, 'combined_dimension': 39,
                   'same_continuous_binary_coordinates': True, 'cache_sha256': cache_hashes,
                   'contract_sha256': sha(output / 'cache/quantization/contract.json'), 'holdout_inspected': False})
        opportunities = pd.read_csv(bundle / 'opportunities.csv').set_index('opportunity_id')
        predictions, selected, signals = [], [], []
        (output / 'decisions').mkdir()
        for fold in config['splits']['walk_forward_dates']:
            groups = fold_groups(fold, split, opportunities)
            for family, matrix in matrices.items():
                def record(candidate):
                    trial = next(a for a in trials if (a['fold_id'], a['feature_set'], a['lambda']) ==
                                 (fold['id'], family, candidate['lambda']))
                    trial.update(candidate, feature_names=list(matrix.columns), train_ids=groups['train'],
                                 validation_ids=groups['validation'], train_samples=len(groups['train']),
                                 validation_samples=len(groups['validation']))
                    write_json(output / 'training_attempts.json', trials)
                scaler, winner = fit_candidates(matrix.loc[groups['train']].to_numpy(),
                    labels.loc[groups['train'], 'base_profit_ratio'].to_numpy() * 10000,
                    matrix.loc[groups['validation']].to_numpy(), labels.loc[groups['validation'], 'base_profit_ratio'].to_numpy() * 10000,
                    LAMBDAS, record)
                require(np.isfinite(winner['validation_mse_bps_squared']), 'Nonfinite validation loss')
                cuts = validation_cutoffs(winner['validation_predictions_bps'])
                selected.append({'fold_id': fold['id'], 'feature_set': family, 'lambda': winner['lambda'],
                                 'alpha': winner['alpha'], 'validation_mse_bps_squared': winner['validation_mse_bps_squared'],
                                 'validation_cutoffs_bps': cuts, 'training_mean_bps': float(labels.loc[groups['train'], 'base_profit_ratio'].mean() * 10000),
                                 'test_samples': len(groups['test']), 'selection_recorded_before_test_prediction': True})
                write_json(output / 'selected_heads.json', selected)
                scores = winner['model'].predict(scaler.transform(matrix.loc[groups['test']].to_numpy()))
                require(np.isfinite(scores).all(), 'Nonfinite test scores')
                gates = {key: scores >= value for key, value in cuts.items()}
                gates['economic'] = scores > 0
                for i, (ident, score) in enumerate(zip(groups['test'], scores)):
                    predictions.append({'fold_id': fold['id'], 'feature_set': family, 'opportunity_id': ident,
                                        'score_bps': float(score), 'gate50': int(gates['rank50'][i]), 'gate25': int(gates['rank25'][i]),
                                        'gate75': int(gates['rank75'][i]), 'economic': int(gates['economic'][i])})
                for gate, actions in gates.items():
                    mode = f'{family}_{gate}'
                    path = Path('decisions') / f'{fold["id"]}_{mode}.json'
                    write_json(output / path, {'schema_version': 1, 'mode': 'ordinary_features_gate', 'decisions': [
                        {'signal_at': ident, 'direction': int(opportunities.loc[ident, 'direction']), 'exposure_fraction': float(action)}
                        for ident, action in zip(groups['test'], actions)]})
                    signals.append({'fold_id': fold['id'], 'mode': mode, 'family': family, 'gate': gate,
                                    'path': path.as_posix(), 'sha256': sha(output / path), 'selected_count': int(actions.sum()),
                                    'opportunities': len(actions), 'primary': gate in ('rank50', 'economic'), 'stress_reuses_base_signal': True})
                print(json.dumps({'fold': fold['id'], 'family': family, 'fits_completed': sum(a['status'] == 'completed' for a in trials)}), flush=True)
        require(len(trials) == 32 and all(a['status'] == 'completed' for a in trials)
                and len(selected) == 8 and len(signals) == 32 and len(predictions) == 2182, 'Incomplete registered budget')
        pd.DataFrame(predictions).to_csv(output / 'development_test_predictions.csv', index=False)
        write_json(output / 'signals_manifest.json', {'schema_version': 1, 'signals': signals, 'final_holdout_inspected': False})
        write_json(output / 'head_acceptance.json', verify_output(output, check_registered=False))
        registry['status'] = 'heads_fitted_pending_engine_backtests'
        registry['selection']['all_attempts'] = trials
        registry['artifacts']['preparation_hashes'] = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
        write_json(registry_path, registry)
    except Exception as error:
        registry.update(status='failed_preparation', result={'conclusion': 'invalid_due_to_audit', 'failure_reason': repr(error)})
        registry['selection']['all_attempts'] = trials
        write_json(registry_path, registry)
        raise


def verify_output(output, check_registered=True):
    """Recompute all saved head predictions and selection without fitting models."""
    output = output.resolve()
    require(output.is_relative_to((ROOT / 'research/runs').resolve()), 'Verification outside runs')
    registry = read_json(ROOT / 'research/registry' / f'{output.name}.json')
    if check_registered:
        for relative, value in registry['artifacts']['preparation_hashes'].items():
            p = (output / relative).resolve()
            require(p.is_relative_to(output) and sha(p) == value, 'Preparation changed: ' + relative)
    config, common = configs(output, registry)
    bundle, snapshot = verified_bundle(config, registry)
    labels, split, _ = development_labels(config)
    features = pd.read_csv(output / 'baseline_reuse/development_features.csv').set_index('opportunity_id')
    matrices, contract, hashes = quant_inputs(output, features, bundle, snapshot, registry)
    attempts = read_json(output / 'training_attempts.json')
    selected = read_json(output / 'selected_heads.json')
    predictions = pd.read_csv(output / 'development_test_predictions.csv')
    signals = read_json(output / 'signals_manifest.json')['signals']
    require(len(attempts) == 32 and len(selected) == 8 and len(signals) == 32 and len(predictions) == 2182, 'Audit budget differs')
    opp = pd.read_csv(bundle / 'opportunities.csv').set_index('opportunity_id')
    checked = 0
    for family, reference in matrices.items():
        saved_matrix = pd.read_csv(output / f'features_{family}.csv').set_index('opportunity_id')
        require(saved_matrix.index.tolist() == reference.index.tolist() and tuple(saved_matrix.columns) == tuple(reference.columns)
                and np.allclose(saved_matrix, reference, rtol=1e-12, atol=1e-12), 'Feature CSV differs from accepted cache')
        for fold in config['splits']['walk_forward_dates']:
            groups = fold_groups(fold, split, opp)
            tx = saved_matrix.loc[groups['train']].to_numpy()
            vx = saved_matrix.loc[groups['validation']].to_numpy()
            scaler = StandardScaler().fit(tx)
            local = []
            for lam in LAMBDAS:
                matches = [a for a in attempts if (a['fold_id'], a['feature_set'], a['lambda']) == (fold['id'], family, lam)]
                require(len(matches) == 1, 'Missing or duplicate candidate')
                a = matches[0]
                require(a['status'] == 'completed' and a['alpha'] == len(tx) * lam and a['train_ids'] == groups['train']
                        and a['validation_ids'] == groups['validation'] and a['feature_names'] == list(saved_matrix.columns), 'Candidate membership/design')
                for name, actual in [('scaler_mean', scaler.mean_), ('scaler_var', scaler.var_), ('scaler_scale', scaler.scale_)]:
                    require(np.allclose(a[name], actual, rtol=1e-12, atol=1e-12), 'Train-only scaler mismatch')
                require(a['scaler_n_samples_seen'] == len(tx), 'Scaler sample count differs')
                # Ridge with positive alpha has a unique centered solution. Check
                # its normal equations without another fitting call or candidate.
                z = scaler.transform(tx)
                z_center = z - z.mean(axis=0)
                train_y = labels.loc[groups['train'], 'base_profit_ratio'].to_numpy() * 10000
                yc = train_y - train_y.mean()
                coef = np.asarray(a['coef'])
                residual = z_center.T @ (z_center @ coef - yc) + a['alpha'] * coef
                bound = max(1., float(np.max(np.abs(z_center.T @ yc)))) * 1e-9
                require(np.isfinite(coef).all() and np.max(np.abs(residual)) <= bound
                        and np.isclose(a['intercept'], train_y.mean() - z.mean(axis=0) @ coef, rtol=1e-10, atol=1e-9),
                        'Saved coefficients do not solve registered training Ridge objective')
                prediction = ((vx - a['scaler_mean']) / a['scaler_scale']) @ np.asarray(a['coef']) + a['intercept']
                y = labels.loc[groups['validation'], 'base_profit_ratio'].to_numpy() * 10000
                require(np.isfinite(prediction).all() and np.allclose(prediction, a['validation_predictions_bps'], rtol=1e-10, atol=1e-9)
                        and np.isclose(np.mean((prediction - y) ** 2), a['validation_mse_bps_squared'], rtol=1e-10), 'Saved coefficient/validation loss mismatch')
                local.append(a)
                checked += 1
            winner = choose_candidate(local)
            heads = [h for h in selected if (h['fold_id'], h['feature_set']) == (fold['id'], family)]
            require(len(heads) == 1 and heads[0]['lambda'] == winner['lambda']
                    and heads[0]['validation_mse_bps_squared'] == winner['validation_mse_bps_squared']
                    and heads[0]['selection_recorded_before_test_prediction'], 'Validation-only selection mismatch')
            cuts = validation_cutoffs(np.asarray(winner['validation_predictions_bps']))
            require(cuts == heads[0]['validation_cutoffs_bps'], 'Cutoff mismatch')
            x = saved_matrix.loc[groups['test']].to_numpy()
            score = ((x - winner['scaler_mean']) / winner['scaler_scale']) @ np.asarray(winner['coef']) + winner['intercept']
            saved = predictions.loc[predictions.fold_id.eq(fold['id']) & predictions.feature_set.eq(family)].sort_values('opportunity_id')
            require(saved.opportunity_id.tolist() == groups['test'] and np.allclose(score, saved.score_bps, rtol=1e-10, atol=1e-9), 'Saved test predictions mismatch')
            gates = {k: score >= v for k, v in cuts.items()}
            gates['economic'] = score > 0
            for gate, actions in gates.items():
                column = gate.replace('rank', 'gate') if gate != 'economic' else gate
                require(np.array_equal(saved[column], actions.astype(int)), 'Saved prediction actions mismatch')
                matching = [s for s in signals if (s['fold_id'], s['family'], s['gate']) == (fold['id'], family, gate)]
                require(len(matching) == 1, 'Missing or duplicate signal')
                s = matching[0]
                path = (output / s['path']).resolve()
                require(path.is_relative_to(output / 'decisions') and sha(path) == s['sha256'], 'Sidecar hash/path differs')
                payload = read_json(path)
                rows = payload['decisions']
                require(set(payload) == {'schema_version', 'mode', 'decisions'} and payload['schema_version'] == 1
                        and payload['mode'] == 'ordinary_features_gate' and [r['signal_at'] for r in rows] == groups['test'], 'Sidecar membership/schema')
                require(all(set(r) == {'signal_at', 'direction', 'exposure_fraction'} and type(r['direction']) is int
                        and r['direction'] == int(opp.loc[r['signal_at'], 'direction']) for r in rows), 'Sidecar outcome leakage/direction')
                require(np.array_equal([r['exposure_fraction'] for r in rows], actions.astype(float))
                        and s['selected_count'] == int(actions.sum()) and s['opportunities'] == len(actions)
                        and s['stress_reuses_base_signal'], 'Sidecar actions/stress differs')
    return {'status': 'passed', 'independently_recomputed_heads': checked, 'selected_heads': 8, 'test_predictions': 2182,
            'signals': 32, 'formal_refits': 0, 'cache_sha256': hashes, 'holdout_inspected': False,
            'scope': 'saved_coefficient_prediction_and_validation_selection_audit',
            'checks': ['same_20_quant_coordinates', 'cache_contract_source_hashes', 'unique_development_IDs',
                       'train_only_scaler', 'all_32_saved_coefficients_and_Ridge_normal_equations', 'validation_MSE_only_choice', 'fixed_validation_cutoffs',
                       'saved_test_score_and_sidecar_actions', 'base_stress_action_reuse']}


def verify_synthetic():
    """Small causality/choice check using no research files or outcome data."""
    rng = np.random.default_rng(17)
    tx, vx = rng.normal(size=(20, 4)), rng.normal(size=(6, 4)) + 10
    ty, vy = tx[:, 0] * 2, np.arange(6.)
    a, b = [], []
    scaler, _ = fit_candidates(tx, ty, vx, vy, LAMBDAS, a.append)
    other, _ = fit_candidates(tx, ty, vx * 9 + 2, vy + 1000, LAMBDAS, b.append)
    require(np.array_equal(scaler.mean_, tx.mean(0)) and np.array_equal(other.mean_, scaler.mean_), 'Validation changed scaler')
    ca, cb = [x for x in a if x['status'] == 'completed'], [x for x in b if x['status'] == 'completed']
    require(all(np.array_equal(x['coef'], y['coef']) and x['intercept'] == y['intercept'] for x, y in zip(ca, cb)), 'Validation entered model fitting')
    z = scaler.transform(tx)
    z -= z.mean(axis=0)
    for candidate in ca:
        coef = np.asarray(candidate['coef'])
        residual = z.T @ (z @ coef - (ty - ty.mean())) + candidate['alpha'] * coef
        require(np.max(np.abs(residual)) <= max(1., np.max(np.abs(z.T @ (ty - ty.mean())))) * 1e-9,
                'Normal-equation audit tolerance')
    require(choose_candidate([{'lambda': lam, 'validation_mse_bps_squared': 1.} for lam in LAMBDAS])['lambda'] == 1., 'Exact tie choice')
    cuts = validation_cutoffs(np.array([1., 2., 2., 4.]))
    require(cuts['rank50'] == 2. and np.sum(np.array([1., 2., 2., 4.]) >= cuts['rank50']) == 3, 'Include cutoff ties')
    print(json.dumps({'status': 'passed', 'synthetic_candidates': 8, 'formal_fits': 0, 'research_data_opened': False,
                      'checks': ['validation_perturbation_model_invariance', 'train_only_scaler', 'Ridge_normal_equations',
                                 'exact_lambda_tie', 'include_cutoff_ties']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir', type=Path)
    group.add_argument('--verify-output', type=Path)
    group.add_argument('--verify-synthetic', action='store_true')
    args = parser.parse_args()
    if args.verify_synthetic:
        verify_synthetic()
    elif args.verify_output:
        print(json.dumps(verify_output(args.verify_output)))
    else:
        prepare(args.output_dir)
