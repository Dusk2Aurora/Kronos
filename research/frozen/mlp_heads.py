"""Pre-registered CPU MLP readouts of immutable development-only frozen caches.

Synthetic verification never opens research data. Formal preparation refuses an
existing attempts file; failed candidates and all seed members remain auditable.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
from prepare_e00r import validation_cutoffs
from ordinary_features_v2 import FEATURE_NAMES, features_for_opportunity
from encoder import load_cache, state_sha, digest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import yaml

FAMILIES = {'mlp19': None, 'mlp_pretrained': 'pretrained',
            **{f'mlp_random_s{s}': f'random_s{s}' for s in (17, 29, 43)}}
SEEDS = [17, 29, 43]
DECAYS = [0.001, 0.01, 0.1]
CUTOFF = pd.Timestamp('2026-04-01T00:00:00Z')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def configure_torch():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def validate_head(protocol):
    expected = {'type': 'MLP', 'hidden_units': 16, 'activation': 'Tanh', 'epochs': 300,
                'learning_rate': 0.003, 'betas': [0.9, 0.999], 'eps': 1e-8,
                'weight_decay_candidates': DECAYS, 'seeds': SEEDS, 'device': 'cpu',
                'dtype': 'float32', 'threads': 1, 'deterministic_algorithms': True,
                'optimizer': 'AdamW', 'target_scaling': 'train_mean_std_ddof0',
                'selection': 'lowest_validation_ensemble_MSE_bps_squared',
                'tie_break': 'larger_weight_decay', 'ensemble': 'arithmetic_mean_all_three_seeds',
                'no_early_stopping': True, 'weight_decay_scope': 'all_parameters_including_bias',
                'loss': 'mean_squared_error_on_training_standardized_target', 'batch': 'full_training_fold',
                'preprocessing': 'StandardScaler_fit_training_fold_only', 'epoch_selection': False}
    for key, value in expected.items():
        require(protocol['head'].get(key) == value, 'Unregistered head setting: ' + key)
    require(protocol['variants'] == FAMILIES, 'Family/cache mapping differs')
    require(protocol['status'] == 'locked' and protocol['stage'] == 'frozen_development'
            and protocol['final_holdout_access'] == 'forbidden', 'Protocol scope differs')


def new_model(width, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(width, 16), torch.nn.Tanh(),
                                   torch.nn.Linear(16, 1)).to(device='cpu', dtype=torch.float32)


def predict(model, scaled_x, mean, std):
    model.eval()
    with torch.inference_mode():
        out = model(torch.as_tensor(scaled_x, dtype=torch.float32)).squeeze(1).numpy().astype(float)
    result = out * std + mean
    require(np.isfinite(result).all(), 'Nonfinite prediction')
    return result


def fit_member(tx, ty, vx, vy, decay, seed, epochs=300, history=None):
    model = new_model(tx.shape[1], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=decay)
    xt, yt = torch.as_tensor(tx, dtype=torch.float32), torch.as_tensor(ty, dtype=torch.float32)
    xv, yv = torch.as_tensor(vx, dtype=torch.float32), torch.as_tensor(vy, dtype=torch.float32)
    history = [] if history is None else history
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(xt).squeeze(1) - yt) ** 2)
        require(torch.isfinite(loss).item(), 'Nonfinite training loss')
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                'Nonfinite gradient')
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            tl = torch.mean((model(xt).squeeze(1) - yt) ** 2).item()
            vl = torch.mean((model(xv).squeeze(1) - yv) ** 2).item()
        require(np.isfinite([tl, vl]).all(), 'Nonfinite post-update loss')
        history.append({'epoch': epoch + 1, 'train_mse_scaled': tl, 'validation_mse_scaled': vl, 'all_gradients_finite': True})
    return model, history


def choose_candidate(candidates):
    require(all(c['status'] == 'completed' for c in candidates), 'Incomplete candidate grid')
    return min(candidates, key=lambda c: (c['validation_mse_bps_squared'], -c['weight_decay']))


def labels_for_ids(bundle, ids):
    """Never converts a forbidden or unrequested outcome (including test labels)."""
    allowed = set(ids)
    require(allowed and all(pd.Timestamp(x) < CUTOFF for x in allowed), 'Forbidden label ID')
    values = {}
    with (bundle / 'labels_forbidden_as_features.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            ident = row['opportunity_id']
            if ident in allowed:
                require(ident not in values, 'Duplicate label')
                require(pd.Timestamp(row['labelable_at']) < CUTOFF, 'Holdout labelability')
                values[ident] = float(row['base_profit_ratio']) * 10000
    require(set(values) == allowed, 'Incomplete requested labels')
    result = np.array([values[x] for x in ids], dtype=float)
    require(np.isfinite(result).all(), 'Nonfinite requested labels')
    return result


def verified_inputs(output, config, protocol, registry):
    old_run = (ROOT / protocol['cache_reuse']['run']).resolve()
    old_path = ROOT / protocol['cache_reuse']['registry']
    require(sha(old_path) == registry['provenance']['parent_frozen_registry_sha256'], 'Parent registry changed')
    old = read_json(old_path)
    require(old['status'] == 'completed_exploratory_development', 'Parent not accepted')
    old_hashes = old['artifacts']['preparation_hashes']
    def verified_old(relative):
        expected = old_hashes.get(relative)
        require(expected and sha(old_run / relative) == expected, 'Parent artifact changed: ' + relative)
        return old_run / relative
    old_config = yaml.safe_load(verified_old('provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    for key in ['market', 'timing', 'fixed_strategy', 'splits', 'labels_and_costs', 'backtest_engine']:
        require(config[key] == old_config[key], 'Immutable experiment contract differs: ' + key)
    require(sha(old_run / 'provenance/common_config.yaml') == old['configuration']['sha256'], 'Old protocol changed')
    require(sha(old_run / 'provenance/experiment_config.yaml') == old['parent_configuration']['sha256'], 'Old config changed')
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    manifest = read_json(bundle / 'manifest.json')
    require(sha(bundle / 'manifest.json') == registry['provenance']['label_manifest_sha256'], 'Label manifest changed')
    for name in ['labels_forbidden_as_features.csv', 'opportunities.csv', 'split_index.csv']:
        require(sha(bundle / name) == manifest['artifact_sha256'][name], 'Label bundle changed: ' + name)
    snapshot = Path(manifest['sources']['snapshot'])
    if not snapshot.is_absolute():
        snapshot = ROOT / snapshot
    require(sha(snapshot / 'manifest.json') == manifest['sources']['snapshot_manifest_sha256'] and
            sha(snapshot / 'candles.csv') == manifest['sources']['snapshot_csv_sha256']['candles.csv'], 'Snapshot changed')
    split = pd.read_csv(bundle / 'split_index.csv', dtype=str)
    split = split.loc[split.fold_id.isin([f['id'] for f in config['splits']['walk_forward_dates']]) & split.included.eq('True')].copy()
    require(set(split.role) == {'train', 'validation', 'test'}, 'Unexpected split roles')
    ids = sorted(set(split.opportunity_id))
    require(all(pd.Timestamp(x) < CUTOFF for x in ids), 'Holdout split')
    opportunities = pd.read_csv(bundle / 'opportunities.csv').set_index('opportunity_id')
    require(opportunities.index.is_unique, 'Duplicate opportunities')
    opportunities = opportunities.loc[ids]
    feature_relative = 'baseline_reuse/development_features.csv'
    original = verified_old(feature_relative)
    target = output / 'baseline_reuse'
    target.mkdir()
    shutil.copy2(original, target / 'development_features.csv')
    features = pd.read_csv(original).set_index('opportunity_id')
    require(tuple(features.columns) == FEATURE_NAMES and features.index.is_unique and sorted(features.index) == ids,
            'Ordinary19 membership/columns differ')
    candles = pd.read_csv(snapshot / 'candles.csv', dtype=str,
                          usecols=['bar_open_at', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    candles = candles.loc[pd.to_datetime(candles.bar_open_at, utc=True) < CUTOFF]
    for ident in [ids[0], ids[len(ids) // 2], ids[-1]]:
        actual = features_for_opportunity(candles, opportunities.loc[ident])
        require(np.allclose(features.loc[ident], list(actual.values()), rtol=1e-10, atol=1e-12), 'Ordinary source window differs')
    matrices = {'mlp19': features.astype(float)}
    contracts = {}
    reused_hashes = {feature_relative: sha(original)}
    for family, folder in FAMILIES.items():
        if folder is None:
            continue
        cache = old_run / 'cache' / folder
        for p in cache.rglob('*'):
            if p.is_file():
                relative = p.relative_to(old_run).as_posix()
                verified_old(relative)
                reused_hashes[relative] = sha(p)
        hidden = load_cache(cache)
        contract = read_json(cache / 'contract.json')
        require(contract['protocol_sha256'] == old['configuration']['sha256'] and
                contract['initial_config_sha256'] == old['parent_configuration']['sha256'], 'Cache parent provenance differs')
        for key, p in {'snapshot_manifest_sha256': snapshot / 'manifest.json', 'candles_sha256': snapshot / 'candles.csv',
                       'opportunities_sha256': bundle / 'opportunities.csv', 'split_index_sha256': bundle / 'split_index.csv'}.items():
            require(contract['sources'][key] == sha(p), 'Cache source changed: ' + key)
        require(hidden.index.is_unique and sorted(hidden.index) == ids and hidden.shape == (len(ids), 512), 'Cache membership/shape')
        require(contract['model']['variant'] == ('pretrained' if folder == 'pretrained' else 'random') and
                contract['model']['seed'] == (None if folder == 'pretrained' else int(folder.rsplit('s', 1)[1])), 'Cache variant/seed')
        matrices[family] = features.join(hidden, validate='one_to_one').astype(float)
        contracts[family] = contract
    require(len({c['model']['model_state_sha256'] for c in contracts.values()}) == 4 and
            len({c['model']['tokenizer_state_sha256'] for c in contracts.values()}) == 1, 'Backbone/tokenizer control differs')
    for key in ['sources', 'dtype', 'normalization', 'model_config', 'pooling', 'time_fields']:
        values = [c['sources'] if key == 'sources' else c['model'][key] for c in contracts.values()]
        require(all(x == values[0] for x in values), 'Cache architecture differs: ' + key)
    for family, matrix in matrices.items():
        require(np.isfinite(matrix).all().all(), 'Nonfinite features')
        matrix.to_csv(output / f'features_{family}.csv')
    write_json(output / 'encoding_head_input_audit.json', {'status': 'passed', 'rows': len(ids),
               'dimensions': {k: v.shape[1] for k, v in matrices.items()}, 'parent_registry_sha256': sha(old_path),
               'reused_artifact_sha256': reused_hashes, 'development_ids_sha256': digest(ids),
               'ordinary_source_windows_recomputed': 3, 'new_encoding': False, 'holdout_inspected': False})
    return matrices, split, opportunities, bundle


def prepare(output):
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs'), 'Output outside research/runs')
    registry_path = ROOT / 'research/registry' / (output.name + '.json')
    registry = read_json(registry_path)
    require(registry['status'] == 'protocol_locked_pending_heads', 'Unexpected preparation status')
    require(not (output / 'training_attempts.json').exists(), 'Refuse to overwrite attempts')
    protocol = yaml.safe_load((output / 'provenance/common_config.yaml').read_text(encoding='utf-8'))
    config = yaml.safe_load((output / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    validate_head(protocol)
    require(config['stage'] == 'frozen_development' and config['evaluation_readiness']['final_holdout_evaluation_allowed'] is False, 'Scope')
    for key, filename in [('configuration', 'common_config.yaml'), ('parent_configuration', 'experiment_config.yaml')]:
        require(sha(output / 'provenance' / filename) == registry[key]['sha256'] and
                sha(Path(registry[key]['path'])) == registry[key]['sha256'], 'Configuration changed')
    configure_torch()
    source_dir = output / 'provenance/heads_source'
    source_dir.mkdir()
    paths = [Path(__file__), Path(__file__).with_name('encoder.py'),
             ROOT / 'research/baselines/prepare_e00.py', ROOT / 'research/baselines/prepare_e00r.py',
             ROOT / 'research/baselines/ordinary_features.py', ROOT / 'research/baselines/ordinary_features_v2.py']
    for p in paths:
        relative = p.relative_to(ROOT).as_posix()
        registry['provenance']['canonical_source_sha256'][relative] = sha(p)
        dest = source_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    attempts = [{'fold_id': f['id'], 'feature_set': family, 'weight_decay': d, 'seed': s, 'status': 'planned'}
                for f in config['splits']['walk_forward_dates'] for family in FAMILIES for d in DECAYS for s in SEEDS]
    registry['selection']['all_attempts'] = attempts
    registry['status'] = 'heads_running'
    write_json(registry_path, registry)
    write_json(output / 'training_attempts.json', attempts)
    selected, candidates, predictions, constituent_predictions, signals = [], [], [], [], []
    try:
        matrices, split, opp, bundle = verified_inputs(output, config, protocol, registry)
        (output / 'models').mkdir()
        (output / 'decisions').mkdir()
        for fold in config['splits']['walk_forward_dates']:
            groups = {r: sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq(r), 'opportunity_id']) for r in ['train', 'validation', 'test']}
            for role, ids in groups.items():
                start, end = map(pd.Timestamp, fold[role])
                require(ids and all(start <= pd.Timestamp(x) < end and pd.Timestamp(opp.loc[x, 'exit_at']) < end and
                                    pd.Timestamp(opp.loc[x, 'labelable_at']) < end for x in ids), 'Unpurged role')
            require(not set(groups['train']) & set(groups['validation']) and
                    not set(groups['test']) & (set(groups['train']) | set(groups['validation'])), 'Role overlap')
            ty = labels_for_ids(bundle, groups['train'])
            vy = labels_for_ids(bundle, groups['validation'])
            mean, std = float(ty.mean()), float(ty.std(ddof=0))
            require(std > 0 and np.isfinite(std), 'Degenerate target standard deviation')
            for family, matrix in matrices.items():
                scaler = StandardScaler().fit(matrix.loc[groups['train']].to_numpy())
                tx = scaler.transform(matrix.loc[groups['train']].to_numpy()).astype(np.float32)
                vx = scaler.transform(matrix.loc[groups['validation']].to_numpy()).astype(np.float32)
                local, member_models = [], {}
                for decay in DECAYS:
                    members = []
                    for seed in SEEDS:
                        a = next(a for a in attempts if a['fold_id'] == fold['id'] and a['feature_set'] == family and a['weight_decay'] == decay and a['seed'] == seed)
                        a.update(status='running', train_ids=groups['train'], validation_ids=groups['validation'],
                                 train_samples=len(ty), validation_samples=len(vy), feature_names=list(matrix.columns),
                                 scaler_mean=scaler.mean_.tolist(), scaler_var=scaler.var_.tolist(), scaler_scale=scaler.scale_.tolist(),
                                 target_mean_bps=mean, target_std_bps=std, target_std_ddof=0, epochs=300, input_dimension=matrix.shape[1],
                                 parameter_count=16 * matrix.shape[1] + 33,
                                 numeric_environment={'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
                                                      'dtype': 'float32', 'threads': torch.get_num_threads(),
                                                      'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()})
                        write_json(output / 'training_attempts.json', attempts)
                        a['loss_history'] = []
                        try:
                            model, history = fit_member(tx, (ty - mean) / std, vx, (vy - mean) / std, decay, seed, history=a['loss_history'])
                            tp, vp = predict(model, tx, mean, std), predict(model, vx, mean, std)
                            for h in history:
                                h['train_mse_bps_squared'] = h['train_mse_scaled'] * std ** 2
                                h['validation_mse_bps_squared'] = h['validation_mse_scaled'] * std ** 2
                            path = Path('models') / f'{fold["id"]}_{family}_wd{decay}_s{seed}.pt'
                            torch.save(model.state_dict(), output / path)
                            a.update(status='completed', loss_history=history, train_predictions_bps=tp.tolist(),
                                     validation_predictions_bps=vp.tolist(), train_mse_bps_squared=float(np.mean((tp - ty) ** 2)),
                                     validation_mse_bps_squared=float(np.mean((vp - vy) ** 2)), model_path=path.as_posix(),
                                     model_file_sha256=sha(output / path), model_state_sha256=state_sha(model))
                            members.append(vp)
                            member_models[(decay, seed)] = model
                        except Exception as error:
                            a.update(status='failed', error=repr(error))
                            raise
                        finally:
                            write_json(output / 'training_attempts.json', attempts)
                    vp = np.mean(members, axis=0)
                    candidate = {'fold_id': fold['id'], 'feature_set': family, 'weight_decay': decay,
                                 'seeds': SEEDS, 'status': 'completed', 'validation_predictions_bps': vp.tolist(),
                                 'validation_mse_bps_squared': float(np.mean((vp - vy) ** 2))}
                    local.append(candidate)
                    candidates.append(candidate)
                    write_json(output / 'ensemble_candidates.json', candidates)
                winner = choose_candidate(local)
                cuts = validation_cutoffs(np.asarray(winner['validation_predictions_bps']))
                selected.append({**winner, 'validation_cutoffs_bps': cuts, 'training_mean_bps': mean,
                                 'test_samples': len(groups['test']), 'primary': family == 'mlp_pretrained',
                                 'selection_recorded_before_test_predictions': True, 'test_labels_loaded': False})
                # Commit validation choice before producing any test prediction; test labels are never needed here.
                write_json(output / 'selected_heads.json', selected)
                sx = scaler.transform(matrix.loc[groups['test']].to_numpy()).astype(np.float32)
                scores_by_seed = []
                for seed in SEEDS:
                    score = predict(member_models[(winner['weight_decay'], seed)], sx, mean, std)
                    scores_by_seed.append(score)
                    for ident, value in zip(groups['test'], score):
                        constituent_predictions.append({'fold_id': fold['id'], 'feature_set': family, 'seed': seed,
                                                        'opportunity_id': ident, 'score_bps': float(value)})
                scores = np.mean(scores_by_seed, axis=0)
                gates = {key: scores >= value for key, value in cuts.items()}
                gates['economic'] = scores > 0
                for i, (ident, score) in enumerate(zip(groups['test'], scores)):
                    predictions.append({'fold_id': fold['id'], 'feature_set': family, 'opportunity_id': ident,
                                        'score_bps': float(score), 'gate50': int(gates['rank50'][i]),
                                        'gate25': int(gates['rank25'][i]), 'gate75': int(gates['rank75'][i]), 'economic': int(gates['economic'][i])})
                for gate, actions in gates.items():
                    mode = family + '_' + gate
                    path = Path('decisions') / f'{fold["id"]}_{mode}.json'
                    write_json(output / path, {'schema_version': 1, 'mode': 'ordinary_features_gate', 'decisions': [
                        {'signal_at': ident, 'direction': int(opp.loc[ident, 'direction']), 'exposure_fraction': float(action)}
                        for ident, action in zip(groups['test'], actions)]})
                    signals.append({'fold_id': fold['id'], 'mode': mode, 'family': family, 'gate': gate,
                                    'path': path.as_posix(), 'sha256': sha(output / path), 'selected_count': int(actions.sum()),
                                    'opportunities': len(actions), 'primary': gate in ('rank50', 'economic'), 'stress_reuses_base_signal': True})
                print(json.dumps({'fold': fold['id'], 'family': family, 'fits_completed': sum(a['status'] == 'completed' for a in attempts)}), flush=True)
        require(len(attempts) == 180 and all(a['status'] == 'completed' for a in attempts) and len(selected) == 20 and len(signals) == 80,
                'Incomplete fit/signal budget')
        require(len(predictions) == 5455 and len(constituent_predictions) == 16365, 'Unexpected test membership')
        pd.DataFrame(predictions).to_csv(output / 'development_test_predictions.csv', index=False)
        pd.DataFrame(constituent_predictions).to_csv(output / 'constituent_test_predictions.csv', index=False)
        write_json(output / 'signals_manifest.json', {'schema_version': 1, 'signals': signals, 'final_holdout_inspected': False})
        write_json(output / 'head_acceptance.json', verify_output(output, check_registered=False))
        registry['status'] = 'heads_fitted_pending_engine_backtests'
        registry['selection']['all_attempts'] = attempts
        registry['artifacts']['preparation_hashes'] = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
        write_json(registry_path, registry)
        print(json.dumps({'status': registry['status'], 'attempts': len(attempts), 'selected_ensembles': len(selected), 'test_predictions': len(predictions), 'signals': len(signals)}))
    except Exception as error:
        registry.update(status='failed_preparation', result={'conclusion': 'invalid_due_to_audit', 'failure_reason': repr(error)})
        registry['selection']['all_attempts'] = attempts
        write_json(registry_path, registry)
        raise


def verify_output(output, check_registered=True):
    """Reload every saved head and independently verify scaler, choice and actions.

    Reads only train/validation outcome values. It never retrains a head or opens
    test outcomes; accepted preparation hashes protect every saved artifact.
    """
    configure_torch()
    registry = read_json(ROOT / 'research/registry' / (output.name + '.json'))
    if check_registered:
        for relative, expected in registry['artifacts']['preparation_hashes'].items():
            p = (output / relative).resolve()
            require(p.is_relative_to(output) and sha(p) == expected, 'Preparation changed: ' + relative)
    config = yaml.safe_load((output / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    protocol = yaml.safe_load((output / 'provenance/common_config.yaml').read_text(encoding='utf-8'))
    validate_head(protocol)
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    require(sha(bundle / 'manifest.json') == registry['provenance']['label_manifest_sha256'], 'Label manifest changed')
    manifest = read_json(bundle / 'manifest.json')
    require(all(sha(bundle / n) == manifest['artifact_sha256'][n] for n in ['split_index.csv', 'opportunities.csv', 'labels_forbidden_as_features.csv']), 'Bundle changed')
    opp = pd.read_csv(bundle / 'opportunities.csv').set_index('opportunity_id')
    split = pd.read_csv(bundle / 'split_index.csv', dtype=str)
    split = split.loc[split.included.eq('True') & split.fold_id.isin([f['id'] for f in config['splits']['walk_forward_dates']])]
    attempts = read_json(output / 'training_attempts.json')
    candidates = read_json(output / 'ensemble_candidates.json')
    selected = read_json(output / 'selected_heads.json')
    predicted = pd.read_csv(output / 'development_test_predictions.csv')
    members_saved = pd.read_csv(output / 'constituent_test_predictions.csv')
    signals = read_json(output / 'signals_manifest.json')['signals']
    require(len(attempts) == 180 and len(candidates) == 60 and len(selected) == 20 and len(signals) == 80, 'Audit budget differs')
    require(len(predicted) == 5455 and len(members_saved) == 16365, 'Audit prediction count differs')
    checked_members = 0
    for fold in config['splits']['walk_forward_dates']:
        groups = {r: sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq(r), 'opportunity_id']) for r in ['train', 'validation', 'test']}
        ty = labels_for_ids(bundle, groups['train'])
        vy = labels_for_ids(bundle, groups['validation'])
        for family in FAMILIES:
            matrix = pd.read_csv(output / f'features_{family}.csv').set_index('opportunity_id')
            require(matrix.index.is_unique and tuple(matrix.columns[:19]) == FEATURE_NAMES and matrix.shape[1] == (19 if family == 'mlp19' else 531), 'Feature matrix contract')
            require(sorted(matrix.index) == sorted(set(split.opportunity_id)), 'Feature membership')
            scaler = StandardScaler().fit(matrix.loc[groups['train']].to_numpy())
            mean, std = float(ty.mean()), float(ty.std(ddof=0))
            local, reloaded = [], {}
            for decay in DECAYS:
                vp_seeds = []
                for seed in SEEDS:
                    matches = [a for a in attempts if (a['fold_id'], a['feature_set'], a['weight_decay'], a['seed']) == (fold['id'], family, decay, seed)]
                    require(len(matches) == 1, 'Missing/duplicate attempt')
                    a = matches[0]
                    require(a['status'] == 'completed' and a['train_ids'] == groups['train'] and a['validation_ids'] == groups['validation'], 'Fit membership/status')
                    require(a['epochs'] == 300 and len(a['loss_history']) == 300 and a['target_std_ddof'] == 0, 'Epoch/target contract')
                    require(all(h['epoch'] == i + 1 and h['all_gradients_finite'] for i, h in enumerate(a['loss_history'])), 'Gradient/epoch diagnostics')
                    require(np.allclose(a['scaler_mean'], scaler.mean_, rtol=1e-12, atol=1e-12) and np.allclose(a['scaler_var'], scaler.var_, rtol=1e-12, atol=1e-12)
                            and np.allclose(a['scaler_scale'], scaler.scale_, rtol=1e-12, atol=1e-12), 'Train-only scaler audit')
                    require(a['target_mean_bps'] == mean and a['target_std_bps'] == std, 'Train-only target audit')
                    model_path = (output / a['model_path']).resolve()
                    require(model_path.is_relative_to(output) and sha(model_path) == a['model_file_sha256'], 'Saved model changed')
                    model = new_model(matrix.shape[1], seed)
                    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
                    require(state_sha(model) == a['model_state_sha256'], 'State hash mismatch')
                    tp = predict(model, scaler.transform(matrix.loc[groups['train']].to_numpy()).astype(np.float32), mean, std)
                    vp = predict(model, scaler.transform(matrix.loc[groups['validation']].to_numpy()).astype(np.float32), mean, std)
                    require(np.allclose(tp, a['train_predictions_bps'], rtol=1e-6, atol=1e-5) and np.allclose(vp, a['validation_predictions_bps'], rtol=1e-6, atol=1e-5), 'Reloaded prediction mismatch')
                    require(np.isclose(np.mean((vp - vy) ** 2), a['validation_mse_bps_squared'], rtol=1e-8), 'Member validation loss mismatch')
                    vp_seeds.append(vp)
                    reloaded[(decay, seed)] = model
                    checked_members += 1
                vp = np.mean(vp_seeds, axis=0)
                c = next(c for c in candidates if (c['fold_id'], c['feature_set'], c['weight_decay']) == (fold['id'], family, decay))
                require(np.allclose(vp, c['validation_predictions_bps'], rtol=1e-6, atol=1e-5) and np.isclose(np.mean((vp - vy) ** 2), c['validation_mse_bps_squared'], rtol=1e-8), 'Ensemble validation loss mismatch')
                local.append(c)
            winner = choose_candidate(local)
            h = next(h for h in selected if (h['fold_id'], h['feature_set']) == (fold['id'], family))
            require(h['weight_decay'] == winner['weight_decay'] and h['validation_mse_bps_squared'] == winner['validation_mse_bps_squared']
                    and h['selection_recorded_before_test_predictions'] and not h['test_labels_loaded'], 'Winner selection mismatch')
            cuts = validation_cutoffs(np.asarray(winner['validation_predictions_bps']))
            require(cuts == h['validation_cutoffs_bps'], 'Validation cutoff mismatch')
            sx = scaler.transform(matrix.loc[groups['test']].to_numpy()).astype(np.float32)
            member_test = []
            for seed in SEEDS:
                score = predict(reloaded[(winner['weight_decay'], seed)], sx, mean, std)
                saved = members_saved.loc[members_saved.fold_id.eq(fold['id']) & members_saved.feature_set.eq(family) & members_saved.seed.eq(seed)].sort_values('opportunity_id')
                require(saved.opportunity_id.tolist() == groups['test'] and np.allclose(score, saved.score_bps, rtol=1e-6, atol=1e-5), 'Member test mismatch')
                member_test.append(score)
            score = np.mean(member_test, axis=0)
            saved = predicted.loc[predicted.fold_id.eq(fold['id']) & predicted.feature_set.eq(family)].sort_values('opportunity_id')
            require(saved.opportunity_id.tolist() == groups['test'] and np.allclose(score, saved.score_bps, rtol=1e-6, atol=1e-5), 'Ensemble test mismatch')
            gates = {k: score >= v for k, v in cuts.items()}
            gates['economic'] = score > 0
            for gate, actions in gates.items():
                column = gate.replace('rank', 'gate') if gate != 'economic' else gate
                require(np.array_equal(saved[column].to_numpy(), actions.astype(int)), 'Prediction gate mismatch')
                signal = next(s for s in signals if (s['fold_id'], s['family'], s['gate']) == (fold['id'], family, gate))
                payload = read_json(output / signal['path'])
                require(sha(output / signal['path']) == signal['sha256'] and set(payload) == {'schema_version', 'mode', 'decisions'}, 'Sidecar integrity/schema')
                require([d['signal_at'] for d in payload['decisions']] == groups['test'] and
                        all(set(d) == {'signal_at', 'direction', 'exposure_fraction'} and type(d['direction']) is int
                            and d['direction'] == int(opp.loc[d['signal_at'], 'direction']) for d in payload['decisions']), 'Sidecar membership/outcomes/direction')
                require(np.array_equal([d['exposure_fraction'] for d in payload['decisions']], actions.astype(float)) and
                        signal['selected_count'] == int(actions.sum()) and signal['opportunities'] == len(actions) and signal['stress_reuses_base_signal'], 'Sidecar action/stress mismatch')
    return {'status': 'passed', 'reloaded_models': checked_members, 'selected_ensembles': 20, 'selected_constituents': 60,
            'test_predictions': len(predicted), 'signals': len(signals), 'formal_refits': 0,
            'test_labels_loaded': False, 'holdout_inspected': False,
            'checks': ['model_bytes_and_state_hashes', 'train_only_scaler_target_stats', 'exact_member_IDs',
                       'all_seed_validation_ensemble_MSE', 'validation_only_winner_and_cutoffs', 'member_test_reload',
                       'ensemble_test_arithmetic_mean', 'sidecar_actions_and_stress_reuse']}


def verify_synthetic():
    configure_torch()
    rng = np.random.default_rng(17)
    x, vx = rng.normal(size=(24, 5)), rng.normal(size=(8, 5)) + 20
    y = x[:, 0] * 30 + 8
    scaler = StandardScaler().fit(x)
    mean, std = float(y.mean()), float(y.std(ddof=0))
    tx, sv = scaler.transform(x).astype(np.float32), scaler.transform(vx).astype(np.float32)
    models, predictions = [], []
    for seed in SEEDS:
        model, _ = fit_member(tx, (y - mean) / std, sv, np.zeros(len(vx)), .01, seed, epochs=3)
        models.append(model)
        predictions.append(predict(model, sv, mean, std))
    repeated, _ = fit_member(tx, (y - mean) / std, sv, np.ones(len(vx)) * 1e6, .01, 17, epochs=3)
    require(state_sha(repeated) == state_sha(models[0]), 'Validation labels entered training')
    require(np.array_equal(scaler.mean_, x.mean(0)) and mean == y.mean(), 'Training-only preprocessing')
    changed, _ = fit_member(tx, (y - mean) / std, sv * 9 + 6, np.ones(len(vx)), .01, 17, epochs=3)
    require(state_sha(changed) == state_sha(models[0]), 'Validation features entered training')
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / 'model.pt'
        torch.save(models[0].state_dict(), p)
        restored = new_model(5, 43)
        restored.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
        require(np.array_equal(predict(restored, sv, mean, std), predictions[0]), 'Save/reload changed predictions')
    ensemble = np.mean(predictions, axis=0)
    require(np.array_equal(ensemble, np.stack(predictions).mean(0)), 'Seed ensemble')
    winner = choose_candidate([{'status': 'completed', 'validation_mse_bps_squared': 1., 'weight_decay': d} for d in DECAYS])
    require(winner['weight_decay'] == .1, 'Exact tie-break')
    cuts = validation_cutoffs(ensemble)
    require(set(cuts) == {'rank25', 'rank50', 'rank75'}, 'Validation cutoffs')
    print(json.dumps({'status': 'synthetic_checks_passed', 'formal_fits': 0, 'research_data_opened': False,
                      'checks': ['train_only_scaler_target', 'validation_label_and_feature_perturbation_model_invariance',
                                 'all_seed_arithmetic_ensemble', 'exact_tie_larger_decay', 'save_reload_exact_prediction', 'validation_cutoffs']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir', type=Path)
    group.add_argument('--verify-synthetic', action='store_true')
    group.add_argument('--verify-output', type=Path)
    args = parser.parse_args()
    if args.verify_synthetic:
        verify_synthetic()
    elif args.verify_output:
        print(json.dumps(verify_output(args.verify_output.resolve())))
    else:
        prepare(args.output_dir)
