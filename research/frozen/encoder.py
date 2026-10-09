"""Offline, per-opportunity frozen historical encoding (no forecasting API)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model.kronos import Kronos, KronosTokenizer, calc_time_stamps

FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount']
CUTOFF = pd.Timestamp('2026-04-01T00:00:00Z')
SEEDS = (17, 29, 43)
REVISIONS = {'model': '901c26c1332695a2a8f243eb2f37243a37bea320',
             'tokenizer': '0e0117387f39004a9016484a186a908917e22426'}
HASHES = {
    'model': {'config.json': '5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396',
              'model.safetensors': 'b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020'},
    'tokenizer': {'config.json': '2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2',
                  'model.safetensors': '59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee'}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + canonical(list(array.shape)).encode()
                          + array.tobytes()).hexdigest()


def immutable_bytes(path, data):
    """Publish complete bytes atomically, never replace an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f'Immutable artifact differs: {path}')
        return
    handle, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def immutable_json(path, value):
    immutable_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def state_sha(model):
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(array_sha(value.detach().cpu().numpy()).encode())
    return hasher.hexdigest()


def load_models(variant='pretrained', seed=None, device='cpu'):
    """Verify pinned cached artifacts before offline loading or random construction."""
    require(variant in ('pretrained', 'random'), 'Unknown backbone variant')
    require((variant == 'random' and seed in SEEDS) or
            (variant == 'pretrained' and seed is None), 'Invalid variant/seed')
    version = json.loads((ROOT / 'research/initialization/version_manifest.json').read_text())
    paths = {}
    for kind in ('model', 'tokenizer'):
        entry = version['weights'][kind]
        require(entry['revision'] == REVISIONS[kind], f'Unexpected {kind} revision')
        paths[kind] = ROOT / entry['local_cache']
        require(paths[kind].name == REVISIONS[kind], 'Cache revision path mismatch')
        for filename, expected in HASHES[kind].items():
            require(sha(paths[kind] / filename) == expected, f'Changed {kind}/{filename}')
    tokenizer = KronosTokenizer.from_pretrained(str(paths['tokenizer']), local_files_only=True)
    model_config = json.loads((paths['model'] / 'config.json').read_text())
    if variant == 'pretrained':
        model = Kronos.from_pretrained(str(paths['model']), local_files_only=True)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = Kronos(**model_config)
    weight_sha = state_sha(model)
    tokenizer_sha = state_sha(tokenizer)
    for module in (model, tokenizer):
        module.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = {'variant': variant, 'seed': seed, 'model_revision': REVISIONS['model'],
                'tokenizer_revision': REVISIONS['tokenizer'], 'pinned_files': HASHES,
                'model_config': model_config, 'model_state_sha256': weight_sha,
                'tokenizer_state_sha256': tokenizer_sha, 'dtype': 'float32',
                'normalization': 'per_window_numpy_float64_ddof0_eps1e-5_clip5_v1',
                'pooling': 'decode_s1_context_last', 'time_fields': ['minute', 'hour', 'weekday', 'day', 'month'],
                'tf32': False, 'autocast': False, 'device': str(device), 'torch': torch.__version__, 'numpy': np.__version__}
    return tokenizer, model, metadata


def prepare_window(candles, opportunity):
    """Timestamp selection deliberately does not trust stored row offsets."""
    start = pd.Timestamp(opportunity['history_start_at'])
    end = pd.Timestamp(opportunity['history_end_exclusive'])
    boundary = pd.Timestamp(opportunity['decision_boundary_at'])
    decision = pd.Timestamp(opportunity['decision_at'])
    require(start.tzinfo is not None and end.tzinfo is not None, 'UTC-aware boundaries required')
    require(end == boundary and end < CUTOFF and start == end - pd.Timedelta(hours=256),
            'Wrong or holdout window boundary')
    require(str(opportunity['opportunity_id']) == boundary.strftime('%Y-%m-%dT%H:%M:%SZ'), 'ID/boundary mismatch')
    require(boundary.hour in (4, 12, 20) and boundary.minute == boundary.second == 0,
            'Wrong decision schedule')
    require(decision == boundary + pd.Timedelta(seconds=60), 'Wrong decision availability delay')
    frame = candles.loc[(candles['bar_open_at'] >= start) & (candles['bar_open_at'] < end)].copy()
    times = pd.DatetimeIndex(frame['bar_open_at'])
    expected = pd.date_range(start, periods=256, freq='h')
    require(len(frame) == 256 and times.equals(expected), 'Missing, duplicated, unsorted or shifted bars')
    require((pd.to_datetime(frame['bar_close_at'], utc=True) == times + pd.Timedelta(hours=1)).all(),
            'Wrong bar close boundary')
    require((pd.to_datetime(frame['available_at'], utc=True) <= decision).all(), 'Unavailable historical bar')
    require((frame['confirm'].astype(str) == '1').all(), 'Unconfirmed historical bar')
    require(frame['amount'].notna().all(), 'Missing actual amount')
    values = frame[FIELDS].to_numpy(dtype=np.float64)
    require(np.isfinite(values).all() and (values[:, :4] > 0).all()
            and (values[:, 4:] >= 0).all(), 'Invalid OHLCVA values')
    require((values[:, 1] >= np.maximum(values[:, 0], values[:, 3])).all()
            and (values[:, 2] <= np.minimum(values[:, 0], values[:, 3])).all(), 'Impossible OHLC bounds')
    normalized = np.clip((values - values.mean(axis=0)) / (values.std(axis=0, ddof=0) + 1e-5), -5, 5)
    stamps = calc_time_stamps(pd.Series(times)).to_numpy(dtype=np.float32)
    identity = {'opportunity_id': str(opportunity['opportunity_id']), 'asset': 'BTC-USDT-SWAP',
                'window_start': start.isoformat(), 'window_end_exclusive': end.isoformat(),
                'history_values_sha256': array_sha(values), 'history_timestamps_sha256': digest(times.astype(str).tolist())}
    return normalized.astype(np.float32), stamps, identity


def encode_batch(tokenizer, model, windows):
    require(bool(windows), 'Empty encoding batch')
    require(not tokenizer.training and not model.training, 'Models must be eval')
    require(all(p.dtype == torch.float32 for module in (tokenizer, model) for p in module.parameters()),
            'Only float32 parameters are registered')
    require(not any(p.requires_grad for module in (tokenizer, model) for p in module.parameters()),
            'Models must be frozen')
    device = next(model.parameters()).device
    values = torch.from_numpy(np.stack([w[0] for w in windows])).to(device)
    stamps = torch.from_numpy(np.stack([w[1] for w in windows])).to(device)
    require(values.shape[1:] == (256, 6) and stamps.shape[1:] == (256, 5), 'Wrong history shape')
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        s1, s2 = tokenizer.encode(values, half=True)
        _, context = model.decode_s1(s1, s2, stamp=stamps)
        result = context[:, -1, :].cpu().numpy().copy()
    require(result.shape == (len(windows), model.d_model) and np.isfinite(result).all(), 'Invalid context')
    return result


def load_development_inputs(config):
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for filename in ('opportunities.csv', 'split_index.csv'):
        require(sha(bundle / filename) == manifest['artifact_sha256'][filename], 'Changed ' + filename)
    snapshot = Path(manifest['sources']['snapshot'])
    if not snapshot.is_absolute():
        snapshot = ROOT / snapshot
    require(sha(snapshot / 'manifest.json') == manifest['sources']['snapshot_manifest_sha256'], 'Changed snapshot manifest')
    require(sha(snapshot / 'candles.csv') == manifest['sources']['snapshot_csv_sha256']['candles.csv'], 'Changed candles')
    split = pd.read_csv(bundle / 'split_index.csv', dtype=str)
    selected = split[(split.fold_id.isin(['WF01', 'WF02', 'WF03', 'WF04']))
                     & split.role.isin(['train', 'validation', 'test']) & (split.included == 'True')]
    require((pd.to_datetime(selected.segment_end_exclusive, utc=True) <= CUTOFF).all(), 'Development fold reaches holdout')
    ids = sorted(set(selected.opportunity_id))
    opportunities = pd.read_csv(bundle / 'opportunities.csv', dtype=str)
    require(not opportunities.opportunity_id.duplicated().any(), 'Duplicate opportunity IDs')
    opportunities = opportunities.set_index('opportunity_id', drop=False).loc[ids]
    require((pd.to_datetime(opportunities.decision_boundary_at, utc=True) < CUTOFF).all(), 'Holdout opportunity')
    # No labels/profit/funding table is read by this module.
    candles = pd.read_csv(snapshot / 'candles.csv', dtype=str, usecols=['bar_open_at', 'bar_close_at', 'available_at', 'confirm'] + FIELDS)
    candles['bar_open_at'] = pd.to_datetime(candles['bar_open_at'], utc=True)
    candles = candles[candles.bar_open_at < CUTOFF].copy()
    for field in FIELDS:
        candles[field] = pd.to_numeric(candles[field], errors='raise')
    sources = {'snapshot_manifest_sha256': sha(snapshot / 'manifest.json'),
               'candles_sha256': sha(snapshot / 'candles.csv'), 'opportunities_sha256': sha(bundle / 'opportunities.csv'),
               'split_index_sha256': sha(bundle / 'split_index.csv'), 'development_ids_sha256': digest(ids),
               'snapshot': str(snapshot), 'bundle': str(bundle)}
    return candles, opportunities, sources


def verify_small_batch(tokenizer, model, candles, opportunities, atol=1e-4, rtol=1e-4, batch_size=8):
    require(len(opportunities) >= batch_size, 'Insufficient verification opportunities')
    rows = [row for _, row in opportunities.iloc[:batch_size].iterrows()]
    windows = [prepare_window(candles, row) for row in rows]
    reference = np.concatenate([encode_batch(tokenizer, model, [window]) for window in windows])
    repeat = np.concatenate([encode_batch(tokenizer, model, [window]) for window in windows])
    batch = encode_batch(tokenizer, model, windows)
    device = next(model.parameters()).device
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        tokens_batch = tokenizer.encode(torch.from_numpy(np.stack([w[0] for w in windows])).to(device), half=True)
        tokens_single = [tokenizer.encode(torch.from_numpy(w[0][None]).to(device), half=True) for w in windows]
    token_equal = all(torch.equal(tokens_batch[k], torch.cat([t[k] for t in tokens_single])) for k in (0, 1))
    require(token_equal, 'Batch changes discrete tokenizer tokens; registered batching rejected')
    require(np.allclose(reference, repeat, atol=atol, rtol=rtol), 'Repeat mismatch')
    require(np.allclose(reference, batch, atol=atol, rtol=rtol), 'Batch mismatch')
    future = candles.copy()
    mask = future.bar_open_at >= pd.Timestamp(rows[0]['history_end_exclusive'])
    future.loc[mask, FIELDS] = future.loc[mask, FIELDS] * 7 + 11
    changed = prepare_window(future, rows[0])
    require(np.array_equal(changed[0], windows[0][0]) and changed[2] == windows[0][2], 'Future perturbation affected window')
    require(np.allclose(reference[:1], encode_batch(tokenizer, model, [changed]), atol=atol, rtol=rtol), 'Future perturbation affected hidden')
    start = pd.Timestamp(rows[0]['history_start_at'])
    inside = candles.index[candles.bar_open_at == start][0]
    checks = {}
    for name, modified, row in [('gap', candles.drop(index=inside), rows[0]),
                                 ('amount', candles.assign(amount=candles.amount.mask(candles.index == inside)), rows[0]),
                                 ('id', candles, {**rows[0].to_dict(), 'opportunity_id': 'wrong'}),
                                 ('duplicate', pd.concat([candles, candles.loc[[inside]]]), rows[0]),
                                 ('unconfirmed', candles.assign(confirm=candles['confirm'].mask(candles.index == inside, 0)), rows[0]),
                                 ('availability', candles.assign(available_at=candles.available_at.mask(candles.index == inside, '2026-04-01T00:00:00Z')), rows[0]),
                                 ('boundary', candles, {**rows[0].to_dict(), 'history_start_at': (start + pd.Timedelta(hours=1)).isoformat()})]:
        try:
            prepare_window(modified, row)
        except ValueError:
            checks[name] = 'rejected'
        else:
            raise ValueError(f'{name} rejection failed')
    return {'status': 'passed', 'rows': [r['opportunity_id'] for r in rows], 'shape': list(reference.shape),
            'atol': atol, 'rtol': rtol, 'repeat_max_abs': float(np.max(np.abs(reference-repeat))),
            'batch_max_abs': float(np.max(np.abs(reference-batch))), 'batch_size': batch_size, 'batch_tokens_identical': token_equal, 'future_perturbation': 'unchanged',
            'negative_checks': checks}


def read_chunk(path, keys, ids, width):
    with np.load(path, allow_pickle=False) as chunk:
        require(chunk['keys'].tolist() == keys and chunk['ids'].tolist() == ids, 'Chunk key or ID mismatch')
        features = chunk['features']
        require(features.dtype == np.float32 and features.shape == (len(ids), width)
                and np.isfinite(features).all(), 'Chunk shape/dtype/finite mismatch')
        require(array_sha(features) == str(chunk['features_sha256']), 'Corrupt chunk features')
        return features.copy()


def validate_protocol(protocol, batch_size, chunk_size):
    e = protocol['encoding']
    require(protocol['stage'] == 'frozen_development' and protocol['final_holdout_access'] == 'forbidden', 'Wrong protocol stage')
    require(e['window_bars'] == 256 and e['fields'] == FIELDS and e['dtype'] == 'float32'
            and e['autocast'] is False and e['tf32'] is False, 'Wrong encoding numeric contract')
    require(e['normalization'] == {'context': 'historical_lookback_only', 'std_function': 'numpy.std',
                                 'ddof': 0, 'epsilon': 1e-5, 'clip_abs': 5.0}, 'Wrong normalization')
    require(e['pooling'] == 'decode_s1_last_valid_context' and e['atol'] == e['rtol'] == 1e-4,
            'Wrong pooling or registered tolerance')
    require(e['batch_size'] == batch_size and e['chunk_size'] == chunk_size, 'CLI batching differs from protocol')
    require(pd.Timestamp(e['development_end_exclusive']) == CUTOFF and protocol['variants']['E02R']['seeds'] == list(SEEDS),
            'Wrong cutoff or random seeds')


def verify_raw_aliases(snapshot, candles, opportunities):
    """Compare a few development history bars to raw OKX volCcy/volCcyQuote fields."""
    from decimal import Decimal
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    sampled = opportunities.iloc[[0, len(opportunities)//2, len(opportunities)-1]]
    times = set(pd.Timestamp(row['history_start_at']) for _, row in sampled.iterrows())
    expected = {str(int(t.timestamp()*1000)): candles.loc[candles.bar_open_at == t].iloc[0] for t in times}
    found = {}
    for source in manifest['sources']:
        if 'history-candles?' not in source['url']:
            continue
        path = snapshot / source['path']
        require(sha(path) == source['sha256'], 'Changed raw candle source')
        payload = json.loads(path.read_text(encoding='utf-8'))
        for raw in payload.get('data', []):
            if raw[0] not in expected:
                continue
            row = expected[raw[0]]
            require(Decimal(raw[6]) == Decimal(str(row['volume'])) and
                    Decimal(raw[7]) == Decimal(str(row['amount'])), 'Raw volume/amount aliases disagree')
            found[raw[0]] = {'bar_open_at': row['bar_open_at'].isoformat(), 'raw_path': source['path'],
                             'raw_sha256': source['sha256'], 'volume_source': 'volCcy', 'amount_source': 'volCcyQuote'}
        if len(found) == len(expected):
            break
    require(len(found) == len(expected), 'Raw alias samples incomplete')
    return list(found.values())


def run(protocol_path, output, variant='pretrained', seed=None, device='cpu', batch_size=8, chunk_size=128, verify_only=False):
    require(batch_size > 0 and chunk_size > 0, 'Positive batch and chunk sizes required')
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT / 'research/runs'), 'Output must be under research/runs')
    protocol = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
    require(protocol.get('status', protocol.get('protocol', {}).get('status')) == 'locked', 'Protocol must be locked')
    validate_protocol(protocol, batch_size, chunk_size)
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    frozen = config['frozen_representation']
    require(config['stage'] == 'frozen_development' and config['evaluation_readiness']['final_holdout_evaluation_allowed'] is False,
            'Wrong stage or holdout permissions')
    require(config['evaluation_readiness']['ready_to_encode'] and frozen['enabled_in_current_stage']
            and frozen['inference_in_current_stage'] == 'allowed'
            and frozen['feature_extraction_in_current_stage'] == 'allowed', 'Stage blocks frozen encoding')
    candles, opportunities, sources = load_development_inputs(config)
    tokenizer, model, metadata = load_models(variant, seed, device)
    code = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
            for p in [Path(__file__), ROOT / 'model/kronos.py', ROOT / 'model/module.py']}
    contract = {'schema_version': 1, 'sources': sources, 'model': metadata, 'code_sha256': code,
                'protocol_sha256': sha(protocol_path), 'initial_config_sha256': sha(config_path),
                'environment_lock_sha256': sha(ROOT / 'research/environment/requirements.lock.txt'),
                'cutoff_exclusive': CUTOFF.isoformat(), 'batch_size': batch_size, 'chunk_size': chunk_size,
                'mode': 'verify_only' if verify_only else 'development_encoding'}
    immutable_json(output / 'contract.json', contract)
    for path in [Path(__file__), ROOT / 'model/kronos.py', ROOT / 'model/module.py',
                 protocol_path, config_path, ROOT / 'research/environment/requirements.lock.txt']:
        immutable_bytes(output / 'source' / path.name, path.read_bytes())
    immutable_json(output / 'source' / 'hashes.json', {path.name: sha(path) for path in
                    [Path(__file__), ROOT / 'model/kronos.py', ROOT / 'model/module.py', protocol_path, config_path,
                     ROOT / 'research/environment/requirements.lock.txt']})
    if variant == 'random':
        # Keep exact initialization, not only a seed; no pretrained backbone loader is used.
        from safetensors.torch import save
        tensors = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        immutable_bytes(output / 'random_initialization.safetensors', save(tensors))
    checks = verify_small_batch(tokenizer, model, candles, opportunities, protocol['encoding']['atol'], protocol['encoding']['rtol'], batch_size)
    checks['raw_source_aliases'] = verify_raw_aliases(Path(sources['snapshot']), candles, opportunities)
    immutable_json(output / 'verification.json', checks)
    if verify_only:
        return checks
    started = time.perf_counter()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    chunks = []
    rows = [row for _, row in opportunities.iterrows()]
    for offset in range(0, len(rows), chunk_size):
        subset = rows[offset:offset + chunk_size]
        windows = [prepare_window(candles, row) for row in subset]
        ids = [w[2]['opportunity_id'] for w in windows]
        keys = [digest({'contract': contract, 'window': w[2]}) for w in windows]
        path = output / f'chunk_{offset // chunk_size:05d}.npz'
        if path.exists():
            features = read_chunk(path, keys, ids, model.d_model)
        else:
            features = np.concatenate([encode_batch(tokenizer, model, windows[i:i+batch_size])
                                       for i in range(0, len(windows), batch_size)])
            buffer = io.BytesIO()
            np.savez_compressed(buffer, ids=np.asarray(ids), keys=np.asarray(keys), features=features,
                                features_sha256=np.asarray(array_sha(features)))
            immutable_bytes(path, buffer.getvalue())
            read_chunk(path, keys, ids, model.d_model)
        print(f'chunk {offset // chunk_size:05d}: {offset + len(ids)}/{len(rows)} opportunities', flush=True)
        chunks.append({'path': path.name, 'sha256': sha(path), 'rows': len(ids), 'first_id': ids[0],
                       'last_id': ids[-1], 'features_sha256': array_sha(features)})
    require(state_sha(model) == metadata['model_state_sha256'] and state_sha(tokenizer) == metadata['tokenizer_state_sha256'],
            'Frozen weights changed during encoding')
    manifest = {'status': 'complete', 'contract_sha256': sha(output / 'contract.json'), 'rows': len(rows),
                'shape': [len(rows), model.d_model], 'dtype': 'float32', 'chunks': chunks,
                'verification_sha256': sha(output / 'verification.json'),
                'random_initialization_sha256': sha(output / 'random_initialization.safetensors') if variant == 'random' else None}
    immutable_json(output / 'manifest.json', manifest)
    runtime = {'seconds': time.perf_counter()-started,
               'opportunities_per_second': len(rows) / max(time.perf_counter()-started, 1e-9),
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
               'device_name': torch.cuda.get_device_name() if str(device).startswith('cuda') else 'cpu'}
    # A resumed execution gets a new runtime record; scientific artifacts remain immutable.
    immutable_json(output / f'runtime_{time.time_ns()}.json', runtime)
    return manifest


def load_cache(output):
    """Return float32 DataFrame indexed by unique opportunity_id, checking all hashes."""
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    require(manifest['status'] == 'complete', 'Incomplete encoding cache')
    require(sha(output / 'contract.json') == manifest['contract_sha256'], 'Changed cache contract')
    require(sha(output / 'verification.json') == manifest['verification_sha256'], 'Changed verification')
    if manifest.get('random_initialization_sha256'):
        require(sha(output / 'random_initialization.safetensors') == manifest['random_initialization_sha256'], 'Changed random initialization')
    values, all_ids = [], []
    for entry in manifest['chunks']:
        path = output / entry['path']
        require(path.parent.resolve() == output.resolve(), 'Unsafe chunk path')
        require(sha(path) == entry['sha256'], 'Changed chunk bytes')
        with np.load(path, allow_pickle=False) as chunk:
            ids, keys = chunk['ids'].tolist(), chunk['keys'].tolist()
        require(len(ids) == entry['rows'] and len(keys) == len(ids)
                and len(set(keys)) == len(keys), 'Invalid chunk index')
        require(ids[0] == entry['first_id'] and ids[-1] == entry['last_id'], 'Changed chunk ID boundary')
        matrix = read_chunk(path, keys, ids, manifest['shape'][1])
        require(array_sha(matrix) == entry['features_sha256'], 'Changed feature checksum')
        values.append(matrix)
        all_ids.extend(ids)
    require(len(all_ids) == manifest['rows'] and len(set(all_ids)) == len(all_ids)
            and all_ids == sorted(all_ids), 'Invalid cache ID merge')
    matrix = np.concatenate(values)
    require(list(matrix.shape) == manifest['shape'], 'Wrong cache total shape')
    return pd.DataFrame(matrix, index=pd.Index(all_ids, name='opportunity_id'),
                        columns=[f'hidden_{i:03d}' for i in range(matrix.shape[1])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variant', choices=['pretrained', 'random'], default='pretrained')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    result = run(args.protocol, args.output, args.variant, args.seed, args.device,
                 args.batch_size, args.chunk_size, args.verify_only)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
