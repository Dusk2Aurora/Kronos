"""Frozen same-coordinate tokenizer diagnostics; no labels, fitting or generation.

Default execution verifies eight development windows. --encode explicitly writes
the full registered development cache. Neither mode loads backbone weights.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model.kronos import KronosTokenizer
from research.frozen.encoder import (
    CUTOFF, FIELDS, HASHES, REVISIONS, array_sha, digest, immutable_bytes,
    immutable_json, load_development_inputs, prepare_window, require, sha,
    state_sha, verify_raw_aliases,
)

QUANTIZATION = {
    'continuous': 'L2_normalized_quant_embed_u_last20',
    'quantized': 'BSQuantizer_q_last20', 'pooling': 'last_valid_position',
    'dimension': 20, 'dtype': 'float32', 'batch_size': 8, 'chunk_size': 128,
    'atol': 1e-4, 'rtol': 1e-4, 'zero_rule': 'u_gt_zero_else_negative',
    'token_ids': 'explicit_s1_s2_half_true',
}
NORMALIZATION = {'context': 'historical_lookback_only', 'std_function': 'numpy.std',
                 'ddof': 0, 'epsilon': 1e-5, 'clip_abs': 5.0}
ARRAY_NAMES = ('u', 'q', 's1', 's2')


def validate_protocol(protocol):
    require(protocol.get('status') == 'locked', 'Quantization protocol must be locked')
    require(protocol['stage'] == 'frozen_development'
            and protocol['final_holdout_access'] == 'forbidden', 'Wrong stage/holdout permission')
    for key in ('backbone_training', 'tokenizer_training', 'future_path_generation'):
        require(protocol[key] == 'forbidden', f'Forbidden operation enabled: {key}')
    require(protocol['quantization'] == QUANTIZATION, 'Changed quantization contract')
    e = protocol['encoding']
    require(e['window_bars'] == 256 and e['fields'] == FIELDS and e['dtype'] == 'float32'
            and e['autocast'] is False and e['tf32'] is False
            and e['normalization'] == NORMALIZATION
            and pd.Timestamp(e['development_end_exclusive']) == CUTOFF,
            'Changed inherited input/numeric/holdout contract')


def load_tokenizer(device='cpu'):
    version_path = ROOT / 'research/initialization/version_manifest.json'
    version = json.loads(version_path.read_text(encoding='utf-8'))
    entry = version['weights']['tokenizer']
    path = ROOT / entry['local_cache']
    require(entry['revision'] == REVISIONS['tokenizer'] and path.name == REVISIONS['tokenizer'],
            'Unexpected tokenizer revision')
    for filename, expected in HASHES['tokenizer'].items():
        require(sha(path / filename) == expected, f'Changed tokenizer/{filename}')
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    require(config['d_in'] == 6 and config['d_model'] == 256
            and config['s1_bits'] == config['s2_bits'] == 10, 'Wrong pinned dimensions')
    tokenizer = KronosTokenizer.from_pretrained(str(path), local_files_only=True)
    tokenizer.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = {'revision': REVISIONS['tokenizer'], 'pinned_files': HASHES['tokenizer'],
                'config': config, 'state_sha256': state_sha(tokenizer),
                'version_manifest_sha256': sha(version_path), 'device': str(device),
                'torch': torch.__version__, 'numpy': np.__version__,
                'dtype': 'float32', 'autocast': False, 'tf32': False}
    return tokenizer, metadata


def encode_batch(tokenizer, windows, atol=1e-4, rtol=1e-4):
    """Return last-position u/q and IDs, auditing all 256 token positions."""
    require(bool(windows) and not tokenizer.training, 'Nonempty windows and eval tokenizer required')
    require(all(p.dtype == torch.float32 and not p.requires_grad for p in tokenizer.parameters()),
            'Tokenizer must be frozen float32')
    device = next(tokenizer.parameters()).device
    values = torch.from_numpy(np.stack([w[0] for w in windows])).to(device)
    require(values.shape[1:] == (256, 6) and torch.isfinite(values).all().item()
            and values.abs().max().item() <= 5, 'Invalid standardized window')
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        z = tokenizer.embed(values)
        for layer in tokenizer.encoder:
            z = layer(z)
        z = tokenizer.quant_embed(z)
        u = F.normalize(z, dim=-1)
        _, q, ids = tokenizer.tokenizer(z, half=True, collect_metrics=False)
        original = tokenizer.encode(values, half=True)
        require(all(torch.equal(ids[k], original[k]) for k in (0, 1)),
                'Diagnostic chain differs from original tokenizer IDs')
        expected = torch.where(u > 0, torch.ones_like(u), -torch.ones_like(u)) / (20 ** .5)
        require(torch.allclose(q, expected, atol=atol, rtol=rtol), 'Wrong sign quantization')
        require(torch.allclose(q, tokenizer.indices_to_bits(ids, half=True), atol=atol, rtol=rtol),
                'Quantized codes differ from explicit half-token reconstruction')
        require(u.shape == q.shape == (len(windows), 256, 20)
                and torch.isfinite(u).all().item() and torch.isfinite(q).all().item(),
                'Invalid bottleneck shape/values')
        require(torch.allclose(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=atol, rtol=rtol),
                'Non-unit quantized norm')
        # F.normalize keeps sub-epsilon vectors below unit norm; do not alter that boundary.
        norm = z.norm(dim=-1)
        target_norm = norm / norm.clamp_min(1e-12)
        require(torch.allclose(u.norm(dim=-1), target_norm, atol=atol, rtol=rtol),
                'Wrong continuous normalization')
        for token in ids:
            require(token.shape == (len(windows), 256) and token.dtype == torch.int64
                    and token.min().item() >= 0 and token.max().item() < 1024, 'Invalid token IDs')
        out = {'u': u[:, -1, :].cpu().numpy().copy(),
               'q': q[:, -1, :].cpu().numpy().copy(),
               's1': ids[0][:, -1].cpu().numpy().copy(),
               's2': ids[1][:, -1].cpu().numpy().copy()}
        audit = {'all_position_ids': [t.cpu().numpy().copy() for t in ids],
                 'sign_max_abs': float((q - expected).abs().max().item()),
                 'zero_u_coordinates': int((u == 0).sum().item()),
                 'sub_epsilon_vectors': int((norm < 1e-12).sum().item())}
    return out, audit


def verify_small_batch(tokenizer, candles, opportunities):
    rows = [row for _, row in opportunities.iloc[:8].iterrows()]
    require(len(rows) == 8, 'Eight verification opportunities required')
    windows = [prepare_window(candles, row) for row in rows]
    batch, audit = encode_batch(tokenizer, windows)
    repeat, repeat_audit = encode_batch(tokenizer, windows)
    single = [encode_batch(tokenizer, [window]) for window in windows]
    deviations = {}
    for key in ARRAY_NAMES:
        reference = np.concatenate([result[0][key] for result in single])
        if key in ('s1', 's2'):
            require(np.array_equal(reference, batch[key]) and np.array_equal(batch[key], repeat[key]),
                    'Batch/repeat last IDs mismatch')
        else:
            require(np.allclose(reference, batch[key], atol=1e-4, rtol=1e-4)
                    and np.allclose(batch[key], repeat[key], atol=1e-4, rtol=1e-4),
                    'Batch/repeat continuous representation mismatch')
            deviations[key] = {'batch_max_abs': float(np.max(np.abs(reference-batch[key]))),
                               'repeat_max_abs': float(np.max(np.abs(repeat[key]-batch[key])))}
    for k in (0, 1):
        reference = np.concatenate([result[1]['all_position_ids'][k] for result in single])
        require(np.array_equal(reference, audit['all_position_ids'][k])
                and np.array_equal(reference, repeat_audit['all_position_ids'][k]),
                'Batch/repeat all-position IDs mismatch')
    future = candles.copy()
    mask = future.bar_open_at >= pd.Timestamp(rows[0]['history_end_exclusive'])
    future.loc[mask, FIELDS] = future.loc[mask, FIELDS] * 7 + 11
    changed = prepare_window(future, rows[0])
    require(np.array_equal(changed[0], windows[0][0]) and changed[2] == windows[0][2],
            'Future perturbation affected window')
    after, after_audit = encode_batch(tokenizer, [changed])
    for key in ARRAY_NAMES:
        equal = (np.array_equal(after[key], single[0][0][key]) if key in ('s1', 's2') else
                 np.allclose(after[key], single[0][0][key], atol=1e-4, rtol=1e-4))
        require(equal, 'Future perturbation affected representation')
    require(all(np.array_equal(after_audit['all_position_ids'][k], single[0][1]['all_position_ids'][k])
                for k in (0, 1)), 'Future perturbation affected all-position IDs')
    start = pd.Timestamp(rows[0]['history_start_at'])
    inside = candles.index[candles.bar_open_at == start][0]
    negatives = {}
    cases = [('gap', candles.drop(index=inside), rows[0]),
             ('amount', candles.assign(amount=candles.amount.mask(candles.index == inside)), rows[0]),
             ('duplicate', pd.concat([candles, candles.loc[[inside]]]), rows[0]),
             ('unconfirmed', candles.assign(confirm=candles['confirm'].mask(candles.index == inside, 0)), rows[0]),
             ('availability', candles.assign(available_at=candles.available_at.mask(candles.index == inside, CUTOFF.isoformat())), rows[0]),
             ('id', candles, {**rows[0].to_dict(), 'opportunity_id': 'wrong'}),
             ('boundary', candles, {**rows[0].to_dict(), 'history_start_at': (start+pd.Timedelta(hours=1)).isoformat()}),
             ('holdout', candles, {**rows[0].to_dict(), 'history_end_exclusive': CUTOFF.isoformat(),
                                   'decision_boundary_at': CUTOFF.isoformat(),
                                   'history_start_at': (CUTOFF-pd.Timedelta(hours=256)).isoformat()})]
    for name, frame, row in cases:
        try:
            prepare_window(frame, row)
        except ValueError:
            negatives[name] = 'rejected'
        else:
            raise ValueError(f'{name} rejection failed')
    return {'status': 'passed', 'rows': [w[2]['opportunity_id'] for w in windows],
            'atol': 1e-4, 'rtol': 1e-4, 'deviations': deviations,
            'original_all_position_ids_identical': True, 'batch_all_position_ids_identical': True,
            'future_perturbation': 'unchanged', 'negative_checks': negatives,
            'sign_max_abs': audit['sign_max_abs'], 'zero_u_coordinates': audit['zero_u_coordinates'],
            'sub_epsilon_vectors': audit['sub_epsilon_vectors']}


def read_chunk(path, ids, keys):
    with np.load(path, allow_pickle=False) as chunk:
        require(chunk['ids'].tolist() == ids and chunk['keys'].tolist() == keys, 'Chunk IDs/keys mismatch')
        out = {key: chunk[key].copy() for key in ARRAY_NAMES}
        for key, value in out.items():
            shape, dtype = ((len(ids), 20), np.float32) if key in ('u', 'q') else ((len(ids),), np.int64)
            require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(),
                    f'Invalid cached {key}')
            require(array_sha(value) == str(chunk[key+'_sha256']), f'Corrupt cached {key}')
        require(all(((out[k] >= 0) & (out[k] < 1024)).all() for k in ('s1', 's2')), 'Cached IDs out of range')
        bits = np.arange(10)
        reconstructed = np.concatenate([((out[k][:, None] >> bits) & 1) for k in ('s1', 's2')], axis=1)
        reconstructed = (reconstructed.astype(np.float32)*2-1) / np.float32(20**.5)
        require(np.allclose(out['q'], reconstructed, atol=1e-4, rtol=1e-4), 'Cached q/token inconsistency')
        require(np.allclose(out['q'], np.where(out['u'] > 0, 1, -1)/(20**.5), atol=1e-4, rtol=1e-4),
                'Cached u/q sign inconsistency')
    return out


def parent_scope(protocol, ids, sources):
    parent = (ROOT / protocol['cache_reuse']['run'] / 'cache/pretrained').resolve()
    require(parent.is_relative_to(ROOT / 'research/runs'), 'Unsafe parent path')
    manifest_path, contract_path = parent / 'manifest.json', parent / 'contract.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['status'] == 'complete' and sha(contract_path) == manifest['contract_sha256'],
            'Invalid parent cache contract')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    require(contract['sources'] == sources, 'Snapshot/split/input sources differ from parent')
    require(contract['model']['tokenizer_revision'] == REVISIONS['tokenizer']
            and contract['model']['pinned_files']['tokenizer'] == HASHES['tokenizer'],
            'Parent tokenizer differs from pinned original')
    parent_ids = []
    for entry in manifest['chunks']:
        path = parent / entry['path']
        require(path.resolve().parent == parent and sha(path) == entry['sha256'], 'Changed parent chunk')
        with np.load(path, allow_pickle=False) as chunk:
            parent_ids.extend(chunk['ids'].tolist())
    require(parent_ids == ids and len(ids) == manifest['rows'] == 2460, 'Scope differs from parent DEV2460')
    return {'path': str(parent), 'manifest_sha256': sha(manifest_path),
            'contract_sha256': sha(contract_path), 'development_ids_sha256': digest(ids),
            'tokenizer_state_sha256': contract['model']['tokenizer_state_sha256']}


def run(protocol_path, output, device='cpu', encode=False):
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT / 'research/runs'), 'Output must be inside research/runs')
    protocol = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
    validate_protocol(protocol)
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(config['stage'] == 'frozen_development' and config['evaluation_readiness']['ready_to_encode']
            and config['evaluation_readiness']['final_holdout_evaluation_allowed'] is False,
            'Stage blocks frozen development diagnostics')
    frozen = config['frozen_representation']
    require(frozen['enabled_in_current_stage'] and frozen['inference_in_current_stage'] == 'allowed'
            and frozen['feature_extraction_in_current_stage'] == 'allowed', 'Frozen extraction blocked')
    registry_path = ROOT / 'research/registry/QUANT_20261008_v1.json'
    if encode:
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
        require(registry['experiment_id'] == 'QUANT_20261008_v1'
                and registry['stage'] == 'frozen_development'
                and registry['configuration']['sha256'] == sha(protocol_path), 'Missing/incompatible registration')
    candles, opportunities, sources = load_development_inputs(config)
    ids = opportunities.opportunity_id.tolist()
    parent = parent_scope(protocol, ids, sources)
    tokenizer, metadata = load_tokenizer(device)
    require(metadata['state_sha256'] == parent['tokenizer_state_sha256'], 'Parent tokenizer state differs')
    paths = [Path(__file__), ROOT / 'research/frozen/encoder.py', ROOT / 'model/kronos.py',
             ROOT / 'model/module.py', protocol_path, config_path,
             ROOT / 'research/environment/requirements.lock.txt',
             ROOT / 'research/initialization/version_manifest.json']
    code = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}
    contract = {'schema_version': 1, 'experiment_id': 'QUANT_20261008_v1',
                'sources': sources, 'parent_scope': parent, 'tokenizer': metadata,
                'source_sha256': code, 'protocol_sha256': sha(protocol_path),
                'initial_config_sha256': sha(config_path), 'cutoff_exclusive': CUTOFF.isoformat(),
                'scope': 'DEV2460_no_holdout_no_labels', 'quantization': QUANTIZATION,
                'fields': FIELDS, 'normalization': NORMALIZATION, 'window_bars': 256,
                'last_position': 255, 'coordinates': 'same_quant_embed_20',
                'mode': 'development_encoding' if encode else 'engineering_verification'}
    immutable_json(output / 'contract.json', contract)
    for path in paths:
        immutable_bytes(output / 'source' / path.relative_to(ROOT), path.read_bytes())
    checks = verify_small_batch(tokenizer, candles, opportunities)
    checks['raw_source_aliases'] = verify_raw_aliases(Path(sources['snapshot']), candles, opportunities)
    require(state_sha(tokenizer) == metadata['state_sha256'], 'Tokenizer state changed during verification')
    checks['tokenizer_state_unchanged'] = True
    immutable_json(output / 'verification.json', checks)
    if not encode:
        return checks
    chunks = []
    for offset in range(0, len(opportunities), 128):
        subset = opportunities.iloc[offset:offset+128]
        windows = [prepare_window(candles, row) for _, row in subset.iterrows()]
        chunk_ids = [w[2]['opportunity_id'] for w in windows]
        keys = [digest({'contract': contract, 'window': w[2],
                        'normalized_window_sha256': array_sha(w[0])}) for w in windows]
        path = output / f'chunk_{offset//128:05d}.npz'
        if path.exists():
            values = read_chunk(path, chunk_ids, keys)
        else:
            batches = [encode_batch(tokenizer, windows[i:i+8])[0] for i in range(0, len(windows), 8)]
            values = {key: np.concatenate([batch[key] for batch in batches]) for key in ARRAY_NAMES}
            payload = {'ids': np.asarray(chunk_ids), 'keys': np.asarray(keys), **values,
                       'windows': np.asarray([json.dumps(w[2], sort_keys=True) for w in windows]),
                       'normalized_window_sha256': np.asarray([array_sha(w[0]) for w in windows])}
            payload.update({key+'_sha256': np.asarray(array_sha(value)) for key, value in values.items()})
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **payload)
            immutable_bytes(path, buffer.getvalue())
            read_chunk(path, chunk_ids, keys)
        chunks.append({'path': path.name, 'sha256': sha(path), 'rows': len(chunk_ids),
                       'first_id': chunk_ids[0], 'last_id': chunk_ids[-1],
                       'arrays_sha256': {k: array_sha(v) for k, v in values.items()}})
        print(f'chunk {offset//128:05d}: {offset+len(chunk_ids)}/{len(ids)}', flush=True)
    require(state_sha(tokenizer) == metadata['state_sha256'], 'Tokenizer state changed during encoding')
    manifest = {'status': 'complete', 'rows': len(ids), 'shape': [len(ids), 20], 'dtype': 'float32',
                'development_ids_sha256': digest(ids), 'chunks': chunks,
                'contract_sha256': sha(output/'contract.json'),
                'verification_sha256': sha(output/'verification.json'), 'tokenizer_state_unchanged': True}
    immutable_json(output / 'manifest.json', manifest)
    load_cache(output)
    return manifest


def load_cache(output):
    output = Path(output).resolve()
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['status'] == 'complete' and manifest['rows'] == 2460, 'Incomplete DEV2460 cache')
    require(sha(output/'contract.json') == manifest['contract_sha256']
            and sha(output/'verification.json') == manifest['verification_sha256'], 'Changed contract/verification')
    contract = json.loads((output/'contract.json').read_text(encoding='utf-8'))
    require(contract['quantization'] == QUANTIZATION and contract['normalization'] == NORMALIZATION,
            'Changed cache numeric contract')
    ids, matrices = [], {k: [] for k in ARRAY_NAMES}
    for entry in manifest['chunks']:
        path = output / entry['path']
        require(path.resolve().parent == output and sha(path) == entry['sha256'], 'Changed/unsafe chunk')
        with np.load(path, allow_pickle=False) as chunk:
            chunk_ids, keys = chunk['ids'].tolist(), chunk['keys'].tolist()
            windows = [json.loads(w) for w in chunk['windows'].tolist()]
            normalized_hashes = chunk['normalized_window_sha256'].tolist()
        require(len(chunk_ids) == entry['rows'] and chunk_ids[0] == entry['first_id']
                and chunk_ids[-1] == entry['last_id'] and len(keys) == len(chunk_ids)
                and len(windows) == len(normalized_hashes) == len(chunk_ids), 'Invalid chunk index')
        expected_keys = [digest({'contract': contract, 'window': w, 'normalized_window_sha256': h})
                         for w, h in zip(windows, normalized_hashes)]
        require(keys == expected_keys and [w['opportunity_id'] for w in windows] == chunk_ids,
                'Window/key provenance mismatch')
        require(all(pd.Timestamp(w['window_end_exclusive']) < CUTOFF for w in windows), 'Holdout in cache')
        values = read_chunk(path, chunk_ids, keys)
        require({k: array_sha(v) for k, v in values.items()} == entry['arrays_sha256'], 'Changed array manifest')
        ids.extend(chunk_ids)
        for key in ARRAY_NAMES:
            matrices[key].append(values[key])
    require(len(ids) == 2460 and ids == sorted(set(ids))
            and digest(ids) == manifest['development_ids_sha256']
            == contract['sources']['development_ids_sha256']
            == contract['parent_scope']['development_ids_sha256'], 'Changed total opportunity scope')
    index = pd.Index(ids, name='opportunity_id')
    return {name: pd.DataFrame(np.concatenate(matrices[key]), index=index,
                              columns=[f'{name}_{i:02d}' for i in range(20)])
            for key, name in (('u', 'continuous'), ('q', 'binary'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', '--output-dir', dest='output', type=Path, required=True,
                        help='Immutable cache/verification directory inside research/runs')
    parser.add_argument('--device', default='cpu')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--encode', action='store_true', help='Encode registered DEV2460; default is verification only')
    mode.add_argument('--verify-only', action='store_true', help='Verify eight development windows only (default)')
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.output, args.device, args.encode), indent=2))


if __name__ == '__main__':
    main()
