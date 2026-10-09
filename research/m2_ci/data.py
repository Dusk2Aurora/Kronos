"""DEV-only preparation. Future RV values are kept outside predictor artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'research/m2_ci/config.yaml'
DEFAULT_RUN = ROOT / 'research/runs/M2_CONDITIONAL_INCREMENT_01'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def config():
    registry = read(ROOT / 'research/registry/M2_CONDITIONAL_INCREMENT_01.json')
    digest = sha(CONFIG)
    if (registry['configuration']['path'] != CONFIG.relative_to(ROOT).as_posix()
            or registry['configuration']['sha256'] != digest or registry['protocol_sha256'] != digest):
        raise PermissionError('M2 configuration differs from preregistered protocol')
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def frame(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        values.to_csv(stream, index=False, float_format='%.17g')


def new_scope(run):
    run = Path(run).resolve()
    base = (ROOT / 'research/runs').resolve()
    if not run.is_relative_to(base) or not run.name.startswith('M2_CONDITIONAL_INCREMENT_01'):
        raise PermissionError('Only isolated new M2 run directories are writable')
    c = config()
    if c['sources']['development_only_max_time_exclusive'] != '2026-04-01T00:00:00Z':
        raise PermissionError('DEV-only boundary changed')
    review = read(ROOT / 'research/m2_ci/phase_a_independent_review.json')
    if review['status'] != 'PASS' or review['new_Q2_Q3_value_reads']:
        raise PermissionError('Passed independent Phase A required')
    for key, parent, manifest in [('original_03_manifest_sha256', 'parent_03','formal_delivery_manifest.json'),
                                 ('original_03_1_manifest_sha256', 'parent_03_1','delivery_manifest.json')]:
        if sha(ROOT / c['sources'][parent] / manifest) != c['sources'][key]:
            raise ValueError('Changed original delivery manifest')
    if read(ROOT / c['sources']['parent_03'] / 'authorization/formal_terminal.json')['state'] != 'CONSUMED':
        raise PermissionError('Original formal lifecycle must remain terminal CONSUMED')
    if read(ROOT / 'research/researchstate.json')['holdout']['status'] != 'CONSUMED':
        raise PermissionError('Original global holdout state must remain CONSUMED')
    return run, c, review


def role_indices(metadata, start, end):
    """All three clocks are evaluated before target slicing."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    boundary = pd.to_datetime(metadata.decision_boundary_at, utc=True)
    label_end = pd.to_datetime(metadata.label_end, utc=True)
    labelable = pd.to_datetime(metadata.labelable_at, utc=True)
    return np.flatnonzero(((boundary >= start) & (boundary < end)
                          & (label_end < end) & (labelable < end)).to_numpy())


def fold_indices(metadata, c):
    result = {}
    for f in c['roles']['folds']:
        roles = {'fit': role_indices(metadata, c['roles']['train_start'], f['fit_end']),
                 'inner_validation': role_indices(metadata, f['fit_end'], f['evaluation_start']),
                 'evaluation': role_indices(metadata, f['evaluation_start'], f['evaluation_end'])}
        counts = [len(roles[r]) for r in ('fit', 'inner_validation', 'evaluation')]
        if counts != f['expected_counts']:
            raise ValueError(f"{f['id']} counts differ: {counts}")
        if any(np.intersect1d(roles[a], roles[b]).size for a, b in
               [('fit', 'inner_validation'), ('fit', 'evaluation'), ('inner_validation', 'evaluation')]):
            raise ValueError('Overlapping roles')
        result[f['id']] = roles
    if sum(len(v['evaluation']) for v in result.values()) != 1267:
        raise ValueError('Expected exactly 1267 OOF predictions')
    return result


def _review_hashes(review):
    expected = {item['path']: item['expected_sha256'] for audit in review['old_delivery_audits']
                for item in audit['checks'] if item['status'] == 'PASS'}
    c = config()
    parent = ROOT / c['sources']['parent_03']
    manifest = parent / 'formal_delivery_manifest.json'
    if sha(manifest) != c['sources']['original_03_manifest_sha256']:
        raise ValueError('Original formal manifest anchor changed')
    bundle = parent / 'preflight/sealed_bundle.json'
    if sha(bundle) != read(manifest)['sealed_bundle_sha256']:
        raise ValueError('Original nested sealed-bundle anchor changed')
    for path, digest in read(bundle)['files_sha256'].items():
        if path in expected and expected[path] != digest:
            raise ValueError('Conflicting old source binding: ' + path)
        expected[path] = digest
    return expected


def prepare(run=DEFAULT_RUN):
    run, c, review = new_scope(run)  # Before any numerical source read.
    from research.frozen.experiment_03 import inputs
    from research.frozen.experiment_02.features import FEATURE_NAMES, HAR_NAMES, R0_NAMES
    parent = ROOT / c['sources']['parent_03']
    paths = [parent / 'features/opportunities.csv', parent / 'features/ordinary_risk.csv',
             parent / 'labels/development.csv', parent / 'features/pretrained_v2.npz',
             parent / 'features/cache_audit_v2.json',
             ROOT / c['sources']['development_5m'] / 'candles.csv']
    expected = _review_hashes(review)
    for p in paths[:-1]:
        key = p.relative_to(ROOT).as_posix()
        if key not in expected or sha(p) != expected[key]:
            raise ValueError('DEV source not independently byte verified: ' + key)
    five_directory = paths[-1].parent
    five_manifest = read(five_directory / 'manifest.json')
    five_audit = read(five_directory / 'raw_csv_audit.json')
    if (sha(paths[-1]) != review['development_contract']['development_5m_candles_sha256']
            or five_manifest['candles_sha256'] != sha(paths[-1])
            or [pd.Timestamp(v) for v in five_manifest['range']] !=
               [pd.Timestamp('2024-01-01T00:00:00Z'),pd.Timestamp('2026-04-01T00:00:00Z')]
            or five_manifest['instrument'] != 'BTC-USDT-SWAP'
            or five_manifest['bar'] != '5m' or five_audit['status'] != 'PASS'
            or not five_audit['exact_raw_csv_match'] or five_audit['holdout_rows'] != 0):
        raise ValueError('DEV 5m independent review/source manifest chain differs')
    # Original source modules/configs are part of the independently verified delivery.
    for key, digest in expected.items():
        if key.startswith('research/frozen/experiment_03/') and key.endswith(('.py', '.yaml')):
            if sha(ROOT / key) != digest:
                raise ValueError('Changed original implementation: ' + key)
    cache = read(parent / 'features/cache_audit_v2.json')
    if not cache or any(v.get('checks', {}).get('status') != 'passed' for v in cache):
        raise ValueError('Original DEV cache checks did not pass')
    opp = pd.read_csv(paths[0], dtype={'opportunity_id': str}, float_precision='round_trip')
    ids = opp.opportunity_id.to_numpy(dtype=str)
    if len(opp) != 2461 or not opp.opportunity_id.is_unique or set(opp.role) != {'train', 'validation'}:
        raise ValueError('Expected original 2461 DEV-only opportunities')
    end = pd.Timestamp(c['sources']['development_only_max_time_exclusive'])
    if (pd.to_datetime(opp.labelable_at, utc=True) >= end).any():
        raise PermissionError('DEV clock exceeds boundary')
    ordinary = pd.read_csv(paths[1],float_precision='round_trip').set_index('opportunity_id')
    labels = pd.read_csv(paths[2],float_precision='round_trip').set_index('opportunity_id')
    if ordinary.index.tolist() != ids.tolist() or labels.index.tolist() != ids.tolist():
        raise ValueError('DEV source identity/order mismatch')
    with np.load(paths[3], allow_pickle=False) as z:
        if z['ids'].astype(str).tolist() != ids.tolist():
            raise ValueError('Frozen cache IDs differ')
        hidden = z['features'].copy()
    if hidden.shape != (2461, 512) or hidden.dtype != np.float32 or not np.isfinite(hidden).all():
        raise ValueError('Invalid DEV hidden cache')
    hourly = inputs.load_hourly('A')
    times = pd.DatetimeIndex(hourly.bar_open_at)
    windows = []
    for _, row in opp.iterrows():
        market, stamp, _ = inputs._prepare_window(hourly, row, times)
        windows.append(np.concatenate((market, stamp / np.array([59,23,6,31,12], dtype=np.float32)), axis=1))
    windows = np.asarray(windows, dtype=np.float32)
    sample = [0, len(opp)//2, len(opp)-1]
    sampled = opp.iloc[sample]
    rebuilt = inputs.build_features(hourly, sampled, 'A')
    np.testing.assert_allclose(rebuilt.to_numpy(), ordinary.loc[sampled.opportunity_id, rebuilt.columns].to_numpy(), rtol=1e-12, atol=1e-12)
    future_checks = []
    for i in sample:
        row = opp.iloc[i]
        altered = hourly.copy()
        future = altered.bar_open_at >= pd.Timestamp(row.decision_boundary_at)
        altered.loc[future, inputs.FIELDS] *= 1.1
        m, s, _ = inputs._prepare_window(altered, row, times)
        np.testing.assert_array_equal(np.concatenate((m, s / np.array([59,23,6,31,12], dtype=np.float32)), axis=1), windows[i])
        f = inputs.build_features(altered, opp.iloc[[i]], 'A')
        np.testing.assert_array_equal(f.to_numpy(), rebuilt.loc[[row.opportunity_id]].to_numpy())
        future_checks.append(row.opportunity_id)
    five = inputs.load_development_5m().set_index('bar_open_at')
    for ident, row in labels.iterrows():
        entry = pd.Timestamp(row.entry_at)
        w = five.loc[pd.date_range(entry, periods=48, freq='5min')]
        prices = np.r_[float(w.iloc[0].open), w.close.to_numpy(dtype=np.float64)]
        np.testing.assert_allclose(prices, np.asarray(json.loads(row.label_prices_json)), rtol=0, atol=1e-9)
        rv = np.sum(np.diff(np.log(prices)) ** 2, dtype=np.float64)
        np.testing.assert_allclose(rv, row.RV_raw, rtol=1e-12, atol=1e-18)
    roles = fold_indices(opp, c)
    raw = ordinary.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    arrays = dict(ids=ids, windows=windows, ordinary_raw=raw, hidden=hidden,
                  har=ordinary.loc[:, HAR_NAMES].to_numpy(dtype=np.float64),
                  r0=ordinary.loc[:, R0_NAMES].to_numpy(dtype=np.float64))
    if any(not np.isfinite(v).all() for k,v in arrays.items() if k != 'ids'):
        raise ValueError('Nonfinite features')
    (run / 'inputs').mkdir(parents=True, exist_ok=True)
    with (run / 'inputs/features.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    frame(run / 'inputs/metadata.csv', opp)
    # Exact bytes retain the independent original future-label evidence.
    (run / 'labels').mkdir(exist_ok=True)
    with (run / 'labels/development.csv').open('xb') as handle:
        handle.write(paths[2].read_bytes())
    records, thresholds = [], {}
    for fold, rr in roles.items():
        for role, ix in rr.items():
            records.extend(dict(fold=fold, role=role, row=int(i), opportunity_id=ids[i]) for i in ix)
        y = np.maximum(labels.RV_raw.to_numpy(dtype=np.float64)[rr['fit']], 1e-12)
        ewma = ordinary.r0_ewma.to_numpy(dtype=np.float64)[rr['fit']]
        thresholds[fold] = {'RV_q90':float(np.quantile(y,.9)), 'RV_q99':float(np.quantile(y,.99)),
                            'EWMA_quartiles':np.quantile(ewma,[.25,.5,.75]).tolist(),
                            'constant_RV':float(np.clip(y.mean(),1e-12,1)),
                            'target_median':float(np.median(y)), 'fit_count':len(y)}
    frame(run / 'folds/index.csv', pd.DataFrame(records))
    write(run / 'folds/thresholds.json', thresholds)
    write(run / 'inputs/benchmark_ids.json', ids[roles['WF01']['fit'][:8]].tolist())
    files = {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    write(run / 'inputs/prepare_audit.json', {'status':'PASS','rows':2461,'oof_rows':1267,
          'fold_counts':{k:[len(v[r]) for r in ('fit','inner_validation','evaluation')] for k,v in roles.items()},
          'all_49_price_labels_verified':2461,'ordinary_rebuilt_ids':ids[sample].tolist(),
          'future_perturbation_ids':future_checks,'source_files':files,
          'protocol_sha256':sha(CONFIG),'no_Q2_Q3_value_reads':True,
          'feature_names':list(FEATURE_NAMES),'feature_schema':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}})
    return run


def load_features(run=DEFAULT_RUN):
    run, _, _ = new_scope(run)
    if read(run / 'inputs/prepare_audit.json')['status'] != 'PASS':
        raise ValueError('Passed preparation required')
    with np.load(run / 'inputs/features.npz', allow_pickle=False) as z:
        if set(z.files) != {'ids','windows','ordinary_raw','hidden','har','r0'}:
            raise ValueError('Prediction inputs contain forbidden or unknown arrays')
        arrays = {k:z[k].copy() for k in z.files}
    metadata = pd.read_csv(run / 'inputs/metadata.csv',float_precision='round_trip')
    if metadata.opportunity_id.tolist() != arrays['ids'].tolist():
        raise ValueError('Metadata order differs')
    return arrays, metadata
