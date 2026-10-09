"""Locked RISK_03 statistics. This module has no source-data reader or fitting path.

``evaluate_quarters(frame, predictions, output_dir, config=None)`` consumes aligned
arrays supplied by the authorized caller. Training thresholds are optional sealed
inputs, never estimated here. Default configuration is the locked YAML protocol.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'frozen_risk_03_v1.yaml'
COMPARISONS = {'R2_minus_R1': ('R2', 'R1'), 'R2_minus_B2': ('R2', 'B2')}


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))


def weighted_ranking(y, score, weights=None):
    """Exact duplicate-sample AUROC (half credit ties), and tied-threshold AP."""
    y, score = np.asarray(y, dtype=bool), np.asarray(score, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or w.shape != y.shape:
        raise ValueError('ranking arrays must be aligned one-dimensional arrays')
    if not np.isfinite(score).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('scores and nonnegative weights must be finite')
    positive, negative = w[y].sum(), w[~y].sum()
    if positive == 0 or negative == 0:
        return {'AUROC': float('nan'), 'AP': float('nan')}
    order = np.argsort(score, kind='stable')
    s, ys, ws = score[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    pos = np.add.reduceat(ws * ys, starts)
    neg = np.add.reduceat(ws * ~ys, starts)
    auc = np.sum(pos * (np.cumsum(neg) - .5 * neg)) / (positive * negative)
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    ap = np.sum(pos[::-1] / positive * np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0))
    return {'AUROC': float(auc), 'AP': float(ap)}


def stationary_day_counts(n_days, mean_block_days, draws=5000, seed=20261008):
    """Circular stationary bootstrap with geometric runs over ALL calendar days."""
    if n_days < 1 or mean_block_days < 1 or draws < 1:
        raise ValueError('positive day, block and draw counts required')
    rng = np.random.default_rng(seed)
    indices = np.empty((draws, n_days), dtype=np.int32)
    indices[:, 0] = rng.integers(n_days, size=draws)
    for t in range(1, n_days):
        restart = rng.random(draws) < 1. / mean_block_days
        fresh = rng.integers(n_days, size=draws)
        indices[:, t] = np.where(restart, fresh, (indices[:, t - 1] + 1) % n_days)
    counts = np.zeros((draws, n_days), dtype=np.uint16)
    np.add.at(counts, (np.arange(draws)[:, None], indices), 1)
    return counts


def equal_quarter_differences(quarter_differences):
    """Ordinary mean deliberately propagates an invalid single-class quarter."""
    return np.mean(np.asarray(quarter_differences, dtype=float), axis=0)


def classify_status(comparisons, adequate_events=True, invalid_fraction=0.,
                    risk_magnitude_red_flag=False, invalidated=False,
                    effect_minimum=.03, max_invalid_fraction=.01):
    """Apply the registered priority, with each co-primary passing separately."""
    if invalidated:
        return 'INVALIDATED'
    if not adequate_events or invalid_fraction > max_invalid_fraction:
        return 'INCONCLUSIVE'
    def passed(c):
        return (c['mean_delta'] >= effect_minimum and c['ci_lower'] > 0
                and all(v > 0 for v in c['quarter_deltas']))
    r1, b2 = comparisons['R2_minus_R1'], comparisons['R2_minus_B2']
    if passed(r1) and passed(b2):
        return 'RANKING_ONLY' if risk_magnitude_red_flag else 'CONFIRMED_RANKING_INCREMENT'
    if passed(r1) and not passed(b2):
        return 'NONLINEAR_BASELINE_COMPETITIVE'
    if any(c['ci_upper'] <= 0 for c in comparisons.values()):
        return 'NOT_SUPPORTED'
    if all(c['ci_lower'] > 0 for c in comparisons.values()) and any(c['mean_delta'] < effect_minimum for c in comparisons.values()):
        return 'NOT_SUPPORTED'
    return 'INCONCLUSIVE'


def variance_losses(rv_raw, prediction, epsilon=1e-12):
    y, p = np.maximum(np.asarray(rv_raw, dtype=float), epsilon), np.asarray(prediction, dtype=float)
    ratio = y / p
    return {'raw_QLIKE': np.log(p) + ratio,
            'QLIKE_Regret': ratio - np.log(ratio) - 1.,
            'logRV_MSE': (np.log(y) - np.log(p)) ** 2,
            'additive_epsilon_logRV_MSE': (np.log(np.asarray(rv_raw) + epsilon) - np.log(p + epsilon)) ** 2}


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def evaluate_quarters(frame, predictions, output_dir, config=None, *,
                      training_thresholds=None, invalidated=False):
    """Write all attempted draws, paired day counts and JSON; never refit thresholds.

    Frame requires ``decision_at,RV_raw`` and contains only the two test quarters.
    ``predictions`` maps family names to aligned positive absolute RV arrays;
    R1, R2, B2 and ewma are mandatory. All models receive identical day weights.
    ``training_thresholds`` may supply ``high_RV`` (sealed train q90).
    Existing evidence files are never overwritten. Returned summary uses NaN in
    memory and JSON null for undefined statistics. Supplied config is intended
    for synthetic engineering checks or the caller's already-sealed protocol.
    """
    cfg = load_config() if config is None else config
    eps = float(cfg['label']['epsilon'])
    quarters = cfg['roles']['test_quarters']
    if len(quarters) != 2:
        raise ValueError('protocol requires exactly two quarters')
    dates = pd.to_datetime(frame['decision_at'], utc=True, errors='raise')
    raw = np.asarray(frame['RV_raw'], dtype=float)
    if len(raw) == 0 or not np.isfinite(raw).all() or (raw < 0).any() or dates.isna().any() or dates.duplicated().any():
        raise ValueError('RV and unique timestamps must be valid')
    if not {'R1', 'R2', 'B2', 'ewma'}.issubset(predictions):
        raise ValueError('missing mandatory family')
    pred, clip_counts = {}, {}
    lo, hi = float(cfg['head']['prediction_absolute_variance_floor']), float(cfg['head']['prediction_absolute_variance_ceiling'])
    for name, values in predictions.items():
        p = np.asarray(values, dtype=float)
        if p.shape != raw.shape or not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError(f'invalid aligned positive prediction: {name}')
        clip_counts[name] = {'below': int((p < lo).sum()), 'above': int((p > hi).sum())}
        pred[name] = np.clip(p, lo, hi)
    event = np.log((raw + eps) / (pred['ewma'] + eps)) > np.log(2.)
    scores = {name: np.log((p + eps) / (pred['ewma'] + eps)) for name, p in pred.items()}
    masks, day_ids, calendars, qmetrics = [], [], [], []
    covered = np.zeros(len(raw), dtype=np.int8)
    for q in quarters:
        start, end = pd.Timestamp(q['start']), pd.Timestamp(q['end_exclusive'])
        if start.tzinfo is None or end.tzinfo is None or start != start.normalize() or end != end.normalize():
            raise ValueError('quarter boundaries must be timezone-aware UTC calendar boundaries')
        start, end = start.tz_convert('UTC'), end.tz_convert('UTC')
        mask = np.asarray((dates >= start) & (dates < end))
        covered += mask
        for field in ('label_end', 'labelable_at'):
            if field in frame and (pd.to_datetime(frame.loc[mask, field], utc=True) >= end).any():
                raise ValueError('unpurged quarter label boundary')
        calendar = pd.date_range(start, end, freq='D', inclusive='left')
        ids = np.asarray((dates[mask].dt.normalize() - start).dt.days, dtype=int)
        masks.append(mask); day_ids.append(ids); calendars.append(calendar)
        y = event[mask]
        qmetrics.append({'id': q['id'], 'opportunities': int(mask.sum()), 'positive': int(y.sum()),
                         'negative': int((~y).sum()), 'calendar_days': len(calendar),
                         'models': {name: weighted_ranking(y, s[mask]) for name, s in scores.items()}})
    if not np.all(covered == 1):
        raise ValueError('frame must contain only nonoverlapping registered quarters')
    boot = cfg['bootstrap']
    draws = int(boot['iterations'])
    blocks = [int(boot['primary_mean_block_days'])] + list(map(int, boot['sensitivity_block_days']))
    out = Path(output_dir)
    files = ['summary.json'] + [f'{kind}_block{b}.{extension}' for b in blocks for kind, extension in [('draws','csv'), ('multiplicities','npz')]]
    if any((out / f).exists() for f in files):
        raise FileExistsError('immutable statistical evidence already exists')
    out.mkdir(parents=True, exist_ok=True)
    minimum = int(cfg['primary']['minimum_positive_and_negative_events_per_quarter'])
    adequate = all(q['positive'] >= minimum and q['negative'] >= minimum for q in qmetrics)
    losses = {name: {metric: float(np.mean(v)) for metric, v in variance_losses(raw, p, eps).items()} for name, p in pred.items()}
    red_flag = losses['R2']['QLIKE_Regret'] > 1.10 * losses['R1']['QLIKE_Regret']
    summary = {'quarter_metrics': qmetrics, 'variance_losses': losses, 'risk_magnitude_red_flag': red_flag,
               'adequate_events': adequate, 'prediction_clipping': clip_counts, 'bootstrap': {},
               'statistic': 'equal mean of within-quarter AUROC differences', 'thresholds_refit': False}
    if training_thresholds is not None:
        threshold = float(training_thresholds['high_RV'])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('sealed training high_RV threshold must be positive')
        high = np.maximum(raw, eps) > threshold
        summary['high_RV'] = {'sealed_threshold': threshold, 'count': int(high.sum()),
                             'underestimate_fraction': {name: float(np.mean(p[high] / np.maximum(raw[high], eps) < .5)) if high.any() else float('nan') for name, p in pred.items()}}
    for block in blocks:
        table = {'draw': np.arange(draws)}
        archive, deltas, quarter_valid = {}, {}, []
        for qi, (q, mask, ids, calendar) in enumerate(zip(quarters, masks, day_ids, calendars)):
            seed = int(boot['seed']) + block * 1000 + qi
            counts = stationary_day_counts(len(calendar), block, draws, seed)
            archive[f"{q['id']}_counts"] = counts
            archive[f"{q['id']}_utc_days"] = calendar.strftime('%Y-%m-%d').to_numpy(dtype='U10')
            archive[f"{q['id']}_seed"] = np.asarray(seed)
            for name, score in scores.items():
                metrics = [weighted_ranking(event[mask], score[mask], counts[d, ids]) for d in range(draws)]
                for metric in ('AUROC', 'AP'):
                    table[f"{q['id']}_{name}_{metric}"] = np.array([m[metric] for m in metrics])
            valid = np.isfinite(table[f"{q['id']}_R2_AUROC"])
            quarter_valid.append(valid)
            for key, (a, b) in COMPARISONS.items():
                delta = table[f"{q['id']}_{a}_AUROC"] - table[f"{q['id']}_{b}_AUROC"]
                table[f"{q['id']}_{key}"] = delta
                deltas.setdefault(key, []).append(delta)
        joint = np.logical_and.reduce(quarter_valid)
        table['joint_valid'] = joint
        alpha = (1 - float(boot['confidence'])) / 2
        comparisons = {}
        for key, (a, b) in COMPARISONS.items():
            values = equal_quarter_differences(deltas[key])
            table[f'equal_quarter_{key}'] = values
            ci = np.quantile(values[joint], [alpha, 1 - alpha]) if joint.any() else [float('nan')] * 2
            point = [q['models'][a]['AUROC'] - q['models'][b]['AUROC'] for q in qmetrics]
            comparisons[key] = {'quarter_deltas': point, 'mean_delta': float(np.mean(point)), 'ci_lower': float(ci[0]), 'ci_upper': float(ci[1])}
        invalid_fractions = {q['id']: float(np.mean(~valid)) for q, valid in zip(quarters, quarter_valid)}
        invalid_fractions['equal_quarter_joint'] = float(np.mean(~joint))
        worst = max(invalid_fractions.values())
        status = classify_status(comparisons, adequate, worst, red_flag, invalidated,
                                 float(cfg['primary']['effect_minimum_absolute_AUROC']), float(boot['maximum_invalid_fraction']))
        summary['bootstrap'][str(block)] = {'attempted_draws': draws, 'joint_valid_draws': int(joint.sum()),
                                            'invalid_fractions': invalid_fractions, 'comparisons': comparisons, 'status': status}
        pd.DataFrame(table).to_csv(out / f'draws_block{block}.csv', index=False)
        np.savez_compressed(out / f'multiplicities_block{block}.npz', **archive)
    summary['status'] = summary['bootstrap'][str(boot['primary_mean_block_days'])]['status']
    (out / 'summary.json').write_text(json.dumps(_json_clean(summary), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary
