"""Descriptive publication supplement from generated CONSUMED artifacts only.

No fitting, new thresholds/bins, market inputs or primary result changes. The
saved bootstrap date multiplicities are reused without generating random draws.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/runs/FROZEN_RISK_03_v1'
OUTPUT = RUN / 'metrics/publication_auxiliary'
FAMILIES = ('persistence', 'ewma', 'har', 'R1', 'R2', 'B2', 'random_s17', 'random_s29', 'random_s43')
LOSSES = ('raw_QLIKE', 'QLIKE_Regret', 'logRV_MSE', 'additive_epsilon_logRV_MSE', 'surprise_MSE')
EPS = 1e-12
CONFIG_SHA = '7ed8affd7cd624cde89d23d29aaca7ef2ab660167fac7d3dc25fac274e394b7f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def weighted_ranking(y, scores, weights):
    """Independent exact weighted tie-group AUROC and average precision.

    weights: [draw, opportunity]. Single-class/empty draws remain NaN.
    AP sums recall increments times precision at each descending distinct score.
    """
    y, scores = np.asarray(y, bool), np.asarray(scores, float)
    weights = np.atleast_2d(np.asarray(weights, float))
    if weights.shape[1] != len(y) or scores.shape != y.shape or not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Aligned finite scores/nonnegative weights required')
    order = np.argsort(-scores, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order]) != 0) + 1]
    wp = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    wn = np.add.reduceat(weights[:, order] * ~y[order], starts, axis=1)
    p, n = wp.sum(axis=1), wn.sum(axis=1)
    cp, cn = wp.cumsum(axis=1), wn.cumsum(axis=1)
    valid = (p > 0) & (n > 0)
    auc = np.divide((wp * (n[:, None] - cn + .5 * wn)).sum(axis=1), p * n, out=np.full(len(p), np.nan), where=valid)
    precision = np.divide(cp, cp + cn, out=np.zeros_like(cp), where=(cp + cn) > 0)
    ap = np.divide((wp * precision).sum(axis=1), p, out=np.full(len(p), np.nan), where=valid)
    return {'AUROC': auc, 'AP': ap}


def partial_corr(a, b, control):
    # Same residual Pearson definition as experiment_02/evaluate.py residual_corr.
    design = np.column_stack([np.ones(len(control)), control])
    ar = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    br = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    return float(np.corrcoef(ar, br)[0, 1]) if np.std(ar) > 1e-14 and np.std(br) > 1e-14 else np.nan


def run():
    sources = {}

    def source(relative):
        path = RUN / relative
        sources[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
        return path

    def jread(relative):
        return json.loads(source(relative).read_text(encoding='utf-8-sig'))

    terminal = jread('authorization/formal_terminal.json')
    if terminal.get('state') != 'CONSUMED' or terminal.get('evidence', {}).get('numeric_review') != 'PASS':
        raise PermissionError('Publication supplement requires CONSUMED and independent numeric PASS')
    if OUTPUT.exists():
        raise FileExistsError('Preserve immutable publication supplement; output already exists')
    formal = jread('metrics/formal_result.json')
    primary_status = formal['status']
    seal = jread('calibration/sealed_calibration.json')
    seal_copy = dict(seal); digest = seal_copy.pop('seal_sha256')
    canonical = json.dumps(seal_copy, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != digest or seal['protocol_sha256'] != CONFIG_SHA or seal['epsilon'] != EPS:
        raise ValueError('Training calibration seal changed')
    # Bind all existing registered descriptive tables; never recalculate their bins.
    for filename in ('period_losses.csv', 'model_calibration.csv', 'high_extreme_RV_underprediction.csv', 'historical_EWMA_quartile_losses.csv', 'manifest.json'):
        source('diagnostics/holdout/' + filename)
    rows = pd.read_csv(source('diagnostics/holdout/row_metrics_and_timeline.csv'), float_precision='round_trip')
    rows['decision_at'] = pd.to_datetime(rows.decision_at, utc=True)
    if set(rows.family) != set(FAMILIES) or set(rows.quarter) != {'2026Q2', '2026Q3'} or rows.duplicated(['decision_at', 'family']).any():
        raise ValueError('Expected all nine aligned families and both independent quarters')
    if ((rows.decision_at < pd.Timestamp('2026-04-01', tz='UTC')) | (rows.decision_at >= pd.Timestamp('2026-10-01', tz='UTC'))).any():
        raise ValueError('Outside formal scope')
    q90 = seal['thresholds']['train_effective_RV_q90']
    q99 = seal['thresholds']['train_effective_RV_q99']
    point, surprise, extreme, gains, weekly = [], [], [], [], []
    aligned = {}
    for quarter, section in rows.groupby('quarter', sort=True):
        groups = {family: section[section.family == family].sort_values('decision_at').reset_index(drop=True) for family in FAMILIES}
        reference = groups['R2']
        for family, group in groups.items():
            for key in ('decision_at', 'RV_raw', 'RV_effective', 'EWMA_RV'):
                if not np.array_equal(reference[key].to_numpy(), group[key].to_numpy()):
                    raise ValueError('Family alignment/reference mismatch: ' + key)
            if not np.isfinite(group[list(LOSSES) + ['prediction_RV', 'observed_surprise', 'surprise_score']]).all().all():
                raise ValueError('Nonfinite generated values')
            y = group.RV_effective.to_numpy() > q90
            metrics = weighted_ranking(y, group.prediction_RV.to_numpy(), np.ones((1, len(group))))
            point.append({'quarter': quarter, 'family': family, 'N': len(group), 'event_count': int(y.sum()), 'event_fraction': float(y.mean()), 'AUROC': metrics['AUROC'][0], 'AP': metrics['AP'][0], 'threshold_effective_RV': q90})
            a, b = group.observed_surprise.to_numpy(), group.surprise_score.to_numpy()
            surprise.append({'quarter': quarter, 'family': family, 'N': len(group), 'Spearman': float(spearmanr(a, b).statistic) if np.ptp(a) > 1e-14 and np.ptp(b) > 1e-14 else np.nan, 'partial_correlation': partial_corr(a, b, np.log(group.EWMA_RV.to_numpy() + EPS))})
            high = group.RV_effective.to_numpy() > q99
            for loss in LOSSES:
                total = float(group[loss].sum()); selected = float(group.loc[high, loss].sum())
                extreme.append({'quarter': quarter, 'family': family, 'loss': loss, 'N': len(group), 'extreme_N': int(high.sum()), 'quarter_total_loss_sum': total, 'train_q99_extreme_loss_sum': selected, 'extreme_loss_share': selected / total if total != 0 else np.nan, 'train_q99_threshold': q99})
            group = group.copy()
            group['week_start_utc'] = group.decision_at.dt.floor('D') - pd.to_timedelta(group.decision_at.dt.weekday, unit='D')
            for week, sub in group.groupby('week_start_utc', sort=True):
                weekly.append({'quarter': quarter, 'family': family, 'week_start_utc': week, 'N': len(sub), 'first_decision_at': sub.decision_at.min(), 'last_decision_at': sub.decision_at.max(), 'mean_actual_RV_raw': float(sub.RV_raw.mean()), 'mean_actual_RV_effective': float(sub.RV_effective.mean()), 'mean_prediction_RV': float(sub.prediction_RV.mean())})
        for comparator in ('R1', 'B2'):
            high = reference.RV_effective.to_numpy() > q99
            for loss in ('raw_QLIKE', 'QLIKE_Regret'):
                values = groups[comparator][loss].to_numpy() - reference[loss].to_numpy()
                total, tail = float(values.sum()), float(values[high].sum())
                gains.append({'quarter': quarter, 'comparator': comparator, 'contrast': comparator + '_minus_R2_loss', 'loss': loss, 'N': len(values), 'extreme_N': int(high.sum()), 'total_gain_sum': total, 'extreme_gain_sum': tail, 'extreme_gain_share': tail / total if total != 0 else np.nan, 'non_extreme_mean_gain': float(values[~high].mean()) if (~high).any() else np.nan})
        aligned[quarter] = groups
    points = pd.DataFrame(point)
    paired_points = []
    for quarter, section in points.groupby('quarter'):
        values = section.set_index('family')
        for comparator in ('R1', 'B2'):
            paired_points.append({'quarter': quarter, 'comparator': comparator, **{metric + '_R2_minus_comparator': values.loc['R2', metric] - values.loc[comparator, metric] for metric in ('AUROC', 'AP')}})
    draw_tables, intervals = {}, []
    for block in (7, 3, 14):
        table = {}
        with np.load(source(f'bootstrap/multiplicities_block{block}.npz'), allow_pickle=False) as archive:
            for quarter, groups in aligned.items():
                qid = 'TEST_' + quarter[-2:]
                dates = pd.to_datetime(archive[qid + '_utc_days'], utc=True)
                reference = groups['R2']
                ids = dates.get_indexer(reference.decision_at.dt.floor('D'))
                if (ids < 0).any() or dates[0] != pd.Timestamp('2026-04-01' if quarter == '2026Q2' else '2026-07-01', tz='UTC'):
                    raise ValueError('Saved bootstrap calendar mismatch')
                counts = archive[qid + '_counts']
                if len(counts) != 5000 or counts.shape[1] != len(dates) or not (counts.sum(axis=1) == len(dates)).all():
                    raise ValueError('Unexpected saved bootstrap multiplicities')
                weights = counts[:, ids].astype(float)
                event = reference.RV_effective.to_numpy() > q90
                for family, group in groups.items():
                    metric = weighted_ranking(event, group.prediction_RV.to_numpy(), weights)
                    for name, value in metric.items():
                        table[f'{quarter}_{family}_{name}'] = value
                for comparator in ('R1', 'B2'):
                    for metric in ('AUROC', 'AP'):
                        table[f'{quarter}_R2_minus_{comparator}_{metric}'] = table[f'{quarter}_R2_{metric}'] - table[f'{quarter}_{comparator}_{metric}']
        for comparator in ('R1', 'B2'):
            for metric in ('AUROC', 'AP'):
                key = f'R2_minus_{comparator}_{metric}'
                table['equal_quarter_' + key] = (table['2026Q2_' + key] + table['2026Q3_' + key]) / 2
                for period in ('2026Q2', '2026Q3', 'equal_quarter'):
                    values = table[period + '_' + key]; valid = np.isfinite(values)
                    ci = np.quantile(values[valid], [.025, .975]) if valid.any() else [np.nan, np.nan]
                    intervals.append({'period': period, 'block_days': block, 'comparator': comparator, 'metric': metric, 'attempted_draws': len(values), 'valid_draws': int(valid.sum()), 'invalid_fraction': float((~valid).mean()), 'ci_lower': ci[0], 'ci_upper': ci[1], 'diagnostic_only': True})
        table['draw'] = np.arange(5000)
        draw_tables[block] = pd.DataFrame(table)
    # Finish computation before exclusively creating the immutable destination.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    outputs = {'high_rv_ranking.csv': points.rename(columns={'quarter':'period','N':'count','event_count':'positive'}), 'high_rv_paired_points.csv': pd.DataFrame(paired_points).rename(columns={'quarter':'period'}), 'surprise_correlations.csv': pd.DataFrame(surprise).rename(columns={'quarter':'period','N':'count','Spearman':'spearman'}), 'weekly_rv.csv': pd.DataFrame(weekly).rename(columns={'quarter':'period','N':'count','week_start_utc':'week_start','mean_actual_RV_raw':'observed_RV','mean_actual_RV_effective':'observed_RV_effective','mean_prediction_RV':'prediction_RV'}), 'loss_concentration.csv': pd.DataFrame(extreme).assign(threshold_group='TRAIN_q99').rename(columns={'quarter':'period','N':'count','train_q99_extreme_loss_sum':'extreme_loss_sum','extreme_loss_share':'loss_sum_fraction'}), 'extreme_paired_QLIKE_contribution.csv': pd.DataFrame(gains), 'high_rv_paired_intervals.csv': pd.DataFrame(intervals)}
    outputs.update({f'high_rv_draws_block{block}.csv': frame for block, frame in draw_tables.items()})
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / name, index=False)
    sources[str(Path(__file__).relative_to(ROOT)).replace('\\', '/')] = sha(__file__)
    manifest = {'schema_version': 1, 'experiment_id': 'FROZEN_RISK_03_v1', 'status': 'COMPLETE', 'diagnostic_only': True, 'formal_primary_result_unchanged': primary_status, 'protocol_sha256': CONFIG_SHA, 'families': list(FAMILIES), 'input_sha256': sources, 'artifact_sha256': {p.name: sha(p) for p in sorted(OUTPUT.iterdir())}, 'methods': {'high_RV': 'RV_effective > sealed TRAIN q90; absolute predicted RV score; ties averaged; weighted AP distinct-score recall increments', 'bootstrap': 'Reuse existing full UTC day multiplicities, blocks7/3/14, exactly5000 attempted draws; no redraw; both quarters must be finite for equal-quarter contrast; percentile95% diagnostic CI', 'partial_correlation': 'Pearson correlation of residuals after separate OLS of true/predicted surprise on intercept and log(EWMA_RV+eps); same residual definition as experiment_02/evaluate.py; +eps control per third publication scope', 'Spearman': 'Average-rank Spearman; constant surprise undefined, serialized CSV empty', 'extreme': 'TRAIN q99 only; extreme loss/gain sum divided by signed full-quarter sum; ratios may be negative or exceed1; zero denominator undefined; no post-test top1% threshold', 'weekly': 'Monday00:00 UTC week_start; each quarter grouped separately, boundary weeks partial; means are opportunity-weighted'}, 'unit': 'RV=squared_log_return_4h_not_annualized; QLIKE and correlations dimensionless; no revenue/cost interpretation', 'epsilon': EPS, 'threshold_and_bins_sha256': seal['threshold_and_bins_sha256'], 'thresholds_refit': False, 'new_fit_or_random_draws': False}
    with (OUTPUT / 'manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'status': 'COMPLETE', 'output': str(OUTPUT), 'formal_primary_result_unchanged': primary_status}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-generated-only', action='store_true', required=True)
    parser.parse_args()
    run()
