"""Locked RV loss evaluation. No fitting or holdout access occurs here."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-12
CONFIG = Path(__file__).resolve().parents[2] / 'configs/frozen_risk_02_v1.yaml'
LOCKED_SHA = 'bc488c5bb9a3296340595e367c1c474d64f96c45e49c5249efc9f12a52d06425'
FAMILIES = ['persistence','ewma','har','R1','R2','random_s17','random_s29','random_s43']
CORE = ['R1','selected_non_kronos','selected_R0']
KEYS = ['opportunity_id','decision_boundary_at','entry_at','RV_raw','RV_effective']


def losses(y, prediction):
    y = np.maximum(np.asarray(y, dtype=float), EPS)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(prediction).all() or (prediction < EPS).any() or (prediction > 1).any():
        raise ValueError('Invalid target or unclipped prediction')
    ratio = y / prediction
    return np.log(prediction) + ratio, ratio - np.log(ratio) - 1, (np.log(y)-np.log(prediction))**2


def paired(left, right):
    left = left.sort_values('opportunity_id').reset_index(drop=True)
    right = right.sort_values('opportunity_id').reset_index(drop=True)
    if len(left) != len(right) or not left[KEYS].equals(right[KEYS]):
        raise ValueError('Pairing ID/time/target mismatch')
    return left, right


def select_controls(frame):
    """Validation-only selection, deterministic ties in locked family order."""
    copies, selectors = [], {}
    for fid, fold in frame.groupby('fold_id', sort=True):
        val = fold[fold.role == 'validation']
        scores = {f: float(val.loc[val.family == f, 'QLIKE'].mean()) for f in ['persistence','ewma','har','R1']}
        if not all(np.isfinite(list(scores.values()))):
            raise ValueError('Missing validation selection scores')
        selectors[fid] = {}
        for alias, candidates in [('selected_R0',['persistence','ewma','har']), ('selected_non_kronos',['persistence','ewma','har','R1'])]:
            chosen = min(candidates, key=lambda f: scores[f])
            selectors[fid][alias] = {'family': chosen, 'validation_QLIKE': scores[chosen]}
            copy = fold[fold.family == chosen].copy()
            copy['family'] = alias
            copies.append(copy)
    return pd.concat([frame]+copies, ignore_index=True), selectors


def binary_metrics(y, score):
    y = np.asarray(y, dtype=bool)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def rank_corr(a, b):
    # EWMA has identically zero predicted surprise; undefined correlation is explicit.
    return float(spearmanr(a, b).statistic) if np.ptp(a)>1e-14 and np.ptp(b)>1e-14 else np.nan


def residual_corr(a, b, control):
    design = np.column_stack([np.ones(len(control)), control])
    ar = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    br = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    return float(np.corrcoef(ar, br)[0, 1]) if np.std(ar)>1e-14 and np.std(br)>1e-14 else np.nan


def assign_bins(values, train, bins):
    edges = np.quantile(train, np.arange(1,bins)/bins)
    return np.searchsorted(edges, values, side='right'), edges


def day_arrays(left, right, fold_config, role='test'):
    """Retain full UTC calendar; sum/count handles only protocol edge purge."""
    left, right = paired(left, right)
    start, end = pd.to_datetime(fold_config[role], utc=True)
    times = pd.to_datetime(left.decision_boundary_at, utc=True)
    expected = pd.date_range(start, end, freq='8h', inclusive='left') + pd.Timedelta(hours=4)
    expected = expected[(expected >= start) & (expected + pd.Timedelta(hours=5, seconds=60) < end)]
    if not pd.DatetimeIndex(times).sort_values().equals(expected.sort_values()):
        raise ValueError(f'{fold_config["id"]}: incomplete decision-day opportunity schedule')
    days = pd.date_range(start, end, freq='D', inclusive='left')
    values = pd.DataFrame({'day':times.dt.floor('D'), 'gain':left.QLIKE.to_numpy()-right.QLIKE.to_numpy()})
    agg = values.groupby('day').gain.agg(['sum','count']).reindex(days, fill_value=0)
    return agg['sum'].to_numpy(), agg['count'].to_numpy()


def stationary_draws(sums, counts, block, draws, seed):
    """Circular stationary bootstrap; each draw samples N calendar days."""
    rng = np.random.default_rng(seed)
    n = len(sums)
    index = rng.integers(0,n,size=draws)
    total = sums[index].copy()
    count = counts[index].copy()
    for _ in range(1,n):
        restart = rng.random(draws) < 1.0/block
        index = np.where(restart, rng.integers(0,n,size=draws), (index+1)%n)
        total += sums[index]
        count += counts[index]
    if (count == 0).any():
        raise ValueError('Bootstrap sampled no valid opportunities')
    return total / count


def screen(comparison_rows, cfg):
    primary = comparison_rows[comparison_rows.block_days == cfg['bootstrap']['primary_block_days']]
    quarter = primary[primary.fold_id != 'pooled']
    pool = primary[primary.fold_id == 'pooled'].iloc[0]
    if len(quarter) != 4:
        raise ValueError('Screen requires exactly four quarters')
    if (quarter.QLIKE_gain <= 0).all() or pool.CI_upper <= 0:
        return 'FAIL'
    rule = cfg['promotion']
    if (quarter.QLIKE_gain > 0).sum() >= rule['positive_quarters_at_least'] and quarter.regret_improvement_percent.median() >= rule['median_relative_regret_improvement_percent_at_least'] and pool.CI_lower > rule['pooled_primary_CI_lower_gt']:
        return 'PASS'
    return 'INCONCLUSIVE'


def evaluate(run, config=CONFIG, engineering_status='pending'):
    run, config = Path(run), Path(config)
    if hashlib.sha256(config.read_bytes()).hexdigest() != LOCKED_SHA:
        raise ValueError('Locked configuration SHA mismatch')
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    frame = pd.read_csv(run / 'predictions.csv')
    required = set(['fold_id','role','family','prediction']+KEYS)
    if not required.issubset(frame.columns) or frame.duplicated(['fold_id','role','family','opportunity_id']).any():
        raise ValueError('Invalid prediction long table')
    for column in ['decision_boundary_at','entry_at']:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if (frame.entry_at >= pd.Timestamp(cfg['holdout']['start'])).any():
        raise ValueError('Holdout access forbidden')
    if not set(frame.role).issubset({'train','validation','test'}) or set(frame.fold_id) != {f['id'] for f in cfg['folds']}:
        raise ValueError('Unexpected fold or role')
    if not np.isfinite(frame.RV_raw).all() or (frame.RV_raw < 0).any() or not np.array_equal(frame.RV_effective, np.maximum(frame.RV_raw, EPS)):
        raise ValueError('Invalid raw/effective target contract')
    for fid, fold in frame.groupby('fold_id'):
        if set(fold.role) != {'train','validation','test'}:
            raise ValueError('Missing registered role')
        fold_config = next(f for f in cfg['folds'] if f['id'] == fid)
        for role, section in fold.groupby('role'):
            start, end = pd.to_datetime(fold_config[role], utc=True)
            boundary = section.decision_boundary_at
            entry = section.entry_at
            if ((boundary < start) | (boundary >= end) | (entry != boundary + pd.Timedelta(hours=1)) | (entry + pd.Timedelta(hours=4, seconds=60) >= end)).any():
                raise ValueError('Role interval or causal timing contract violated')
            if set(section.family)-set(CORE[1:]) != set(FAMILIES):
                raise ValueError(f'Missing registered family: {fid}/{role}')
            reference = section[section.family == 'R2']
            for _, model in section.groupby('family'):
                paired(model, reference)
    frame['QLIKE'], frame['Regret'], frame['logMSE'] = losses(frame.RV_effective, frame.prediction)
    base = frame[frame.family.isin(FAMILIES)].copy()
    calculated, selectors = select_controls(base)
    if set(CORE[1:]).issubset(set(frame.family)):
        for alias in CORE[1:]:
            left = frame[frame.family == alias].sort_values(['fold_id','role','opportunity_id'])
            right = calculated[calculated.family == alias].sort_values(['fold_id','role','opportunity_id'])
            if not np.array_equal(left.prediction, right.prediction):
                raise ValueError('Supplied selected control differs from validation selection')
    frame = calculated
    selector_path = run/'selector.json'
    if selector_path.exists():
        existing = json.loads(selector_path.read_text(encoding='utf-8'))
        if existing != selectors:
            raise ValueError('Existing selector differs from validation-only selector')
    else:
        selector_path.write_text(json.dumps(selectors,indent=2),encoding='utf-8')
    metrics, calibration, diagnostics, quartiles = [], [], [], []
    for fid, fold in frame.groupby('fold_id', sort=True):
        train_y = fold[(fold.role == 'train') & (fold.family == 'R2')].RV_effective
        q90, q99 = np.quantile(train_y,[.90,.99])
        ewma = {role: part[part.family == 'ewma'] for role,part in fold.groupby('role')}
        for family, model in fold.groupby('family', sort=True):
            train = model[model.role == 'train'].sort_values('opportunity_id')
            train, etrain = paired(train, ewma['train'])
            train_s = np.log((train.prediction.to_numpy()+EPS)/(etrain.prediction.to_numpy()+EPS))
            for role, section in model.groupby('role', sort=True):
                section, e = paired(section, ewma[role])
                auc, ap = binary_metrics(section.RV_effective > q90,section.prediction)
                metrics.append(dict(fold_id=fid,role=role,family=family,N=len(section),QLIKE=section.QLIKE.mean(),Regret=section.Regret.mean(),logMSE=section.logMSE.mean(),q90trainAUROC=auc,q90trainAP=ap,train_q90=q90,train_q99=q99,target_floor_count=int((section.RV_raw<EPS).sum()),prediction_floor_count=int((section.prediction==EPS).sum()),prediction_ceiling_count=int((section.prediction==1).sum())))
                bins, edges = assign_bins(section.prediction, train.prediction,10)
                for bin_id in range(10):
                    subset = section[bins == bin_id]
                    calibration.append(dict(fold_id=fid,role=role,family=family,decile=bin_id+1,N=len(subset),predicted_RV=subset.prediction.mean(),actual_RV=subset.RV_effective.mean(),training_boundaries=json.dumps(edges.tolist())))
                if role != 'test':
                    continue
                true_s = np.log((section.RV_raw.to_numpy()+EPS)/(e.prediction.to_numpy()+EPS))
                pred_s = np.log((section.prediction.to_numpy()+EPS)/(e.prediction.to_numpy()+EPS))
                sauc, sap = binary_metrics(true_s > cfg['evaluation']['surprise']['event_threshold_log_ratio'],pred_s)
                diagnostics.append(dict(fold_id=fid,family=family,N=len(section),surprise_MSE=np.mean((true_s-pred_s)**2),direction_accuracy=np.mean((true_s>0)==(pred_s>0)),surprise_AUROC=sauc,surprise_AP=sap,Spearman=rank_corr(true_s,pred_s),partial_correlation=residual_corr(true_s,pred_s,np.log(e.prediction.to_numpy())),extreme_N=int((section.RV_effective>q99).sum()),extreme_QLIKE=section.loc[section.RV_effective>q99,'QLIKE'].mean(),extreme_Regret=section.loc[section.RV_effective>q99,'Regret'].mean()))
                qbins, qedges = assign_bins(pred_s,train_s,4)
                for bin_id in range(4):
                    quartiles.append(dict(fold_id=fid,family=family,quartile=bin_id+1,N=int((qbins==bin_id).sum()),mean_true_S=float(np.mean(true_s[qbins==bin_id])) if (qbins==bin_id).any() else np.nan,training_boundaries=json.dumps(qedges.tolist())))
    comparisons, extremes, daily = [], [], []
    boot = cfg['evaluation']['bootstrap']
    for comparator_index, comparator in enumerate(CORE+['random_average_loss']):
        pool_draws = {block: [] for block in boot['mean_block_days']}
        fold_gains = []
        for fold_index, fcfg in enumerate(cfg['folds']):
            subset = frame[(frame.fold_id==fcfg['id']) & (frame.role=='test')]
            r2 = subset[subset.family=='R2'].sort_values('opportunity_id').reset_index(drop=True)
            if comparator == 'random_average_loss':
                control = r2.copy()
                seed_models = [paired(subset[subset.family==seed],r2)[0] for seed in ['random_s17','random_s29','random_s43']]
                for loss in ['QLIKE','Regret','logMSE']:
                    control[loss] = np.mean([s[loss].to_numpy() for s in seed_models],axis=0)
            else:
                control, r2 = paired(subset[subset.family==comparator],r2)
            sums, counts = day_arrays(control,r2,fcfg)
            gain = float(control.QLIKE.mean()-r2.QLIKE.mean())
            improvement = float(100*(control.Regret.mean()-r2.Regret.mean())/control.Regret.mean()) if control.Regret.mean()>0 else np.nan
            fold_gains.append(gain)
            train = frame[(frame.fold_id==fcfg['id']) & (frame.role=='train') & (frame.family=='R2')]
            tail = r2.RV_effective > train.RV_effective.quantile(.99)
            tail_gain = float((control.QLIKE-r2.QLIKE)[tail].sum())
            total_gain = float((control.QLIKE-r2.QLIKE).sum())
            extremes.append(dict(fold_id=fcfg['id'],comparator=comparator,N=len(r2),extreme_N=int(tail.sum()),total_QLIKE_gain_sum=total_gain,extreme_QLIKE_gain_sum=tail_gain,extreme_gain_fraction=tail_gain/total_gain if total_gain!=0 else np.nan,non_extreme_mean_gain=float((control.QLIKE-r2.QLIKE)[~tail].mean())))
            for day, value, count in zip(pd.date_range(*pd.to_datetime(fcfg['test'],utc=True),freq='D',inclusive='left'),sums,counts):
                daily.append(dict(fold_id=fcfg['id'],comparator=comparator,day=day,QLIKE_gain_sum=value,N=count))
            for block in boot['mean_block_days']:
                seed = boot['seed']+comparator_index*100000+block*1000+fold_index
                draws = stationary_draws(sums,counts,block,boot['draws'],seed)
                pool_draws[block].append(draws)
                low, high = np.quantile(draws,[.025,.975])
                comparisons.append(dict(comparator=comparator,fold_id=fcfg['id'],block_days=block,N=len(r2),QLIKE_gain=gain,regret_improvement_percent=improvement,positive_direction=gain>0,CI_lower=low,CI_upper=high,seed=seed))
        for block in boot['mean_block_days']:
            draws = np.mean(pool_draws[block],axis=0)
            low, high = np.quantile(draws,[.025,.975])
            comparisons.append(dict(comparator=comparator,fold_id='pooled',block_days=block,N=np.nan,QLIKE_gain=np.mean(fold_gains),regret_improvement_percent=np.nan,positive_direction=np.mean(fold_gains)>0,CI_lower=low,CI_upper=high,seed=np.nan))
    comparisons = pd.DataFrame(comparisons)
    statuses = {c:screen(comparisons[comparisons.comparator==c],cfg['evaluation']) for c in CORE+['random_average_loss']}
    main_status = 'FAIL' if 'FAIL' in [statuses[c] for c in CORE] else ('PASS' if all(statuses[c]=='PASS' for c in CORE) else 'INCONCLUSIVE')
    pretraining = {'PASS':'supports','FAIL':'not_supports','INCONCLUSIVE':'insufficient'}[statuses['random_average_loss']]
    summary = dict(engineering_status=engineering_status,main_status=main_status,pretraining_evidence=pretraining,research_recommendation='independent_validation_required' if main_status=='PASS' else ('stop_current_route' if main_status=='FAIL' else 'stop_finite_budget_evidence_uncertain'),comparators=statuses,development_status=cfg['development_status'],holdout_sealed=True,primary_block_days=7,bootstrap_draws=boot['draws'],random_control='mean of three per-opportunity losses, not predictions')
    for name, result in [('metrics',metrics),('calibration',calibration),('surprise_diagnostics',diagnostics),('surprise_quartiles',quartiles),('extreme_gain_concentration',extremes),('paired_daily_gains',daily)]:
        pd.DataFrame(result).to_csv(run/f'{name}.csv',index=False)
    comparisons.to_csv(run/'paired_comparisons.csv',index=False)
    frame.to_csv(run/'evaluated_predictions.csv',index=False)
    (run/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--config',default=str(CONFIG))
    parser.add_argument('--engineering-status',default='pending')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run,args.config,args.engineering_status),indent=2))

if __name__ == '__main__':
    main()

