"""Finite development fusion and paired diagnostics; never fit base models or read consumed tests."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from research.frozen.experiment_03.statistics import stationary_day_counts

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/runs/M2_CONDITIONAL_INCREMENT_01'
CONFIG = ROOT / 'research/m2_ci/config.yaml'
BASE = ['B5', 'B5_s17', 'B5_s29', 'B5_s43', 'R1', 'R2', 'B2', 'har', 'ewma', 'persistence', 'constant_RV']
EPS = 1e-12

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)): return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v

def write(p, v):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(clean(v), f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

def save(p, frame):
    if Path(p).exists(): raise FileExistsError(str(p))
    frame.to_csv(p, index=False, float_format='%.17g')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def csv(p): return pd.read_csv(p, float_precision='round_trip')

def config():
    r = read(ROOT / 'research/registry/M2_CONDITIONAL_INCREMENT_01.json')
    if sha(CONFIG) != r['protocol_sha256']: raise ValueError('Protocol changed')
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))

def check_source_seal():
    s = read(RUN / 'preflight/source_seal.json')
    if s['status'] != 'PASS' or s['protocol_sha256'] != sha(CONFIG): raise ValueError('Invalid science seal')
    for path, digest in s['files_sha256'].items():
        if sha(ROOT / path) != digest: raise ValueError('Sealed source changed: ' + path)
    return s

def losses(y, p):
    y = np.maximum(np.asarray(y, float), EPS); p = np.asarray(p, float)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all() or (p <= 0).any(): raise ValueError('Invalid aligned arrays')
    ratio = y / p
    return {'raw_QLIKE': np.log(p) + ratio, 'QLIKE_Regret': ratio - np.log(ratio) - 1, 'logRV_MSE': np.log(y / p) ** 2}

def ranking(event, score):
    if len(np.unique(event)) < 2: return {'AUROC': np.nan, 'AP': np.nan}
    return {'AUROC': float(roc_auc_score(event, score)), 'AP': float(average_precision_score(event, score))}

def load_tables(filename):
    p = csv(RUN / 'predictions' / filename)
    y = csv(RUN / 'labels/development.csv').set_index('opportunity_id', drop=False)
    if p.opportunity_id.duplicated().any() or not set(BASE).issubset(p.columns): raise ValueError('Prediction families/identity invalid')
    times = pd.to_datetime(p.decision_at, utc=True)
    if times.max() >= pd.Timestamp('2026-04-01', tz='UTC'): raise PermissionError('Development prediction exceeds scope')
    y = y.loc[p.opportunity_id].copy()
    if y.opportunity_id.tolist() != p.opportunity_id.tolist(): raise ValueError('Prediction label identity mismatch')
    if not np.isfinite(p[BASE].to_numpy()).all() or (p[BASE].to_numpy() <= 0).any(): raise ValueError('Invalid positive RV forecasts')
    p['decision_at'] = times
    return p, y

def choose_weights():
    """Selection never opens original March validation forecasts/labels."""
    c = config(); check_source_seal()
    out = RUN / 'fusion/selected_weights.json'
    if out.exists(): raise FileExistsError('Fusion selection already frozen')
    p, y = load_tables('oof_predictions.csv')
    if len(p) != 1267 or set(p.fold_id) != {x['id'] for x in c['roles']['folds']}: raise ValueError('Unexpected OOF coverage')
    target = y.RV_raw.to_numpy(); b5 = p.B5.to_numpy(); base_losses = losses(target, b5)
    rows, choices = [], {}
    for other in c['fusion']['comparators']:
        trials = []
        for alpha in c['fusion']['alpha_candidates']:
            pred = (1-alpha)*b5 + alpha*p[other].to_numpy()
            loss = losses(target, pred)
            row = {'comparator': other, 'alpha': alpha, 'N': len(p), **{k: float(v.mean()) for k, v in loss.items()}, 'gain_vs_B5': float((base_losses['raw_QLIKE']-loss['raw_QLIKE']).mean()), 'relative_Regret_improvement': float(1-loss['QLIKE_Regret'].mean()/base_losses['QLIKE_Regret'].mean())}
            rows.append(row); trials.append(row)
        choices[other] = min(trials, key=lambda v: (v['raw_QLIKE'], v['alpha']))
    save(RUN / 'fusion/all_weight_candidates.csv', pd.DataFrame(rows))
    write(out, {'status': 'FROZEN_DEVELOPMENT_SELECTION', 'selected': choices, 'all_candidates': rows, 'selection': c['fusion']['selection'], 'tie_break': c['fusion']['tie_break'], 'protocol_sha256': sha(CONFIG), 'oof_predictions_sha256': sha(RUN / 'predictions/oof_predictions.csv'), 'labels_sha256': sha(RUN / 'labels/development.csv'), 'original_VALID_opened_by_this_selection': False, 'evidence_class': c['evidence_class'], 'selected_at_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'status': 'FUSION_WEIGHTS_FROZEN', 'alphas': {k: v['alpha'] for k, v in choices.items()}}))

def classify(gain, alpha, relative, positive_folds, validation_gain, ci_lower, ci_upper, cheap_screens, specificity, minimum_relative=.05):
    if alpha == 0 or gain <= 0 or ci_upper <= 0 or (positive_folds <= 2 and validation_gain <= 0): return 'NO_DEVELOPMENT_INCREMENT'
    primary_pass = relative >= minimum_relative and positive_folds >= 4 and validation_gain > 0 and ci_lower > 0
    if primary_pass and specificity: return 'CANDIDATE_INCREMENT_NOT_CONFIRMED'
    if primary_pass and any(cheap_screens): return 'NON_SPECIFIC_ENSEMBLE_GAIN'
    return 'INCONCLUSIVE'

def _thresholds(labels, ids, ewma):
    y = np.maximum(labels.loc[ids, 'RV_raw'].to_numpy(), EPS)
    return {'q90': float(np.quantile(y, .9)), 'q99': float(np.quantile(y, .99)), 'ewma_cuts': np.quantile(ewma, [.25, .5, .75]).tolist()}

def planning_scenarios(se, effect, months_grid, alpha, n_reference=1267):
    """Conditional scenario only. An identical/zero-variance contrast has no MDE."""
    applicable = alpha > 0 and np.isfinite(se) and se > 0
    rows = []
    for months in months_grid:
        expected_n = months * 365.25 / 12 * 3
        future_se = se * np.sqrt(n_reference / expected_n) if applicable else np.nan
        delta = effect / future_se if applicable else np.nan
        z = norm.ppf(.975)
        power = float(norm.cdf(delta-z) + norm.cdf(-delta-z)) if applicable else np.nan
        rows.append({'months':months, 'expected_opportunities_approx':expected_n,
                     'planning_SE_7day':future_se,
                     'MDE_80power_two_sided95':float((z+norm.ppf(.8))*future_se),
                     'power_at_5pct_base_Regret_approx':power,
                     'planning_status':'CONDITIONAL_SCENARIO' if applicable else 'NOT_APPLICABLE_ALPHA_ZERO_OR_DEGENERATE',
                     'future_stationarity_unproven':True})
    return rows

@threadpool_limits.wrap(limits=1)
def evaluate():
    c = config(); check_source_seal()
    if (RUN / 'metrics/summary.json').exists(): raise FileExistsError('Preserve completed/failed analysis')
    selected = read(RUN / 'fusion/selected_weights.json')
    assert selected['protocol_sha256'] == sha(CONFIG)
    assert selected['oof_predictions_sha256'] == sha(RUN / 'predictions/oof_predictions.csv')
    assert selected['labels_sha256'] == sha(RUN / 'labels/development.csv')
    oo, yy = load_tables('oof_predictions.csv'); va, vy = load_tables('validation_predictions.csv')
    va['fold_id'] = 'VALID'
    pred = pd.concat([oo, va], ignore_index=True)
    labels = pd.concat([yy, vy], ignore_index=True)
    for comparator, choice in selected['selected'].items(): pred['mix_'+comparator] = (1-choice['alpha'])*pred.B5 + choice['alpha']*pred[comparator]
    candidates = []
    for comparator in c['fusion']['comparators']:
        for alpha in c['fusion']['alpha_candidates']:
            key = f'candidate_{comparator}_a{alpha:g}'; pred[key] = (1-alpha)*pred.B5 + alpha*pred[comparator]; candidates.append(key)
    families = BASE + ['mix_'+x for x in c['fusion']['comparators']] + candidates
    all_y = csv(RUN / 'labels/development.csv').set_index('opportunity_id', drop=False)
    # This DEV-only source supplies known historical EWMA, never test data.
    old_x = csv(ROOT / 'research/runs/FROZEN_RISK_03_v1/features/ordinary_risk.csv').set_index('opportunity_id')
    thresholds, fold_def = {}, {}
    dt = pd.to_datetime(all_y.decision_boundary_at, utc=True)
    label_end = pd.to_datetime(all_y.label_end, utc=True); labelable = pd.to_datetime(all_y.labelable_at, utc=True)
    for f in c['roles']['folds']:
        end = pd.Timestamp(f['fit_end']); mask = (dt>=pd.Timestamp(c['roles']['train_start'])) & (dt<end) & (label_end<end) & (labelable<end)
        ids = all_y.index[mask]; assert len(ids) == f['expected_counts'][0]
        thresholds[f['id']] = _thresholds(all_y, ids, np.clip(old_x.loc[ids, 'r0_ewma'], EPS, 1))
        fold_def[f['id']] = (pd.Timestamp(f['evaluation_start']), pd.Timestamp(f['evaluation_end']))
    end = pd.Timestamp(c['roles']['original_train_end']); mask = (dt<end)&(label_end<end)&(labelable<end)
    thresholds['VALID'] = _thresholds(all_y, all_y.index[mask], np.clip(old_x.loc[all_y.index[mask], 'r0_ewma'], EPS, 1))
    fold_def['VALID'] = (end, pd.Timestamp(c['roles']['original_validation_end']))
    write(RUN / 'metrics/FIT_fixed_thresholds.json', thresholds)
    points, row_tables, tail, state = [], [], [], []
    per_loss = {}
    for fid, section in pred.groupby('fold_id', sort=False):
        idx = section.index.to_numpy(); target = labels.iloc[idx].RV_raw.to_numpy(); effective = np.maximum(target, EPS); ewma = section.ewma.to_numpy()
        event = np.log((target+EPS)/(ewma+EPS)) > np.log(2)
        high = effective > thresholds[fid]['q90']; bins = np.searchsorted(thresholds[fid]['ewma_cuts'], ewma, side='right')
        for family in families:
            values = section[family].to_numpy(); ll = losses(target, values); per_loss[(fid, family)] = ll
            score = np.log((values+EPS)/(ewma+EPS)); rank = ranking(event, score); high_rank = ranking(high, values)
            points.append({'fold_id': fid, 'family': family, 'N': len(section), 'positive': int(event.sum()), 'negative': int((~event).sum()), **{k: float(x.mean()) for k, x in ll.items()}, 'surprise_AUROC': rank['AUROC'], 'surprise_AP': rank['AP'], 'absolute_q90_AUROC': high_rank['AUROC'], 'absolute_q90_AP': high_rank['AP'], 'q90_events': int(high.sum())})
            row_tables.append(pd.DataFrame({'opportunity_id': section.opportunity_id.to_numpy(), 'decision_at': section.decision_at.to_numpy(), 'fold_id': fid, 'family': family, 'RV_raw': target, 'RV_effective': effective, 'EWMA_RV': ewma, 'prediction_RV': values, 'surprise_event': event, 'surprise_score': score, **ll}))
            for subset in ['q90', 'q99']:
                pick = effective > thresholds[fid][subset]
                tail.append({'fold_id': fid, 'family': family, 'subset': subset, 'FIT_threshold': thresholds[fid][subset], 'N': int(pick.sum()), 'under_count': int((pick & (values/effective<.5)).sum()), 'under_fraction': float((values[pick]/effective[pick]<.5).mean()) if pick.any() else np.nan, 'mean_prediction': float(values[pick].mean()) if pick.any() else np.nan, 'mean_actual': float(target[pick].mean()) if pick.any() else np.nan, 'Regret_sum': float(ll['QLIKE_Regret'][pick].sum())})
            for bin_id in range(4):
                pick = bins == bin_id
                state.append({'fold_id': fid, 'family': family, 'EWMA_bin': bin_id, 'N': int(pick.sum()), **{k: float(x[pick].mean()) if pick.any() else np.nan for k, x in ll.items()}})
    row = pd.concat(row_tables, ignore_index=True); point = pd.DataFrame(points)
    pooled = []
    for group, ids in [('OOF', [f['id'] for f in c['roles']['folds']]), ('VALID', ['VALID'])]:
        for family in families:
            sub = row[row.fold_id.isin(ids)&row.family.eq(family)]
            rank = ranking(sub.surprise_event.to_numpy(bool), sub.surprise_score.to_numpy())
            pooled.append({'period': group, 'family': family, 'N': len(sub), **{k: float(sub[k].mean()) for k in ['raw_QLIKE','QLIKE_Regret','logRV_MSE']}, 'surprise_AUROC': rank['AUROC'], 'surprise_AP': rank['AP'], 'pooled_ranking_limit': 'fold-specific models/constants; use per-fold ranking for interpretation'})
    save(RUN/'metrics/point_metrics.csv', point); save(RUN/'metrics/pooled_metrics.csv', pd.DataFrame(pooled)); save(RUN/'metrics/row_metrics.csv', row); save(RUN/'metrics/tail_metrics.csv', pd.DataFrame(tail)); save(RUN/'metrics/state_metrics.csv', pd.DataFrame(state))
    contrasts = {'B5_minus_mix_R2': ('B5','mix_R2'), 'B5_minus_mix_R1': ('B5','mix_R1'), 'B5_minus_mix_B2': ('B5','mix_B2'), 'mix_R1_minus_mix_R2': ('mix_R1','mix_R2'), 'mix_B2_minus_mix_R2': ('mix_B2','mix_R2'), 'R2_minus_B5': ('R2','B5')}
    intervals, daily_rows, concentrations = [], [], []
    for block in c['bootstrap']['blocks']:
        table, archive, aggregate = {}, {}, {}
        for fi, (fid, (start, end)) in enumerate(fold_def.items()):
            sub = pred[pred.fold_id.eq(fid)]; calendar = pd.date_range(start, end, freq='D', inclusive='left')
            day_ids = calendar.get_indexer(sub.decision_at.dt.floor('D')); assert (day_ids>=0).all()
            counts = stationary_day_counts(len(calendar), block, 5000, c['bootstrap']['base_seed']+block*1000+fi)
            archive[fid+'_utc_days'] = calendar.astype(str).to_numpy(dtype='U25'); archive[fid+'_counts'] = counts
            day_n = np.bincount(day_ids, minlength=len(calendar)); denom = counts.astype(float) @ day_n
            assert (denom>0).all()
            sums = {family: np.bincount(day_ids, weights=per_loss[(fid,family)]['raw_QLIKE'], minlength=len(calendar)) for family in families}
            for name, (first, second) in contrasts.items():
                day_gain = sums[first]-sums[second]; numerator = counts.astype(float) @ day_gain; values = numerator/denom
                table[fid+'_'+name] = values
                aggregate.setdefault(name, []).append((fid,numerator,denom))
                point_gain = float((per_loss[(fid,first)]['raw_QLIKE']-per_loss[(fid,second)]['raw_QLIKE']).mean())
                ci = np.quantile(values,[.025,.975]); intervals.append({'period':fid,'contrast':name,'block_days':block,'N':len(sub),'point_gain':point_gain,'ci_lower':ci[0],'ci_upper':ci[1],'valid_draws':5000,'invalid_fraction':0.,'selection_conditioned_development':True})
                if block==7:
                    daily_rows.append(pd.DataFrame({'fold_id':fid,'day_utc':calendar,'contrast':name,'N':day_n,'gain_sum':day_gain,'cumulative_gain_sum':np.cumsum(day_gain)}))
                    total=float(day_gain.sum()); best=int(np.argmax(day_gain)); count_other=int(day_n.sum()-day_n[best]); gain_other=float((total-day_gain[best])/count_other) if count_other else np.nan
                    concentrations.append({'fold_id':fid,'contrast':name,'N':len(sub),'total_gain_sum':total,'max_positive_day_gain':max(0.,float(day_gain[best])),'max_positive_day_UTC':str(calendar[best]),'max_7day_gain_sum':float(pd.Series(day_gain).rolling(7,min_periods=1).sum().max()),'max_day_signed_share':float(day_gain[best]/total) if total!=0 else np.nan,'drop_best_positive_day_mean_gain':gain_other if day_gain[best]>0 else point_gain,'diagnostic_no_refit_no_reselection':True})
            for family in families: table[fid+'_loss_'+family] = (counts.astype(float)@sums[family])/denom
        for name, records in aggregate.items():
            take=[v for v in records if v[0]!='VALID']; values=sum(v[1] for v in take)/sum(v[2] for v in take); table['OOF_'+name]=values
            first, second=contrasts[name]; targets = [f['id'] for f in c['roles']['folds']]
            point_gain=float(np.concatenate([per_loss[(fid,first)]['raw_QLIKE']-per_loss[(fid,second)]['raw_QLIKE'] for fid in targets]).mean())
            ci=np.quantile(values,[.025,.975]); intervals.append({'period':'OOF','contrast':name,'block_days':block,'N':1267,'point_gain':point_gain,'ci_lower':ci[0],'ci_upper':ci[1],'valid_draws':5000,'invalid_fraction':0.,'bootstrap_SE':float(values.std(ddof=1)),'selection_conditioned_development':True})
        with (RUN/'bootstrap'/f'multiplicities_block{block}.npz').open('xb') as f:np.savez_compressed(f,**archive)
        save(RUN/'bootstrap'/f'draws_block{block}.csv',pd.DataFrame({'draw':np.arange(5000),**table}))
    ci=pd.DataFrame(intervals);save(RUN/'metrics/paired_intervals.csv',ci);save(RUN/'metrics/daily_paired_gains.csv',pd.concat(daily_rows,ignore_index=True));save(RUN/'metrics/gain_concentration.csv',pd.DataFrame(concentrations))
    pooled=pd.DataFrame(pooled).set_index(['period','family']); screens={}; primary_rows=ci[ci.period.eq('OOF')&ci.block_days.eq(7)].set_index('contrast')
    for comparator in c['fusion']['comparators']:
        gain=float(primary_rows.loc['B5_minus_mix_'+comparator,'point_gain']); base_reg=float(pooled.loc[('OOF','B5'),'QLIKE_Regret']); relative=gain/base_reg
        fold_gains=[float(per_loss[(f['id'],'B5')]['raw_QLIKE'].mean()-per_loss[(f['id'],'mix_'+comparator)]['raw_QLIKE'].mean()) for f in c['roles']['folds']]
        valid_gain=float(per_loss[('VALID','B5')]['raw_QLIKE'].mean()-per_loss[('VALID','mix_'+comparator)]['raw_QLIKE'].mean()); positive=sum(x>0 for x in fold_gains); lower=float(primary_rows.loc['B5_minus_mix_'+comparator,'ci_lower']);upper=float(primary_rows.loc['B5_minus_mix_'+comparator,'ci_upper'])
        screens[comparator]={'alpha':selected['selected'][comparator]['alpha'],'mean_gain':gain,'base_OOF_Regret':base_reg,'relative_Regret_improvement':relative,'fold_gains':fold_gains,'positive_folds':positive,'VALID_gain':valid_gain,'ci_lower_7d':lower,'ci_upper_7d':upper,'primary_screen_pass':relative>=.05 and positive>=4 and valid_gain>0 and lower>0}
    specificity=all(primary_rows.loc[name,'ci_lower']>0 for name in ['mix_R1_minus_mix_R2','mix_B2_minus_mix_R2']); r=screens['R2']
    outcome=classify(r['mean_gain'],r['alpha'],r['relative_Regret_improvement'],r['positive_folds'],r['VALID_gain'],r['ci_lower_7d'],r['ci_upper_7d'],[screens[x]['primary_screen_pass'] for x in ['R1','B2']],specificity)
    se=float(primary_rows.loc['B5_minus_mix_R2','bootstrap_SE']); effect=.05*r['base_OOF_Regret']
    horizon=planning_scenarios(se,effect,c['future_draft']['sample_length_grid_months'],r['alpha'])
    save(RUN/'metrics/future_sample_scenarios.csv',pd.DataFrame(horizon))
    direct_se=float(primary_rows.loc['R2_minus_B5','bootstrap_SE'])
    direct_horizon=planning_scenarios(direct_se,effect,c['future_draft']['sample_length_grid_months'],1.)
    save(RUN/'metrics/future_direct_R2_B5_scenarios.csv',pd.DataFrame(direct_horizon))
    write(RUN/'metrics/summary.json',{'status':outcome,'evidence_class':c['evidence_class'],'protocol_sha256':sha(CONFIG),'N_OOF':1267,'N_original_VALID':len(va),'OOF_primary_weighting':'opportunity_weighted','screens':screens,'specificity_screen_pass':specificity,'no_new_independent_confirmation':True,'Regret_minus_raw_loss_gain_max_abs':float(max(np.max(np.abs((per_loss[(fid,'B5')]['QLIKE_Regret']-per_loss[(fid,'mix_R2')]['QLIKE_Regret'])-(per_loss[(fid,'B5')]['raw_QLIKE']-per_loss[(fid,'mix_R2')]['raw_QLIKE']))) for fid in fold_def)),'bootstrap_caution':'Fixed selected alpha, OOF used for selection: intervals conditional and potentially optimistic; not formal evidence','planning':{'effect_candidate_5pct_base_Regret':effect,'paired_bootstrap_SE':se,'minimum_months_80power_grid':next((v['months'] for v in horizon if v['power_at_5pct_base_Regret_approx']>=.8),None),'normal_approximation_not_actual_power':True},'future_formal_test_approved':False,'old_holdout_status':'CONSUMED'})
    check_source_seal()
    print(json.dumps({'status':outcome,'R2_alpha':r['alpha'],'OOF_gain':r['mean_gain'],'relative_Regret_improvement':r['relative_Regret_improvement']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['select','evaluate']);args=parser.parse_args();{'select':choose_weights,'evaluate':evaluate}[args.action]()
