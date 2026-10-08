"""Summarize completed exploratory E00R development results; never inspect holdout.

Reference opportunity utility is not a compounded portfolio return. No model,
coverage or threshold is selected by this script, and no promotion is automatic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
FOLDS = ('WF01', 'WF02', 'WF03', 'WF04')
CUTOFF = pd.Timestamp('2026-04-01T00:00:00Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def development_labels(config):
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    split = pd.read_csv(bundle / 'split_index.csv', dtype=str)
    split = split.loc[split.fold_id.isin(FOLDS) & split.included.eq('True')].copy()
    allowed = set(split.opportunity_id)
    require(allowed and all(pd.Timestamp(x) < CUTOFF for x in allowed), 'Invalid development membership')
    # Filter strings before any outcome conversion, including labelability.
    rows = []
    with (bundle / 'labels_forbidden_as_features.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['opportunity_id'] in allowed:
                rows.append(row)
    labels = pd.DataFrame(rows).set_index('opportunity_id')
    require(labels.index.is_unique and set(labels.index) == allowed, 'Missing/duplicate development labels')
    require(pd.to_datetime(labels.labelable_at, utc=True).lt(CUTOFF).all(), 'Holdout labelability reached')
    for column in ('base_profit_ratio', 'stress_profit_ratio', 'direction'):
        labels[column] = pd.to_numeric(labels[column], errors='raise')
        require(np.isfinite(labels[column]).all(), 'Nonfinite reference labels')
    for fold in config['splits']['walk_forward_dates']:
        require(fold['id'] in FOLDS, 'Unexpected development fold')
        for role in ('train', 'validation', 'test'):
            ids = split.loc[split.fold_id.eq(fold['id']) & split.role.eq(role), 'opportunity_id']
            require(pd.to_datetime(labels.loc[ids, 'labelable_at'], utc=True).lt(pd.Timestamp(fold[role][1])).all(), 'Unpurged membership')
    return labels, split, bundle


def prediction_summary(predictions, labels, split):
    feature_column, score_column = 'feature_set', 'score_bps'
    require({feature_column, score_column}.issubset(predictions.columns), 'Prediction schema lacks feature_set or score_bps')
    rows = []
    for (fold, variant), group in predictions.groupby(['fold_id', feature_column]):
        ids = group.opportunity_id.tolist()
        expected = set(split.loc[split.fold_id.eq(fold) & split.role.eq('test'), 'opportunity_id'])
        require(len(ids) == len(set(ids)) and set(ids) == expected, 'Prediction test membership differs')
        y = labels.loc[ids, 'base_profit_ratio'].to_numpy() * 10000
        p = pd.to_numeric(group[score_column], errors='raise').to_numpy()
        train_ids = split.loc[split.fold_id.eq(fold) & split.role.eq('train'), 'opportunity_id']
        mean = float(labels.loc[train_ids, 'base_profit_ratio'].mean() * 10000)
        require(np.isfinite(p).all(), 'Nonfinite predictions')
        errors = (p - y) ** 2
        count = max(1, int(np.ceil(len(y) * .05)))
        rows.append({'fold_id': fold, 'head': 'Ridge_validation_selected', 'feature_variant': variant,
                     'samples': len(y), 'MSE_bps_squared': float(errors.mean()),
                     'MAE_bps': float(np.abs(p-y).mean()), 'training_mean_bps': mean,
                     'constant_MSE_bps_squared': float(np.mean((mean-y)**2)),
                     'constant_MAE_bps': float(np.mean(np.abs(mean-y))),
                     'MSE_improvement_bps_squared': float(np.mean((mean-y)**2)-errors.mean()),
                     'Spearman': float(pd.Series(p).corr(pd.Series(y), method='spearman')) if np.std(p) and np.std(y) else None,
                     'mean_prediction_bias_bps': float(np.mean(p-y)),
                     'top_5pct_squared_error_share': float(np.sort(errors)[-count:].sum()/errors.sum()) if errors.sum() else 0})
    return pd.DataFrame(rows)


def matched_random(y, selected, ids, direction):
    """Posthoc month/direction count matching; percentiles are not confidence limits."""
    count = int(selected.sum())
    if count == 0:
        return {'selected_count': 0, 'selected_mean_bps': None, 'random_mean_bps': None,
                'random_p025_bps': None, 'random_p975_bps': None, 'selection_lift_bps': None,
                'opportunity_utility_bps': 0., 'status': 'NA_zero_selection', 'draws': 0}
    strata = pd.DataFrame({'month': pd.to_datetime(ids, utc=True).strftime('%Y-%m'), 'direction': direction})
    groups = [(np.asarray(index), int(selected[np.asarray(index)].sum()))
              for index in strata.groupby(['month', 'direction']).indices.values()]
    values = []
    for seed in (17, 29, 43):
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            total = sum(float(y[rng.choice(index, size=n, replace=False)].sum()) for index, n in groups if n)
            values.append(total/count)
    mean = float(y[selected].mean())
    return {'selected_count': count, 'selected_mean_bps': mean, 'random_mean_bps': float(np.mean(values)),
            'random_p025_bps': float(np.quantile(values, .025)), 'random_p975_bps': float(np.quantile(values, .975)),
            'selection_lift_bps': mean-float(np.mean(values)), 'opportunity_utility_bps': float(y[selected].sum()/len(y)),
            'status': 'posthoc_random_distribution_not_CI', 'draws': len(values)}


def stationary_means(values, block, draws=5000, seed=17):
    """Circular stationary bootstrap of a single complete fold calendar."""
    values = np.asarray(values, dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), 'Invalid bootstrap series')
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), draws)
    totals = values[indexes].copy()
    for _ in range(1, len(values)):
        restart = rng.random(draws) < 1./block
        indexes = np.where(restart, rng.integers(0, len(values), draws), (indexes+1) % len(values))
        totals += values[indexes]
    return totals / len(values)


def reference_summary(output, manifest, labels, split, config):
    rows, random_rows, gates = [], [], {}
    for signal in manifest['signals']:
        fold, mode = signal['fold_id'], signal['mode']
        require(fold in FOLDS, 'Unexpected signal fold')
        ids = sorted(split.loc[split.fold_id.eq(fold) & split.role.eq('test'), 'opportunity_id'])
        path = (output / signal['path']).resolve()
        require(path.is_relative_to(output.resolve()), 'Signal path escapes run')
        payload = read_json(path)
        decisions = payload['decisions']
        # Buy/hold, fractional sizing and long-only controls have different reference utility.
        if mode in ('buy_hold', 'constant_half_exposure', 'vol_target'):
            continue
        actions = {d.get('opportunity_id', d.get('signal_at')): float(d['exposure_fraction']) for d in decisions}
        require(len(actions) == len(decisions) and set(actions).issubset(set(ids)), 'Invalid signal membership')
        selected = np.array([actions.get(x, 0) > 0 for x in ids])
        require(all(v in (0., 1.) for v in actions.values()), 'Reference gate requires binary sizing')
        require(int(selected.sum()) == signal['selected_count'], 'Signal selected count differs')
        require(signal['opportunities'] == len(ids), 'Signal opportunity count differs')
        require(sha(path) == signal['sha256'], 'Signal manifest hash differs')
        gates[(fold, mode)] = (ids, selected)
        for cost in ('base', 'stress'):
            y = labels.loc[ids, cost+'_profit_ratio'].to_numpy()*10000
            rows.append({'fold_id': fold, 'mode': mode, 'cost': cost, 'fee_bps_per_side': 7 if cost == 'base' else 14,
                         'family': signal['family'], 'gate': signal['gate'], 'primary': signal['primary'],
                         'opportunities': len(ids), 'selected_count': int(selected.sum()),
                         'participation': float(selected.mean()),
                         'selected_mean_net_bps': float(y[selected].mean()) if selected.any() else None,
                         'gated_net_bps_per_eligible_opportunity': float(np.mean(y*selected)),
                         'metric_basis': 'independent_reference_labels_not_portfolio_return'})
            if mode.startswith('ridge'):
                random_rows.append({'fold_id': fold, 'mode': mode, 'cost': cost,
                                    **matched_random(y, selected, labels.loc[ids, 'entry_at'].tolist(), labels.loc[ids, 'direction'].to_numpy())})
    comparisons = [('ridge19_rank50', 'mean_random_q50'), ('ridge19_rank50', 'ridge13_rank50'),
                   ('ridge19_economic', 'cash')]
    boot = []
    folds = {f['id']: f for f in config['splits']['walk_forward_dates']}
    for fold in FOLDS:
        for model, control in comparisons:
            require((fold, model) in gates, 'Missing primary model gate')
            ids, a = gates[(fold, model)]
            if control == 'cash':
                b = np.zeros(len(ids))
            elif control == 'mean_random_q50':
                bs = []
                for seed in (17, 29, 43):
                    other, values = gates[(fold, 'random_q50_s'+str(seed))]
                    require(other == ids, 'Random pairing membership differs')
                    bs.append(values)
                b = np.mean(bs, axis=0)
            else:
                other, b = gates[(fold, control)]
                require(other == ids, 'Pairing membership differs')
            calendar = pd.date_range(folds[fold]['test'][0], folds[fold]['test'][1], freq='D', inclusive='left')
            for cost in ('base', 'stress'):
                utility = (a.astype(float)-b)*labels.loc[ids, cost+'_profit_ratio'].to_numpy()*10000
                daily = pd.Series(utility, index=pd.to_datetime(ids, utc=True).floor('D')).groupby(level=0).sum().reindex(calendar, fill_value=0)
                for block in (7, 3, 14):
                    samples = stationary_means(daily.to_numpy(), block)
                    boot.append({'fold_id': fold, 'model': model, 'control': control, 'cost': cost,
                                 'mean_block_days': block, 'draws': 5000, 'seed': 17, 'calendar_days': len(daily),
                                 'approximate_effective_blocks': len(daily)/block,
                                 'daily_mean_difference_bps': float(daily.mean()),
                                 'bootstrap_p025_bps': float(np.quantile(samples, .025)),
                                 'bootstrap_p975_bps': float(np.quantile(samples, .975)),
                                 'scope': 'paired_daily_reference_utility_not_portfolio_return'})
    return pd.DataFrame(rows), pd.DataFrame(random_rows), pd.DataFrame(boot), comparisons


def charts(output, metrics, reference, matched):
    selected = ['ridge19_rank50', 'ridge19_economic', 'cash', 'legacy_E00_logistic_gate']
    legacy = metrics['mode'].eq('ordinary_features_gate')
    metrics = metrics.copy()
    metrics.loc[legacy, 'mode'] = 'legacy_E00_logistic_gate'
    values = []
    for (fold, cost), group in metrics.groupby(['fold_id', 'cost']):
        for mode in selected:
            row = group.loc[group['mode'].eq(mode)]
            require(len(row) == 1, 'Missing/duplicate chart portfolio: '+mode)
            values.append({'fold_id': fold, 'cost': cost, 'mode': mode, 'net_return_pct': float(row.net_return_pct.iloc[0]), 'trades': float(row.trades.iloc[0])})
        for name, modes in [('random_mean', ['random_q50_s'+str(x) for x in (17,29,43)]), ('periodic_mean', ['periodic_offset0','periodic_offset1'])]:
            row = group.loc[group['mode'].isin(modes)]
            require(set(row['mode']) == set(modes), 'Missing chart controls')
            values.append({'fold_id': fold, 'cost': cost, 'mode': name, 'net_return_pct': float(row.net_return_pct.mean()), 'trades': float(row.trades.mean())})
    data = pd.DataFrame(values)
    data.to_csv(output/'portfolio_comparison_chart_values.csv', index=False)
    order = selected[:2]+['random_mean','periodic_mean']+selected[2:]
    fig, axes = plt.subplots(2, 2, figsize=(16,9), constrained_layout=True)
    return_limit = max(float(data.net_return_pct.abs().max()), 1.)
    for col, cost in enumerate(('base','stress')):
        for row, metric in enumerate(('net_return_pct','trades')):
            pivot = data.loc[data.cost.eq(cost)].pivot(index='fold_id', columns='mode', values=metric).reindex(index=FOLDS, columns=order)
            scale = {'vmin': -return_limit, 'vmax': return_limit} if row == 0 else {'vmin': 0}
            im = axes[row,col].imshow(pivot, aspect='auto', cmap='RdYlGn' if row == 0 else 'Blues', **scale)
            axes[row,col].set_xticks(range(len(order)), order, rotation=28, ha='right', fontsize=8)
            axes[row,col].set_yticks(range(4), ['2025 Q2','2025 Q3','2025 Q4','2026 Q1'])
            axes[row,col].set_title(f'{7 if cost == "base" else 14} bps per side | '+('net portfolio return (%)' if row == 0 else 'executed trades (control mean may be fractional)'))
            for (i,j), value in np.ndenumerate(pivot.to_numpy()):
                axes[row,col].text(j,i,f'{value:+.2f}' if row == 0 else f'{value:.1f}',ha='center',va='center',fontsize=9)
            fig.colorbar(im, ax=axes[row,col], shrink=.8)
    fig.suptitle('E00R exploratory development | Official Freqtrade | 10,000 USDT reset per quarter\nFunding included; fee proxy is not price-impact simulation; closed-trade drawdown only')
    for ext in ('png','svg'):
        fig.savefig(output/('portfolio_comparison.'+ext),dpi=160)
    plt.close(fig)
    chart = matched.loc[matched['mode'].isin(('ridge19_rank50','ridge19_economic'))].copy()
    chart.to_csv(output/'matched_random_chart_values.csv',index=False)
    fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,cost in zip(axes,('base','stress')):
        for shift,mode in ((-.08,'ridge19_rank50'),(.08,'ridge19_economic')):
            group = chart.loc[chart.cost.eq(cost)&chart['mode'].eq(mode)].set_index('fold_id').reindex(FOLDS)
            x = np.arange(4)+shift
            ax.vlines(x,group.random_p025_bps,group.random_p975_bps,alpha=.4)
            ax.scatter(x,group.selected_mean_bps,label=mode)
            ax.scatter(x,group.random_mean_bps,marker='_',color='gray')
        ax.axhline(0,color='black',linewidth=.7)
        ax.set_xticks(range(4),['2025 Q2','2025 Q3','2025 Q4','2026 Q1'])
        ax.set_ylabel('Selected independent reference mean net bps')
        ax.set_title(f'{7 if cost == "base" else 14} bps per side | random draw 2.5%-97.5% range, not CI')
        ax.legend()
    fig.suptitle('Matched random: same UTC month x direction counts; zero selection is NA\nNot executable controls; not portfolio return confidence intervals')
    for ext in ('png','svg'):
        fig.savefig(output/('matched_random.'+ext),dpi=160)
    plt.close(fig)
    subset = reference.loc[reference['mode'].isin(order)].copy()
    subset.to_csv(output/'reference_utility_chart_values.csv',index=False)
    fig, axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,cost in zip(axes,('base','stress')):
        for mode in ('ridge19_rank50','ridge19_economic'):
            group=subset.loc[subset.cost.eq(cost)&subset['mode'].eq(mode)].set_index('fold_id').reindex(FOLDS)
            ax.plot(range(4),group.gated_net_bps_per_eligible_opportunity,marker='o',label=mode)
        ax.axhline(0,color='black',linewidth=.7)
        ax.set_xticks(range(4),['2025 Q2','2025 Q3','2025 Q4','2026 Q1'])
        ax.set_ylabel('Reference net bps per eligible opportunity')
        ax.set_title(f'{7 if cost == "base" else 14} bps per side; not portfolio returns')
        ax.legend()
    for ext in ('png','svg'):
        fig.savefig(output/('reference_utility.'+ext),dpi=160)
    plt.close(fig)


def run(output):
    output = output.resolve()
    require((output/'metrics.csv').exists(), 'Official results do not exist; analysis remains pending')
    require(not (output/'analysis_report.json').exists(), 'Refuse to overwrite existing analysis provenance; use a fresh analysis run')
    engine_report = read_json(output/'engine_report.json')
    require(engine_report['status'] == 'verified_engine_pending_analysis' and engine_report['engine_runs'] == 112,
            'Official E00R engine audits incomplete')
    registry = read_json(ROOT/'research/registry'/(output.name+'.json'))
    require(registry['status'] == 'engine_verified_pending_analysis', 'Run registry not ready for analysis')
    for relative, digest in registry['artifacts']['preparation_hashes'].items():
        require(sha(output/relative) == digest, 'Preparation artifact changed: '+relative)
    for relative, digest in registry['artifacts']['engine_hashes'].items():
        require(sha(output/relative) == digest, 'Engine artifact changed: '+relative)
    config = yaml.safe_load((output/'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    require(pd.Timestamp(config['splits']['final_holdout'][0]) == CUTOFF, 'Holdout boundary changed')
    metrics = pd.read_csv(output/'metrics.csv')
    require(set(metrics.fold_id) == set(FOLDS) and set(metrics.cost) == {'base','stress'}, 'Incomplete development engine results')
    require(not metrics.duplicated(['fold_id','mode','cost']).any(), 'Duplicate engine metrics')
    require(len(metrics)==112, 'Expected 112 verified E00R engine exports')
    legacy_path = ROOT/'research/runs/E00_20261008_phase4_v1/metrics.csv'
    old_registry_path = ROOT/'research/registry/E00_20261008_phase4_v1.json'
    old_registry = read_json(old_registry_path)
    require(sha(legacy_path) == old_registry['artifacts']['final_hashes']['metrics.csv'], 'Legacy metrics hash differs from accepted E00 evidence')
    old = pd.read_csv(legacy_path)
    require(set(old.fold_id) == set(FOLDS) and len(old)==48, 'Legacy engine results incomplete')
    require(not set(metrics['mode']) & set(old['mode']), 'Legacy/new portfolio mode collision')
    # Save the analysis source and fixed method plan before diagnostic computation.
    # Engine summaries above have been opened for structural/hash acceptance only.
    # Do not mutate the completed preparation's provenance snapshots.
    execution = output/'analysis_execution_source'
    execution.mkdir()
    shutil.copy2(Path(__file__), execution/'summarize_e00r.py')
    plan = {'analysis_source_sha256':sha(Path(__file__)),
            'preparation_source_sha256':registry['provenance']['canonical_source_sha256'].get('research/baselines/summarize_e00r.py'),
            'development_folds':FOLDS,'holdout_boundary_exclusive':str(CUTOFF),'selection':'validation_only_no_test_selection',
            'matched_random':{'strata':'UTC_entry_month_x_fixed_direction','seeds':[17,29,43],'draws_per_seed':1000,'zero_selection_draws':0},
            'bootstrap':{'comparisons':[['ridge19_rank50','mean_random_q50'],['ridge19_rank50','ridge13_rank50'],['ridge19_economic','cash']],
                         'draws':5000,'seed':17,'primary_mean_block_days':7,'sensitivity_mean_block_days':[3,14],
                         'series':'daily_paired_independent_reference_utility_bps','stratification':'separate_quarters'},
            'interpretation':'exploratory_information_and_profitability_separate_exposure_reduction_not_evidence_of_either; preserve_negative_results; no_automatic_promotion'}
    (execution/'analysis_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    labels, split, bundle = development_labels(config)
    predictions = pd.read_csv(output/'development_test_predictions.csv')
    require(set(predictions.fold_id) == set(FOLDS), 'Unexpected prediction fold')
    heads = read_json(output/'selected_heads.json')
    attempts = read_json(output/'training_attempts.json')
    require(len(heads) == 8 and len(attempts) == 32, 'Unexpected head/attempt count')
    require(set(zip(predictions.fold_id,predictions.feature_set)) == {(f,v) for f in FOLDS for v in ('ridge13','ridge19')}, 'Missing prediction variants')
    for head in heads:
        candidates = [a for a in attempts if a['fold_id'] == head['fold_id'] and a['feature_set'] == head['feature_set']]
        require(len(candidates) == 4 and all(a['status'] == 'completed' for a in candidates), 'Incomplete candidate grid')
        winner = min(candidates,key=lambda a:(a['validation_mse_bps_squared'],-a['lambda']))
        require(winner['lambda'] == head['lambda'] and winner['validation_mse_bps_squared'] == head['validation_mse_bps_squared'], 'Head not selected by validation MSE/tie-break')
    prediction = prediction_summary(predictions,labels,split)
    manifest = read_json(output/'signals_manifest.json')
    reference, random, bootstrap, comparisons = reference_summary(output,manifest,labels,split,config)
    inputs = [output/'metrics.csv',output/'engine_report.json',output/'development_test_predictions.csv',output/'signals_manifest.json',output/'selected_heads.json',output/'training_attempts.json',output/'provenance/experiment_config.yaml',legacy_path,old_registry_path,bundle/'split_index.csv',bundle/'labels_forbidden_as_features.csv',execution/'summarize_e00r.py',execution/'analysis_plan.json']
    inputs += [output/s['path'] for s in manifest['signals']]
    provenance = {str(p): sha(p) for p in inputs}
    for name,table in [('prediction_metrics',prediction),('reference_gate_metrics',reference),('matched_random_summary',random),('paired_block_bootstrap',bootstrap)]:
        table.to_csv(output/(name+'.csv'),index=False)
    charts(output,pd.concat([metrics,old],ignore_index=True),reference,random)
    report = {'status':'completed_exploratory_development_analysis','final_holdout_inspected':False,
              'frozen_representations_extracted':False,'automatic_promotion':False,
              'development_status':'previously_seen_development_not_new_clean_OOS',
              'prespecified_comparisons':comparisons,'no_test_selection':True,
              'prediction_head_selection':'chronological_validation_MSE_only',
              'matched_random':'UTC_month_x_direction_count_matched; 17/29/43 each 1000; percentiles are random distribution, not statistical CI',
              'bootstrap':'fold-stratified paired complete daily calendar; stationary bootstrap 5000 seed17; mean blocks7, sensitivity3/14; approximate within-fold stationarity and limited effective blocks required',
              'reference_utility':'independent reference net labels in bps; selected mean differs from mean gate*return over all eligible opportunities; neither is portfolio return',
              'portfolio':'official Freqtrade; 10,000 USDT reset each quarter; 7/14 bps per-side fee proxy; funding included; closed-trade drawdown does not certify hourly marked-to-market risk',
              'interpretation':'Information value requires evidence against noninformative controls. Profitability requires actual net portfolio gains over cash with stress evidence. Exposure reduction and smaller losses alone establish neither. Negative and sparse results retained; ordinary baseline profitability is not a prerequisite for later frozen-representation research.',
              'evidence_claim':'descriptive_and_uncertainty_evidence_only_no_automatic_information_or_profitability_claim',
              'zero_selection':'selected mean/lift NA; opportunity utility zero; insufficient trade evidence, not engine failure',
              'input_sha256':provenance,'outputs':{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name in ('prediction_metrics.csv','reference_gate_metrics.csv','matched_random_summary.csv','paired_block_bootstrap.csv','portfolio_comparison_chart_values.csv','reference_utility_chart_values.csv','matched_random_chart_values.csv','portfolio_comparison.png','portfolio_comparison.svg','reference_utility.png','reference_utility.svg','matched_random.png','matched_random.svg')}}
    (output/'analysis_report.json').write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'output_dir':str(output),'automatic_promotion':False}))


def verify_synthetic():
    """Small targeted checks without opening research data or running analysis."""
    require(np.array_equal(stationary_means(np.zeros(31),7),np.zeros(5000)), 'Identical paired series must give zero')
    ids = [f'2025-01-{day:02d}T04:00:00Z' for day in range(1,13)]
    y = np.arange(12,dtype=float)
    actions = np.ones(12,dtype=bool)
    full = matched_random(y,actions,ids,np.array([1,-1]*6))
    require(full['selected_count'] == 12 and full['random_mean_bps'] == y.mean()
            and full['random_p025_bps'] == full['random_p975_bps'] == y.mean(), 'Matched random must preserve stratum counts')
    direction = np.array([1,-1]*6)
    partial = np.array([True,False,True,True]+[False]*8)
    stratum_constant = np.where(direction == 1,2.,10.)
    check = matched_random(stratum_constant,partial,ids,direction)
    require(np.isclose(check['random_mean_bps'], stratum_constant[partial].mean())
            and check['random_p025_bps'] == check['random_p975_bps'], 'Partial selection must match direction stratum counts')
    zero = matched_random(y,np.zeros(12,dtype=bool),ids,np.array([1,-1]*6))
    require(zero['selected_mean_bps'] is None and zero['opportunity_utility_bps'] == 0, 'Zero selection handling')
    # Demonstrate pre-conversion filtering: forbidden outcome strings are never converted.
    raw = [{'id':ids[0],'outcome':'2.5'}, {'id':'2026-04-01T04:00:00Z','outcome':'MUST_NOT_PARSE'}]
    allowed = {ids[0]}
    converted = [float(row['outcome']) for row in raw if row['id'] in allowed]
    require(converted == [2.5] and all(pd.Timestamp(x)<CUTOFF for x in allowed), 'Holdout filtering before outcome conversion')
    print(json.dumps({'status':'synthetic_checks_passed','official_analysis_run':False,
                      'checks':['identical_paired_series_zero','matched_random_counts','zero_selection_NA','preconversion_holdout_filter']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir', type=Path)
    group.add_argument('--verify-synthetic', action='store_true')
    args = parser.parse_args()
    if args.verify_synthetic:
        verify_synthetic()
    else:
        run(args.output_dir)
