"""Paired exploratory development analysis; reference uncertainty is not portfolio uncertainty."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
from summarize_e00r import development_labels, prediction_summary, matched_random, stationary_means, clean
from run_e00 import read_result
import argparse
from datetime import datetime, timezone
import shutil
import json
import numpy as np
import pandas as pd
import yaml

SEEDS=(17,29,43)
FOLDS=('WF01','WF02','WF03','WF04')
FAMILIES=('mlp19','mlp_pretrained',*[f'mlp_random_s{s}' for s in SEEDS])
GATES=('rank50','rank25','rank75','economic')
SECONDARY_RUNS=('E00_20261008_phase4_v1','E00R_20261008_phase4_v1','FROZEN_20261008_v1')

def stratum_expected_weights(ids, direction, actions):
    table=pd.DataFrame({'month':pd.to_datetime(ids,utc=True).strftime('%Y-%m'),'direction':direction})
    weights=np.zeros(len(ids))
    for indexes in table.groupby(['month','direction']).indices.values():
        indexes=np.asarray(indexes); weights[indexes]=actions[indexes].mean()
    return weights

def daily(values, ids, fold):
    calendar=pd.date_range(fold['test'][0],fold['test'][1],freq='D',inclusive='left')
    return pd.Series(values,index=pd.to_datetime(ids,utc=True).floor('D')).groupby(level=0).sum().reindex(calendar,fill_value=0)

def run(output):
    output=output.resolve()
    require(output.is_relative_to((ROOT/'research/runs').resolve()) and output!=(ROOT/'research/runs').resolve(),'Output must be a research run')
    regpath=ROOT/'research/registry'/f'{output.name}.json'
    registry=json.loads(regpath.read_text(encoding='utf-8'))
    require(registry['status']=='engine_verified_pending_analysis','Official replay not accepted')
    require(not (output/'analysis_report.json').exists(),'Do not overwrite analysis')
    for category in ['preparation_hashes','engine_hashes']:
        for rel,value in registry['artifacts'][category].items():require(sha(output/rel)==value,'Changed input '+rel)
    config=yaml.safe_load((output/'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    common=yaml.safe_load((output/'provenance/common_config.yaml').read_text(encoding='utf-8'))
    require(sha(output/'provenance/common_config.yaml')==registry['configuration']['sha256'],'Protocol copy/hash mismatch')
    require(set(common['variants'])==set(FAMILIES) and common['head']['type']=='MLP' and common['head']['seeds']==list(SEEDS),'Unexpected MLP design')
    require(common['fit_budget']['new_official_exports']==160 and common['final_holdout_access']=='forbidden','Scope mismatch')
    uncertainty_spec=common['paired_uncertainty']; screen=common['research_screen']
    require(uncertainty_spec['draws']==5000 and uncertainty_spec['seed']==17 and uncertainty_spec['mean_block_days']==7
            and uncertainty_spec['sensitivity_days']==[3,14],'Unsupported uncertainty design; version the analysis')
    require(screen['prediction_median_relative_MSE_improvement_min']==.01 and screen['matched_random_median_lift_bps_min']==1
            and screen['positive_increment_folds_min']==3 and screen['information_equal_fold_primary_block_lower_95_bound_min']==0
            and screen['economic_median_quarterly_increment_vs_ordinary_min_fraction']==.01
            and screen['economic_positive_quarters_vs_cash_min']==3
            and screen['economic_median_closed_trade_drawdown_change_max_fraction']==0,'Unsupported research screen; version the analysis')
    source=output/'provenance/analysis_source'; source.mkdir()
    for p in [Path(__file__),ROOT/'research/baselines/summarize_e00r.py',ROOT/'research/baselines/run_e00.py']:
        shutil.copy2(p,source/p.name)
    write_json(source/'analysis_plan.json',{'protocol_sha256':registry['configuration']['sha256'],
               'primary_predeclared_comparisons':common['primary_comparisons'],
               'paired_bootstrap':common['paired_uncertainty'],'screen':common['research_screen'],
               'no_portfolio_bootstrap':True,'no_model_or_threshold_selection':True})
    labels,split,bundle=development_labels(config)
    inputs={str(output/'provenance/common_config.yaml'):sha(output/'provenance/common_config.yaml')}
    for rel in ('split_index.csv','labels_forbidden_as_features.csv'):
        inputs[str(bundle/rel)]=sha(bundle/rel)
    predictions=pd.read_csv(output/'development_test_predictions.csv')
    require(set(predictions.feature_set)==set(FAMILIES) and set(predictions.fold_id)==set(FOLDS),'Unexpected MLP prediction families/folds')
    pred_summary=prediction_summary(predictions,labels,split)
    require(len(pred_summary)==20,'Expected five independently ensembled MLP families per fold')
    pred_summary['head']='MLP_validation_selected_three_head_seed_mean'
    pred_summary.to_csv(output/'prediction_metrics.csv',index=False)
    origins=[(output,json.loads((output/'signals_manifest.json').read_text(encoding='utf-8')))]
    expected_gates={(f,f'{family}_{gate}') for f in FOLDS for family in FAMILIES for gate in GATES}
    gates={}; reference=[]; matched=[]
    for origin,manifest in origins:
        for s in manifest['signals']:
            fold=s['fold_id']; mode=s['mode']; require(fold in FOLDS,'Unexpected fold')
            ids=sorted(split.loc[split.fold_id.eq(fold)&split.role.eq('test'),'opportunity_id'])
            require((fold,mode) in expected_gates,'Unexpected MLP gate')
            path=(origin/s['path']).resolve(); require(path.is_relative_to(output),'Signal escapes run'); require(sha(path)==s['sha256'],'Changed gate '+mode)
            payload=json.loads(path.read_text(encoding='utf-8'))
            rows=payload['decisions']; require([r['signal_at'] for r in rows]==ids,'Gate membership differs')
            require(all(set(r)=={'signal_at','direction','exposure_fraction'} for r in rows),'Outcome sidecar leak')
            a=np.asarray([r['exposure_fraction'] for r in rows],float)
            require(np.isin(a,[0,1]).all() and int(a.sum())==s['selected_count'],'Gate actions differ')
            require((fold,mode) not in gates,'Duplicate gate')
            gates[(fold,mode)]=(ids,a)
            inputs[str(path)]=sha(path)
            direction=labels.loc[ids,'direction'].to_numpy()
            require(np.array_equal(direction,np.asarray([r['direction'] for r in rows])),'Gate direction differs from fixed momentum')
            entry=labels.loc[ids,'entry_at'].tolist()
            weights=stratum_expected_weights(entry,direction,a)
            for cost in ['base','stress']:
                y=labels.loc[ids,cost+'_profit_ratio'].to_numpy()*10000
                reference.append({'fold_id':fold,'mode':mode,'cost':cost,'opportunities':len(ids),'selected_count':int(a.sum()),
                                  'participation':float(a.mean()),'selected_mean_net_bps':float(y[a>0].mean()) if a.any() else None,
                                  'gated_net_bps_per_eligible_opportunity':float(np.mean(y*a)),
                                  'metric_basis':'independent_official_reference_net_labels_not_portfolio'})
                if mode.startswith('mlp'):
                    result=matched_random(y,a>0,entry,direction)
                    exact=float(y@weights/a.sum()) if a.sum() else None
                    matched.append({'fold_id':fold,'mode':mode,'cost':cost,**result,
                                    'exact_null_mean_bps':exact,
                                    'exact_selection_lift_bps':float(y@a/a.sum()-exact) if a.sum() else None})
    require(set(gates)==expected_gates,'Missing registered MLP gates')
    reference=pd.DataFrame(reference); matched=pd.DataFrame(matched)
    reference.to_csv(output/'reference_gate_metrics.csv',index=False)
    matched.to_csv(output/'matched_random_summary.csv',index=False)
    boots=[]; daily_rows=[]; draws={}
    fold_lookup={f['id']:f for f in config['splits']['walk_forward_dates']}
    for fi,fold_id in enumerate(FOLDS):
        fold=fold_lookup[fold_id]; ids,a=gates[(fold_id,'mlp_pretrained_rank50')]
        direction=labels.loc[ids,'direction'].to_numpy(); entry=labels.loc[ids,'entry_at'].tolist()
        weights=stratum_expected_weights(entry,direction,a)
        random_actions=np.mean([gates[(fold_id,f'mlp_random_s{s}_rank50')][1] for s in SEEDS],axis=0)
        ordinary=gates[(fold_id,'mlp19_rank50')][1]
        ordinary_w=stratum_expected_weights(entry,direction,ordinary)
        scores={family:predictions.loc[predictions.fold_id.eq(fold_id)&predictions.feature_set.eq(family)].set_index('opportunity_id').loc[ids,'score_bps'].to_numpy() for family in ['mlp19','mlp_pretrained',*[f'mlp_random_s{s}' for s in SEEDS]]}
        require(all(len(v)==len(ids) for v in scores.values()),'Prediction pairing mismatch')
        for cost in ['base','stress']:
            y=labels.loc[ids,cost+'_profit_ratio'].to_numpy()*10000
            series={
                'rank50_vs_ordinary':((a-ordinary)*y,'reference_utility_bps_per_calendar_day',1.),
                'rank50_vs_mean_random_backbone':((a-random_actions)*y,'reference_utility_bps_per_calendar_day',1.),
                'rank50_vs_matched_expected_counts':((a-weights)*y,'reference_selection_excess_bps_per_calendar_day',1.),
                'economic_vs_cash':(gates[(fold_id,'mlp_pretrained_economic')][1]*y,'reference_utility_bps_per_calendar_day',1.),
                'economic_vs_ordinary':((gates[(fold_id,'mlp_pretrained_economic')][1]-gates[(fold_id,'mlp19_economic')][1])*y,'reference_utility_bps_per_calendar_day',1.),
                'economic_vs_mean_random_backbone':((gates[(fold_id,'mlp_pretrained_economic')][1]-np.mean([gates[(fold_id,f'mlp_random_s{s}_economic')][1] for s in SEEDS],axis=0))*y,'reference_utility_bps_per_calendar_day',1.)}
            days=len(pd.date_range(fold['test'][0],fold['test'][1],freq='D',inclusive='left'))
            if a.sum() and ordinary.sum():
                series['selection_lift_vs_ordinary']=((a-weights)*y/a.sum()-(ordinary-ordinary_w)*y/ordinary.sum(),'selected_reference_mean_lift_difference_bps',float(days))
            if cost=='base':
                err_pre=(scores['mlp_pretrained']-y)**2
                err_ord=(scores['mlp19']-y)**2
                err_random=np.mean([(scores[f'mlp_random_s{s}']-y)**2 for s in SEEDS],axis=0)
                series['MSE_improvement_vs_ordinary']=(err_ord-err_pre,'MSE_difference_bps_squared',days/len(ids))
                series['MSE_improvement_vs_mean_random_backbone']=(err_random-err_pre,'MSE_difference_bps_squared',days/len(ids))
            for comparison,(v,unit,scale) in series.items():
                calendar_series=daily(v,ids,fold)
                daily_rows.extend({'fold_id':fold_id,'comparison':comparison,'cost':cost,'at':at.isoformat(),'value':float(value),'scale_from_daily_mean':scale,'reported_unit':unit} for at,value in calendar_series.items())
                for block in (7,3,14):
                    samples=stationary_means(calendar_series.to_numpy(),block,draws=5000,seed=17+fi*100)*scale
                    draws.setdefault((comparison,cost,block,unit),[]).append(samples)
                    boots.append({'fold_id':fold_id,'comparison':comparison,'cost':cost,'mean_block_days':block,
                                  'point_difference':float(calendar_series.mean()*scale),'lower_95':float(np.quantile(samples,.025)),
                                  'upper_95':float(np.quantile(samples,.975)),'unit':unit,'scope':'reference_or_prediction_not_compounded_portfolio',
                                  'calendar_days':days,'effective_blocks_approx':days/block,'draws':5000,'seed':17+fi*100})
    for (comparison,cost,block,unit),parts in draws.items():
        require(len(parts)==4,'Aggregate requires all four folds')
        samples=np.mean(parts,axis=0)
        points=[r['point_difference'] for r in boots if r['comparison']==comparison and r['cost']==cost and r['mean_block_days']==block]
        boots.append({'fold_id':'EQUAL_FOLD_MEAN','comparison':comparison,'cost':cost,'mean_block_days':block,
                      'point_difference':float(np.mean(points)),'lower_95':float(np.quantile(samples,.025)),
                      'upper_95':float(np.quantile(samples,.975)),'unit':unit,'scope':'reference_or_prediction_not_compounded_portfolio',
                      'draws':5000,'seed':'17+fold_index*100'})
    bootstrap=pd.DataFrame(boots); bootstrap.to_csv(output/'paired_block_bootstrap.csv',index=False)
    pd.DataFrame(daily_rows).to_csv(output/'paired_daily_series.csv',index=False)
    metrics=pd.read_csv(output/'metrics.csv')
    expected_metrics={(f,f'{family}_{gate}',c) for f in FOLDS for family in FAMILIES for gate in GATES for c in ('base','stress')}
    require(len(metrics)==160 and set(metrics[['fold_id','mode','cost']].itertuples(index=False,name=None))==expected_metrics,'Expected 160 unique new MLP exports')
    secondary=[]
    for run_id in SECONDARY_RUNS:
        old_run=ROOT/'research/runs'/run_id
        old_regpath=ROOT/'research/registry'/f'{run_id}.json'
        old_registry=json.loads(old_regpath.read_text(encoding='utf-8'))
        require(old_registry['status'].startswith('completed'),'Secondary baseline not accepted')
        old_hashes={key.replace('\\','/'):digest for key,digest in old_registry['artifacts']['final_hashes'].items()}
        require(sha(old_run/'metrics.csv')==old_hashes['metrics.csv'],'Secondary baseline metrics changed')
        frame=pd.read_csv(old_run/'metrics.csv')
        require(set(frame.fold_id)==set(FOLDS) and not frame.duplicated(['fold_id','mode','cost']).any(),'Secondary scope mismatch')
        frame.loc[frame['mode'].eq('ordinary_features_gate'),'mode']='legacy_E00_logistic_gate'
        frame['comparison_role']='secondary_explanation_only'
        frame['source_run']=run_id
        secondary.append(frame)
        inputs[str(old_run/'metrics.csv')]=sha(old_run/'metrics.csv')
        inputs[str(old_regpath)]=sha(old_regpath)
    metrics['comparison_role']='predeclared_MLP_primary_or_sensitivity'
    metrics['source_run']=output.name
    combined=pd.concat([metrics,*secondary],ignore_index=True)
    require(not combined.duplicated(['fold_id','mode','cost']).any(),'Duplicate portfolio metrics')
    combined.to_csv(output/'all_portfolio_metrics.csv',index=False)
    concentration=[]
    for row in metrics.itertuples(index=False):
        require(sha(row.archive)==row.archive_sha256,'Official archive hash changed')
        stats=read_result(Path(row.archive)); trades=stats['trades']
        profit=np.array([t['profit_abs'] for t in trades],float)
        pos=profit[profit>0]; top=max(1,int(np.ceil(len(profit)*.05)))
        turnover=sum(float(t['amount'])*(float(t['open_rate'])+float(t['close_rate'])) for t in trades)
        concentration.append({'fold_id':row.fold_id,'mode':row.mode,'cost':row.cost,'trades':len(trades),
                              'positive_profit_top5pct_trade_count_share':float(np.sort(pos)[-top:].sum()/pos.sum()) if pos.sum() else None,
                              'largest_profit_USDT':float(profit.max()) if len(profit) else None,'largest_loss_USDT':float(profit.min()) if len(profit) else None,
                              'gross_entry_exit_notional_USDT':turnover,'turnover_over_initial_wallet':turnover/10000})
    pd.DataFrame(concentration).to_csv(output/'portfolio_concentration_turnover.csv',index=False)
    pairs=[]
    for fold in FOLDS:
        for cost in ('base','stress'):
            for gate in ('rank50','economic'):
                rows=combined.loc[combined.fold_id.eq(fold)&combined.cost.eq(cost)].set_index('mode')
                pre=rows.loc['mlp_pretrained_'+gate]; ordinary=rows.loc['mlp19_'+gate]
                rnd=rows.loc[[f'mlp_random_s{s}_{gate}' for s in SEEDS]]
                pairs.append({'fold_id':fold,'cost':cost,'gate':gate,'MLP_pretrained_return_pct':pre.net_return_pct,
                              'MLP_ordinary19_return_pct':ordinary.net_return_pct,'MLP_random_backbone_seed_mean_return_pct':rnd.net_return_pct.mean(),
                              'increment_vs_ordinary_pp':pre.net_return_pct-ordinary.net_return_pct,
                              'increment_vs_seed_mean_pp':pre.net_return_pct-rnd.net_return_pct.mean(),
                              'closed_drawdown_change_vs_ordinary_pp':pre.closed_trade_max_drawdown_pct-ordinary.closed_trade_max_drawdown_pct,
                              'MLP_pretrained_trades':pre.trades,'MLP_ordinary19_trades':ordinary.trades,'MLP_random_backbone_seed_mean_trades':rnd.trades.mean()})
    paired=pd.DataFrame(pairs); paired.to_csv(output/'paired_portfolio_metrics.csv',index=False)
    prediction_pairs=[]
    for fold in FOLDS:
        rows=pred_summary.loc[pred_summary.fold_id.eq(fold)].set_index('feature_variant')
        pre=rows.loc['mlp_pretrained','MSE_bps_squared']; ordinary=rows.loc['mlp19','MSE_bps_squared']
        random=rows.loc[[f'mlp_random_s{s}' for s in SEEDS],'MSE_bps_squared'].mean()
        prediction_pairs.append({'fold_id':fold,'MLP_pretrained_MSE_bps_squared':pre,'MLP_ordinary19_MSE_bps_squared':ordinary,
                                 'MLP_random_backbone_seed_mean_MSE_bps_squared':random,'relative_MSE_improvement_vs_ordinary':1-pre/ordinary,
                                 'relative_MSE_improvement_vs_seed_mean':1-pre/random})
    pp=pd.DataFrame(prediction_pairs); pp.to_csv(output/'paired_prediction_metrics.csv',index=False)
    selection=matched.loc[matched.cost.eq('base') & matched['mode'].eq('mlp_pretrained_rank50')].sort_values('fold_id')
    info_comps=['rank50_vs_ordinary','rank50_vs_mean_random_backbone','rank50_vs_matched_expected_counts']
    uncertainty=bootstrap.loc[bootstrap.fold_id.eq('EQUAL_FOLD_MEAN')&bootstrap.cost.eq('base')&bootstrap.mean_block_days.eq(7)&bootstrap.comparison.isin(info_comps)]
    require(len(uncertainty)==3,'Missing prespecified uncertainty')
    info_checks={'median_relative_MSE_vs_ordinary_ge_1pct':pp.relative_MSE_improvement_vs_ordinary.median()>=.01,
                 'median_relative_MSE_vs_random_seed_mean_ge_1pct':pp.relative_MSE_improvement_vs_seed_mean.median()>=.01,
                 'positive_prediction_improvement_folds_ge_3':int((pp.relative_MSE_improvement_vs_ordinary>0).sum())>=3 and int((pp.relative_MSE_improvement_vs_seed_mean>0).sum())>=3,
                 'median_matched_selection_lift_ge_1bps':selection.exact_selection_lift_bps.median()>=1,
                 'positive_matched_lift_folds_ge_3':int((selection.exact_selection_lift_bps>0).sum())>=3,
                 'paired_primary_95_lower_positive_all_comparators':bool((uncertainty.lower_95>0).all()),
                 'positive_paired_utility_folds_ge_3_all_comparators':all(int((bootstrap.loc[bootstrap.fold_id.isin(FOLDS)&bootstrap.cost.eq('base')&bootstrap.mean_block_days.eq(7)&bootstrap.comparison.eq(c),'point_difference']>0).sum())>=3 for c in info_comps)}
    economic={}
    for gate in ('rank50','economic'):
        b=paired.loc[paired.gate.eq(gate)&paired.cost.eq('base')]; s=paired.loc[paired.gate.eq(gate)&paired.cost.eq('stress')]
        checks={'positive_base_increment_folds_ge_3':int((b.increment_vs_ordinary_pp>0).sum())>=3,
                'median_base_increment_ge_1pp':b.increment_vs_ordinary_pp.median()>=1,
                'positive_base_quarters_vs_cash_ge_3':int((b.MLP_pretrained_return_pct>0).sum())>=3,
                'stress_median_return_positive':s.MLP_pretrained_return_pct.median()>0,
                'stress_median_increment_positive':s.increment_vs_ordinary_pp.median()>0,
                'median_closed_drawdown_not_worse':b.closed_drawdown_change_vs_ordinary_pp.median()<=0}
        economic[gate]={'checks':checks,'passed':all(checks.values())}
    info_pass=all(info_checks.values())
    report={'status':'completed_exploratory_development_analysis','completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'experiment':output.name,'information_screen':{'checks':info_checks,'passed':info_pass},'economic_screens':economic,
            'conclusion':'support_further_research' if info_pass or any(v['passed'] for v in economic.values()) else 'insufficient_evidence',
            'automatic_promotion':False,'final_holdout_inspected':False,'backbone_or_tokenizer_trained':False,
            'random_seed_selection':False,'protocol_sha256':registry['configuration']['sha256'],
            'limitations':['Development folds previously seen; exploratory historical research.',
                           'Checkpoint training cutoff unproven; no strict ex-ante OOS claim.',
                           'Paired intervals apply to independent reference utility or prediction loss, never compounded portfolio return.',
                           'Stationary bootstrap assumes approximate within-fold stationarity; 7-day blocks leave about 13 effective blocks per quarter.',
                           'E02R preserves pretrained tokenizer; only backbone pretraining is removed.',
                           'Closed-trade equity excludes unrealized intrahour exposure risk; fee proxy does not simulate price impact.'],
            'matched_random':'Month x direction exact expected mean plus 3000 random draws; draw percentiles are not portfolio confidence intervals.',
            'seed_mean':'Arithmetic mean of all three backbone losses/utilities; not mean backbone scores or an executable policy. Each family uses its own three-head-seed score ensemble.',
            'new_MLP_fits':common['fit_budget']['new_MLP_fits'],'selected_ensembles':20,
            'input_sha256':inputs,'metrics_rows':len(combined),'new_official_exports':160,'head_seed_ensemble':[17,29,43],
            'primary_prediction_families':list(FAMILIES),'secondary_baselines_affect_primary_screen':False,
            'paired_calendar':'complete UTC decision-date calendar with zero opportunity days retained',
            'sparse_economic_gate_counts':{f:int(gates[(f,'mlp_pretrained_economic')][1].sum()) for f in FOLDS},
            'failure_policy':'insufficient_evidence_when_prespecified_screen_fails; no added capacity, seed or cutoff search'}
    write_json(output/'analysis_report.json',clean(report))
    print(json.dumps(clean(report),ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',required=True,type=Path)
    run(parser.parse_args().output_dir)
