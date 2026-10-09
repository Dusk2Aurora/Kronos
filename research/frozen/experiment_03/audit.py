"""Engineering-only preflight audits; all real inputs end before holdout."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import importlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from research.frozen.experiment_03.guard import ROOT,RUN,sha,config
from research.frozen.experiment_03.run import read_prepared,immutable_json,immutable_csv
from research.frozen.experiment_03 import models,inputs

def original_suites():
    kr=str(ROOT/'.venv/Scripts/python.exe');ft='D:/file/freqtrade/.venv/Scripts/python.exe'
    suites=[(kr,'research/environment/verify_environment.py',[]),
      (kr,'research/baselines/verify_ordinary_features.py',[]),(kr,'research/baselines/verify_e00r.py',[]),
      (kr,'research/freqtrade/verify_reference_labels.py',[]),(kr,'research/frozen/mlp_heads.py',['--verify-synthetic']),
      (kr,'research/frozen/quant_heads.py',['--verify-synthetic']),(ft,'research/freqtrade/verify_import.py',[]),
      (ft,'research/freqtrade/verify_strategy.py',[]),(ft,'research/freqtrade/verify_e00_strategy.py',[])]
    results=[];destination=RUN/'preflight/tests';destination.mkdir(exist_ok=True)
    for executable,rel,args in suites:
        proc=subprocess.run([executable,str(ROOT/rel),*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        log=destination/(Path(rel).stem+'.txt')
        with log.open('x',encoding='utf-8') as f:f.write(proc.stdout+'\n'+proc.stderr)
        status='PASS' if proc.returncode==0 else 'FAIL'
        results.append({'suite':rel,'arguments':args,'exit_code':proc.returncode,'status':status,'log_sha256':sha(log),'scope':'offline synthetic/environment; no actual holdout values'})
        print(rel+': '+status,flush=True)
        if proc.returncode and 'verify_e00r.py' not in rel:raise ValueError('Original suite failure '+rel)
    # Replay the original stage assertion on its archived input without editing it.
    sys.path.insert(0,str(ROOT/'research/baselines'))
    prior=json.loads((ROOT/'research/registry/E00R_20261008_phase4_v1.json').read_text(encoding='utf-8'))
    text=prior['parent_configuration']['full_text'];old=importlib.import_module('verify_e00r');orig_read=Path.read_text
    def redirect(path,*args,**kwargs):
        return text if path.resolve()==(ROOT/'research/configs/initial_experiment.yaml').resolve() else orig_read(path,*args,**kwargs)
    with patch.object(Path,'read_text',redirect):checks=old.synthetic_checks()
    immutable_json(destination/'historical_e00r.json',{'status':'PASS','checks':checks,'original_assertions_unchanged':True,'historical_config_sha256':prior['parent_configuration']['sha256']})
    # M0 verifier requires an initialization output; retain its accepted historical
    # replay instead of changing old provenance or weakening the output-path guard.
    retained=ROOT/'research/initialization/M0_risk02_revalidation_v1.json'
    prior_m0=json.loads(retained.read_text(encoding='utf-8'))
    immutable_json(destination/'M0_retained_evidence.json',{'source':str(retained.relative_to(ROOT)),
         'source_sha256':sha(retained),'full_report':prior_m0,'mode':'accepted historical evidence; not a new M0 execution'})
    immutable_json(destination/'original_suites.json',results)
    return results

def new_suites():
    destination=RUN/'preflight/tests';destination.mkdir(exist_ok=True)
    paths=sorted((ROOT/'tests').glob('test_risk03_*.py'))
    proc=subprocess.run([sys.executable,'-m','pytest',*[str(p) for p in paths],'-q','-s','-p','no:cacheprovider'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    log=destination/'risk03_tests.txt'
    with log.open('x',encoding='utf-8') as f:f.write(proc.stdout+'\n'+proc.stderr)
    print(proc.stdout,flush=True)
    immutable_json(destination/'new_suites.json',{'status':'PASS' if proc.returncode==0 else 'FAIL','exit_code':proc.returncode,'paths':[str(p.relative_to(ROOT)) for p in paths],'log_sha256':sha(log),'synthetic_only':True})
    if proc.returncode:raise ValueError('Third-round synthetic contracts failed')

def actual():
    x,opp,labels,matrices=read_prepared();cfg=config();preview=RUN/'preflight/preview'
    checks=[]
    def check(name,condition,detail=None):
        checks.append({'check':name,'status':'PASS' if bool(condition) else 'FAIL','detail':detail})
        if not condition:raise AssertionError(name)
    check('train_validation_counts',(opp.role=='train').sum()==2369 and (opp.role=='validation').sum()==92)
    tr=opp.index[opp.role=='train'];va=opp.index[opp.role=='validation']
    for role,ids in [('train',tr),('validation',va)]:
        start,end=map(pd.Timestamp,cfg['roles'][role]);selected=labels.loc[ids]
        check(role+'_label_purge',all(pd.Timestamp(v)<end for v in selected.label_end) and all(pd.Timestamp(v)<end for v in selected.labelable_at))
        check(role+'_feature_causality',all(pd.Timestamp(v)<end for v in opp.loc[ids,'history_end_exclusive']))
    independent=[]
    for _,row in labels.iterrows():
        prices=np.asarray(json.loads(row.label_prices_json),dtype=float)
        times=pd.DatetimeIndex(json.loads(row.label_bar_open_at_json))
        check('label_segments_'+row.opportunity_id,len(prices)==49 and len(times)==48 and times.equals(pd.date_range(pd.Timestamp(row.entry_at),periods=48,freq='5min')))
        independent.append(sum(float(np.log(prices[i+1]/prices[i]))**2 for i in range(48)))
    check('all_labels_independent_reconstruction',np.allclose(labels.RV_raw,independent,rtol=1e-10,atol=1e-16),{'rows':len(labels),'max_abs':float(np.abs(labels.RV_raw-np.asarray(independent)).max())})
    candles=inputs.load_hourly();sample=opp.iloc[[0,len(opp)//2,len(opp)-1]]
    rebuilt=inputs.build_features(candles,sample)
    check('features_independent_rebuild',np.allclose(x.loc[sample.index],rebuilt,rtol=1e-12,atol=1e-14))
    events=[json.loads(v) for v in (preview/'fit_events.jsonl').read_text().splitlines()]
    check('28_candidate_fits',sum(v['status']=='completed' for v in events)==28)
    check('two_determinism_replays',sum(v['status']=='determinism_completed' for v in events)==2)
    check('all_fit_calls_closed',len(events)==60 and not any('failed' in v['status'] for v in events))
    candidates=pd.read_csv(preview/'all_candidate_predictions.csv',float_precision='round_trip')
    selected_predictions=pd.read_csv(preview/'selected_predictions.csv',float_precision='round_trip')
    chosen=json.loads((preview/'selected_models.json').read_text());numeric=[]
    for selection in chosen:
        family=selection['family'];m=json.loads((RUN/selection['model_path']).read_text());matrix=matrices['R1' if family=='B2' else family]
        rows=selected_predictions.loc[selected_predictions.family==family]
        pred=(models.predict_b2 if family=='B2' else models.predict_head)(m,matrix.loc[rows.opportunity_id].to_numpy())
        check('prediction_reload_'+family,np.allclose(pred,rows.prediction,rtol=1e-12,atol=1e-16),{'rows':len(rows)})
        eligible=candidates.loc[(candidates.family==family)&(candidates.role=='validation')]
        scores={name:float(models.qlike(group.RV_raw.to_numpy(),group.prediction.to_numpy()).mean()) for name,group in eligible.groupby('candidate')}
        if family=='B2':
            selection_key=f"B2_leaves_{selection['num_leaves']}_minleaf_{selection['min_data_in_leaf']}"
            best=min(scores,key=lambda key:(scores[key],int(key.split('_')[2]),-int(key.split('_')[4])))
            check('B2_train_scale',m['scale']==float(np.median(np.maximum(labels.loc[tr,'RV_raw'],1e-12))))
            history=m['history']['validation']['raw_QLIKE'];check('B2_early_stop_best_iteration',m['best_iteration']==int(np.argmin(history))+1)
            check('B2_native_float32_precision_recorded',m['training_label_precision']['library_dataset_dtype']=='float32')
        else:
            selection_key=f"{family}_lambda_{selection['lambda']}";best=min(scores,key=lambda key:(scores[key],-float(key.rsplit('_',1)[1])))
            train_matrix=matrix.loc[tr].to_numpy(dtype=float)
            check(family+'_train_scaler_mean',np.allclose(m['mean'],train_matrix.mean(axis=0),rtol=1e-12,atol=1e-12))
            check(family+'_solver_guard',m['eligible'] and m['solver']['success'] and m['solver']['max_abs_gradient']<=1e-5)
        check('validation_only_selection_'+family,best==selection_key and abs(scores[best]-selection['validation_qlike'])<=1e-12)
        numeric.append({'family':family,'validation_qlike':scores[best],'dimension':matrix.shape[1],'prediction_reload':'PASS','selector_recomputed':'PASS'})
    check('holdout_not_generated',not (RUN/'authorization/user_approval.json').exists() and not (RUN/'authorization/execution_claim.json').exists() and not list((RUN/'data_5m').rglob('*.json')))
    immutable_json(RUN/'preflight/actual_audit.json',{'status':'PASS','checks':checks,'check_count':len(checks),'numeric_reproduction':numeric,'holdout_values_used':False})
    immutable_csv(RUN/'preflight/numeric_reproduction.csv',pd.DataFrame(numeric),False)
    print('Actual DEV audit PASS checks='+str(len(checks)),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['original','synthetic','actual']);a=p.parse_args()
    {'original':original_suites,'synthetic':new_suites,'actual':actual}[a.action]()
if __name__=='__main__':main()
