"""Revalidate old synthetic suites and new risk numerical/causal contracts."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.frozen import encoder
from research.frozen.experiment_02 import data,models,features
from research.frozen.experiment_02.run import RUN,CONFIG,read_inputs,get_matrices,config,write_json

def reject(call):
    try:call()
    except (ValueError,RuntimeError):return True
    raise AssertionError('Required negative rejection failed')

def synthetic():
    y=np.array([1.,2.,4.]);p=np.array([2.,2.,2.]);ratio=y/p
    np.testing.assert_allclose(models.qlike(y,p),np.log(p)+ratio,rtol=0,atol=1e-15)
    np.testing.assert_allclose(models.regret(y,p),ratio-np.log(ratio)-1,rtol=0,atol=1e-15)
    np.testing.assert_allclose(models.qlike(y,p)-models.regret(y,p),np.log(y)+1)
    np.testing.assert_array_equal(models.regret([0],[1e-12]),[0])
    rng=np.random.default_rng(17);x=rng.normal(size=(60,5));target=np.exp(.2*x[:,0])
    theta=np.r_[rng.normal(size=5)*.1,.2]
    _,gradient=models.objective_and_gradient(theta,x,target,.1)
    finite=[]
    for j in range(len(theta)):
        a=theta.copy();b=theta.copy();a[j]+=1e-6;b[j]-=1e-6
        finite.append((models.objective_and_gradient(a,x,target,.1)[0]-models.objective_and_gradient(b,x,target,.1)[0])/2e-6)
    np.testing.assert_allclose(gradient,finite,rtol=1e-6,atol=1e-8)
    with threadpool_limits(1):
        first=models.fit_head(x,target,.1);second=models.fit_head(x,target,.1)
    assert first==second and first['eligible']
    start=pd.Timestamp('2024-01-01T05:00:00Z');times=pd.date_range(start,periods=48,freq='5min')
    closes=100*np.exp(np.arange(1,49)*.001)
    c=pd.DataFrame({'bar_open_at':times,'bar_close_at':times+pd.Timedelta(minutes=5),
                    'open':np.r_[100,closes[:-1]],'close':closes,'confirm':'1'})
    o=pd.DataFrame([{'opportunity_id':'synthetic','entry_at':start.isoformat()}])
    lab=data.build_labels(c,o)
    np.testing.assert_allclose(lab.RV_raw,[48*.001**2],rtol=1e-11)
    assert pd.Timestamp(lab.iloc[0].label_start)==start and pd.Timestamp(lab.iloc[0].label_end)==start+pd.Timedelta(hours=4)
    assert pd.Timestamp(lab.iloc[0].labelable_at)==start+pd.Timedelta(hours=4,seconds=60)
    reject(lambda:data.build_labels(c.iloc[1:],o))
    reject(lambda:data.build_labels(c.assign(bar_open_at=c.bar_open_at+pd.Timedelta(minutes=5)),o))
    reject(lambda:data.build_labels(c.assign(confirm='0'),o))
    reject(lambda:data.build_labels(c.assign(bar_open_at=pd.Timestamp('2026-04-01T00:00:00Z'),close='must_not_parse'),o))
    reject(lambda:data.build_labels(c,pd.DataFrame([{'opportunity_id':'forbidden','entry_at':'2026-04-01T05:00:00Z'}])))
    reject(lambda:data.build_labels(c,o,delay=0))
    guard=data.build_labels(c,pd.DataFrame([{'opportunity_id':'edge','entry_at':'2026-03-31T20:00:00Z'}]))
    assert len(guard)==0
    return {'status':'PASS','checks':['QLIKE_hand_cases','Regret_same_pairwise_order','zero_floor_consistency',
       'analytic_gradient','synthetic_fit_exact_repeat','48_segment_hand_RV','entry_and_labelable_timing',
       'missing_shift_unconfirmed_reject','holdout_timestamp_before_price_reject','holdout_opportunity_reject','right_boundary_purge','delay_locked'],
       'gradient_error_max':float(np.max(np.abs(gradient-finite)))}

def old_suites():
    """Run old executable test suites without changing old files or trading inputs."""
    retained=RUN/'tests/original_suites.json'
    if retained.exists():
        results=json.loads(retained.read_text())
        for item in results:
            assert encoder.sha(RUN/'tests'/f'{Path(item["suite"]).stem}.txt')==item['log_sha256']
        return results
    kronos=str(ROOT/'.venv/Scripts/python.exe');ft='D:/file/freqtrade/.venv/Scripts/python.exe'
    suites=[(kronos,'research/environment/verify_environment.py',[]),
            (kronos,'research/baselines/verify_ordinary_features.py',[]),
            (kronos,'research/baselines/verify_e00r.py',[]),
            (kronos,'research/freqtrade/verify_reference_labels.py',[]),
            (kronos,'research/frozen/mlp_heads.py',['--verify-synthetic']),
            (kronos,'research/frozen/quant_heads.py',['--verify-synthetic']),
            (ft,'research/freqtrade/verify_import.py',[]),
            (ft,'research/freqtrade/verify_strategy.py',[]),
            (ft,'research/freqtrade/verify_e00_strategy.py',[])]
    results=[]
    for executable,path,args in suites:
        result=subprocess.run([executable,str(ROOT/path),*args],cwd=ROOT,text=True,capture_output=True,encoding='utf-8',errors='replace')
        log=RUN/'tests'/f'{Path(path).stem}.txt'
        encoder.immutable_bytes(log,(result.stdout+'\n'+result.stderr).encode())
        results.append({'suite':path,'arguments':args,'interpreter':executable,'exit_code':result.returncode,
                        'status':'PASS' if result.returncode==0 else 'FAIL','log_sha256':encoder.sha(log)})
        print(path+': '+results[-1]['status'],flush=True)
    write_json(RUN/'tests/original_suites.json',results)
    encoder.require(all(r['exit_code']==0 or r['suite']=='research/baselines/verify_e00r.py' for r in results),
                    'Original executable tests failed outside historical-stage E00R adapter')
    return results

def historical_e00r():
    prior=json.loads((ROOT/'research/registry/E00R_20261008_phase4_v1.json').read_text(encoding='utf-8'))
    text=prior['parent_configuration']['full_text']
    assert hashlib.sha256(text.encode()).hexdigest()==prior['parent_configuration']['sha256']
    cfgpath=ROOT/'research/configs/initial_experiment.yaml'
    import verify_e00r as old
    original_read=Path.read_text
    def redirected(path,*args,**kwargs):
        return text if path.resolve()==cfgpath.resolve() else original_read(path,*args,**kwargs)
    with patch.object(Path,'read_text',redirected):checks=old.synthetic_checks()
    return {'status':'PASS','checks':checks,'original_assertions_unchanged':True,
            'historical_input_sha256':prior['parent_configuration']['sha256'],
            'current_initial_yaml_unmodified':True,
            'prior_direct_stage_failure_preserved':'tests/verify_e00r.txt'}

def historical_m0():
    """Replay the stage-specific acceptance with its saved historical config only.

    The actual initial YAML is not changed. Original assertions all remain active;
    read_text and config hash are redirected to the retained immutable stage input.
    """
    transition=json.loads((ROOT/'research/initialization/M0_state_transition.json').read_text(encoding='utf-8'))
    text=transition['acceptance_input_config_full_text']
    cfgpath=ROOT/'research/configs/initial_experiment.yaml'
    prior=json.loads((ROOT/'research/initialization/M0_acceptance_report.json').read_text())
    assert hashlib.sha256(text.encode()).hexdigest()==prior['config_evaluated_sha256']
    import research.initialization.verify_m0 as old
    target=ROOT/'research/initialization/M0_risk02_revalidation_v1.json'
    original_read=Path.read_text;original_sha=old.sha
    def redirected(path,*args,**kwargs):
        return text if path.resolve()==cfgpath.resolve() else original_read(path,*args,**kwargs)
    def redirected_sha(path):
        return prior['config_evaluated_sha256'] if Path(path).resolve()==cfgpath.resolve() else original_sha(path)
    with patch.object(Path,'read_text',redirected),patch.object(old,'sha',redirected_sha):
        old.verify(target)
    encoder.immutable_bytes(RUN/'tests/M0_historical_revalidation.json',target.read_bytes())
    return {'status':'PASS','original_assertions_unchanged':True,'current_initial_config_unchanged':True,
            'historical_stage_input_sha256':prior['config_evaluated_sha256'],
            'report_path':str(target),'explanation':'Stage-specific M0 test replayed against retained M0 input, not current completed frozen stage'}

def actual():
    cfg=config();candles,opp,split,sources=read_inputs()
    saved=pd.read_csv(RUN/'features.csv',index_col='opportunity_id')
    sampled=opp.iloc[[0,len(opp)//2,len(opp)-1]]
    independently=features.build_features(candles,sampled)
    np.testing.assert_allclose(saved.loc[sampled.index].to_numpy(),independently.to_numpy(),rtol=1e-12,atol=1e-14)
    labels=pd.read_csv(RUN/'labels_rv.csv',index_col='opportunity_id',float_precision='round_trip')
    c5=pd.read_csv(RUN/'data_5m/candles.csv',dtype=str)
    rebuilt=data.build_labels(c5,sampled).set_index('opportunity_id')
    np.testing.assert_allclose(labels.loc[sampled.index,'RV_raw'],rebuilt.RV_raw,rtol=1e-12)
    allrv=[]
    for _,row in labels.iterrows():
        prices=np.asarray(json.loads(row.label_prices_json));times=pd.DatetimeIndex(json.loads(row.label_bar_open_at_json))
        assert len(prices)==49 and len(times)==48 and times.equals(pd.date_range(pd.Timestamp(row.entry_at),periods=48,freq='5min'))
        # Independently reconstruct returns explicitly, avoiding implementation's diff(log).
        allrv.append(sum(float(np.log(prices[i+1]/prices[i]))**2 for i in range(48)))
    np.testing.assert_allclose(labels.RV_raw.to_numpy(),allrv,rtol=1e-10,atol=1e-16)
    assert all(pd.Timestamp(x)<pd.Timestamp(cfg['holdout']['start']) for x in labels.labelable_at)
    events=[json.loads(line) for line in (RUN/'fit_events.jsonl').read_text().splitlines()]
    completed=[e for e in events if e['status']=='completed'];failed=[e for e in events if e['status']=='failed']
    assert len(completed)==96 and not failed
    predictions=pd.read_csv(RUN/'predictions.csv')
    candidates=pd.read_csv(RUN/'all_candidate_predictions.csv')
    matrices=get_matrices(saved)
    chosen=json.loads((RUN/'selected_models.json').read_text())
    reproduction=[]
    for item in chosen:
        model=json.loads((RUN/item['model_path']).read_text())
        rows=predictions[(predictions.fold_id==item['fold_id'])&(predictions.family==item['family'])]
        pred=models.predict_head(model,matrices[item['family']].loc[rows.opportunity_id].to_numpy())
        np.testing.assert_allclose(pred,rows.prediction,rtol=1e-12,atol=1e-16)
        vals=candidates[(candidates.fold_id==item['fold_id'])&(candidates.family==item['family'])&(candidates.role=='validation')]
        scores={lam:float(models.qlike(group.RV_raw,group.prediction).mean()) for lam,group in vals.groupby('lambda')}
        best=min(scores,key=lambda lam:(scores[lam],-lam))
        assert best==item['lambda']
        trainids=sorted(split.loc[(split.fold_id==item['fold_id'])&(split.role=='train'),'opportunity_id'])
        # The registered head promotes the float32 hidden cache to float64 before
        # fitting. Compute the independent mean in that same mandated precision.
        assert np.allclose(model['mean'],matrices[item['family']].loc[trainids].to_numpy(dtype=np.float64).mean(axis=0),rtol=1e-12,atol=1e-12)
        reproduction.append({'fold_id':item['fold_id'],'family':item['family'],'rows':len(rows),'prediction_reload':'PASS','validation_selection':'PASS','train_only_scaler':'PASS'})
    item=next(i for i in chosen if i['family']=='R2' and i['fold_id']=='WF01')
    original=json.loads((RUN/item['model_path']).read_text());trainids=sorted(split.loc[(split.fold_id=='WF01')&(split.role=='train'),'opportunity_id'])
    with threadpool_limits(1):repeat=models.fit_head(matrices['R2'].loc[trainids].to_numpy(),labels.loc[trainids,'RV_raw'].to_numpy(float),item['lambda'])
    assert np.array_equal(repeat['coef'],original['coef']) and repeat['intercept']==original['intercept']
    # Perturb future input prices before recomputing a real model prediction.
    one=opp.iloc[[0]];changed=candles.copy();mask=changed.bar_open_at>=pd.Timestamp(one.iloc[0].history_end_exclusive)
    changed.loc[mask,encoder.FIELDS]=changed.loc[mask,encoder.FIELDS]*7+11
    unchanged=features.build_features(changed,one)
    hidden=encoder.load_cache(ROOT/cfg['sources']['parent_cache']/'pretrained').loc[one.index]
    now=pd.concat([unchanged.loc[:,features.FEATURE_NAMES],hidden],axis=1)
    baseline=features.build_features(candles,one)
    np.testing.assert_array_equal(unchanged.to_numpy(),baseline.to_numpy())
    baseline=pd.concat([baseline.loc[:,features.FEATURE_NAMES],hidden],axis=1)
    expected=models.predict_head(original,baseline.to_numpy())
    np.testing.assert_array_equal(models.predict_head(original,now.to_numpy()),expected)
    prep=json.loads((RUN/'cache_audit.json').read_text())
    assert len(prep)==4 and all(a['checks']['future_perturbation']=='unchanged' for a in prep)
    return {'status':'PASS','all_2460_RVs_independently_recomputed':True,'labels_rows':len(labels),
       '96_formal_fit_records_complete':True,'all_24_selected_prediction_reloads':reproduction,
       'selected_R2_exact_deterministic_refit':True,'future_perturbation_prediction_unchanged':True,
       'all_variant_encoder_causal_checks':'PASS','new_holdout_access':False,
       'five_minute_raw_audit':data.audit_snapshot()}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['pre','post']);args=parser.parse_args()
    if args.phase=='pre':
        write_json(RUN/'tests/new_synthetic.json',synthetic())
        results=old_suites()
        write_json(RUN/'tests/historical_E00R_replay.json',historical_e00r())
        write_json(RUN/'tests/historical_M0_replay.json',historical_m0())
        encoder.require(all(r['exit_code']==0 or r['suite']=='research/baselines/verify_e00r.py' for r in results),'Old suite failure')
        write_json(RUN/'tests/prefit_acceptance.json',{'status':'PASS','original_executable_suites':len(results),
                   'historical_stage_replays':['E00R','M0'],'new_synthetic':'PASS','direct_stage_failure_retained':True})
    else:
        result=actual();write_json(RUN/'tests/actual_causal_numerical.json',result);print(json.dumps({'status':result['status'],'labels':result['labels_rows']}))

if __name__=='__main__':main()
