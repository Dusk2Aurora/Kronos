"""User-authorized, finite post-hoc B5 supplement. Never reruns parent models."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch, yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from research.frozen.experiment_03_1 import b5_model as model
from research.frozen.experiment_03 import inputs as parent_inputs
from research.frozen import encoder
from research.delivery.frozen_risk_03_auxiliary import weighted_ranking
RUN=ROOT/'research/runs/FROZEN_RISK_03_1_B5_v1'
PARENT=ROOT/'research/runs/FROZEN_RISK_03_v1'
CONFIG=ROOT/'research/configs/frozen_risk_03_1_b5_v1.yaml'
CONFIG_SHA='4636e4f7cc5db9ab3a9e425b518c5f2674322a8d67293bed1ab358e8cd96741c'
OLD=['persistence','ewma','har','R1','R2','B2','random_s17','random_s29','random_s43']
SEEDS=[17,29,43]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def frame(path,value):
    assert not Path(path).exists(),str(path)
    value.to_csv(path,index=False,float_format='%.17g')
def loadcsv(path):return pd.read_csv(path,float_precision='round_trip')
def cfg():
    assert sha(CONFIG)==CONFIG_SHA,'Supplement protocol changed'
    assert read(PARENT/'authorization/formal_terminal.json')['state']=='CONSUMED'
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
def verify_parent():
    manifest=read(PARENT/'formal_delivery_manifest.json')
    assert manifest['status']=='COMPLETE'
    for name,digest in manifest['source_and_artifact_sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(PARENT/'formal_delivery_manifest.json')
def lock_check():
    latest=RUN/'preflight/pretrain_review_seal.json'
    seal=read(latest if latest.exists() else RUN/'preflight/source_seal.json')
    for name,digest in seal['files_sha256'].items():assert sha(ROOT/name)==digest,name
    return seal
def prepare():
    c=cfg();parent_hash=verify_parent()
    paths=['features/opportunities.csv','features/ordinary_risk.csv','features/holdout_ordinary_risk.csv','labels/development.csv','labels/holdout.csv','predictions/holdout.csv','calibration/sealed_calibration.json']
    sources={str((PARENT/p).relative_to(ROOT)).replace('\\','/'):sha(PARENT/p) for p in paths}
    src=ROOT/c['inputs']['source'];manifest=read(src.parent/'manifest.json')
    assert sha(src)==manifest['files']['candles.csv']
    sources[str(src.relative_to(ROOT)).replace('\\','/')]=sha(src)
    candles=loadcsv(src)
    for col in ['bar_open_at','bar_close_at','available_at']:candles[col]=pd.to_datetime(candles[col],utc=True)
    assert candles.bar_open_at.is_unique and candles.bar_open_at.is_monotonic_increasing
    assert candles.bar_open_at.max()<pd.Timestamp('2026-10-01',tz='UTC')
    assert (candles.bar_close_at==candles.bar_open_at+pd.Timedelta(hours=1)).all()
    oppdev=loadcsv(PARENT/'features/opportunities.csv')
    labelsdev=loadcsv(PARENT/'labels/development.csv');labelstest=loadcsv(PARENT/'labels/holdout.csv')
    xdev=loadcsv(PARENT/'features/ordinary_risk.csv').set_index('opportunity_id')
    xtest=loadcsv(PARENT/'features/holdout_ordinary_risk.csv').set_index('opportunity_id')
    x=pd.concat([xdev,xtest]);opps=pd.concat([oppdev,labelstest[oppdev.columns]],ignore_index=True)
    labels=pd.concat([labelsdev,labelstest],ignore_index=True).set_index('opportunity_id',drop=False)
    assert opps.opportunity_id.is_unique and x.index.is_unique and labels.index.is_unique
    assert len(opps)==3008 and len(labels)==3008
    assert set(opps.role)=={'train','validation','TEST_Q2','TEST_Q3'}
    assert opps.role.eq('TEST_Q2').sum()==272 and opps.role.eq('TEST_Q3').sum()==275
    features=list(parent_inputs.FEATURE_NAMES);assert len(features)==33
    tr=opps.role.eq('train').to_numpy();va=opps.role.eq('validation').to_numpy()
    assert (tr.sum(),va.sum())==(2369,92)
    raw=x.loc[opps.opportunity_id,features].to_numpy(np.float64)
    scaler=StandardScaler().fit(raw[tr]);ordinary=scaler.transform(raw).astype(np.float32)
    write(RUN/'inputs/train_scaler.json',{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'features':features,'fitted_role':'train','N':int(tr.sum())})
    times=pd.DatetimeIndex(candles.bar_open_at);windows=[];identities=[]
    for i,row in opps.iterrows():
        start=pd.Timestamp(row.history_start_at);end=pd.Timestamp(row.history_end_exclusive)
        roleend=pd.Timestamp(row.segment_end_exclusive)
        assert pd.Timestamp(row.label_end)<roleend and pd.Timestamp(row.labelable_at)<roleend
        assert pd.Timestamp(row.entry_at)==end+pd.Timedelta(hours=1)
        assert pd.Timestamp(row.label_end)==end+pd.Timedelta(hours=5)
        normalized,_,identity=parent_inputs._prepare_window(candles,row,times)
        t=pd.date_range(start,periods=256,freq='h')
        calendar=np.stack([t.minute,t.hour,t.dayofweek,t.day,t.month],axis=1).astype(np.float64)/np.asarray(c['inputs']['calendar_divisors'])
        windows.append(np.concatenate([normalized,calendar.astype(np.float32)],axis=1));identities.append(identity)
    windows=np.asarray(windows,np.float32)
    for pos in (0,len(opps)//2,len(opps)-1):
        row=opps.iloc[pos];changed=candles.copy();mask=changed.bar_open_at>=pd.Timestamp(row.history_end_exclusive)
        changed.loc[mask,encoder.FIELDS]=changed.loc[mask,encoder.FIELDS]*7+11
        altered,_,_=parent_inputs._prepare_window(changed,row,times)
        assert np.array_equal(altered,windows[pos,:,:6])
    for ident,row in labels.iterrows():
        prices=np.asarray(json.loads(row.label_prices_json),float);assert len(prices)==49
        computed=np.sum(np.diff(np.log(prices))**2)
        assert computed==row.RV_raw,ident
    with (RUN/'inputs/windows.npz').open('xb') as f:np.savez_compressed(f,windows=windows,ordinary=ordinary,ids=opps.opportunity_id.to_numpy(dtype='U20'),roles=opps.role.to_numpy(dtype='U12'))
    frame(RUN/'inputs/sample_identities.csv',opps);frame(RUN/'labels/reused_risk_labels.csv',labels.reset_index(drop=True))
    write(RUN/'inputs/window_identities.json',identities)
    write(RUN/'preflight/inputs_audit.json',{'status':'PASS','N':len(opps),'train':int(tr.sum()),'validation':int(va.sum()),'test':int((~(tr|va)).sum()),'source_sha256':sources,'parent_delivery_sha256':parent_hash,'all_labels_49_prices_48_returns_verified':True,'all_window_clock_availability_and_roles_checked':True,'future_perturbation_ids':opps.iloc[[0,len(opps)//2,len(opps)-1]].opportunity_id.tolist(),'normalization':'only6market channels normalized in historicalwindow;calendar fixedscales','post_hoc':True})
    scientific=[CONFIG,Path(__file__).resolve(),ROOT/'research/frozen/experiment_03_1/b5_model.py',ROOT/'tests/test_risk03_1_b5_model.py',ROOT/'research/frozen/experiment_03/inputs.py',ROOT/'research/frozen/encoder.py',ROOT/'research/delivery/frozen_risk_03_auxiliary.py']
    scientific+=list((RUN/'inputs').glob('*'))+list((RUN/'labels').glob('*'))+[RUN/'preflight/inputs_audit.json']
    write(RUN/'preflight/source_seal.json',{'status':'PASS','protocol_sha256':CONFIG_SHA,'files_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in scientific},'parent_delivery_sha256':parent_hash,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'locked_at_utc':now()})
    print('B5_INPUTS_PREPARED',len(opps),flush=True)
def arrays():
    with np.load(RUN/'inputs/windows.npz',allow_pickle=False) as f:parts={k:f[k] for k in f.files}
    labels=loadcsv(RUN/'labels/reused_risk_labels.csv').set_index('opportunity_id').loc[parts['ids']]
    return parts,labels
def train():
    c=cfg();lock_check()
    assert read(RUN/'preflight/model_tests.json')['status']=='PASS'
    write(RUN/'models/training_claim.json',{'state':'CLAIMED','at_utc':now(),'budget_fits':12,'protocol_sha256':CONFIG_SHA})
    a,labels=arrays();tr=a['roles']=='train';va=a['roles']=='validation'
    y=labels.RV_raw.to_numpy();candidate_scores=[];allpred=[];resources=[]
    journal=RUN/'models/fit_events.jsonl'
    def event(value):
        with journal.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(value,at_utc=now()))+'\n')
    for width in c['model']['width_candidates']:
        for wd in c['model']['weight_decay_candidates']:
            predictions=[];metadata=[]
            tag=f'w{width}_wd{wd}'
            for seed in SEEDS:
                ident=tag+f'_s{seed}';event({'candidate':ident,'state':'STARTED'})
                try:
                    p=model.fit_b5(a['windows'][tr],a['ordinary'][tr],y[tr],a['windows'][va],a['ordinary'][va],y[va],width,wd,seed,'cuda')
                    model.save_b5(p,RUN/'models'/f'{ident}.pt')
                    model2=model.load_b5(RUN/'models'/f'{ident}.pt',device='cuda')
                    pv=model.predict_b5(p,a['windows'][va],a['ordinary'][va])
                    reload=model.predict_b5(model2,a['windows'][va],a['ordinary'][va])
                    assert np.array_equal(pv['raw_score'],reload['raw_score'])
                    predictions.append(pv['prediction']);metadata.append(p['metadata'])
                    frame(RUN/'models'/f'history_{ident}.csv',pd.DataFrame(p['history']))
                    write(RUN/'models'/f'audit_{ident}.json',dict(status='PASS',metadata=p['metadata'],resources=p['resources'],reload_exact=True))
                    resources.append(dict(p['resources'],width=width,weight_decay=wd,seed=seed))
                    allpred.append(pd.DataFrame({'opportunity_id':a['ids'][va],'candidate':ident,'prediction_RV':pv['prediction']}))
                    event({'candidate':ident,'state':'COMPLETED','best_epoch':p['metadata']['best_epoch'],'epochs_run':p['resources']['epochs_run']})
                    print('B5_CANDIDATE',ident,'epoch',p['metadata']['best_epoch'],'VALID',p['metadata']['best_valid_raw_qlike'],flush=True)
                    del p,model2;torch.cuda.empty_cache()
                except BaseException as e:
                    event({'candidate':ident,'state':'FAILED','error':repr(e)});raise
            ensemble=np.mean(predictions,axis=0);loss=model.raw_qlike(y[va],ensemble)
            candidate_scores.append({'width':width,'weight_decay':wd,'validation_qlike':loss,'seed_metadata':metadata})
    selected=dict(min(candidate_scores,key=lambda v:(v['validation_qlike'],v['width'],-v['weight_decay'])))
    selected.update(structure_scores=candidate_scores,selection_basis='VALID-only mean RV ensemble')
    write(RUN/'models/selected.json',selected)
    frame(RUN/'models/all_candidate_validation_predictions.csv',pd.concat(allpred,ignore_index=True))
    write(RUN/'models/training_resources.json',resources)
    write(RUN/'models/candidate_audit.json',{'status':'PASS','candidate_fits':12,'max_possible_epochs':1440,'actual_total_epochs':sum(r['epochs_run'] for r in resources),'validation_only_selection':True,'selected_sha256':sha(RUN/'models/selected.json'),'no_fits_after_supplement_test_evaluation':True})
    print('B5_TRAINING_COMPLETE',selected['width'],selected['weight_decay'],flush=True)
def evaluate():
    model.configure(17,'cuda')
    c=cfg();lock_check();verify_parent();sel=read(RUN/'models/selected.json')
    assert read(RUN/'models/candidate_audit.json')['status']=='PASS'
    write(RUN/'predictions/evaluation_claim.json',{'state':'CLAIMED','post_hoc':True,'selected_sha256':sha(RUN/'models/selected.json'),'at_utc':now()})
    a,labels=arrays();tr=a['roles']=='train';test=~np.isin(a['roles'],['train','validation'])
    predicts={};train_predicts={};audits=[]
    for seed in SEEDS:
        ident=f"w{sel['width']}_wd{sel['weight_decay']}_s{seed}";p=model.load_b5(RUN/'models'/f'{ident}.pt',device='cuda')
        v=model.predict_b5(p,a['windows'][test],a['ordinary'][test]);tv=model.predict_b5(p,a['windows'][tr],a['ordinary'][tr])
        repeat=model.predict_b5(p,a['windows'][test],a['ordinary'][test]);assert np.array_equal(v['raw_score'],repeat['raw_score'])
        differences=[]
        for pos in (0,int(test.sum())//2,int(test.sum())-1):
            one=model.predict_b5(p,a['windows'][test][pos:pos+1],a['ordinary'][test][pos:pos+1]);differences.append(float(np.max(np.abs(one['raw_score']-v['raw_score'][pos:pos+1]))))
            assert np.allclose(one['raw_score'],v['raw_score'][pos:pos+1],atol=1e-4,rtol=1e-4)
        predicts[f'B5_s{seed}']=v['prediction'];train_predicts[f'B5_s{seed}']=tv['prediction']
        audits.append({'seed':seed,'status':'PASS','repeat_exact':True,'single_batch_max_abs':max(differences),'clip_counts':v['clip_counts']})
    predicts['B5']=np.mean([predicts[f'B5_s{s}'] for s in SEEDS],axis=0)
    train_predicts['B5']=np.mean([train_predicts[f'B5_s{s}'] for s in SEEDS],axis=0)
    write(RUN/'preflight/inference_audit.json',{'status':'PASS','checks':audits,'ensemble':'mean original-RV forecasts, not mean AUROC'})
    old=loadcsv(PARENT/'predictions/holdout.csv');assert old.opportunity_id.tolist()==a['ids'][test].tolist()
    predictions=old[['opportunity_id','decision_at','decision_boundary_at','role','entry_at']+OLD].copy()
    for family,value in predicts.items():predictions[family]=value
    frame(RUN/'predictions/test_predictions_only.csv',predictions)
    frame(RUN/'predictions/train_B5_predictions_only.csv',pd.DataFrame({'opportunity_id':a['ids'][tr],**train_predicts}))
    stats(predictions,labels.loc[a['ids'][test]],train_predicts,labels.loc[a['ids'][tr]],loadcsv(PARENT/'features/ordinary_risk.csv').set_index('opportunity_id').loc[a['ids'][tr],'r0_ewma'].to_numpy())
    print('B5_SUPPLEMENT_EVALUATION_COMPLETE',flush=True)
def stats(predictions,labels,train_predictions,trainlabels,train_ewma):
    seal=read(PARENT/'calibration/sealed_calibration.json');families=OLD+['B5','B5_s17','B5_s29','B5_s43'];rows=[];quarters=[];under=[];calibration=[]
    for quarter,role in [('2026Q2','TEST_Q2'),('2026Q3','TEST_Q3')]:
        mask=predictions.role.eq(role).to_numpy();raw=labels.RV_raw.to_numpy()[mask];effective=np.maximum(raw,1e-12);ewma=predictions.ewma.to_numpy()[mask]
        event=np.log((raw+1e-12)/(ewma+1e-12))>np.log(2)
        for family in families:
            p=predictions[family].to_numpy()[mask];ratio=effective/p;ql=np.log(p)+ratio;reg=ratio-np.log(ratio)-1;mse=(np.log(effective)-np.log(p))**2
            score=np.log((p+1e-12)/(ewma+1e-12));N=len(p);quarters.append(dict(quarter=quarter,family=family,N=N,positive=int(event.sum()),negative=int((~event).sum()),AUROC=float(roc_auc_score(event,score)),AP=float(average_precision_score(event,score)),raw_QLIKE=float(ql.mean()),QLIKE_Regret=float(reg.mean()),logRV_MSE=float(mse.mean())))
            rows.append(pd.DataFrame({'opportunity_id':predictions.opportunity_id.to_numpy()[mask],'decision_at':predictions.decision_at.to_numpy()[mask],'quarter':quarter,'family':family,'RV_raw':raw,'RV_effective':effective,'EWMA_RV':ewma,'prediction_RV':p,'surprise_score':score,'surprise_event':event,'raw_QLIKE':ql,'QLIKE_Regret':reg,'logRV_MSE':mse}))
            for subset,q in [('q90',.9),('q99',.99)]:
                cutoff=seal['thresholds'][f'train_effective_RV_q{int(q*100)}'];s=effective>cutoff
                under.append(dict(quarter=quarter,family=family,subset=subset,count=int(s.sum()),underprediction_count=int((s&(p/effective<.5)).sum()),underprediction_fraction=float((p[s]/effective[s]<.5).mean()) if s.any() else np.nan,mean_prediction_RV=float(p[s].mean()) if s.any() else np.nan,mean_observed_RV=float(raw[s].mean()) if s.any() else np.nan))
            if family.startswith('B5'):
                for kind,trainvalues,testvalues,quantiles in [('prediction_decile',train_predictions[family],p,np.arange(.1,1,.1)),('score_quartile',np.log((train_predictions[family]+1e-12)/(train_ewma+1e-12)),score,[.25,.5,.75])]:
                    cuts=np.unique(np.quantile(trainvalues,quantiles));bins=np.searchsorted(cuts,testvalues,side='right')
                    for binid in range(len(cuts)+1):
                        s=bins==binid;calibration.append(dict(quarter=quarter,family=family,bin_type=kind,bin_id=binid,count=int(s.sum()),mean_prediction_RV=float(p[s].mean()) if s.any() else np.nan,mean_observed_RV=float(raw[s].mean()) if s.any() else np.nan,event_fraction=float(event[s].mean()) if s.any() else np.nan))
    qm=pd.DataFrame(quarters);rowmetrics=pd.concat(rows,ignore_index=True);frame(RUN/'metrics/quarter_metrics.csv',qm);frame(RUN/'metrics/row_metrics.csv',rowmetrics);frame(RUN/'metrics/underprediction.csv',pd.DataFrame(under));frame(RUN/'metrics/calibration.csv',pd.DataFrame(calibration))
    contrasts={'R2_minus_B5':('R2','B5'),'B5_minus_R1':('B5','R1'),'B5_minus_B2':('B5','B2')};comparisons={}
    for block in (7,3,14):
        data={};draws={};weightpath=PARENT/'bootstrap'/f'multiplicities_block{block}.npz'
        with np.load(weightpath,allow_pickle=False) as weights:
            for quarter in ('2026Q2','2026Q3'):
                role='TEST_'+quarter[-2:];days=pd.to_datetime(weights[role+'_utc_days'],utc=True);count=weights[role+'_counts'];aligned={f:rowmetrics[(rowmetrics.quarter==quarter)&(rowmetrics.family==f)].sort_values('decision_at') for f in ['R1','R2','B2','B5']}
                ref=aligned['R2'];ids=days.get_indexer(pd.to_datetime(ref.decision_at,utc=True).dt.floor('D'));assert (ids>=0).all()
                metrics={f:weighted_ranking(ref.surprise_event.to_numpy(bool),g.surprise_score.to_numpy(),count[:,ids]) for f,g in aligned.items()}
                for name,(first,second) in contrasts.items():draws[quarter+'_'+name]=metrics[first]['AUROC']-metrics[second]['AUROC']
        for name,(first,second) in contrasts.items():
            values=(draws['2026Q2_'+name]+draws['2026Q3_'+name])/2;draws['equal_quarter_'+name]=values;valid=np.isfinite(values)
            deltas=[float(qm[(qm.quarter==q)&(qm.family==first)].AUROC.iloc[0]-qm[(qm.quarter==q)&(qm.family==second)].AUROC.iloc[0]) for q in ('2026Q2','2026Q3')]
            ci=np.quantile(values[valid],[.025,.975]);data[name]={'quarter_deltas':deltas,'mean_delta':float(np.mean(deltas)),'ci_lower':float(ci[0]),'ci_upper':float(ci[1]),'valid_draws':int(valid.sum()),'invalid_fraction':float((~valid).mean())}
        comparisons[str(block)]=data;frame(RUN/'bootstrap'/f'draws_block{block}.csv',pd.DataFrame(draws))
    write(RUN/'metrics/summary.json',{'status':'POST_HOC_B5_SUPPLEMENT','evidence_class':'POST_HOC_EXPLORATORY_NOT_NEW_INDEPENDENT_CONFIRMATION','original_primary_status':read(PARENT/'metrics/formal_result.json')['status'],'quarter_metrics':[{'quarter':q,'N':int(qm[qm.quarter==q].N.iloc[0]),'positive':int(qm[qm.quarter==q].positive.iloc[0]),'negative':int(qm[qm.quarter==q].negative.iloc[0])} for q in ('2026Q2','2026Q3')],'comparisons':comparisons,'selected_structure':{k:read(RUN/'models/selected.json')[k] for k in ('width','weight_decay')},'high_rv':{'q90_threshold':seal['thresholds']['train_effective_RV_q90'],'q99_threshold':seal['thresholds']['train_effective_RV_q99']},'bootstrap_new_random_draws':0,'threshold_refit_on_test':False})
def benchmark():
    model.configure(17,'cuda')
    cfg();lock_check();sel=read(RUN/'models/selected.json');a,_=arrays();idx=np.flatnonzero(a['roles']=='train')[:8];x=a['windows'][idx];ordinary=a['ordinary'][idx];result=[]
    def bench(family,fun,parameters,checkpoint_bytes):
        for _ in range(3):fun()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        for _ in range(20):fun()
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t)/20
        result.append({'family':family,'parameters':parameters,'seconds_per_batch8':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint_bytes':checkpoint_bytes})
    loaded=[]
    for seed in SEEDS:
        path=RUN/'models'/f"w{sel['width']}_wd{sel['weight_decay']}_s{seed}.pt";p=model.load_b5(path,device='cuda');loaded.append(p)
        bench(f'B5_s{seed}',lambda:model.predict_b5(p,x,ordinary,batch_size=8),sum(t.numel() for t in p['model'].parameters()),path.stat().st_size)
    bench('B5',lambda:np.mean([model.predict_b5(p,x,ordinary,batch_size=8)['prediction'] for p in loaded],axis=0),sum(sum(t.numel() for t in p['model'].parameters()) for p in loaded),sum((RUN/'models'/f"w{sel['width']}_wd{sel['weight_decay']}_s{seed}.pt").stat().st_size for seed in SEEDS))
    del p,loaded;torch.cuda.empty_cache()
    tokenizer,backbone,meta=encoder.load_models('pretrained',None,'cuda')
    sample=loadcsv(RUN/'inputs/sample_identities.csv').iloc[idx];windows=[]
    for pos,row in sample.iterrows():
        stamps=encoder.calc_time_stamps(pd.Series(pd.date_range(pd.Timestamp(row.history_start_at),periods=256,freq='h'))).to_numpy(np.float32)
        windows.append((a['windows'][pos,:,:6],stamps,{}))
    bench('Kronos_tokenizer_backbone',lambda:encoder.encode_batch(tokenizer,backbone,windows),sum(p.numel() for mod in (tokenizer,backbone) for p in mod.parameters()),sum((ROOT/e['local_cache']/'model.safetensors').stat().st_size for e in read(ROOT/'research/initialization/version_manifest.json')['weights'].values()))
    training=read(RUN/'models/training_resources.json')
    write(RUN/'provenance/resources.json',{'status':'PASS','candidate_training_seconds_total':sum(r['elapsed_seconds'] for r in training),'training':training,'inference':result,'same_training_window_ids':sample.opportunity_id.tolist(),'gpu':torch.cuda.get_device_name(0),'resource_comparability_limit':'B5 deliberately smaller; pretraining compute/data and model capacity differ; inference benchmark excludes original guard metadata overhead and R2 linear head; peak allocated includes resident models and differs from incremental peak','torch':torch.__version__,'repetitions':20,'warmups':3})
    print('B5_RESOURCES_MEASURED',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train','evaluate','benchmark']);args=p.parse_args()
    try:globals()[args.command]()
    except BaseException as e:
        path=RUN/'provenance'/f'failure_{args.command}_{time.time_ns()}.json'
        write(path,{'state':'FAILED','command':args.command,'error':repr(e),'traceback':traceback.format_exc(),'at_utc':now()});raise
