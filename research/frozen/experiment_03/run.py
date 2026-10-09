"""Fixed DEV preparation and 28-candidate train/validation preview; no test access."""
from __future__ import annotations
import argparse
import io
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.frozen import encoder
from research.frozen.experiment_03.guard import ROOT,RUN,config,sha,new_json,now,require_scope
from research.frozen.experiment_03 import inputs,models

def immutable_json(path,value): encoder.immutable_json(path,value)
def immutable_csv(path,frame,index=True): encoder.immutable_bytes(path,frame.to_csv(index=index).encode())

def audit_and_extend_caches(candles,opp):
    cfg=config(); parent=ROOT/cfg['sources']['parent_cache']; result={}; audits=[]
    windows={ident:inputs.prepare_window(candles,row) for ident,row in opp.iterrows()}
    for variant in ('pretrained','random_s17','random_s29','random_s43'):
        directory=parent/variant; contract=json.loads((directory/'contract.json').read_text())
        matrix=encoder.load_cache(directory)
        for rel,expected in contract['code_sha256'].items():
            if sha(ROOT/rel)!=expected:raise ValueError('Parent cache code changed '+rel)
        if contract['environment_lock_sha256']!=sha(ROOT/'research/environment/requirements.lock.txt'):
            raise ValueError('Parent cache environment changed')
        snapshot=ROOT/cfg['sources']['one_hour_snapshot']
        if contract['sources']['candles_sha256']!=sha(snapshot/'candles.csv') or contract['sources']['snapshot_manifest_sha256']!=sha(snapshot/'manifest.json'):
            raise ValueError('Parent cache source snapshot changed')
        if contract['protocol_sha256']!=sha(ROOT/'research/configs/frozen_comparison_v1.yaml'):
            raise ValueError('Parent cache protocol changed')
        keys={}
        for path in sorted(directory.glob('chunk_*.npz')):
            with np.load(path,allow_pickle=False) as chunk:
                keys.update(zip(chunk['ids'].tolist(),chunk['keys'].tolist()))
        common=sorted(set(opp.index)&set(matrix.index)); missing=sorted(set(opp.index)-set(matrix.index))
        for ident in common:
            if encoder.digest({'contract':contract,'window':windows[ident][2]})!=keys[ident]:
                raise ValueError('Parent cache historical window identity mismatch '+ident)
        seed=None if variant=='pretrained' else int(variant.split('s')[-1])
        tokenizer,backbone,metadata=encoder.load_models('pretrained' if seed is None else 'random',seed,'cuda')
        for key in ('model_state_sha256','tokenizer_state_sha256','dtype','normalization','pooling','model_revision','tokenizer_revision','tf32','autocast'):
            if metadata[key]!=contract['model'][key]:raise ValueError('Pinned model cache mismatch '+key)
        sampled=[common[0],common[len(common)//2],common[-1]]
        independently=encoder.encode_batch(tokenizer,backbone,[windows[i] for i in sampled])
        expected=matrix.loc[sampled].to_numpy()
        if not np.allclose(independently,expected,atol=1e-4,rtol=1e-4):raise ValueError('Independent cache re-encode mismatch')
        checks=encoder.verify_small_batch(tokenizer,backbone,candles,opp,1e-4,1e-4,8)
        extended=matrix.loc[common].copy()
        if missing:
            batch=int(cfg['encoding']['batch_size'])
            added=np.concatenate([encoder.encode_batch(tokenizer,backbone,[windows[i] for i in missing[j:j+batch]]) for j in range(0,len(missing),batch)])
            repeated=np.concatenate([encoder.encode_batch(tokenizer,backbone,[windows[i] for i in missing[j:j+batch]]) for j in range(0,len(missing),batch)])
            if not np.allclose(added,repeated,atol=1e-4,rtol=1e-4):raise ValueError('New DEV cache repeat failure')
            extended=pd.concat([extended,pd.DataFrame(added,index=missing,columns=matrix.columns)])
        extended=extended.loc[opp.index]
        if encoder.state_sha(backbone)!=metadata['model_state_sha256'] or encoder.state_sha(tokenizer)!=metadata['tokenizer_state_sha256']:
            raise ValueError('Frozen weights changed')
        buffer=io.BytesIO();np.savez_compressed(buffer,ids=np.asarray(opp.index,dtype='U20'),features=extended.to_numpy(dtype=np.float32))
        output=RUN/'features'/f'{variant}_v2.npz';encoder.immutable_bytes(output,buffer.getvalue())
        audit={'variant':variant,'status':'PASS','parent_total_rows':len(matrix),'reused_rows':len(common),
               'new_DEV_ids':missing,'all_reused_window_keys_verified':len(common),'independent_ids':sampled,
               'max_reencode_abs':float(np.abs(independently.astype(float)-expected.astype(float)).max()),
               'parent_contract_sha256':sha(directory/'contract.json'),'parent_manifest_sha256':sha(directory/'manifest.json'),
               'output_sha256':sha(output),'model':metadata,'checks':checks}
        audits.append(audit);result[variant]=extended
        print('Cache verified '+variant+' reused='+str(len(common))+' new_DEV='+str(len(missing)),flush=True)
        del tokenizer,backbone
    immutable_json(RUN/'features/cache_audit_v2.json',audits)
    return result

def prepare():
    require_scope(config()['roles']['validation'][1],'A')
    if (RUN/'preflight/preparation.json').exists():raise ValueError('DEV preparation already published; retained immutable')
    candles=inputs.load_hourly(); five=inputs.load_development_5m()
    opp=pd.concat([inputs.opportunities(candles,'train'),inputs.opportunities(candles,'validation')]).set_index('opportunity_id',drop=False).sort_index()
    if not opp.index.is_unique:raise ValueError('Role overlap')
    labels=inputs.build_labels(five,opp)
    if labels.index.name!='opportunity_id':labels=labels.set_index('opportunity_id',drop=False)
    if set(labels.index)!=set(opp.index):raise ValueError('Uniform label coverage missing')
    labels=labels.loc[opp.index]
    feature=inputs.build_features(candles,opp)
    cache=audit_and_extend_caches(candles,opp)
    causal=[]
    for position in (0,len(opp)//2,len(opp)-1):
        one=opp.iloc[[position]]; altered=candles.copy()
        mask=altered.bar_open_at>=pd.Timestamp(one.iloc[0].history_end_exclusive)
        altered.loc[mask,encoder.FIELDS]=altered.loc[mask,encoder.FIELDS]*7+11
        other=inputs.build_features(altered,one)
        if not np.array_equal(other.to_numpy(),feature.loc[one.index].to_numpy()):raise ValueError('Future changed ordinary risk features')
        causal.append(str(one.index[0]))
    immutable_csv(RUN/'features/ordinary_risk.csv',feature)
    immutable_csv(RUN/'features/opportunities.csv',opp,index=False)
    immutable_csv(RUN/'labels/development.csv',labels,index=False)
    train=opp.index[opp.role=='train']; validation=opp.index[opp.role=='validation']
    raw=labels.loc[train,'RV_raw'].to_numpy(float); effective=np.maximum(raw,1e-12)
    thresholds={'train_q90':float(np.quantile(effective,.9)), 'train_q99':float(np.quantile(effective,.99)),
                'historical_ewma_quartiles':np.quantile(models.clip_predictions(feature.loc[train,'r0_ewma'].to_numpy())[0],[.25,.5,.75]).tolist(),
                'train_count':len(train),'fitted_from':'TRAIN only'}
    immutable_json(RUN/'calibration/training_thresholds.json',thresholds)
    immutable_json(RUN/'preflight/preparation.json',{'status':'PASS','train':len(train),'validation':len(validation),
        'total':len(opp),'model_feature_count':33,'R0_not_fitted':True,'future_perturbation_ids':causal,
        'label_floor_count':int((raw<1e-12).sum()),'input_policy':'Timestamp filtered before numerical parsing',
        'holdout_values_used':False,'feature_sha256':sha(RUN/'features/ordinary_risk.csv'),
        'labels_sha256':sha(RUN/'labels/development.csv'),'opportunities_sha256':sha(RUN/'features/opportunities.csv')})
    print('Prepared TRAIN='+str(len(train))+' VALID='+str(len(validation)),flush=True)

def read_prepared():
    prep=json.loads((RUN/'preflight/preparation.json').read_text())
    for name,key in [('features/ordinary_risk.csv','feature_sha256'),('labels/development.csv','labels_sha256'),('features/opportunities.csv','opportunities_sha256')]:
        if sha(RUN/name)!=prep[key]:raise ValueError('Prepared data changed '+name)
    x=pd.read_csv(RUN/'features/ordinary_risk.csv',index_col='opportunity_id',float_precision='round_trip')
    opp=pd.read_csv(RUN/'features/opportunities.csv',dtype=str).set_index('opportunity_id',drop=False)
    labels=pd.read_csv(RUN/'labels/development.csv',float_precision='round_trip').set_index('opportunity_id',drop=False)
    matrices={'har':x.loc[:,inputs.HAR_NAMES],'R1':x.loc[:,inputs.FEATURE_NAMES]}
    cache_audits={a['variant']:a for a in json.loads((RUN/'features/cache_audit_v2.json').read_text())}
    for family,variant in [('R2','pretrained'),('random_s17','random_s17'),('random_s29','random_s29'),('random_s43','random_s43')]:
        if sha(RUN/'features'/f'{variant}_v2.npz')!=cache_audits[variant]['output_sha256']:
            raise ValueError('Published hidden feature hash changed '+variant)
        with np.load(RUN/'features'/f'{variant}_v2.npz',allow_pickle=False) as part:
            if part['ids'].tolist()!=x.index.tolist():raise ValueError('Feature IDs changed')
            hidden=pd.DataFrame(part['features'],index=x.index,columns=[f'kronos_{i:03d}' for i in range(512)])
        matrices[family]=pd.concat([matrices['R1'],hidden],axis=1)
    return x,opp,labels,matrices

def fit(phase='A'):
    cfg=config();require_scope(cfg['roles']['validation'][1],'A')
    if phase not in ('A','B'):raise ValueError('Unknown fit phase')
    if phase=='B':require_scope(cfg['roles']['holdout'][1],'B',purpose='training')
    pretests=json.loads((RUN/'preflight/tests/new_suites.json').read_text())
    if pretests['status']!='PASS':raise ValueError('Synthetic preflight blocks fitting')
    destination=RUN/('preflight/preview' if phase=='A' else 'models/formal');destination.mkdir(parents=True,exist_ok=True)
    journal=destination/'fit_events.jsonl'
    if journal.exists():raise ValueError('Fit journal already exists: cannot conceal or rerun attempts')
    x,opp,labels,matrices=read_prepared()
    groups={role:opp.index[opp.role==role].tolist() for role in ('train','validation')}
    tr,va=groups['train'],groups['validation'];y=labels.RV_raw
    predictions=[];selected=[];all_candidates=[];clipping=[]
    def record(event):
        with journal.open('a',encoding='utf-8') as f:f.write(json.dumps({'at_utc':now(),'context':'DEV_preview' if phase=='A' else 'formal_training',**event})+'\n')
    def predrows(family,role,values,candidate=None):
        ids=groups[role];frame=opp.loc[ids,['opportunity_id','decision_at','decision_boundary_at','role']].copy()
        frame['RV_raw']=y.loc[ids].to_numpy(float);frame['prediction']=values;frame['family']=family
        if candidate is not None:frame['candidate']=candidate
        return frame.reset_index(drop=True)
    with threadpool_limits(limits=1):
      for family in cfg['models']['linear_families']:
        matrix=matrices[family];candidates=[]
        for lam in cfg['head']['lambda_candidates']:
            key=f'{family}_lambda_{lam}';record({'family':family,'lambda':lam,'status':'running'});start=time.perf_counter()
            try:
                model=models.fit_head(matrix.loc[tr].to_numpy(),y.loc[tr].to_numpy(float),lam)
                model.update(family=family,feature_columns=matrix.columns.tolist(),train_count=len(tr))
                immutable_json(destination/(key+'.json'),model)
                if not model['eligible']:raise ValueError('Solver convergence/gradient guard failure')
                preds={role:models.predict_head(model,matrix.loc[ids].to_numpy()) for role,ids in groups.items()}
                loss=float(models.qlike(y.loc[va].to_numpy(float),preds['validation']).mean())
                candidates.append({'model':model,'validation_qlike':loss,'predictions':preds,'model_path':str((destination/(key+'.json')).relative_to(RUN))})
                for role in groups:all_candidates.append(predrows(family,role,preds[role],key))
                record({'family':family,'lambda':lam,'status':'completed','validation_qlike':loss,'seconds':time.perf_counter()-start,'gradient':model['solver']['max_abs_gradient']})
            except Exception as error:
                record({'family':family,'lambda':lam,'status':'failed','error':repr(error)});raise
        chosen=models.select_candidate(candidates)
        selected.append({'family':family,'lambda':chosen['model']['lambda'],'validation_qlike':chosen['validation_qlike'],'model_path':chosen['model_path'],'dimension':matrix.shape[1]})
        for role,ids in groups.items():
            predictions.append(predrows(family,role,chosen['predictions'][role]))
            _,stats=models.predict_head_with_stats(chosen['model'],matrix.loc[ids].to_numpy());clipping.append({'family':family,'role':role,**stats})
        print('Selected '+family+' lambda='+str(chosen['model']['lambda']),flush=True)
      candidates=[]
      for leaves,minleaf in models.B2_GRID:
        key=f'B2_leaves_{leaves}_minleaf_{minleaf}';event={'family':'B2','num_leaves':leaves,'min_data_in_leaf':minleaf}
        record({**event,'status':'running'});start=time.perf_counter()
        try:
            m=models.fit_b2(matrices['R1'].loc[tr].to_numpy(),y.loc[tr].to_numpy(float),matrices['R1'].loc[va].to_numpy(),y.loc[va].to_numpy(float),leaves,minleaf)
            immutable_json(destination/(key+'.json'),m);encoder.immutable_bytes(destination/(key+'.txt'),m['model_text'].encode())
            immutable_csv(destination/(key+'_history.csv'),pd.DataFrame({'iteration':np.arange(1,m['training_iterations']+1),'train_QLIKE':m['history']['train']['raw_QLIKE'],'validation_QLIKE':m['history']['validation']['raw_QLIKE']}),False)
            preds={role:models.predict_b2(m,matrices['R1'].loc[ids].to_numpy()) for role,ids in groups.items()}
            candidates.append({'model':m,'validation_qlike':m['validation_qlike'],'predictions':preds,'model_path':str((destination/(key+'.json')).relative_to(RUN))})
            for role in groups:all_candidates.append(predrows('B2',role,preds[role],key))
            record({**event,'status':'completed','validation_qlike':m['validation_qlike'],'best_iteration':m['best_iteration'],'training_iterations':m['training_iterations'],'seconds':time.perf_counter()-start})
        except Exception as error:record({**event,'status':'failed','error':repr(error)});raise
      chosen=models.select_b2(candidates);m=chosen['model']
      selected.append({'family':'B2','num_leaves':m['num_leaves'],'min_data_in_leaf':m['min_data_in_leaf'],'best_iteration':m['best_iteration'],'validation_qlike':m['validation_qlike'],'model_path':chosen['model_path'],'dimension':33})
      for role,ids in groups.items():
          predictions.append(predrows('B2',role,chosen['predictions'][role]))
          _,stats=models.predict_b2_with_stats(m,matrices['R1'].loc[ids].to_numpy());clipping.append({'family':'B2','role':role,**stats})
      for family,col in [('persistence','r0_persistence'),('ewma','r0_ewma')]:
          for role,ids in groups.items():
              pred,stats=models.clip_predictions(x.loc[ids,col].to_numpy());predictions.append(predrows(family,role,pred));clipping.append({'family':family,'role':role,**stats})
      # Registered determinism replay exactly one selected R2 and one B2.
      replays=[]
      for family in ('R2','B2'):
          s=next(v for v in selected if v['family']==family); original=json.loads((RUN/s['model_path']).read_text())
          record({'family':family,'status':'determinism_running'})
          try:
              if family=='R2':
                  replay=models.fit_head(matrices[family].loc[tr].to_numpy(),y.loc[tr].to_numpy(float),s['lambda'])
                  keys=('mean','scale','coef','intercept','median','lambda','eligible','solver','prediction_clip','target_floor_count')
                  same=all(replay[k]==original[k] for k in keys)
              else:
                  replay=models.fit_b2(matrices['R1'].loc[tr].to_numpy(),y.loc[tr].to_numpy(float),matrices['R1'].loc[va].to_numpy(),y.loc[va].to_numpy(float),s['num_leaves'],s['min_data_in_leaf'])
                  same=replay==original
          except Exception as error:
              record({'family':family,'status':'determinism_failed','error':repr(error)});raise
          immutable_json(destination/(family+'_determinism_replay.json'),replay)
          record({'family':family,'status':'determinism_completed','exact_identical':same});replays.append({'family':family,'exact_identical':same})
          if not same:raise ValueError('Deterministic refit mismatch '+family)
    immutable_csv(destination/'all_candidate_predictions.csv',pd.concat(all_candidates),False)
    immutable_csv(destination/'selected_predictions.csv',pd.concat(predictions),False)
    immutable_json(destination/'selected_models.json',selected);immutable_json(destination/'clipping.json',clipping)
    immutable_json(destination/'fit_audit.json',{'status':'PASS','candidate_fits':28,'determinism_refits':2,'replays':replays,'failed':[],'train':len(tr),'validation':len(va),'holdout_values_used':False})
    if phase=='B':
        expected=json.loads((RUN/'preflight/preview/selected_models.json').read_text())
        for old,new in zip(expected,selected):
            if {k:v for k,v in old.items() if k!='model_path'}!={k:v for k,v in new.items() if k!='model_path'}:raise ValueError('Formal selection differs from sealed preview')
            if sha(RUN/old['model_path'])!=sha(RUN/new['model_path']):raise ValueError('Formal model differs from sealed preview')
        formal_files={s['model_path']:sha(RUN/s['model_path']) for s in selected}
        formal_files[str((destination/'selected_models.json').relative_to(RUN))]=sha(destination/'selected_models.json')
        immutable_json(destination/'preview_identity_check.json',{'status':'PASS','all_selected_models_exact':True,'files_sha256':formal_files})
    return selected

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','fit']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:fit()
if __name__=='__main__':main()
