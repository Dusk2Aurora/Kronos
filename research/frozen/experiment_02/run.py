"""Strict development-only preparation, cache audit and registered risk fitting."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from research.frozen import encoder
from research.frozen.experiment_02 import features, models

RUN=ROOT/'research/runs/FROZEN_RISK_02_v1'
CONFIG=ROOT/'research/configs/frozen_risk_02_v1.yaml'
CUTOFF=pd.Timestamp('2026-04-01T00:00:00Z')
FAMILIES=('har','R1','R2','random_s17','random_s29','random_s43')

def write_json(path,value):
    encoder.immutable_json(path,value)

def config():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    reg=json.loads((RUN/'protocol_registration.json').read_text())
    encoder.require(encoder.sha(CONFIG)==reg['protocol_sha256'] and cfg['status']=='locked','Changed locked protocol')
    encoder.require(cfg['holdout']['read_5m'] is False and cfg['holdout']['predict'] is False,'Holdout denied')
    return cfg

def read_inputs():
    """Timestamp-filter each CSV record before parsing any market values."""
    cfg=config()
    bundle=ROOT/cfg['sources']['opportunity_bundle']
    snapshot=ROOT/cfg['sources']['one_hour_snapshot']
    bm=json.loads((bundle/'manifest.json').read_text())
    sm=json.loads((snapshot/'manifest.json').read_text())
    for name in ('opportunities.csv','split_index.csv'):
        encoder.require(encoder.sha(bundle/name)==bm['artifact_sha256'][name],'Changed source '+name)
    encoder.require(encoder.sha(snapshot/'manifest.json')==bm['sources']['snapshot_manifest_sha256'],'Snapshot manifest mismatch')
    encoder.require(encoder.sha(snapshot/'candles.csv')==bm['sources']['snapshot_csv_sha256']['candles.csv'],'Snapshot candle hash mismatch')
    splits=[]
    with (bundle/'split_index.csv').open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['fold_id'] in ('WF01','WF02','WF03','WF04') and row['included']=='True':
                encoder.require(pd.Timestamp(row['segment_end_exclusive'])<=CUTOFF,'Holdout split')
                encoder.require(pd.Timestamp(row['opportunity_id'])<CUTOFF,'Holdout id')
                splits.append(row)
    split=pd.DataFrame(splits)
    allowed=set(split.opportunity_id)
    opportunities=[]
    with (bundle/'opportunities.csv').open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            if pd.Timestamp(row['decision_boundary_at'])>=CUTOFF:
                continue
            if row['opportunity_id'] in allowed:
                opportunities.append(row)
    opp=pd.DataFrame(opportunities).set_index('opportunity_id',drop=False).sort_index()
    encoder.require(opp.index.is_unique and set(opp.index)==allowed,'Opportunity coverage')
    candle_rows=[]
    fields=['bar_open_at','bar_close_at','available_at','confirm']+encoder.FIELDS
    with (snapshot/'candles.csv').open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            if pd.Timestamp(row['bar_open_at'])>=CUTOFF:
                continue
            candle_rows.append({key:row[key] for key in fields})
    candles=pd.DataFrame(candle_rows)
    candles['bar_open_at']=pd.to_datetime(candles.bar_open_at,utc=True)
    for field in encoder.FIELDS: candles[field]=pd.to_numeric(candles[field],errors='raise')
    sources={'snapshot_manifest_sha256':encoder.sha(snapshot/'manifest.json'),
             'candles_sha256':encoder.sha(snapshot/'candles.csv'),
             'opportunities_sha256':encoder.sha(bundle/'opportunities.csv'),
             'split_index_sha256':encoder.sha(bundle/'split_index.csv'),
             'development_ids_sha256':encoder.digest(sorted(allowed)),
             'input_policy':'record timestamp filtered before numeric OHLCVA parsing; no profit/funding table access',
             'raw_scope_caveat':'prior snapshot includes holdout raw bytes; only file hash and DEV numerical rows used here'}
    return candles,opp,split,sources

def audit_caches(candles,opp,sources):
    cfg=config(); parent=ROOT/cfg['sources']['parent_cache']
    cache={}; audits=[]
    windows=[encoder.prepare_window(candles,row) for _,row in opp.iterrows()]
    old_protocol=ROOT/'research/configs/frozen_comparison_v1.yaml'
    for variant in ('pretrained','random_s17','random_s29','random_s43'):
        directory=parent/variant
        contract=json.loads((directory/'contract.json').read_text())
        encoder.require(contract['cutoff_exclusive']==CUTOFF.isoformat(),'Wrong cache cutoff')
        encoder.require(contract['protocol_sha256']==encoder.sha(old_protocol),'Changed cache protocol')
        encoder.require(contract['environment_lock_sha256']==encoder.sha(ROOT/'research/environment/requirements.lock.txt'),'Environment changed')
        for field in ('snapshot_manifest_sha256','candles_sha256','opportunities_sha256','split_index_sha256','development_ids_sha256'):
            encoder.require(contract['sources'][field]==sources[field],'Cache source mismatch '+field)
        for rel,expected in contract['code_sha256'].items():
            encoder.require(encoder.sha(ROOT/rel)==expected,'Cache source code changed '+rel)
        for field,expected in [('dtype','float32'),('normalization','per_window_numpy_float64_ddof0_eps1e-5_clip5_v1'),
                               ('pooling','decode_s1_context_last'),('model_revision',cfg['encoding']['model_revision']),
                               ('tokenizer_revision',cfg['encoding']['tokenizer_revision'])]:
            encoder.require(contract['model'][field]==expected,'Cache model contract '+field)
        matrix=encoder.load_cache(directory)
        encoder.require(matrix.index.tolist()==opp.index.tolist(),'Cache IDs changed')
        rebuilt=[encoder.digest({'contract':contract,'window':w[2]}) for w in windows]
        actual=[]
        for path in sorted(directory.glob('chunk_*.npz')):
            with np.load(path,allow_pickle=False) as part: actual.extend(part['keys'].tolist())
        encoder.require(rebuilt==actual,'Cache window identity mismatch')
        seed=None if variant=='pretrained' else int(variant.split('s')[-1])
        tokenizer,model,metadata=encoder.load_models('pretrained' if seed is None else 'random',seed,'cuda')
        encoder.require(metadata['model_state_sha256']==contract['model']['model_state_sha256']
                        and metadata['tokenizer_state_sha256']==contract['model']['tokenizer_state_sha256'],'Changed weights')
        sampled=opp.iloc[[0,len(opp)//2,len(opp)-1]]
        independent=encoder.encode_batch(tokenizer,model,[encoder.prepare_window(candles,row) for _,row in sampled.iterrows()])
        expected=matrix.loc[sampled.index].to_numpy()
        encoder.require(np.allclose(independent,expected,atol=1e-4,rtol=1e-4),'Cache independent mismatch')
        old_checks=encoder.verify_small_batch(tokenizer,model,candles,opp,1e-4,1e-4,8)
        audits.append({'variant':variant,'status':'PASS','rows':len(matrix),'windows_rebuilt':len(rebuilt),
                       'independent_ids':sampled.index.tolist(),'independent_max_abs':float(np.abs(independent-expected).max()),
                       'contract_sha256':encoder.sha(directory/'contract.json'),
                       'manifest_sha256':encoder.sha(directory/'manifest.json'),'checks':old_checks,
                       'model':metadata})
        cache[variant]=matrix
        print('cache audit PASS '+variant,flush=True)
        del tokenizer,model
    return cache,audits

def prepare():
    encoder.require(not (RUN/'features.csv').exists(),'Preparation already published')
    candles,opp,split,sources=read_inputs()
    cache,audits=audit_caches(candles,opp,sources)
    x=features.build_features(candles,opp)
    encoder.require(x.shape==(len(opp),35),'Feature dimensions')
    causal=[]
    for position in (0,len(opp)//2,len(opp)-1):
        one=opp.iloc[[position]]
        altered=candles.copy()
        mask=altered.bar_open_at>=pd.Timestamp(one.iloc[0].history_end_exclusive)
        altered.loc[mask,encoder.FIELDS]=altered.loc[mask,encoder.FIELDS]*7+11
        check=features.build_features(altered,one)
        encoder.require(np.array_equal(check.to_numpy(),x.loc[one.index].to_numpy()),'Future changed ordinary risk features')
        causal.append(str(one.index[0]))
    for name,frame in [('features.csv',x),('opportunities_development.csv',opp.reset_index(drop=True)),('split_development.csv',split)]:
        encoder.immutable_bytes(RUN/name,frame.to_csv(index=name=='features.csv').encode())
    write_json(RUN/'cache_audit.json',audits)
    write_json(RUN/'preparation_audit.json',{'status':'PASS','rows':len(opp),'feature_model_columns':list(features.FEATURE_NAMES),
               'HAR_columns':list(features.HAR_NAMES),'R0_columns':list(features.R0_NAMES),
               'future_perturbation_unchanged_ids':causal,'sources':sources,
               'no_new_holdout_read_encoded_predicted':True,'feature_sha256':encoder.sha(RUN/'features.csv')})
    print(json.dumps({'prepared':len(x),'features':len(features.FEATURE_NAMES)}),flush=True)

def provenance():
    paths=sorted((ROOT/'research/frozen/experiment_02').glob('*.py'))
    paths += [ROOT/'research/baselines/ordinary_features.py',ROOT/'research/baselines/ordinary_features_v2.py',
              ROOT/'research/frozen/encoder.py',ROOT/'model/kronos.py',ROOT/'model/module.py',CONFIG,
              ROOT/'research/environment/requirements.lock.txt',ROOT/'research/initialization/version_manifest.json']
    hashes={str(path.relative_to(ROOT)).replace('\\','/'):encoder.sha(path) for path in paths}
    for path in paths:
        target=RUN/'provenance/source'/path.relative_to(ROOT)
        encoder.immutable_bytes(target,path.read_bytes())
    dirty=[]
    lines=subprocess.check_output(['git','-c','core.quotePath=false','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,encoding='utf-8').splitlines()
    for line in lines:
        path=ROOT/line[3:]
        if path.is_file():dirty.append({'path':line[3:],'status':line[:2],'sha256':encoder.sha(path)})
    write_json(RUN/'provenance/formal_source_manifest.json',{'registered_protocol_sha256':encoder.sha(CONFIG),
               'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               'canonical_source_sha256':hashes,'dirty_files_and_sha256':dirty,
               'interpreter':sys.executable,'blas_threads':1,'created_at_utc':datetime.now(timezone.utc).isoformat()})

def get_matrices(x):
    result={'har':x.loc[:,features.HAR_NAMES], 'R1':x.loc[:,features.FEATURE_NAMES]}
    parent=ROOT/config()['sources']['parent_cache']
    for family,variant in [('R2','pretrained'),('random_s17','random_s17'),('random_s29','random_s29'),('random_s43','random_s43')]:
        hidden=encoder.load_cache(parent/variant)
        result[family]=pd.concat([result['R1'],hidden.loc[x.index]],axis=1)
    return result

def fit():
    from research.frozen.experiment_02.data import build_labels,audit_snapshot
    encoder.require(not (RUN/'predictions.csv').exists(),'Formal fitting already published')
    cfg=config()
    preparation=json.loads((RUN/'preparation_audit.json').read_text())
    pretests=json.loads((RUN/'tests/prefit_acceptance.json').read_text())
    encoder.require(preparation['status']=='PASS' and pretests['status']=='PASS','Preparation and causal tests must pass before fitting')
    encoder.require(encoder.sha(RUN/'features.csv')==preparation['feature_sha256'],'Prepared features changed')
    x=pd.read_csv(RUN/'features.csv',index_col='opportunity_id')
    opp=pd.read_csv(RUN/'opportunities_development.csv',dtype=str).set_index('opportunity_id',drop=False)
    split=pd.read_csv(RUN/'split_development.csv',dtype=str)
    # Collector publishes a complete trusted dataset; validate its immutable content first.
    ddir=RUN/'data_5m'
    dm=json.loads((ddir/'manifest.json').read_text())
    da=json.loads((ddir/'audit.json').read_text())
    encoder.require(da.get('status') in ('PASS','passed','complete'),'5min coverage audit blocks formal run')
    encoder.require(encoder.sha(ddir/'candles.csv')==dm['candles_sha256'],'5min CSV changed')
    write_json(RUN/'independent_data_prefit_audit.json',audit_snapshot(ddir))
    candles5=pd.read_csv(ddir/'candles.csv',dtype=str)
    labels=build_labels(candles5,opp,delay=60)
    if labels.index.name!='opportunity_id':labels=labels.set_index('opportunity_id',drop=False)
    encoder.require(set(labels.index)==set(opp.index),'All models require complete identical label coverage')
    encoder.immutable_bytes(RUN/'labels_rv.csv',labels.to_csv().encode())
    provenance()
    matrices=get_matrices(x)
    predictions=[]; all_candidates=[]; selected=[]; selectors=[]; clips=[]; split_audit=[]; failed=[]
    journal=RUN/'fit_events.jsonl'
    encoder.require(not journal.exists(),'Formal events already exist; do not hide partial attempt')
    def record(event):
        with journal.open('a',encoding='utf-8') as handle:handle.write(json.dumps(event,ensure_ascii=False)+'\n')
    def predframe(fold,role,family,ids,pred):
        result=opp.loc[ids,['opportunity_id','decision_boundary_at','decision_at','entry_at']].copy().reset_index(drop=True)
        result['fold_id']=fold;result['role']=role;result['family']=family
        result['RV_raw']=labels.loc[ids,'RV_raw'].to_numpy(dtype=float)
        result['RV_effective']=np.maximum(result.RV_raw,1e-12)
        result['prediction']=pred
        return result
    with threadpool_limits(limits=1):
      for fold in cfg['folds']:
        fid=fold['id'];groups={}
        for role in ('train','validation','test'):
            ids=sorted(split.loc[(split.fold_id==fid)&(split.role==role),'opportunity_id'])
            start,end=map(pd.Timestamp,fold[role])
            for ident in ids:
                lab=labels.loc[ident]
                encoder.require(start<=pd.Timestamp(opp.loc[ident,'decision_boundary_at'])<end
                                and pd.Timestamp(lab['label_end'])<end and pd.Timestamp(lab['labelable_at'])<end,'Label purge failed')
                encoder.require(pd.Timestamp(opp.loc[ident,'entry_at'])==pd.Timestamp(opp.loc[ident,'decision_boundary_at'])+pd.Timedelta(hours=1),'Entry shift')
            groups[role]=ids
        encoder.require(not any(set(groups[a])&set(groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]),'Fold role overlap')
        split_audit.append({'fold_id':fid,'role_counts':{k:len(v) for k,v in groups.items()},'purge':'PASS','disjoint':'PASS'})
        val_scores={}; fold_predictions={}
        for family,column in [('persistence','r0_persistence'),('ewma','r0_ewma')]:
            fold_predictions[family]={}
            for role,ids in groups.items():
                pred,stats=models.clip_predictions(x.loc[ids,column].to_numpy())
                fold_predictions[family][role]=pred
                predictions.append(predframe(fid,role,family,ids,pred))
                clips.append({'fold_id':fid,'role':role,'family':family,**stats})
            val_scores[family]=float(models.qlike(labels.loc[groups['validation'],'RV_raw'].to_numpy(float),fold_predictions[family]['validation']).mean())
        for family in FAMILIES:
            matrix=matrices[family];candidates=[]
            for lam in cfg['head']['lambda_candidates']:
                event={'fold_id':fid,'family':family,'lambda':lam,'status':'running'};record(event)
                before=time.perf_counter()
                try:
                    model=models.fit_head(matrix.loc[groups['train']].to_numpy(),labels.loc[groups['train'],'RV_raw'].to_numpy(float),lam)
                    model.update({'fold_id':fid,'family':family,'feature_columns':matrix.columns.tolist(),'train_count':len(groups['train'])})
                    write_json(RUN/'models'/f'{fid}_{family}_lambda_{lam}.json',model)
                    encoder.require(model['eligible'],'Registered convergence guard failed')
                    preds={role:models.predict_head(model,matrix.loc[ids].to_numpy()) for role,ids in groups.items()}
                    val_loss=float(models.qlike(labels.loc[groups['validation'],'RV_raw'].to_numpy(float),preds['validation']).mean())
                    candidates.append({'model':model,'validation_qlike':val_loss,'predictions':preds})
                    for role,ids in groups.items():
                        frame=predframe(fid,role,family,ids,preds[role]);frame['lambda']=lam
                        all_candidates.append(frame)
                    record({**event,'status':'completed','validation_qlike':val_loss,'seconds':time.perf_counter()-before,
                            'max_abs_gradient':model['solver']['max_abs_gradient']})
                except Exception as error:
                    record({**event,'status':'failed','error':repr(error)});failed.append({**event,'error':repr(error)})
                    raise
            chosen=models.select_candidate(candidates);model=chosen['model'];val_scores[family]=chosen['validation_qlike']
            fold_predictions[family]=chosen['predictions']
            selected.append({'fold_id':fid,'family':family,'lambda':model['lambda'],'validation_qlike':chosen['validation_qlike'],
                             'model_path':str((RUN/'models'/f'{fid}_{family}_lambda_{model["lambda"]}.json').relative_to(RUN)),
                             'dimension':len(model['coef']),'train_count':len(groups['train'])})
            for role,ids in groups.items():
                predictions.append(predframe(fid,role,family,ids,chosen['predictions'][role]))
                _,stats=models.predict_head_with_stats(model,matrix.loc[ids].to_numpy())
                clips.append({'fold_id':fid,'role':role,'family':family,**stats})
            print(f'{fid} {family} selected lambda={model["lambda"]}; {len(candidates)} candidates complete',flush=True)
        for name,pool in [('selected_R0',['persistence','ewma','har']),('selected_non_kronos',['persistence','ewma','har','R1'])]:
            chosen=min(pool,key=lambda family:(val_scores[family],pool.index(family)))
            selectors.append({'fold_id':fid,'selector':name,'chosen_family':chosen,'validation_qlike':val_scores[chosen],
                              'candidate_validation_qlike':{f:val_scores[f] for f in pool}})
            for role,ids in groups.items():predictions.append(predframe(fid,role,name,ids,fold_predictions[chosen][role]))
    encoder.immutable_bytes(RUN/'predictions.csv',pd.concat(predictions,ignore_index=True).to_csv(index=False).encode())
    encoder.immutable_bytes(RUN/'all_candidate_predictions.csv',pd.concat(all_candidates,ignore_index=True).to_csv(index=False).encode())
    write_json(RUN/'selected_models.json',selected);write_json(RUN/'selectors.json',selectors)
    write_json(RUN/'prediction_clipping.json',clips)
    write_json(RUN/'fit_audit.json',{'status':'PASS','formal_fits':96,'completed_fits':sum(1 for line in journal.read_text().splitlines() if json.loads(line)['status']=='completed'),
                                  'failed':failed,'split_audit':split_audit,'holdout_access':False,
                                  'data_manifest_sha256':encoder.sha(ddir/'manifest.json'),'labels_sha256':encoder.sha(RUN/'labels_rv.csv')})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','fit'])
    args=parser.parse_args()
    if args.phase=='prepare':prepare()
    else:fit()

if __name__=='__main__':main()
