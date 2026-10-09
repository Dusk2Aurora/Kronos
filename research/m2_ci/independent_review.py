"""Post-run independent DEV numerical review; never train, select anew or open tests."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/runs/M2_CONDITIONAL_INCREMENT_01'
CONFIG = ROOT / 'research/m2_ci/config.yaml'
EPS = 1e-12
BASE = ['B5','B5_s17','B5_s29','B5_s43','R1','R2','B2','har','ewma','persistence','constant_RV']
CONTRASTS = {'B5_minus_mix_R2':('B5','mix_R2'), 'B5_minus_mix_R1':('B5','mix_R1'),
    'B5_minus_mix_B2':('B5','mix_B2'), 'mix_R1_minus_mix_R2':('mix_R1','mix_R2'),
    'mix_B2_minus_mix_R2':('mix_B2','mix_R2'), 'R2_minus_B5':('R2','B5')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def csv(path): return pd.read_csv(path, float_precision='round_trip')


def loss(y,p):
    y = np.maximum(np.asarray(y,float),EPS); p = np.asarray(p,float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(p<=0):
        raise ValueError('Invalid independent loss arrays')
    return {'raw_QLIKE':np.log(p)+y/p, 'QLIKE_Regret':y/p-np.log(y/p)-1,
            'logRV_MSE':(np.log(y)-np.log(p))**2}


def rank(event, score):
    # Tied-score grouped rank arithmetic, independent of sklearn production API.
    event=np.asarray(event,bool); score=np.asarray(score,float)
    pos=event.sum(); neg=(~event).sum()
    if not pos or not neg: return np.nan,np.nan
    order=np.argsort(score,kind='stable'); s=score[order]; y=event[order]
    begin=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    n=np.diff(np.r_[begin,len(s)]); p=np.add.reduceat(y.astype(int),begin); q=n-p
    auc=np.sum(p*(np.cumsum(q)-q/2))/(pos*neg)
    tp=np.cumsum(p[::-1]); total=np.cumsum(n[::-1])
    ap=np.sum(p[::-1]/pos*tp/total)
    return float(auc),float(ap)


def roles(meta,start,end):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    return np.flatnonzero(((pd.to_datetime(meta.decision_boundary_at,utc=True)>=start)&
        (pd.to_datetime(meta.decision_boundary_at,utc=True)<end)&
        (pd.to_datetime(meta.label_end,utc=True)<end)&
        (pd.to_datetime(meta.labelable_at,utc=True)<end)).to_numpy())


def xmatrix(features,family,ix):
    if family=='har': return features['har'][ix]
    if family=='R2': return np.concatenate((features['ordinary_raw'][ix],features['hidden'][ix]),axis=1).astype(float)
    return features['ordinary_raw'][ix]


def day_counts(days, block, seed):
    rng=np.random.default_rng(seed); ix=np.empty((5000,days),np.int32)
    ix[:,0]=rng.integers(days,size=5000)
    for j in range(1,days):
        restart=rng.random(5000)<1/block; fresh=rng.integers(days,size=5000)
        ix[:,j]=np.where(restart,fresh,(ix[:,j-1]+1)%days)
    counts=np.zeros((5000,days),np.uint16)
    np.add.at(counts,(np.arange(5000)[:,None],ix),1)
    return counts


class Audit:
    def __init__(self): self.checks=[]; self.files={}
    def ok(self,name,condition,detail=None):
        self.checks.append({'name':name,'status':'PASS' if bool(condition) else 'FAIL', 'detail':detail})
        if not condition: raise AssertionError(name)
    def close(self,name,actual,expected,rtol=1e-10,atol=1e-12):
        actual=np.asarray(actual); expected=np.asarray(expected)
        self.ok(name,actual.shape==expected.shape and np.allclose(actual,expected,rtol=rtol,atol=atol,equal_nan=True))
    def bind(self,path,expected=None):
        path=Path(path); digest=sha(path)
        key=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
        self.files[key]=digest
        if expected is not None: self.ok('byte_hash:'+key,digest==expected)
        return digest


def predict(a,features,ix,selected,scaler,device):
    # Only feature arrays and locked model artifacts enter prediction APIs.
    from research.frozen.experiment_03 import models
    from research.frozen.experiment_03_1 import b5_model as b5
    output={}
    for family in ('har','R1','R2','B2'):
        model=read(selected[family]); a.bind(selected[family])
        output[family]=(models.predict_b2 if family=='B2' else models.predict_head)(model,xmatrix(features,family,ix))
    ordinary=((features['ordinary_raw'][ix]-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])).astype(np.float32)
    for seed,path in selected['B5'].items():
        a.bind(path); payload=b5.load_b5(path,device=device)
        output['B5_s'+seed]=b5.predict_b5(payload,features['windows'][ix],ordinary,device=device)['prediction']
        del payload
    output['B5']=np.mean([output['B5_s'+str(s)] for s in (17,29,43)],axis=0,dtype=float)
    output['ewma']=np.clip(features['r0'][ix,1],EPS,1)
    output['persistence']=np.clip(features['r0'][ix,0],EPS,1)
    output['constant_RV']=np.full(len(ix),scaler['constant_RV'])
    return output


@threadpool_limits.wrap(limits=1)
def review(a):
    from research.frozen.experiment_03 import models
    from research.frozen.experiment_03_1 import b5_model as b5
    import torch
    torch.set_num_threads(1)
    a.ok('locked_inference_runtime',os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
        and torch.are_deterministic_algorithms_enabled() and torch.get_num_threads()==1)
    c=yaml.safe_load(CONFIG.read_text(encoding='utf-8')); protocol=a.bind(CONFIG)
    registry=read(ROOT/'research/registry/M2_CONDITIONAL_INCREMENT_01.json')
    a.ok('protocol_registry',registry['protocol_sha256']==protocol)
    a.ok('future_unapproved',c['future_draft']['approved'] is False and registry['future_formal_test_approved'] is False)
    a.ok('old_holdout_consumed',read(ROOT/'research/researchstate.json')['holdout']['status']=='CONSUMED')
    seal=read(RUN/'preflight/source_seal.json'); a.bind(RUN/'preflight/source_seal.json')
    a.ok('training_source_seal',seal['status']=='PASS' and seal['protocol_sha256']==protocol)
    for path,digest in seal['files_sha256'].items(): a.bind(ROOT/path,digest)
    result_seal_path=RUN/'provenance/training_result_seal.json'; result_seal=read(result_seal_path)
    result_seal_sha=a.bind(result_seal_path)
    a.ok('training_result_seal_identity',result_seal['status']=='PASS' and result_seal['protocol_sha256']==protocol
        and result_seal['source_seal_sha256']==sha(RUN/'preflight/source_seal.json')
        and result_seal['alpha_selection_not_started'] is True and result_seal['new_fits']==0)
    result_files=result_seal['files_sha256']; a.ok('training_result_seal_nonempty',bool(result_files))
    required=[p for p in (RUN/'models').rglob('*') if p.is_file()]+[RUN/'predictions/oof_predictions.csv',RUN/'preflight/source_seal.json']
    for path in required: a.ok('training_result_seal_coverage:'+path.relative_to(ROOT).as_posix(),path.relative_to(ROOT).as_posix() in result_files)
    for relative,digest in result_files.items():
        path=(ROOT/relative).resolve()
        a.ok('training_result_seal_scope:'+relative,path.is_relative_to(RUN))
        a.bind(path,digest)
    claim_path=RUN/'provenance/review_runtime_manifest.json'; claim=read(claim_path); a.bind(claim_path)
    a.ok('review_wrapper_claim_identity',claim['status']=='RUNNING' and claim['action']=='review'
        and Path(claim['run_directory']).resolve()==RUN.resolve() and claim['protocol_sha256']==protocol
        and claim['source_seal_sha256']==sha(RUN/'preflight/source_seal.json')
        and claim['training_result_seal_sha256']==result_seal_sha
        and claim['training_result_files_sha256']==result_files
        and claim['fit_events_sha256']==sha(RUN/'models/fit_events.jsonl')
        and claim['selected_weights_sha256']==sha(RUN/'fusion/selected_weights.json')
        and claim['wrapper_sha256']==sha(ROOT/'research/m2_ci/execute_locked.py')
        and claim['new_fitting'] is False and claim['new_selection'] is False and claim['automatic_retry'] is False)
    flags=claim['runtime']
    a.ok('review_wrapper_actual_runtime',flags['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        and flags['cuda_matmul_allow_tf32'] is False and flags['cudnn_allow_tf32'] is False
        and flags['cudnn_benchmark'] is False and flags['cudnn_deterministic'] is True
        and flags['deterministic_algorithms'] is True and flags['CPU_threads']==1
        and flags['tensor_dtype']=='float32' and flags['selection_and_loss_dtype']=='float64'
        and flags['autocast'] is False and all(p['num_threads']==1 for p in flags['threadpools']))
    with np.load(RUN/'inputs/features.npz',allow_pickle=False) as z: features={k:z[k].copy() for k in z.files}
    meta=csv(RUN/'inputs/metadata.csv'); labels=csv(RUN/'labels/development.csv')
    a.ok('feature_schema',set(features)=={'ids','windows','ordinary_raw','hidden','har','r0'})
    a.ok('dev_identity',len(meta)==2461 and meta.opportunity_id.tolist()==features['ids'].tolist()==labels.opportunity_id.tolist())
    a.ok('dev_boundary',(pd.to_datetime(meta.labelable_at,utc=True)<pd.Timestamp('2026-04-01',tz='UTC')).all())
    a.ok('feature_dimensions',features['ordinary_raw'].shape==(2461,33) and features['hidden'].shape==(2461,512) and features['windows'].shape==(2461,256,11))
    for key,v in features.items():
        if key!='ids': a.ok('finite_features:'+key,np.isfinite(v).all())
    fold_index=csv(RUN/'folds/index.csv'); thresholds=read(RUN/'folds/thresholds.json')
    oo=csv(RUN/'predictions/oof_predictions.csv'); va=csv(RUN/'predictions/validation_predictions.csv')
    for name,frame,n in [('OOF',oo,1267),('VALID',va,92)]:
        a.ok('prediction_identity:'+name,len(frame)==n and frame.opportunity_id.is_unique and not any(x in frame for x in ('RV_raw','labelable_at','label_end')))
    actual_epochs=actual_iterations=0; names=[]; fold_defs={}; fit_roles={}
    for fc in c['roles']['folds']:
        fid=fc['id']; folder=RUN/'models'/fid
        rr={'fit':roles(meta,c['roles']['train_start'],fc['fit_end']),
            'inner_validation':roles(meta,fc['fit_end'],fc['evaluation_start']),
            'evaluation':roles(meta,fc['evaluation_start'],fc['evaluation_end'])}
        fit_roles[fid]=rr['fit']; fold_defs[fid]=(fc['evaluation_start'],fc['evaluation_end'])
        a.ok('role_counts:'+fid,[len(rr[k]) for k in rr]==fc['expected_counts'])
        for role,ix in rr.items(): a.close('fold_index:'+fid+':'+role,fold_index[(fold_index.fold==fid)&(fold_index.role==role)].row,ix,0,0)
        tr,iv,ev=(rr[k] for k in rr); y=np.maximum(labels.RV_raw.to_numpy()[tr],EPS)
        rebuilt=StandardScaler().fit(features['ordinary_raw'][tr]); scaling=read(folder/'scaling_audit.json')
        a.close('scaler_mean:'+fid,scaling['mean'],rebuilt.mean_); a.close('scaler_scale:'+fid,scaling['scale'],rebuilt.scale_)
        a.close('target_median:'+fid,scaling['target_median'],np.median(y)); a.close('constant_RV:'+fid,scaling['constant_RV'],np.clip(y.mean(),EPS,1))
        a.ok('scaler_role:'+fid,scaling['fit_count']==len(tr) and scaling['fitted_role']=='fit' and scaling['global_scaler_used'] is False)
        for key,value in [('RV_q90',np.quantile(y,.9)),('RV_q99',np.quantile(y,.99)),('EWMA_quartiles',np.quantile(features['r0'][tr,1],[.25,.5,.75]))]: a.close('prepare_threshold:'+fid+':'+key,thresholds[fid][key],value)
        selected_record=read(folder/'selected_models.json'); selected=selected_record['selected']
        a.ok('selection_role:'+fid,selected_record['selection_role']=='inner_validation' and selected_record['no_refit'] is True)
        inner=csv(folder/'inner_validation_predictions.csv'); records={v['candidate']:v for v in selected_record['candidates']}
        for family in ('har','R1','R2','B2'):
            candidates=[]
            paths=list(folder.glob(f'{family}_lambda_*.json')) if family!='B2' else list(folder.glob('B2_leaves_*_minleaf_*.json'))
            a.ok('candidate_count:'+fid+':'+family,len(paths)==4)
            for path in paths:
                name=fid+'/'+path.stem; names.append(name); model=read(path)
                a.close('candidate_FIT_median:'+name,model['median'],np.median(y))
                if family!='B2':
                    candidate_scaler=StandardScaler().fit(xmatrix(features,family,tr))
                    a.close('candidate_FIT_mean:'+name,model['mean'],candidate_scaler.mean_)
                    a.close('candidate_FIT_scale:'+name,model['scale'],candidate_scaler.scale_)
                else:
                    a.close('B2_FIT_target_scale:'+name,model['scale'],np.median(y))
                pred=(models.predict_b2 if family=='B2' else models.predict_head)(model,xmatrix(features,family,iv))
                saved=inner[inner.candidate.eq(name)]
                a.ok('inner_ids:'+name,saved.opportunity_id.tolist()==features['ids'][iv].tolist())
                a.close('inner_predictions:'+name,saved.prediction,pred)
                score=float(loss(labels.RV_raw.to_numpy()[iv],pred)['raw_QLIKE'].mean())
                a.close('inner_selection_score:'+name,records[name]['inner_validation_qlike'],score)
                a.ok('eligible:'+name,model['eligible'] is True)
                if family=='B2':
                    actual_iterations+=model['training_iterations']; a.close('B2_saved_score:'+name,model['validation_qlike'],score)
                    history=np.asarray(model['history']['validation']['raw_QLIKE'])
                    a.ok('B2_best_iteration:'+name,model['best_iteration']==int(np.argmin(history))+1 and len(history)==model['training_iterations'])
                    a.close('B2_best_iteration_score:'+name,history[model['best_iteration']-1],score)
                    key=(score,model['num_leaves'],-model['min_data_in_leaf'])
                else: key=(score,-model['lambda'])
                candidates.append((key,path))
            chosen=min(candidates,key=lambda v:v[0])[1]
            a.ok('selected_inner_min:'+fid+':'+family,Path(selected[family]).resolve()==chosen.resolve())
        ordinary=rebuilt.transform(features['ordinary_raw'][iv]).astype(np.float32)
        for seed in (17,29,43):
            name=fid+f'/B5_s{seed}'; names.append(name); saved=read(folder/f'B5_s{seed}.json')
            payload=b5.load_b5(selected['B5'][str(seed)],device=c['models']['B5_device']); md=payload['metadata']
            a.ok('B5_checkpoint_metadata:'+name,md==saved['metadata'] and md['width']==32 and md['weight_decay']==.001 and md['seed']==seed)
            pred=b5.predict_b5(payload,features['windows'][iv],ordinary,device=c['models']['B5_device'])['prediction']
            a.close('B5_inner_reload:'+name,inner[inner.candidate.eq(name)].prediction,pred)
            score=float(loss(labels.RV_raw.to_numpy()[iv],pred)['raw_QLIKE'].mean())
            history=saved['history']; best=min(history,key=lambda v:(v['valid_raw_qlike'],v['epoch']))
            a.ok('B5_best_epoch:'+name,md['best_epoch']==best['epoch'] and len(history)==saved['resources']['epochs_run'] and len(history)<=120)
            a.close('B5_best_score:'+name,md['best_valid_raw_qlike'],score)
            a.close('B5_target_scale:'+name,md['target_scale'],np.median(y))
            actual_epochs+=len(history); del payload
        expected=predict(a,features,ev,selected,scaling,c['models']['B5_device']); section=oo[oo.fold_id.eq(fid)]
        a.ok('outer_ids:'+fid,section.opportunity_id.tolist()==features['ids'][ev].tolist())
        for family,pred in expected.items(): a.close('outer_prediction:'+fid+':'+family,section[family],pred)
    events=[json.loads(line) for line in (RUN/'models/fit_events.jsonl').read_text().splitlines()]
    a.ok('95_exact_fit_lifecycle',len(events)==190 and len(names)==95 and len(set(names))==95)
    for number,(started,ended) in enumerate(zip(events[::2],events[1::2]),1):
        a.ok('fit_pair:'+str(number),started['state']=='STARTED' and ended['state']=='COMPLETED' and started['candidate']==ended['candidate'] and started['candidate'] in names and started['fit_number']==ended['fit_number']==number and pd.Timestamp(started['at_utc'])<=pd.Timestamp(ended['at_utc']))
    a.ok('fit_journal_exact_candidates',set(e['candidate'] for e in events[::2])==set(names))
    audit=read(RUN/'models/candidate_audit.json'); resources=audit['resources']
    a.ok('actual_fit_budget',audit['fits']==95 and audit['linear_fits']==60 and audit['B2_fits']==20 and audit['B5_fits']==15 and len(resources)==95)
    a.ok('actual_iterations_epochs',actual_epochs==audit['actual_B5_epochs']<=1800 and actual_iterations==audit['actual_B2_iterations']<=6000)
    for row in resources:
        fid,name=row['candidate'].split('/'); path=RUN/'models'/fid/(name+('.pt' if row['family']=='B5' else '.json'))
        a.ok('candidate_resource:'+row['candidate'],row['elapsed_seconds']>0 and row['checkpoint_bytes']==path.stat().st_size)
    original=read(RUN/'predictions/validation_audit.json'); tr=np.flatnonzero(meta.role.eq('train')); vi=np.flatnonzero(meta.role.eq('validation'))
    a.ok('original_VALID_no_fit',original['fits']==0 and len(tr)==2369 and len(vi)==92)
    parent=ROOT/c['sources']['parent_03']; bparent=ROOT/c['sources']['parent_03_1']
    original_heads={r['family']:r for r in read(parent/'models/formal/selected_models.json')}
    for family in ('har','R1','R2','B2'):
        source=Path(original_heads[family]['model_path']); source=source if source.is_absolute() else parent/source
        a.ok('VALID_original_model:'+family,Path(original['selected'][family]).resolve()==source.resolve())
    for seed in (17,29,43): a.ok('VALID_original_B5:'+str(seed),Path(original['selected']['B5'][str(seed)]).resolve()==(bparent/f'models/w32_wd0.001_s{seed}.pt').resolve())
    scaler=read(ROOT/c['sources']['parent_03_1']/'inputs/train_scaler.json'); rebuilt=StandardScaler().fit(features['ordinary_raw'][tr])
    a.close('VALID_scaler_mean',scaler['mean'],rebuilt.mean_); a.close('VALID_scaler_scale',scaler['scale'],rebuilt.scale_)
    scaler['constant_RV']=float(np.clip(np.maximum(labels.RV_raw.to_numpy()[tr],EPS).mean(),EPS,1))
    valid_expected=predict(a,features,vi,original['selected'],scaler,c['models']['B5_device'])
    a.ok('VALID_ids',va.opportunity_id.tolist()==features['ids'][vi].tolist())
    for family,pred in valid_expected.items(): a.close('VALID_prediction:'+family,va[family],pred)
    fit_roles['VALID']=tr; fold_defs['VALID']=(c['roles']['original_train_end'],c['roles']['original_validation_end'])
    weights=read(RUN/'fusion/selected_weights.json'); weight_path=RUN/'fusion/selected_weights.json'
    a.ok('result_seal_before_alpha_selection',pd.Timestamp(result_seal['sealed_at_utc'])<pd.Timestamp(weights['selected_at_utc'])<pd.Timestamp(claim['at_utc']))
    a.ok('selection_frozen_before_VALID',pd.Timestamp(weights['selected_at_utc']).timestamp()<(RUN/'predictions/validation_predictions.csv').stat().st_mtime)
    a.ok('selection_OOF_hash',weights['oof_predictions_sha256']==sha(RUN/'predictions/oof_predictions.csv') and weights['labels_sha256']==sha(RUN/'labels/development.csv') and weights['original_VALID_opened_by_this_selection'] is False)
    target=labels.set_index('opportunity_id').loc[oo.opportunity_id].RV_raw.to_numpy(); base=loss(target,oo.B5)
    trials=csv(RUN/'fusion/all_weight_candidates.csv'); a.ok('18_alpha_candidates',len(trials)==18)
    for family in c['fusion']['comparators']:
        candidates=[]
        for alpha in c['fusion']['alpha_candidates']:
            ll=loss(target,(1-alpha)*oo.B5.to_numpy()+alpha*oo[family].to_numpy()); row=trials[trials.comparator.eq(family)&trials.alpha.eq(alpha)]
            a.ok('alpha_trial_unique:'+family+str(alpha),len(row)==1)
            for key,value in ll.items(): a.close('alpha_loss:'+family+str(alpha)+key,row.iloc[0][key],value.mean())
            candidates.append((float(ll['raw_QLIKE'].mean()),alpha))
        best=min(candidates); a.close('alpha_min:'+family,weights['selected'][family]['alpha'],best[1])
    frame=pd.concat([oo,va.assign(fold_id='VALID')],ignore_index=True)
    labels_by_id=labels.set_index('opportunity_id'); row_targets=labels_by_id.loc[frame.opportunity_id].RV_raw.to_numpy()
    families=BASE.copy()
    for family in c['fusion']['comparators']:
        key='mix_'+family; alpha=weights['selected'][family]['alpha']; frame[key]=(1-alpha)*frame.B5+alpha*frame[family]; families.append(key)
    for family in c['fusion']['comparators']:
        for alpha in c['fusion']['alpha_candidates']:
            key=f'candidate_{family}_a{alpha:g}'; frame[key]=(1-alpha)*frame.B5+alpha*frame[family]; families.append(key)
    a.ok('32_families',len(families)==32)
    point=csv(RUN/'metrics/point_metrics.csv'); pool=csv(RUN/'metrics/pooled_metrics.csv'); row_table=csv(RUN/'metrics/row_metrics.csv')
    tail=csv(RUN/'metrics/tail_metrics.csv'); state=csv(RUN/'metrics/state_metrics.csv'); fit_threshold=read(RUN/'metrics/FIT_fixed_thresholds.json')
    losses_by_fold={}; thresholds_by_fold={}
    for fid in fold_defs:
        pick=frame.fold_id.eq(fid).to_numpy(); sub=frame.loc[pick]; y=row_targets[pick]; effective=np.maximum(y,EPS); ewma=sub.ewma.to_numpy()
        fit_y=np.maximum(labels.RV_raw.to_numpy()[fit_roles[fid]],EPS); fit_ewma=np.clip(features['r0'][fit_roles[fid],1],EPS,1)
        threshold={'q90':np.quantile(fit_y,.9),'q99':np.quantile(fit_y,.99),'ewma_cuts':np.quantile(fit_ewma,[.25,.5,.75])}; thresholds_by_fold[fid]=threshold
        for k,v in threshold.items(): a.close('FIT_threshold:'+fid+k,fit_threshold[fid][k],v)
        event=np.log((y+EPS)/(ewma+EPS))>np.log(2); bins=np.searchsorted(threshold['ewma_cuts'],ewma,side='right')
        for family in families:
            prediction=sub[family].to_numpy(); ll=loss(y,prediction); losses_by_fold[fid,family]=ll
            r=point[point.fold_id.eq(fid)&point.family.eq(family)].iloc[0]
            saved_rows=row_table[row_table.fold_id.eq(fid)&row_table.family.eq(family)]
            a.ok('metric_row_ids:'+fid+family,saved_rows.opportunity_id.tolist()==sub.opportunity_id.tolist())
            score=np.log((prediction+EPS)/(ewma+EPS)); auc,ap=rank(event,score); high_auc,high_ap=rank(effective>threshold['q90'],prediction)
            for k,v in ll.items(): a.close('point:'+fid+family+k,r[k],v.mean()); a.close('row_loss:'+fid+family+k,saved_rows[k],v)
            for k,v in [('surprise_AUROC',auc),('surprise_AP',ap),('absolute_q90_AUROC',high_auc),('absolute_q90_AP',high_ap)]: a.close('ranking:'+fid+family+k,r[k],v)
            a.close('event_score:'+fid+family,saved_rows.surprise_score,score); a.close('event:'+fid+family,saved_rows.surprise_event.to_numpy(bool),event,0,0)
            if family=='constant_RV':
                inverse_auc,inverse_ap=rank(event,-np.log(ewma+EPS))
                a.close('constant_surprise_denominator_auc:'+fid,auc,inverse_auc); a.close('constant_surprise_denominator_ap:'+fid,ap,inverse_ap)
                if len(np.unique(effective>threshold['q90']))==2: a.close('constant_absolute_no_ranking:'+fid,high_auc,.5)
            for quantile in ('q90','q99'):
                selected=effective>threshold[quantile]; t=tail[tail.fold_id.eq(fid)&tail.family.eq(family)&tail.subset.eq(quantile)].iloc[0]
                expected={'N':selected.sum(),'FIT_threshold':threshold[quantile],'under_count':(selected&(prediction/effective<.5)).sum(),
                    'under_fraction':(prediction[selected]/effective[selected]<.5).mean() if selected.any() else np.nan,
                    'mean_prediction':prediction[selected].mean() if selected.any() else np.nan,'mean_actual':y[selected].mean() if selected.any() else np.nan,'Regret_sum':ll['QLIKE_Regret'][selected].sum()}
                for k,v in expected.items(): a.close('tail:'+fid+family+quantile+k,t[k],v)
            for bin_id in range(4):
                selected=bins==bin_id; s=state[state.fold_id.eq(fid)&state.family.eq(family)&state.EWMA_bin.eq(bin_id)].iloc[0]
                a.close('state_n:'+fid+family+str(bin_id),s.N,selected.sum())
                for k,v in ll.items(): a.close('state:'+fid+family+str(bin_id)+k,s[k],v[selected].mean() if selected.any() else np.nan)
    for period in ('OOF','VALID'):
        pick=~frame.fold_id.eq('VALID') if period=='OOF' else frame.fold_id.eq('VALID')
        sub=frame.loc[pick]; y=row_targets[pick]; event=np.log((y+EPS)/(sub.ewma.to_numpy()+EPS))>np.log(2)
        for family in families:
            r=pool[pool.period.eq(period)&pool.family.eq(family)].iloc[0]; ll=loss(y,sub[family])
            for k,v in ll.items(): a.close('pool:'+period+family+k,r[k],v.mean())
            auc,ap=rank(event,np.log((sub[family].to_numpy()+EPS)/(sub.ewma.to_numpy()+EPS)))
            a.close('pool_auc:'+period+family,r.surprise_AUROC,auc); a.close('pool_ap:'+period+family,r.surprise_AP,ap)
    intervals=csv(RUN/'metrics/paired_intervals.csv'); recomputed={}
    daily=csv(RUN/'metrics/daily_paired_gains.csv'); concentration=csv(RUN/'metrics/gain_concentration.csv')
    for fid,(start,end) in fold_defs.items():
        calendar=pd.date_range(start,end,freq='D',inclusive='left'); sub=frame[frame.fold_id.eq(fid)]
        day=calendar.get_indexer(pd.to_datetime(sub.decision_at,utc=True).dt.floor('D')); n=np.bincount(day,minlength=len(calendar))
        for name,(first,second) in CONTRASTS.items():
            opportunity_gain=losses_by_fold[fid,first]['raw_QLIKE']-losses_by_fold[fid,second]['raw_QLIKE']
            gain=np.bincount(day,weights=opportunity_gain,minlength=len(calendar)); saved=daily[daily.fold_id.eq(fid)&daily.contrast.eq(name)]
            a.ok('daily_calendar:'+fid+name,pd.to_datetime(saved.day_utc,utc=True).tolist()==calendar.tolist())
            for key,value in [('N',n),('gain_sum',gain),('cumulative_gain_sum',np.cumsum(gain))]: a.close('daily_gain:'+fid+name+key,saved[key],value)
            best=int(np.argmax(gain)); total=float(gain.sum()); count_other=int(n.sum()-n[best]); gain_other=float((total-gain[best])/count_other) if count_other else np.nan
            row=concentration[concentration.fold_id.eq(fid)&concentration.contrast.eq(name)].iloc[0]
            expected={'N':len(sub),'total_gain_sum':total,'max_positive_day_gain':max(0.,float(gain[best])),
                'max_7day_gain_sum':float(pd.Series(gain).rolling(7,min_periods=1).sum().max()),
                'max_day_signed_share':float(gain[best]/total) if total else np.nan,
                'drop_best_positive_day_mean_gain':gain_other if gain[best]>0 else opportunity_gain.mean()}
            for key,value in expected.items(): a.close('concentration:'+fid+name+key,row[key],value)
            a.ok('concentration_bestday:'+fid+name,pd.Timestamp(row.max_positive_day_UTC)==calendar[best] and bool(row.diagnostic_no_refit_no_reselection))
    for block in c['bootstrap']['blocks']:
        draws=csv(RUN/f'bootstrap/draws_block{block}.csv'); aggregates={name:[] for name in CONTRASTS}
        with np.load(RUN/f'bootstrap/multiplicities_block{block}.npz',allow_pickle=False) as archive:
            for fi,(fid,(start,end)) in enumerate(fold_defs.items()):
                calendar=pd.date_range(start,end,freq='D',inclusive='left'); sub=frame[frame.fold_id.eq(fid)]
                day=calendar.get_indexer(pd.to_datetime(sub.decision_at,utc=True).dt.floor('D')); a.ok('bootstrap_day_scope:'+fid+str(block),(day>=0).all())
                counts=day_counts(len(calendar),block,c['bootstrap']['base_seed']+block*1000+fi)
                a.close('shared_multiplicities:'+fid+str(block),archive[fid+'_counts'],counts,0,0)
                a.ok('calendar_including_empty_days:'+fid+str(block),archive[fid+'_utc_days'].tolist()==calendar.astype(str).tolist())
                n=np.bincount(day,minlength=len(calendar)); denom=counts.astype(float)@n; a.ok('valid_draws:'+fid+str(block),(denom>0).all())
                sums={family:np.bincount(day,weights=losses_by_fold[fid,family]['raw_QLIKE'],minlength=len(calendar)) for family in families}
                for family in families: a.close('draw_family:'+fid+str(block)+family,draws[fid+'_loss_'+family],counts.astype(float)@sums[family]/denom)
                for name,(first,second) in CONTRASTS.items():
                    numerator=counts.astype(float)@(sums[first]-sums[second]); values=numerator/denom
                    sample_draws=np.array([0,2500,4999]); opportunity_weights=counts[sample_draws][:,day].astype(float)
                    delta=losses_by_fold[fid,first]['raw_QLIKE']-losses_by_fold[fid,second]['raw_QLIKE']
                    direct=opportunity_weights@delta/opportunity_weights.sum(axis=1)
                    a.close('draw_direct_opportunity_weights:'+fid+str(block)+name,values[sample_draws],direct)
                    a.close('draw_contrast:'+fid+str(block)+name,draws[fid+'_'+name],values)
                    aggregates[name].append((fid,numerator,denom)); recomputed[fid,block,name]=values
                    row=intervals[intervals.period.eq(fid)&intervals.block_days.eq(block)&intervals.contrast.eq(name)].iloc[0]
                    a.close('CI:'+fid+str(block)+name,[row.ci_lower,row.ci_upper],np.quantile(values,[.025,.975]))
                    a.close('contrast_point:'+fid+str(block)+name,row.point_gain,(losses_by_fold[fid,first]['raw_QLIKE']-losses_by_fold[fid,second]['raw_QLIKE']).mean())
                    a.ok('interval_counts:'+fid+str(block)+name,row.N==len(sub) and row.valid_draws==5000 and row.invalid_fraction==0)
            for name,records in aggregates.items():
                take=[r for r in records if r[0]!='VALID']; values=sum(v[1] for v in take)/sum(v[2] for v in take); recomputed['OOF',block,name]=values
                a.close('weighted_OOF_draw:'+str(block)+name,draws['OOF_'+name],values)
                row=intervals[intervals.period.eq('OOF')&intervals.block_days.eq(block)&intervals.contrast.eq(name)].iloc[0]
                a.close('weighted_OOF_CI:'+str(block)+name,[row.ci_lower,row.ci_upper],np.quantile(values,[.025,.975])); a.close('weighted_OOF_SE:'+str(block)+name,row.bootstrap_SE,values.std(ddof=1))
                first,second=CONTRASTS[name]
                gain=np.concatenate([losses_by_fold[fid,first]['raw_QLIKE']-losses_by_fold[fid,second]['raw_QLIKE'] for fid in fold_defs if fid!='VALID']).mean()
                a.close('weighted_OOF_point:'+str(block)+name,row.point_gain,gain)
                a.ok('OOF_interval_counts:'+str(block)+name,row.N==1267 and row.valid_draws==5000 and row.invalid_fraction==0)
    summary=read(RUN/'metrics/summary.json'); screens={}
    for comparator in c['fusion']['comparators']:
        gains=[float((losses_by_fold[fid,'B5']['raw_QLIKE']-losses_by_fold[fid,'mix_'+comparator]['raw_QLIKE']).mean()) for fid in fit_roles if fid!='VALID']
        gain=np.concatenate([losses_by_fold[fid,'B5']['raw_QLIKE']-losses_by_fold[fid,'mix_'+comparator]['raw_QLIKE'] for fid in fit_roles if fid!='VALID']).mean()
        baseline=np.concatenate([losses_by_fold[fid,'B5']['QLIKE_Regret'] for fid in fit_roles if fid!='VALID']).mean()
        vg=(losses_by_fold['VALID','B5']['raw_QLIKE']-losses_by_fold['VALID','mix_'+comparator]['raw_QLIKE']).mean(); lower,upper=np.quantile(recomputed['OOF',7,'B5_minus_mix_'+comparator],[.025,.975]); positive=sum(g>0 for g in gains)
        screens[comparator]={'gain':gain,'relative':gain/baseline,'positive':positive,'validation':vg,'lower':lower,'upper':upper,'alpha':weights['selected'][comparator]['alpha'],'pass':gain/baseline>=.05 and positive>=4 and vg>0 and lower>0}
        recorded=summary['screens'][comparator]
        for k,v in [('mean_gain',gain),('base_OOF_Regret',baseline),('relative_Regret_improvement',gain/baseline),('positive_folds',positive),('VALID_gain',vg),('ci_lower_7d',lower),('ci_upper_7d',upper)]: a.close('screen:'+comparator+k,recorded[k],v)
        a.ok('primary_screen:'+comparator,recorded['primary_screen_pass']==screens[comparator]['pass'])
    specificity=all(np.quantile(recomputed['OOF',7,name],.025)>0 for name in ('mix_R1_minus_mix_R2','mix_B2_minus_mix_R2')); r=screens['R2']
    if r['alpha']==0 or r['gain']<=0 or r['upper']<=0 or (r['positive']<=2 and r['validation']<=0): decision='NO_DEVELOPMENT_INCREMENT'
    elif r['pass'] and specificity: decision='CANDIDATE_INCREMENT_NOT_CONFIRMED'
    elif r['pass'] and any(screens[k]['pass'] for k in ('R1','B2')): decision='NON_SPECIFIC_ENSEMBLE_GAIN'
    else: decision='INCONCLUSIVE'
    a.ok('independent_project_decision',summary['status']==decision and summary['specificity_screen_pass']==specificity)
    scenarios=csv(RUN/'metrics/future_sample_scenarios.csv'); se=recomputed['OOF',7,'B5_minus_mix_R2'].std(ddof=1); effect=.05*baseline
    a.ok('planning_grid',scenarios.months.tolist()==c['future_draft']['sample_length_grid_months'])
    a.close('planning_summary_SE',summary['planning']['paired_bootstrap_SE'],se); a.close('planning_summary_effect',summary['planning']['effect_candidate_5pct_base_Regret'],effect)
    for row in scenarios.itertuples():
        applicable=r['alpha']>0 and np.isfinite(se) and se>0; future_se=se*np.sqrt(1267/(row.months*365.25/12*3)) if applicable else np.nan
        delta=effect/future_se if applicable else np.nan; power=norm.cdf(delta-norm.ppf(.975))+norm.cdf(-delta-norm.ppf(.975)) if applicable else np.nan
        a.close('planning_SE:'+str(row.months),row.planning_SE_7day,future_se); a.close('planning_MDE:'+str(row.months),row.MDE_80power_two_sided95,(norm.ppf(.975)+norm.ppf(.8))*future_se); a.close('planning_two_sided_power:'+str(row.months),row.power_at_5pct_base_Regret_approx,power)
        a.ok('planning_applicability:'+str(row.months),row.planning_status==('CONDITIONAL_SCENARIO' if applicable else 'NOT_APPLICABLE_ALPHA_ZERO_OR_DEGENERATE'))
    minimum=next((int(row.months) for row in scenarios.itertuples() if row.power_at_5pct_base_Regret_approx>=.8),None)
    a.ok('primary_planning_scenario_not_authorization',summary['planning']['minimum_months_80power_grid']==minimum and summary['planning']['normal_approximation_not_actual_power'] is True and summary['future_formal_test_approved'] is False and summary['old_holdout_status']=='CONSUMED')
    direct_scenarios=csv(RUN/'metrics/future_direct_R2_B5_scenarios.csv'); direct_se=recomputed['OOF',7,'R2_minus_B5'].std(ddof=1)
    a.ok('H3_direct_scenario_grid',direct_scenarios.months.tolist()==c['future_draft']['sample_length_grid_months'])
    for row in direct_scenarios.itertuples():
        applicable=np.isfinite(direct_se) and direct_se>0
        future_se=direct_se*np.sqrt(1267/(row.months*365.25/12*3)) if applicable else np.nan
        delta=effect/future_se if applicable else np.nan
        power=norm.cdf(delta-norm.ppf(.975))+norm.cdf(-delta-norm.ppf(.975)) if applicable else np.nan
        for key,value in [('planning_SE_7day',future_se),('MDE_80power_two_sided95',(norm.ppf(.975)+norm.ppf(.8))*future_se),('power_at_5pct_base_Regret_approx',power)]: a.close('H3_direct_scenario:'+str(row.months)+key,getattr(row,key),value)
        a.ok('H3_direct_applicability:'+str(row.months),row.planning_status==('CONDITIONAL_SCENARIO' if applicable else 'NOT_APPLICABLE_ALPHA_ZERO_OR_DEGENERATE'))
    observed_csv=RUN/'metrics/future_observed_gain_scenarios.csv'; observed_json=RUN/'metrics/future_observed_gain_scenarios.json'
    observed=csv(observed_csv); supplement=read(observed_json)
    a.ok('observed_supplement_binding',supplement['status']=='COMPLETE' and supplement['csv_sha256']==sha(observed_csv)
        and supplement['source_sha256']==sha(ROOT/'research/m2_ci/planning_supplement.py')
        and supplement['future_formal_test_approved'] is False and supplement['changed_selection_models_or_thresholds'] is False and supplement['new_fits']==0)
    for relative,digest in supplement['input_sha256'].items(): a.bind(ROOT/relative,digest)
    a.ok('observed_supplement_12rows',len(observed)==12)
    json_rows=pd.DataFrame(supplement['rows'])
    a.ok('observed_json_csv_identity',len(json_rows)==len(observed) and json_rows.contrast.tolist()==observed.contrast.tolist() and json_rows.months.tolist()==observed.months.tolist())
    for key in ('N_reference','observed_point_gain','paired_bootstrap_SE_7day','expected_opportunities_approx','planning_SE_7day','power_at_observed_signed_effect_two_sided95','nominal_N80_at_observed_effect','nominal_months80_at_observed_effect'):
        a.close('observed_json_csv_numeric:'+key,pd.to_numeric(json_rows[key]),observed[key])
    for contrast in ('B5_minus_mix_R2','R2_minus_B5'):
        sample=observed[observed.contrast.eq(contrast)]; first,second=CONTRASTS[contrast]
        gain=np.concatenate([losses_by_fold[fid,first]['raw_QLIKE']-losses_by_fold[fid,second]['raw_QLIKE'] for fid in fold_defs if fid!='VALID']).mean()
        observed_se=recomputed['OOF',7,contrast].std(ddof=1)
        applicable=np.isfinite(observed_se) and observed_se>0 and np.isfinite(gain) and gain!=0 and (contrast!='B5_minus_mix_R2' or r['alpha']>0)
        n80=1267*((norm.ppf(.975)+norm.ppf(.8))*observed_se/abs(gain))**2 if applicable else np.nan
        a.ok('observed_scenario_grid:'+contrast,sample.months.tolist()==[3,6,9,12,18,24])
        for row in sample.itertuples():
            future_se=observed_se*np.sqrt(1267/(row.months*365.25/12*3)) if applicable else np.nan
            delta=gain/future_se if applicable else np.nan
            power=norm.cdf(delta-norm.ppf(.975))+norm.cdf(-delta-norm.ppf(.975)) if applicable else np.nan
            for key,value in [('N_reference',1267),('expected_opportunities_approx',row.months*365.25/12*3),('observed_point_gain',gain),('paired_bootstrap_SE_7day',observed_se),('planning_SE_7day',future_se),('power_at_observed_signed_effect_two_sided95',power),('nominal_N80_at_observed_effect',n80),('nominal_months80_at_observed_effect',n80/(365.25/12*3))]: a.close('observed_scenario:'+contrast+str(row.months)+key,getattr(row,key),value)
            a.ok('observed_scenario_status:'+contrast+str(row.months),row.planning_status==('POST_HOC_OBSERVED_EFFECT_SCENARIO' if applicable else 'NOT_APPLICABLE_ZERO_EFFECT_OR_DEGENERATE'))
            a.ok('observed_scenario_limits:'+contrast+str(row.months),row.selection_conditioned_optimistic and row.future_stationarity_unproven and row.positive_contrast_means==('R2_mix_lower_loss_than_B5' if contrast=='B5_minus_mix_R2' else 'direct_R2_higher_loss_than_B5'))
    a.bind(ROOT/'research/m2_ci/planning_supplement.py'); a.bind(observed_csv); a.bind(observed_json)
    bench=read(RUN/'metrics/resource_benchmark.json'); a.ok('resource_scope',bench['status']=='PASS' and bench['selected_fold']=='WF05' and bench['no_target_table_reads'] is True and bench['full_R2_encoder_cached_hidden_used'] is False)
    a.ok('benchmark_common8',bench['common_input_ids']==features['ids'][fit_roles['WF01'][:8]].tolist() and len(bench['records'])==12)
    selected=read(RUN/'models/WF05/selected_models.json')['selected']; b5_parameters={}
    for seed,path in selected['B5'].items():
        payload=b5.load_b5(path,device='cpu'); b5_parameters['B5_s'+seed]=sum(p.numel() for p in payload['model'].parameters()); del payload
    b5_total=sum(b5_parameters.values()); head_parameters={}
    for family in ('har','R1','R2'): head_parameters[family]=len(read(selected[family])['coef'])+1
    booster=models._lightgbm().Booster(model_str=read(selected['B2'])['model_text'])
    def tree_size(node):
        if 'left_child' not in node: return 1,1
        ln,ll=tree_size(node['left_child']); rn,rl=tree_size(node['right_child']); return 1+ln+rn,ll+rl
    sizes=[tree_size(t['tree_structure']) for t in booster.dump_model()['tree_info']]; nodes=sum(x[0] for x in sizes); leaves=sum(x[1] for x in sizes)
    for row in bench['records']:
        values=np.asarray(row['elapsed_seconds']); a.ok('benchmark_raw:'+row['name'],len(values)==20 and (values>0).all() and row['warmups']==3 and row['batch_windows']==8 and row['CPU_threads']==1 and row['CUDA_synchronize'] is True)
        for key,value in [('batch_mean_seconds',values.mean()),('batch_median_seconds',np.median(values)),('batch_p95_seconds',np.quantile(values,.95)),('per_window_mean_seconds',values.mean()/8)]: a.close('benchmark_summary:'+row['name']+key,row[key],value)
        a.ok('benchmark_checkpoint_bytes:'+row['name'],row['checkpoint_bytes']==sum(Path(p).stat().st_size for p in set(row['checkpoint_paths'])))
        name=row['name']
        if name in b5_parameters: expected_parameters=b5_parameters[name]
        elif name=='B5_ensemble': expected_parameters=b5_total
        elif name.endswith('_head_CPU'): expected_parameters=None if name=='B2_head_CPU' else head_parameters[name.split('_')[0]]
        elif name=='R2_tokenizer_encoder_CUDA': expected_parameters=0
        else:
            family=name.split('_')[2]; expected_parameters=b5_total+head_parameters.get(family,0)
        a.ok('benchmark_actual_parameters:'+name,row['trainable_parameters_at_fit']==expected_parameters)
        if name in ('B2_head_CPU','full_B5_B2_fusion'): a.ok('benchmark_B2_tree_size:'+name,row['B2_nodes']==nodes and row['B2_leaves']==leaves)
        a.ok('benchmark_memory_scope:'+name,row['cuda_peak_reserved_bytes']>=row['cuda_peak_allocated_bytes']>0 and row['rss_sampled_max_bytes']>=max(row['rss_before_bytes'],row['rss_after_bytes'])>0 and bool(row['measurement_scope']))
    # Independently replay the benchmark's uncached DEV encoder path. The fixed
    # tolerances are specified by the review contract and never widened on failure.
    from research.frozen import encoder
    common_ix=fit_roles['WF01'][:8]; encoder_windows=[]
    for ix in common_ix:
        start=pd.Timestamp(meta.iloc[ix].history_start_at); end=pd.Timestamp(meta.iloc[ix].history_end_exclusive)
        timestamps=pd.date_range(start,periods=256,freq='h')
        a.ok('fresh_encoder_history:'+str(ix),timestamps[-1]+pd.Timedelta(hours=1)==end and end<pd.Timestamp('2026-04-01',tz='UTC'))
        stamps=encoder.calc_time_stamps(pd.Series(timestamps)).to_numpy(dtype=np.float32)
        encoder_windows.append((features['windows'][ix,:,:6].copy(),stamps))
    tokenizer,backbone,encoder_identity=encoder.load_models('pretrained',device='cuda')
    for key in ('model_revision','tokenizer_revision','model_state_sha256','tokenizer_state_sha256','normalization','pooling','dtype','tf32','autocast'):
        a.ok('fresh_encoder_identity:'+key,encoder_identity[key]==bench['encoder_metadata'][key])
    fresh_hidden=encoder.encode_batch(tokenizer,backbone,encoder_windows)
    cached_hidden=features['hidden'][common_ix]
    max_difference=float(np.max(np.abs(fresh_hidden.astype(float)-cached_hidden.astype(float))))
    a.ok('fresh_encoder_vs_DEV_cache',np.allclose(fresh_hidden,cached_hidden,rtol=1e-5,atol=1e-5),
        {'max_abs_difference':max_difference,'rtol':1e-5,'atol':1e-5,
         'common_input_ids':features['ids'][common_ix].tolist(),'encoder_identity':encoder_identity})
    r2_head=read(selected['R2']); ordinary=features['ordinary_raw'][common_ix]
    fresh_prediction=models.predict_head(r2_head,np.concatenate((ordinary,fresh_hidden),axis=1).astype(float))
    cached_prediction=models.predict_head(r2_head,np.concatenate((ordinary,cached_hidden),axis=1).astype(float))
    a.ok('fresh_R2_original_unit_prediction_vs_cached',np.allclose(fresh_prediction,cached_prediction,rtol=1e-5,atol=1e-12),
        {'max_abs_difference':float(np.max(np.abs(fresh_prediction-cached_prediction))),
         'rtol':1e-5,'atol':1e-12,'checkpoint':selected['R2']})
    del tokenizer,backbone
    for path in [RUN/'fusion/selected_weights.json',RUN/'models/candidate_audit.json',RUN/'metrics/summary.json',RUN/'metrics/resource_benchmark.json',RUN/'predictions/oof_predictions.csv',RUN/'predictions/validation_predictions.csv']:
        a.bind(path)
    for directory in ('models','predictions','fusion','metrics','bootstrap'):
        for path in (RUN/directory).rglob('*'):
            if path.is_file() and path.relative_to(ROOT).as_posix() not in a.files: a.bind(path)
    return {'decision':decision,'families':32,'OOF_rows':1267,'VALID_rows':92,'fits_replayed':0,'selected_predictions_recomputed':True,'bootstrap_draws_per_stream':5000}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=['review']); args=parser.parse_args()
    output=RUN/'independent_review/numeric_review.json'
    if output.exists(): raise FileExistsError('Preserve existing independent numeric review')
    a=Audit(); status='PASS'; error=None; result=None
    try: result=review(a)
    except BaseException as exc: status='FAIL'; error=repr(exc)
    output.parent.mkdir(parents=True,exist_ok=True)
    document={'schema_version':1,'experiment_id':'M2_CONDITIONAL_INCREMENT_01','status':status,
        'at_utc':datetime.now(timezone.utc).isoformat(),'review_source_sha256':sha(__file__),
        'check_count':len(a.checks),'checks':a.checks,'source_and_artifact_sha256':a.files,
        'result':result,'error':error,'no_new_fitting':True,'new_Q2_Q3_value_reads':False,
        'oracle':'independent loss/ranking/roles/selection/bootstrap/decision/planning arithmetic; old prediction APIs only'}
    with output.open('x',encoding='utf-8') as f: json.dump(document,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'status':status,'check_count':len(a.checks),'error':error,'output':str(output)}))
    return int(status!='PASS')


if __name__=='__main__': raise SystemExit(main())
