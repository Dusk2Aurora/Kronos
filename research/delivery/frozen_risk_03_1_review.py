"""Independent saved-artifact review of B5; no fit, research retry or random draw."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/runs/FROZEN_RISK_03_1_B5_v1'
PARENT = ROOT / 'research/runs/FROZEN_RISK_03_v1'
FAMILIES = ['persistence','ewma','har','R1','R2','B2','random_s17','random_s29','random_s43','B5','B5_s17','B5_s29','B5_s43']
CONTRASTS = {'R2_minus_B5': ('R2','B5'), 'B5_minus_R1': ('B5','R1'), 'B5_minus_B2': ('B5','B2')}
INPUTS = {}
CHECKS = []

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def record(path):
    path = Path(path)
    INPUTS[path.relative_to(ROOT).as_posix()] = sha(path)
    return path

def read(path): return json.loads(record(path).read_text(encoding='utf-8-sig'))
def csv(path): return pd.read_csv(record(path), float_precision='round_trip')
def near(a,b,name):
    if not np.allclose(a,b,rtol=1e-11,atol=1e-12,equal_nan=True):
        raise AssertionError(name + ': numerical mismatch')

def check(condition,name):
    if not condition: raise AssertionError(name)
    CHECKS.append(name)


def recovery_checks(events):
    """Verify the operational exception independently; journal alone is appendable."""
    shutdown_dir=RUN/'provenance/shutdown_20261009'
    shutdown=read(shutdown_dir/'shutdown_manifest.json')
    archived=record(shutdown_dir/'fit_events.jsonl').read_bytes()
    journal=record(RUN/'models/fit_events.jsonl').read_bytes()
    journal_name=(RUN/'models/fit_events.jsonl').relative_to(ROOT).as_posix()
    check(sha(shutdown_dir/'fit_events.jsonl')==shutdown['source_and_artifact_sha256'][journal_name], 'archived journal shutdown SHA')
    check(journal.startswith(archived) and len(archived.splitlines())==11 and journal.endswith(b'\n'), 'shutdown eleven-event byte prefix retained')
    tags=[f'w{w}_wd{wd}_s{s}' for w in (16,32) for wd in (.0001,.001) for s in (17,29,43)]
    interrupted=tags[5]
    expected=[(tag,state) for tag in tags[:5] for state in ('STARTED','COMPLETED')]
    expected += [(interrupted,'STARTED'),(interrupted,'INTERRUPTED')]
    expected += [(tag,state) for tag in tags[5:] for state in ('STARTED','COMPLETED')]
    check([(e['candidate'],e['state']) for e in events]==expected, 'exact 26-event order: 13 attempts 12 completions one user interruption')
    check(shutdown['completed_candidate_count']==5 and shutdown['interrupted_candidate']==interrupted, 'shutdown five completed and sixth interrupted')
    for tag in tags[:5]:
        for name in (f'{tag}.pt',f'history_{tag}.csv',f'audit_{tag}.json'):
            path=RUN/'models'/name
            key=path.relative_to(ROOT).as_posix()
            check(sha(record(path))==shutdown['source_and_artifact_sha256'][key], 'reused original artifact byte SHA '+name)
    backup=read(RUN/'provenance/pre_recovery_tool_versions/manifest.json')
    for name,entry in backup.items():
        check(sha(record(ROOT/entry['archived_path']))==entry['sha256'], 'pre-recovery archive SHA '+name)
        if name.startswith('research/delivery/'):
            check(entry['sha256']==shutdown['source_and_artifact_sha256'][name], 'original delivery tool shutdown SHA '+name)
    seal_path=RUN/'preflight/recovery_seal_20261009.json'
    seal=read(seal_path)
    check(seal['status']=='PASS' and seal['completed_candidates']==tags[:5] and seal['remaining_candidates']==tags[5:], 'recovery sealed original candidate order')
    check((seal['candidate_fits'],seal['fit_attempts_including_user_interruption'],seal['interrupted_attempts'])==(12,13,1) and seal['scientific_selection_unchanged'] is True, 'recovery scientific budget and all attempts disclosed')
    check(seal['protocol_sha256']==shutdown['protocol_sha256'] and seal['parent_delivery_sha256']==sha(PARENT/'formal_delivery_manifest.json'), 'recovery protocol and parent SHA unchanged')
    check(seal['journal_prefix_bytes']==len(archived) and seal['journal_prefix_sha256']==sha(shutdown_dir/'fit_events.jsonl'), 'recovery seal original journal prefix binding')
    check(journal_name in seal['files_sha256'], 'journal explicitly sealed before recovery')
    for name,digest in seal['files_sha256'].items():
        actual=hashlib.sha256(journal[:seal['journal_prefix_bytes']]).hexdigest() if name==journal_name else sha(record(ROOT/name))
        check(actual==digest, 'recovery sealed SHA '+name)
    amendment=ROOT/seal['amendment']
    check(seal['amendment']=='research/frozen/experiment_03_1/OPERATIONAL_RESUME_20261009.md' and seal['files_sha256'][seal['amendment']]==sha(record(amendment)), 'operational amendment sealed SHA')
    claim=read(RUN/'models/recovery_claim_20261009.json')
    check(claim['state']=='CLAIMED' and claim['recovery_seal_sha256']==sha(seal_path) and claim['remaining_fits']==7 and claim['protocol_sha256']==seal['protocol_sha256'], 'exclusive recovery claim source binding')
    check((claim['candidate_fits'],claim['fit_attempts_including_user_interruption'],claim['interrupted_attempts'])==(12,13,1), 'recovery claim all attempts')
    partial=read(RUN/'models/interrupted_attempt_resources.json')
    check(partial['candidate']==interrupted and partial['status']=='USER_SHUTDOWN_PARTIAL_ATTEMPT' and partial['shutdown_manifest_sha256']==sha(shutdown_dir/'shutdown_manifest.json'), 'interruption resource provenance')
    check(partial['interrupted_at_utc']==shutdown['at_utc'], 'interruption timestamp matches shutdown')
    near(partial['pre_pause_partial_fit_wall_seconds_approx'],shutdown['pre_pause_partial_fit_wall_seconds_approx'], 'partial wall time matches shutdown')
    near(partial['pre_pause_partial_fit_wall_seconds_approx'],430.165747,'partial pre-pause approximate wall seconds')
    check(partial['partial_fit_epochs_run'] is None and partial['gpu_compute_seconds'] is None and partial['pause_idle_seconds'] is None, 'interrupted epochs GPU compute and idle unknown')
    check(partial['wall_time_is_not_gpu_compute'] is True and partial['pause_idle_excluded_from_compute'] is True and partial['included_in_completed_training_resources'] is False and partial['matched_compute_budget_claim'] is False, 'partial wall time not GPU compute or matched compute budget')
    interruption=events[11]
    check(interruption['shutdown_manifest_sha256']==partial['shutdown_manifest_sha256'] and interruption['epochs_run'] is None and interruption['disk_checkpoint_available'] is False, 'interrupted attempt event disclosure')
    recovered=read(RUN/'models/recovered_candidate_audit.json')
    check(recovered['status']=='PASS' and recovered['recovery_seal_sha256']==sha(seal_path) and recovered['validation_N']==92 and recovered['test_prediction_generated'] is False, 'five reused candidates VALID-only recovery audit')
    check([x['candidate'] for x in recovered['candidates']]==tags[:5], 'all five recovered candidate audits retained')
    return seal

def independent_prediction(path, windows, ordinary):
    # No model module import: reconstruct the sealed architecture using functional operations.
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stored = torch.load(record(path), map_location=device, weights_only=True)
    state = stored['state_dict']; metadata = stored['metadata']; outputs = []
    with torch.no_grad():
        for start in range(0,len(windows),64):
            x = torch.from_numpy(windows[start:start+64]).to(device).transpose(1,2)
            ordinary_batch = torch.from_numpy(ordinary[start:start+64]).to(device)
            h = F.conv1d(F.pad(x,(2,0)),state['input_conv.weight'],state['input_conv.bias'])
            for block,dilation in enumerate((1,2,4,8,16,32,64)):
                h = h + F.gelu(F.conv1d(F.pad(h,(2*dilation,0)),state[f'blocks.{block}.weight'],state[f'blocks.{block}.bias'],dilation=dilation))
            z = F.linear(torch.cat((h.transpose(1,2)[:,-1],ordinary_batch),dim=1),state['head.weight'],state['head.bias']).squeeze(-1)
            outputs.append(z.cpu().numpy())
    z = np.concatenate(outputs).astype(np.float64)
    return np.clip(np.exp(np.clip(z+np.log(metadata['target_scale']),np.log(1e-12),0)),1e-12,1),metadata,device


def resource_checks(selected, candidate_audit, train_ids):
    """Check saved measurements and accounting only; never rerun the benchmark."""
    training=read(RUN/'models/training_resources.json')
    resource=read(RUN/'provenance/resources.json')
    expected=[(w,wd,s) for w in (16,32) for wd in (.0001,.001) for s in (17,29,43)]
    check(len(training)==12 and [(r['width'],r['weight_decay'],r['seed']) for r in training]==expected, 'all twelve training resource rows in original candidate order')
    fields=('elapsed_seconds','parameters','epochs_run','threads','cuda_peak_allocated_bytes')
    for row,(width,wd,seed) in zip(training,expected):
        tag=f'w{width}_wd{wd}_s{seed}'
        audit=read(RUN/'models'/f'audit_{tag}.json')['resources']
        check(set(row)==set(audit)|{'width','weight_decay','seed'}, 'training resource schema equals candidate audit '+tag)
        for field in fields:
            if field=='elapsed_seconds':
                near(row[field],audit[field], 'saved elapsed matches candidate audit '+tag)
                check(np.isfinite(row[field]) and row[field]>0, 'positive finite candidate elapsed '+tag)
            else:
                check(row[field]==audit[field], 'saved resource matches candidate audit '+tag+' '+field)
        check(row['parameters']>0 and row['threads']==1 and 1<=row['epochs_run']<=120 and row['cuda_peak_allocated_bytes']>0, 'candidate measured resource ranges '+tag)
    check(resource['status']=='PASS' and resource['training']==training, 'benchmark training records exactly equal completed twelve resource rows')
    elapsed=sum(r['elapsed_seconds'] for r in training)
    near(resource['candidate_training_seconds_total'],elapsed, 'training total equals only twelve completed fits without partial wall time')
    check(candidate_audit['actual_total_epochs']==sum(r['epochs_run'] for r in training) and candidate_audit['interrupted_partial_epochs_unknown'] is True, 'completed epoch total keeps partial epochs unknown')
    check(resource['same_training_window_ids']==list(train_ids[:8]) and len(resource['same_training_window_ids'])==8, 'inference benchmark uses identical first eight TRAIN IDs')
    check(resource['warmups']==3 and resource['repetitions']==20, 'saved benchmark three warmups and twenty repetitions')
    inference=resource['inference']
    families=['B5_s17','B5_s29','B5_s43','B5','Kronos_tokenizer_backbone']
    check(len(inference)==5 and [r['family'] for r in inference]==families, 'all five inference resource families retained')
    by_family={r['family']:r for r in inference}
    params=[];sizes=[]
    for row in inference:
        check(np.isfinite(row['seconds_per_batch8']) and row['seconds_per_batch8']>0 and row['parameters']>0 and row['checkpoint_bytes']>0 and row['peak_allocated_bytes']>0, 'positive finite saved inference resources '+row['family'])
    for seed in (17,29,43):
        tag=f"w{selected['width']}_wd{selected['weight_decay']}_s{seed}"
        path=record(RUN/'models'/f'{tag}.pt')
        audit=read(RUN/'models'/f'audit_{tag}.json')
        row=by_family[f'B5_s{seed}']
        check(row['parameters']==audit['resources']['parameters'] and row['checkpoint_bytes']==path.stat().st_size, 'selected seed benchmark actual checkpoint bytes and audit parameters '+tag)
        params.append(row['parameters']);sizes.append(path.stat().st_size)
    check(by_family['B5']['parameters']==sum(params) and by_family['B5']['checkpoint_bytes']==sum(sizes), 'ensemble parameters and checkpoint bytes sum all three selected seeds')

def review():
    # Publication-only adaptations are bound at review/publication/final delivery,
    # while the recovery source seal retains the byte-archived original tools.
    record(Path(__file__).resolve())
    manifest = read(PARENT/'formal_delivery_manifest.json')
    hashes = manifest['source_and_artifact_sha256']
    for path,digest in hashes.items():
        actual=sha(ROOT/path);INPUTS[path]=actual
        check(actual==digest,'parent SHA '+path)
    check(len(hashes)==561,'all 561 original formal delivery artifacts retained')
    check(read(PARENT/'authorization/formal_terminal.json')['state']=='CONSUMED','original terminal CONSUMED')
    original = read(PARENT/'metrics/formal_result.json')['status']
    summary = read(RUN/'metrics/summary.json')
    check(summary['original_primary_status']==original,'original formal status unchanged')
    check(summary['bootstrap_new_random_draws']==0 and summary['threshold_refit_on_test'] is False,'no new draws or TEST threshold refit')
    seal = read(RUN/'preflight/pretrain_review_seal.json')
    for path,digest in seal['files_sha256'].items(): check(sha(ROOT/path)==digest,'pretraining sealed SHA '+path)
    with np.load(record(RUN/'inputs/windows.npz'),allow_pickle=False) as f: a={k:f[k] for k in f.files}
    identities=csv(RUN/'inputs/sample_identities.csv')
    labels=csv(RUN/'labels/reused_risk_labels.csv').set_index('opportunity_id',drop=False)
    check(labels.index.is_unique and identities.opportunity_id.tolist()==a['ids'].tolist(),'all sample IDs unique and array identity order exact')
    labels=labels.loc[a['ids']]
    tr=a['roles']=='train'; va=a['roles']=='validation'; te=np.isin(a['roles'],['TEST_Q2','TEST_Q3'])
    check((tr.sum(),va.sum(),te.sum())==(2369,92,547),'TRAIN VALID TEST sample counts')
    check(labels.role.tolist()==a['roles'].tolist(),'labels roles align all arrays')
    raw=labels.RV_raw.to_numpy()
    if 'RV_effective' in labels:near(labels.RV_effective,np.maximum(raw,1e-12),'effective RV floor all labels')
    for row in labels.itertuples():
        prices=np.asarray(json.loads(row.label_prices_json),float)
        check(len(prices)==49 and np.isfinite(prices).all() and (prices>0).all(),'49 valid raw prices '+str(row.opportunity_id))
        near(np.sum(np.diff(np.log(prices))**2),row.RV_raw,'48-return raw RV '+str(row.opportunity_id))
    scaler=read(RUN/'inputs/train_scaler.json')
    check(len(scaler['features'])==33,'exactly 33 ordinary features')
    allfeatures=pd.concat([csv(PARENT/'features/ordinary_risk.csv'),csv(PARENT/'features/holdout_ordinary_risk.csv')]).set_index('opportunity_id')
    ordinary=allfeatures.loc[a['ids'],scaler['features']].to_numpy(float)
    independent_scaler=StandardScaler().fit(ordinary[tr])
    near(independent_scaler.mean_,scaler['mean'],'TRAIN-only means');near(independent_scaler.scale_,scaler['scale'],'TRAIN-only scales')
    near(independent_scaler.transform(ordinary).astype(np.float32),a['ordinary'],'TRAIN-scaled ordinary all roles')
    check(scaler['fitted_role']=='train' and scaler['N']==2369,'scaler fit role metadata')
    median=float(np.median(np.maximum(raw[tr],1e-12)))
    selected=read(RUN/'models/selected.json'); candidates=csv(RUN/'models/all_candidate_validation_predictions.csv')
    events=[json.loads(line) for line in record(RUN/'models/fit_events.jsonl').read_text().splitlines() if line.strip()]
    recovery_seal=recovery_checks(events)
    scores=[]
    for width in (16,32):
        for wd in (.0001,.001):
            predictions=[]
            for seed in (17,29,43):
                tag=f'w{width}_wd{wd}_s{seed}'
                expected_states=['STARTED','INTERRUPTED','STARTED','COMPLETED'] if tag=='w16_wd0.001_s43' else ['STARTED','COMPLETED']
                check([e['state'] for e in events if e['candidate']==tag]==expected_states,'all attempts for '+tag)
                audit=read(RUN/'models'/f'audit_{tag}.json');meta=audit['metadata']
                near(meta['target_scale'],median,'TRAIN effective median '+tag)
                check(audit['status']=='PASS' and audit['reload_exact'] is True and meta['finite_checks_passed'] is True,'candidate finite/reload audit '+tag)
                history=csv(RUN/'models'/f'history_{tag}.csv')
                check(np.isfinite(history[['train_scaled_objective','train_raw_qlike','valid_raw_qlike']].to_numpy()).all(),'all candidate history losses finite '+tag)
                best=history.loc[history.valid_raw_qlike.idxmin()]
                check(int(best.epoch)==meta['best_epoch'] and len(history)==audit['resources']['epochs_run']<=120,'earliest minimum VALID checkpoint '+tag)
                near(best.valid_raw_qlike,meta['best_valid_raw_qlike'],'best VALID loss '+tag)
                sub=candidates[candidates.candidate==tag].set_index('opportunity_id')
                check(sub.index.is_unique and set(sub.index)==set(a['ids'][va]),'VALID-only candidate prediction IDs '+tag)
                p=sub.loc[a['ids'][va],'prediction_RV'].to_numpy();predictions.append(p)
                near(np.mean(np.log(p)+np.maximum(raw[va],1e-12)/p),meta['best_valid_raw_qlike'],'saved VALID prediction loss '+tag)
            p=np.mean(predictions,axis=0);loss=float(np.mean(np.log(p)+np.maximum(raw[va],1e-12)/p));scores.append((loss,width,-wd))
            savedscore=next(item for item in selected['structure_scores'] if item['width']==width and item['weight_decay']==wd)
            near(savedscore['validation_qlike'],loss,'independent structure VALID loss '+str(width)+str(wd))
    winner=min(scores)
    check((winner[1],-winner[2])==(selected['width'],selected['weight_decay']),'four-structure three-seed VALID mean selection')
    near(winner[0],selected['validation_qlike'],'selected VALID loss')
    check(len(candidates)==12*92,'all 12 candidates VALID predictions retained')
    candidate_audit=read(RUN/'models/candidate_audit.json')
    check(candidate_audit['candidate_fits']==12 and candidate_audit['selected_sha256']==sha(RUN/'models/selected.json'),'candidate audit selected SHA and budget')
    check(candidate_audit['fit_attempts_including_user_interruption']==13 and candidate_audit['interrupted_attempts']==1 and candidate_audit['journal_event_counts']=={'STARTED':13,'COMPLETED':12,'INTERRUPTED':1},'candidate audit all attempts and journal counts')
    check(candidate_audit['recovery_seal_sha256']==sha(RUN/'preflight/recovery_seal_20261009.json') and candidate_audit['validation_only_selection'] is True and candidate_audit['interrupted_partial_epochs_unknown'] is True, 'candidate recovery seal VALID-only and unknown partial epochs')
    resources=read(RUN/'models/training_resources.json')
    check(len(resources)==12 and candidate_audit['actual_total_epochs']==sum(r['epochs_run'] for r in resources) and candidate_audit['max_possible_epochs']==1440, 'completed epoch totals exclude unknown interrupted partial fit')
    check(candidate_audit['actual_total_epochs_scope']=='12 completed fits only; interrupted partial epoch count unknown' and candidate_audit['original_journal_prefix_preserved'] is True, 'resource scope and preserved journal explicit')
    resource_checks(selected,candidate_audit,a['ids'][tr])
    pred=csv(RUN/'predictions/test_predictions_only.csv');trainpred=csv(RUN/'predictions/train_B5_predictions_only.csv')
    check(pred.opportunity_id.tolist()==a['ids'][te].tolist() and trainpred.opportunity_id.tolist()==a['ids'][tr].tolist(),'TRAIN and TEST saved predictions ID order exact')
    oldpred=csv(PARENT/'predictions/holdout.csv')
    check(oldpred.opportunity_id.tolist()==pred.opportunity_id.tolist(),'parent TEST identity exact')
    check(pred.role.tolist()==a['roles'][te].tolist(),'TEST roles align array roles')
    for field in ('decision_at','decision_boundary_at','role','entry_at'):
        check(pred[field].tolist()==oldpred[field].tolist(),'parent TEST metadata unchanged '+field)
    for family in FAMILIES[:9]: near(pred[family],oldpred[family],'unchanged original prediction '+family)
    devices=[]
    for seed in (17,29,43):
        path=RUN/'models'/f"w{selected['width']}_wd{selected['weight_decay']}_s{seed}.pt"
        testp,meta,device=independent_prediction(path,a['windows'][te],a['ordinary'][te]);devices.append(device)
        trainp,_,_=independent_prediction(path,a['windows'][tr],a['ordinary'][tr])
        # Reconstruct float32 functional forward with identical 64-sample batches.
        near(testp,pred[f'B5_s{seed}'],'independent checkpoint TEST seed '+str(seed))
        near(trainp,trainpred[f'B5_s{seed}'],'independent checkpoint TRAIN seed '+str(seed))
    for frame in (pred,trainpred): near(frame.B5,frame[[f'B5_s{s}' for s in (17,29,43)]].to_numpy().mean(axis=1),'RV arithmetic ensemble')
    qm=csv(RUN/'metrics/quarter_metrics.csv').set_index(['quarter','family'])
    rm=csv(RUN/'metrics/row_metrics.csv').set_index(['quarter','family','opportunity_id'])
    under=csv(RUN/'metrics/underprediction.csv').set_index(['quarter','family','subset'])
    calibration=csv(RUN/'metrics/calibration.csv').set_index(['quarter','family','bin_type','bin_id'])
    thresholds=read(PARENT/'calibration/sealed_calibration.json')['thresholds']
    for q in (90,99):
        near(np.quantile(np.maximum(raw[tr],1e-12),q/100),thresholds[f'train_effective_RV_q{q}'],'TRAIN q'+str(q))
        near(summary['high_rv'][f'q{q}_threshold'],thresholds[f'train_effective_RV_q{q}'],'summary sealed TRAIN threshold q'+str(q))
    train_ewma=allfeatures.loc[a['ids'][tr],'r0_ewma'].to_numpy(float)
    rawtest=raw[te];derived={};points={}
    for quarter,role in [('2026Q2','TEST_Q2'),('2026Q3','TEST_Q3')]:
        mask=pred.role.eq(role).to_numpy();rv=rawtest[mask];effective=np.maximum(rv,1e-12);ewma=pred.ewma.to_numpy()[mask]
        event=np.log((rv+1e-12)/(ewma+1e-12))>np.log(2)
        summaryquarter=next(item for item in summary['quarter_metrics'] if item['quarter']==quarter)
        check((summaryquarter['N'],summaryquarter['positive'],summaryquarter['negative'])==(len(rv),int(event.sum()),int((~event).sum())),'summary quarter sample/event counts '+quarter)
        derived[quarter]={'event':event,'dates':pd.to_datetime(pred.decision_at[mask],utc=True).dt.floor('D'),'scores':{}}
        for family in FAMILIES:
            p=pred[family].to_numpy()[mask];check(np.isfinite(p).all() and (p>0).all(),'positive finite prediction '+quarter+family)
            score=np.log((p+1e-12)/(ewma+1e-12));ratio=effective/p
            fields={'RV_raw':rv,'RV_effective':effective,'EWMA_RV':ewma,'prediction_RV':p,'surprise_score':score,'surprise_event':event,'raw_QLIKE':np.log(p)+ratio,'QLIKE_Regret':ratio-np.log(ratio)-1,'logRV_MSE':(np.log(effective)-np.log(p))**2}
            saved=rm.loc[(quarter,family)].loc[pred.opportunity_id[mask]]
            for field,value in fields.items(): near(saved[field],value,'row '+quarter+family+field)
            point={'N':len(p),'positive':int(event.sum()),'negative':int((~event).sum()),'AUROC':roc_auc_score(event,score),'AP':average_precision_score(event,score),**{k:float(fields[k].mean()) for k in ('raw_QLIKE','QLIKE_Regret','logRV_MSE')}}
            for field,value in point.items():near(qm.loc[(quarter,family),field],value,'point '+quarter+family+field)
            points[quarter,family]=point['AUROC'];derived[quarter]['scores'][family]=score
            for subset,q in [('q90',90),('q99',99)]:
                high=effective>thresholds[f'train_effective_RV_q{q}'];u=under.loc[(quarter,family,subset)]
                values={'count':int(high.sum()),'underprediction_count':int((high&(p/effective<.5)).sum()),'underprediction_fraction':float((p[high]/effective[high]<.5).mean()) if high.any() else np.nan,'mean_prediction_RV':p[high].mean() if high.any() else np.nan,'mean_observed_RV':rv[high].mean() if high.any() else np.nan}
                for field,value in values.items():near(u[field],value,'underprediction '+quarter+family+subset+field)
            if family.startswith('B5'):
                for kind,tvalues,values,quantiles in [('prediction_decile',trainpred[family].to_numpy(),p,np.arange(.1,1,.1)),('score_quartile',np.log((trainpred[family].to_numpy()+1e-12)/(train_ewma+1e-12)),score,[.25,.5,.75])]:
                    cuts=np.unique(np.quantile(tvalues,quantiles));bins=np.searchsorted(cuts,values,side='right')
                    for binid in range(len(cuts)+1):
                        b=bins==binid;row=calibration.loc[(quarter,family,kind,binid)]
                        vals={'count':int(b.sum()),'mean_prediction_RV':p[b].mean() if b.any() else np.nan,'mean_observed_RV':rv[b].mean() if b.any() else np.nan,'event_fraction':event[b].mean() if b.any() else np.nan}
                        for field,value in vals.items():near(row[field],value,'TRAIN-only calibration '+quarter+family+kind+str(binid)+field)
    check(len(qm)==26 and len(rm)==547*13 and len(under)==52,'all 13 families x 2 quarters metric row counts')
    bootstrap=[]
    for block in (7,3,14):
        saved=csv(RUN/'bootstrap'/f'draws_block{block}.csv');draws={}
        with np.load(record(PARENT/'bootstrap'/f'multiplicities_block{block}.npz'),allow_pickle=False) as weights:
            for quarter,role in [('2026Q2','TEST_Q2'),('2026Q3','TEST_Q3')]:
                data=derived[quarter];days=pd.to_datetime(weights[role+'_utc_days'],utc=True);w=weights[role+'_counts'].astype(float)
                check(w.shape==(5000,len(days)),'all 5000 bootstrap attempts '+quarter+str(block))
                ids=days.get_indexer(data['dates']);check((ids>=0).all(),'UTC calendar day alignment '+quarter+str(block))
                event=data['event'];pos=np.flatnonzero(event);neg=np.flatnonzero(~event)
                dailypos=np.bincount(ids[pos],minlength=len(days));dailyneg=np.bincount(ids[neg],minlength=len(days))
                denominator=(w@dailypos)*(w@dailyneg);metrics={}
                for family in ('R1','R2','B2','B5'):
                    score=data['scores'][family];wins=(score[pos,None]>score[None,neg]).astype(float)+.5*(score[pos,None]==score[None,neg])
                    matrix=np.zeros((len(days),len(days)))
                    np.add.at(matrix,(ids[pos,None],ids[None,neg]),wins)
                    numerator=((w@matrix)*w).sum(axis=1)
                    values=np.divide(numerator,denominator,out=np.full(5000,np.nan),where=denominator>0);metrics[family]=values
                    for draw in (0,2499,4999):
                        expanded=np.repeat(np.arange(len(event)),w[draw,ids].astype(int))
                        if len(np.unique(event[expanded]))==2:near(roc_auc_score(event[expanded],score[expanded]),values[draw],'expanded sklearn '+quarter+family+str(block)+str(draw))
                check((denominator>0).all(),'zero invalid bootstrap draws '+quarter+str(block))
                for name,(first,second) in CONTRASTS.items():
                    key=quarter+'_'+name;draws[key]=metrics[first]-metrics[second];near(saved[key],draws[key],'all saved quarter draws '+str(block)+key)
            for name,(first,second) in CONTRASTS.items():
                values=(draws['2026Q2_'+name]+draws['2026Q3_'+name])/2
                near(saved['equal_quarter_'+name],values,'all saved equal-quarter draws '+str(block)+name)
                expected=summary['comparisons'][str(block)][name];deltas=[points[q,first]-points[q,second] for q in ('2026Q2','2026Q3')]
                near(expected['quarter_deltas'],deltas,'quarter deltas '+name);near(expected['mean_delta'],np.mean(deltas),'mean delta '+name)
                ci=np.quantile(values[np.isfinite(values)],[.025,.975]);near([expected['ci_lower'],expected['ci_upper']],ci,'independent CI '+str(block)+name)
                check(expected['valid_draws']==5000 and expected['invalid_fraction']==0,'all CI attempts valid '+str(block)+name)
                bootstrap.append({'block_days':block,'comparison':name,'attempts':5000,'all_saved_draws_match':True,'CI_matches':True})
    return {'status':'PASS','evidence_class':'POST_HOC_EXPLORATORY','checks':CHECKS,'inputs_sha256':INPUTS,
            'all_bootstrap_attempts_verified':True,'all_saved_draw_matches':True,'bootstrap':bootstrap,
            'no_new_fit':True,'new_random_draws':0,'independent_checkpoint_inference_devices':devices,
            'recovery_checks_passed':True,'candidate_fits':12,'fit_attempts_including_user_interruption':13,'interrupted_attempts':1,
            'original_formal_status':original,'original_delivery_artifacts_verified':len(hashes)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-generated-only',action='store_true',required=True)
    parser.parse_args()
    destination=RUN/'independent_review/numeric_review_v2.json'
    if destination.exists():raise FileExistsError('Independent review immutable output exists: '+str(destination))
    try: result=review()
    except Exception as exc:
        result={'status':'FAIL','error':repr(exc),'evidence_class':'POST_HOC_EXPLORATORY','checks':CHECKS,'inputs_sha256':INPUTS,'all_bootstrap_attempts_verified':False,'no_new_fit':True,'new_random_draws':0}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'path':str(destination),'checks':len(CHECKS),'error':result.get('error')},ensure_ascii=False))
    return int(result['status']!='PASS')

if __name__=='__main__':raise SystemExit(main())
