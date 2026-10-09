"""Bounded 95-fit M2 runner; no outer target reaches fitting or inference."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# Must be established before any unchanged B5/Torch import.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from research.m2_ci import data


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_seal(run):
    run, _, _ = data.new_scope(run)
    seal = data.read(run / 'preflight/source_seal.json')
    if seal.get('status') != 'PASS' or seal.get('protocol_sha256') != data.sha(data.CONFIG):
        raise PermissionError('Passed current source seal required')
    files = seal.get('files_sha256', {})
    if not files:
        raise PermissionError('Empty source seal')
    for key in ('code_review', 'tests'):
        check = seal.get(key, {})
        if check.get('status') != 'PASS' or check.get('path') not in files:
            raise PermissionError('Sealed code review and critical tests required')
        if data.read(data.ROOT / check['path']).get('status') != 'PASS':
            raise PermissionError('Failed code review or tests')
    required = [data.CONFIG, Path(__file__), Path(data.__file__),
                run / 'inputs/features.npz', run / 'inputs/metadata.csv',
                run / 'inputs/prepare_audit.json', run / 'labels/development.csv',
                run / 'folds/index.csv', run / 'folds/thresholds.json']
    for p in required:
        if p.resolve().relative_to(data.ROOT).as_posix() not in files:
            raise PermissionError('Required source/input missing from seal: ' + str(p))
    for p, digest in files.items():
        path = (data.ROOT / p).resolve()
        if not path.is_relative_to(data.ROOT) or data.sha(path) != digest:
            raise PermissionError('Changed sealed source/input: ' + p)
    return seal


def journal(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(value, at_utc=now()), allow_nan=False) + '\n')
        handle.flush()


def feature_matrix(features, family, indices):
    if family == 'har':
        return features['har'][indices]
    if family in ('R1','B2'):
        return features['ordinary_raw'][indices]
    if family == 'R2':
        return np.concatenate((features['ordinary_raw'][indices], features['hidden'][indices]), axis=1).astype(np.float64)
    raise ValueError('Unknown family')


def fit_role_targets(labels, indices):
    """Call only with FIT or inner-validation indices; no outer bundle to fit."""
    return labels.RV_raw.to_numpy(dtype=np.float64)[indices].copy()


def predict_only(features, indices, selected, scaler, *, device='cuda'):
    """Inference accepts features, integer identities and frozen artifacts only."""
    from research.frozen.experiment_03 import models
    from research.frozen.experiment_03_1 import b5_model as b5
    result = {}
    for family in ('har','R1','R2','B2'):
        model = data.read(selected[family])
        fn = models.predict_b2 if family == 'B2' else models.predict_head
        result[family] = fn(model, feature_matrix(features, family, indices))
    ordinary = ((features['ordinary_raw'][indices] - np.asarray(scaler['mean'])) / np.asarray(scaler['scale'])).astype(np.float32)
    predictions, identity = [], {}
    for seed, path in selected['B5'].items():
        model = b5.load_b5(path, device=device)
        first = b5.predict_b5(model, features['windows'][indices], ordinary, device=device)
        repeat = b5.predict_b5(model, features['windows'][indices], ordinary, device=device)
        sample = np.unique([0,len(indices)//2,len(indices)-1])
        single = b5.predict_b5(model, features['windows'][indices[sample]], ordinary[sample], device=device, batch_size=1)
        np.testing.assert_array_equal(first['raw_score'], repeat['raw_score'])
        np.testing.assert_array_equal(first['prediction'], repeat['prediction'])
        np.testing.assert_allclose(first['raw_score'][sample], single['raw_score'], rtol=1e-4, atol=1e-4)
        identity[str(seed)] = {'status':'PASS','repeat_raw_score_max_abs':float(np.max(np.abs(first['raw_score']-repeat['raw_score']))),
                'single_batch_raw_score_max_abs':float(np.max(np.abs(first['raw_score'][sample]-single['raw_score']))),
                'checked_ids':features['ids'][indices[sample]].tolist(),'atol':1e-4,'rtol':1e-4}
        pred = first['prediction']
        result['B5_s' + str(seed)] = pred
        predictions.append(pred)
        del model
    result['B5'] = np.mean(predictions, axis=0, dtype=np.float64)
    result['persistence'] = np.clip(features['r0'][indices,0],1e-12,1)
    result['ewma'] = np.clip(features['r0'][indices,1],1e-12,1)
    result['constant_RV'] = np.full(len(indices), scaler['constant_RV'], dtype=np.float64)
    return result, identity


def prediction_frame(metadata, indices, predictions, fold_id, logical_ready_at):
    cols = ['opportunity_id','decision_boundary_at','decision_at','entry_at','history_start_at','history_end_exclusive']
    frame = metadata.iloc[indices][cols].copy().reset_index(drop=True)
    frame['fold_id'] = fold_id
    frame['model_logical_ready_at'] = logical_ready_at
    for k,v in predictions.items():
        if len(v) != len(frame) or not np.isfinite(v).all() or (v < 1e-12).any() or (v > 1).any():
            raise ValueError('Invalid prediction: ' + k)
        frame[k] = v
    return frame


@threadpool_limits.wrap(limits=1)
def train_oof(run=data.DEFAULT_RUN):
    run, c, _ = data.new_scope(run)
    seal = verify_seal(run)
    features, metadata = data.load_features(run)
    folds = data.fold_indices(metadata, c)
    labels = pd.read_csv(run / 'labels/development.csv',float_precision='round_trip')
    if labels.opportunity_id.tolist() != features['ids'].tolist():
        raise ValueError('Labels order differs')
    from research.frozen.experiment_03 import models
    from research.frozen.experiment_03_1 import b5_model as b5
    out = run / 'models'
    out.mkdir(exist_ok=True)
    data.write(out / 'training_claim.json', {'state':'CLAIMED','at_utc':now(),
               'budget_fits':95,'protocol_sha256':data.sha(data.CONFIG),
               'seal_sha256':data.sha(run / 'preflight/source_seal.json')})
    events = out / 'fit_events.jsonl'
    resources, identities, predictions = [], {}, []
    count, epochs, iterations = 0, 0, 0
    active_candidate = None

    def execute(name, function):
        nonlocal count, active_candidate
        if count >= 95:
            raise RuntimeError('Fit budget exhausted')
        count += 1
        active_candidate = name
        journal(events, {'state':'STARTED','candidate':name,'fit_number':count})
        start = time.perf_counter()
        try:
            value = function()
        except BaseException as exc:
            journal(events, {'state':'FAILED','candidate':name,'error':repr(exc)})
            active_candidate = None
            raise
        return value, time.perf_counter()-start

    def complete(name):
        nonlocal active_candidate
        if active_candidate != name:
            raise RuntimeError('Candidate lifecycle mismatch')
        journal(events, {'state':'COMPLETED','candidate':name,'fit_number':count})
        active_candidate = None

    try:
        for fold_cfg in c['roles']['folds']:
            fold = fold_cfg['id']
            rr = folds[fold]
            tr, va, ev = (rr[k] for k in ('fit','inner_validation','evaluation'))
            # Only these role-sliced arrays can reach the unchanged fit functions.
            ytrain, yval = fit_role_targets(labels,tr), fit_role_targets(labels,va)
            folder = out / fold
            folder.mkdir()
            scaler_object = StandardScaler().fit(features['ordinary_raw'][tr])
            mean, scale = scaler_object.mean_, scaler_object.scale_
            np.testing.assert_allclose(mean, features['ordinary_raw'][tr].mean(axis=0), rtol=1e-12, atol=1e-12)
            scaler = {'mean':mean.tolist(),'scale':scale.tolist(),'fit_count':len(tr),
                      'fit_ids_sha256':data.array_sha(features['ids'][tr]),
                      'scaler_mean_sha256':data.array_sha(mean),'scaler_scale_sha256':data.array_sha(scale),
                      'target_median':float(np.median(np.maximum(ytrain,1e-12))),
                      'target_median_sha256':data.array_sha(np.array([np.median(np.maximum(ytrain,1e-12))])),
                      'constant_RV':float(np.clip(np.maximum(ytrain,1e-12).mean(),1e-12,1)),
                      'FIT_labelable_at_max':str(pd.to_datetime(metadata.iloc[tr].labelable_at,utc=True).max()),
                      'FIT_label_end_max':str(pd.to_datetime(metadata.iloc[tr].label_end,utc=True).max()),
                      'fold_id':fold,'fitted_role':'fit','global_scaler_used':False}
            data.write(folder / 'scaling_audit.json',dict(scaler,status='PASS'))
            selected = {'B5':{}}
            inner_predictions = []
            candidate_records = []
            for family in c['models']['linear_families']:
                candidates = []
                for lam in c['models']['linear_lambda_candidates']:
                    name = f'{fold}/{family}_lambda_{lam}'
                    model, elapsed = execute(name, lambda:models.fit_head(feature_matrix(features,family,tr),ytrain,lam))
                    path = folder / f'{family}_lambda_{lam}.json'
                    data.write(path,model)
                    if not model['eligible']:
                        raise RuntimeError('Registered linear candidate failed eligibility; no retry: ' + name)
                    score = None
                    if model['eligible']:
                        p = models.predict_head(model,feature_matrix(features,family,va))
                        np.testing.assert_array_equal(p,models.predict_head(data.read(path),feature_matrix(features,family,va)))
                        score = float(np.mean(models.qlike(yval,p)))
                        candidates.append({'model':model,'validation_qlike':score,'path':str(path)})
                        inner_predictions.append(pd.DataFrame({'opportunity_id':features['ids'][va], 'candidate':name,'prediction':p}))
                    resources.append({'candidate':name,'family':family,'elapsed_seconds':elapsed,
                                      'checkpoint_bytes':path.stat().st_size,'parameters':len(model['coef'])+1,
                                      'solver_iterations':model['solver']['iterations']})
                    candidate_records.append({'candidate':name,'eligible':model['eligible'],'inner_validation_qlike':score})
                    complete(name)
                selected[family] = models.select_candidate(candidates)['path']
            candidates = []
            for leaves, minleaf in c['models']['B2_structures']:
                name = f'{fold}/B2_leaves_{leaves}_minleaf_{minleaf}'
                model, elapsed = execute(name,lambda:models.fit_b2(feature_matrix(features,'B2',tr),ytrain,
                      feature_matrix(features,'B2',va),yval,leaves,minleaf))
                path = folder / f'B2_leaves_{leaves}_minleaf_{minleaf}.json'
                data.write(path,model)
                p = models.predict_b2(data.read(path),feature_matrix(features,'B2',va))
                score = float(np.mean(models.qlike(yval,p)))
                np.testing.assert_allclose(score,model['validation_qlike'],rtol=1e-12,atol=1e-12)
                iterations += model['training_iterations']
                candidates.append({'model':model,'validation_qlike':score,'path':str(path)})
                inner_predictions.append(pd.DataFrame({'opportunity_id':features['ids'][va],'candidate':name,'prediction':p}))
                resources.append({'candidate':name,'family':'B2','elapsed_seconds':elapsed,'checkpoint_bytes':path.stat().st_size,
                                  'training_iterations':model['training_iterations'],'parameters':None})
                candidate_records.append({'candidate':name,'eligible':True,'inner_validation_qlike':score})
                complete(name)
            selected['B2'] = models.select_b2(candidates)['path']
            ordinary_tr = scaler_object.transform(features['ordinary_raw'][tr]).astype(np.float32)
            ordinary_va = scaler_object.transform(features['ordinary_raw'][va]).astype(np.float32)
            for seed in c['models']['B5_seeds']:
                name = f'{fold}/B5_s{seed}'
                payload, elapsed = execute(name,lambda:b5.fit_b5(features['windows'][tr],ordinary_tr,ytrain,
                    features['windows'][va],ordinary_va,yval,width=32,weight_decay=.001,seed=seed,
                    device=c['models']['B5_device'],max_epochs=120,patience=15,batch_size=64,lr=.001))
                path = folder / f'B5_s{seed}.pt'
                if path.exists():
                    raise FileExistsError(path)
                # Reserve exclusively before old writer; it writes only this new file.
                with path.open('xb'):
                    pass
                b5.save_b5(payload,path)
                loaded = b5.load_b5(path,device=c['models']['B5_device'])
                p = b5.predict_b5(payload,features['windows'][va],ordinary_va)['prediction']
                reloaded = b5.predict_b5(loaded,features['windows'][va],ordinary_va)['prediction']
                np.testing.assert_array_equal(p,reloaded)
                score = b5.raw_qlike(yval,reloaded)
                np.testing.assert_allclose(score,payload['metadata']['best_valid_raw_qlike'],rtol=1e-12,atol=1e-12)
                data.write(folder / f'B5_s{seed}.json',{'metadata':payload['metadata'],'history':payload['history'],
                                                       'resources':payload['resources'],'reload_identity':'PASS'})
                selected['B5'][str(seed)] = str(path)
                epochs += payload['resources']['epochs_run']
                resources.append(dict(payload['resources'],candidate=name,family='B5',checkpoint_bytes=path.stat().st_size,
                                      elapsed_seconds=elapsed))
                inner_predictions.append(pd.DataFrame({'opportunity_id':features['ids'][va],'candidate':name,'prediction':p}))
                candidate_records.append({'candidate':name,'eligible':True,'inner_validation_qlike':score})
                complete(name)
                del payload, loaded
                gc.collect()
            data.write(folder / 'selected_models.json',{'selected':selected,'fold_id':fold,
                       'selection_role':'inner_validation','no_refit':True,'scaling_audit':scaler,
                       'candidates':candidate_records})
            data.frame(folder / 'inner_validation_predictions.csv',pd.concat(inner_predictions,ignore_index=True))
            # All family selections are frozen before touching the outer features.
            outer, inference_identity = predict_only(features,ev,selected,scaler,device=c['models']['B5_device'])
            data.write(folder / 'inference_audit.json',{'status':'PASS','fold_id':fold,'B5':inference_identity})
            predictions.append(prediction_frame(metadata,ev,outer,fold,fold_cfg['evaluation_start']))
            identities[fold] = {'selected':selected,'source_sha256':{str(p):data.sha(p) for family,p in selected.items() if family != 'B5'},
                                'B5_sha256':{s:data.sha(p) for s,p in selected['B5'].items()},'evaluation_count':len(ev)}
            verify_seal(run)
        if count != 95 or epochs > 1800 or iterations > 6000:
            raise RuntimeError('Actual registered budget violated')
        result = pd.concat(predictions,ignore_index=True)
        if len(result) != 1267 or not result.opportunity_id.is_unique:
            raise RuntimeError('OOF completeness failed')
        data.frame(run / 'predictions/oof_predictions.csv',result)
        data.write(out / 'selected_fold_map.json',identities)
        data.write(out / 'candidate_audit.json',{'status':'PASS','fits':count,'linear_fits':60,'B2_fits':20,'B5_fits':15,
                   'actual_B5_epochs':epochs,'actual_B2_iterations':iterations,'fit_event_pairs':95,
                   'outer_targets_passed_to_fit_or_inference':False,'all_inner_reload_identity':'PASS',
                   'outer_B5_repeat_single_batch_identity':'PASS','deterministic_refits':0,'resources':resources,
                   'protocol_sha256':seal['protocol_sha256']})
        verify_seal(run)
    except BaseException as exc:
        if active_candidate is not None:
            journal(events, {'state':'FAILED','candidate':active_candidate,'error':repr(exc)})
            active_candidate = None
        data.write(out / 'training_failure.json',{'status':'FAILED','error':repr(exc),'fits_started':count,
                   'actual_B5_epochs':epochs,'actual_B2_iterations':iterations,'retry_allowed':False,'at_utc':now()})
        raise
    return result


def validation(run=data.DEFAULT_RUN):
    """Original TRAIN frozen artifacts only. This action performs zero fits."""
    run,c,_ = data.new_scope(run)
    verify_seal(run)
    features, metadata = data.load_features(run)
    tr = np.flatnonzero(metadata.role.to_numpy() == 'train')
    va = np.flatnonzero(metadata.role.to_numpy() == 'validation')
    if (len(tr),len(va)) != (2369,92):
        raise ValueError('Original DEV roles differ')
    parent = data.ROOT / c['sources']['parent_03']
    bparent = data.ROOT / c['sources']['parent_03_1']
    original = {row['family']:row for row in data.read(parent / 'models/formal/selected_models.json')}
    selected = {}
    for family in ('har','R1','R2','B2'):
        item = original[family]
        path = item['model_path']
        selected[family] = str(parent / path) if not Path(path).is_absolute() else path
    bselected = data.read(bparent / 'models/selected.json')
    if bselected['width'] != 32 or bselected['weight_decay'] != .001:
        raise ValueError('Original B5 selected structure changed')
    selected['B5'] = {str(s):str(bparent / f'models/w32_wd0.001_s{s}.pt') for s in (17,29,43)}
    scaler = data.read(bparent / 'inputs/train_scaler.json')
    rebuilt = StandardScaler().fit(features['ordinary_raw'][tr])
    np.testing.assert_allclose(rebuilt.mean_,scaler['mean'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(rebuilt.scale_,scaler['scale'],rtol=1e-12,atol=1e-12)
    labels = pd.read_csv(run / 'labels/development.csv',usecols=['opportunity_id','RV_raw'],float_precision='round_trip')
    scaler['constant_RV'] = float(np.clip(np.maximum(fit_role_targets(labels,tr),1e-12).mean(),1e-12,1))
    p, identity = predict_only(features,va,selected,scaler,device=c['models']['B5_device'])
    data.write(run / 'predictions/validation_inference_audit.json',{'status':'PASS','fold_id':'VALID','B5':identity})
    output = prediction_frame(metadata,va,p,'VALID',c['roles']['original_train_end'])
    data.frame(run / 'predictions/validation_predictions.csv',output)
    data.write(run / 'predictions/validation_audit.json',{'status':'PASS','fits':0,'rows':92,'original_train_scaler_verified':True,
                 'selected':selected,'validation_targets_in_prediction_interface':False})
    verify_seal(run)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train-oof','validation','benchmark'])
    parser.add_argument('--run',type=Path,default=data.DEFAULT_RUN)
    args = parser.parse_args()
    if args.action == 'prepare':
        data.prepare(args.run)
    elif args.action == 'train-oof':
        train_oof(args.run)
    elif args.action == 'validation':
        validation(args.run)
    else:
        from research.m2_ci.benchmark import benchmark
        benchmark(args.run)


if __name__ == '__main__':
    main()
