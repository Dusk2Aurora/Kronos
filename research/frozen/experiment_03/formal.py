"""Only executable one-shot formal entrypoint. No approval is created by this code."""
from __future__ import annotations
import argparse
import io
import json
import traceback
import subprocess
import numpy as np
import pandas as pd
from research.frozen import encoder
from research.frozen.experiment_03 import guard,inputs,models,statistics
from research.frozen.experiment_03.run import fit,read_prepared,immutable_json,immutable_csv

def encode_test(candles,opp):
    guard.require_scope(guard.config()['roles']['holdout'][1],'B')
    windows=[inputs.prepare_window(candles,row,phase='B') for _,row in opp.iterrows()]
    result={}; cfg=guard.config(); batch=int(cfg['encoding']['batch_size'])
    for variant in ('pretrained','random_s17','random_s29','random_s43'):
        seed=None if variant=='pretrained' else int(variant.split('s')[-1])
        tokenizer,backbone,meta=encoder.load_models('pretrained' if seed is None else 'random',seed,'cuda')
        chunks=[encoder.encode_batch(tokenizer,backbone,windows[i:i+batch]) for i in range(0,len(windows),batch)]
        values=np.concatenate(chunks);sample=[0,len(opp)//2,len(opp)-1]
        separate=np.concatenate([encoder.encode_batch(tokenizer,backbone,[windows[i]]) for i in sample])
        repeated=encoder.encode_batch(tokenizer,backbone,[windows[i] for i in sample])
        if not np.allclose(separate,values[sample],atol=1e-4,rtol=1e-4) or not np.allclose(repeated,values[sample],atol=1e-4,rtol=1e-4):
            raise ValueError('Formal frozen encode repeat/batch mismatch '+variant)
        if encoder.state_sha(backbone)!=meta['model_state_sha256'] or encoder.state_sha(tokenizer)!=meta['tokenizer_state_sha256']:
            raise ValueError('Formal frozen weights mutated')
        buffer=io.BytesIO();np.savez_compressed(buffer,ids=np.asarray(opp.index,dtype='U20'),features=values,window_identities=np.asarray([encoder.canonical(w[2]) for w in windows],dtype=str))
        out=guard.RUN/'features'/('holdout_'+variant+'.npz');encoder.immutable_bytes(out,buffer.getvalue())
        immutable_json(guard.RUN/'features'/('holdout_'+variant+'_audit.json'),{'status':'PASS','rows':len(opp),'model':meta,'features_sha256':guard.sha(out),
             'sample_ids':opp.index[sample].tolist(),'repeat_max_abs':float(np.abs(repeated-values[sample]).max()),'single_batch_max_abs':float(np.abs(separate-values[sample]).max())})
        result[variant]=pd.DataFrame(values,index=opp.index,columns=[f'kronos_{i:03d}' for i in range(512)])
        del tokenizer,backbone
    return result

def predict_test(feature,hidden,selected):
    guard.require_scope(guard.config()['roles']['holdout'][1],'B')
    pred={};clipping=[]
    for s in selected:
        family=s['family'];model=json.loads((guard.RUN/s['model_path']).read_text())
        matrix=feature.loc[:,inputs.FEATURE_NAMES]
        if family=='har':matrix=feature.loc[:,inputs.HAR_NAMES]
        elif family=='R2':matrix=pd.concat([matrix,hidden['pretrained']],axis=1)
        elif family.startswith('random_'):matrix=pd.concat([matrix,hidden[family]],axis=1)
        values,stats=(models.predict_b2_with_stats if family=='B2' else models.predict_head_with_stats)(model,matrix.to_numpy())
        pred[family]=values;clipping.append({'family':family,**stats})
    for family,col in [('persistence','r0_persistence'),('ewma','r0_ewma')]:
        values,stats=models.clip_predictions(feature[col].to_numpy());pred[family]=values;clipping.append({'family':family,**stats})
    immutable_json(guard.RUN/'predictions/holdout_clipping.json',clipping)
    return pred

def independent_numbers(frame,pred,summary):
    """Independent pair-comparison AUROC for every saved primary bootstrap draw."""
    guard.require_scope(guard.config()['roles']['holdout'][1],'B')
    from sklearn.metrics import roc_auc_score,average_precision_score
    checks=[];eps=1e-12; y=np.log((frame.RV_raw.to_numpy()+eps)/(pred['ewma']+eps))>np.log(2)
    # Independent every-label reconstruction from retained traded prices/times.
    independent=[]
    for _,row in frame.iterrows():
        prices=np.asarray(json.loads(row.label_prices_json),dtype=float);times=pd.DatetimeIndex(json.loads(row.label_bar_open_at_json))
        if len(prices)!=49 or len(times)!=48 or not times.equals(pd.date_range(pd.Timestamp(row.entry_at),periods=48,freq='5min')):
            raise AssertionError('Independent formal label segment/window mismatch')
        independent.append(sum(float(np.log(prices[i+1]/prices[i]))**2 for i in range(48)))
    if not np.allclose(frame.RV_raw,independent,rtol=1e-10,atol=1e-16):raise AssertionError('Independent all-label RV mismatch')
    checks.append({'check':'all_labels_48_price_ratio_returns','rows':len(frame),'status':'PASS'})
    target=np.maximum(frame.RV_raw.to_numpy(),eps)
    for family,p in pred.items():
        ratio=target/p
        independent_losses={'raw_QLIKE':float(np.mean(np.log(p)+ratio)),
            'QLIKE_Regret':float(np.mean(ratio-np.log(ratio)-1)),
            'logRV_MSE':float(np.mean(np.log(target/p)**2)),
            'additive_epsilon_logRV_MSE':float(np.mean(np.log((frame.RV_raw.to_numpy()+eps)/(p+eps))**2))}
        for name,value in independent_losses.items():
            if not np.isclose(value,summary['variance_losses'][family][name],rtol=1e-12,atol=1e-12):raise AssertionError('Independent variance metric mismatch')
        checks.append({'check':'all_variance_losses','family':family,'status':'PASS'})
    dates=pd.to_datetime(frame.decision_at,utc=True);cfg=guard.config()
    scores={f:np.log((v+eps)/(pred['ewma']+eps)) for f,v in pred.items()}
    for qi,q in enumerate(cfg['roles']['test_quarters']):
        start,end=pd.Timestamp(q['start']),pd.Timestamp(q['end_exclusive']);mask=((dates>=start)&(dates<end)).to_numpy();yy=y[mask]
        ids=(dates[mask].dt.normalize()-start).dt.days.to_numpy()
        for family in scores:
            s=scores[family][mask];point=summary['quarter_metrics'][qi]['models'][family]
            if yy.any() and (~yy).any():
                a=roc_auc_score(yy,s);p=average_precision_score(yy,s)
                if not np.isclose(a,point['AUROC'],atol=1e-12,rtol=0) or not np.isclose(p,point['AP'],atol=1e-12,rtol=0):raise AssertionError('Independent point ranking mismatch')
        for block in (7,3,14):
            with np.load(guard.RUN/'bootstrap'/f'multiplicities_block{block}.npz',allow_pickle=False) as f:counts=f[q['id']+'_counts']
            draws=pd.read_csv(guard.RUN/'bootstrap'/f'draws_block{block}.csv',float_precision='round_trip')
            wp=counts[:,ids[yy]].astype(float);wn=counts[:,ids[~yy]].astype(float);den=wp.sum(axis=1)*wn.sum(axis=1)
            valid=den>0
            for family in ('R1','R2','B2'):
                s=scores[family][mask];comparison=(s[yy,None]>s[None,~yy]).astype(float)+.5*(s[yy,None]==s[None,~yy])
                numer=np.einsum('di,ij,dj->d',wp,comparison,wn,optimize=True)
                result=np.divide(numer,den,out=np.full(len(den),np.nan),where=valid)
                expected=draws[q['id']+'_'+family+'_AUROC'].to_numpy()
                if not np.allclose(result,expected,atol=1e-12,rtol=0,equal_nan=True):raise AssertionError('Independent every-draw paired ranking mismatch')
                checks.append({'quarter':q['id'],'block_days':block,'family':family,'draws':len(result),'method':'pairwise_positive_negative_comparison_matrix','status':'PASS'})
    # Recompute equal-quarter contrasts and each percentile interval from saved draws.
    for block in (7,3,14):
        table=pd.read_csv(guard.RUN/'bootstrap'/f'draws_block{block}.csv',float_precision='round_trip')
        for key,(a,b) in statistics.COMPARISONS.items():
            values=np.mean([table[q['id']+'_'+a+'_AUROC']-table[q['id']+'_'+b+'_AUROC'] for q in cfg['roles']['test_quarters']],axis=0)
            valid=np.isfinite(values);ci=np.quantile(values[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
            registered=summary['bootstrap'][str(block)]['comparisons'][key]
            if not np.allclose(ci,[registered['ci_lower'],registered['ci_upper']],atol=1e-12,rtol=0,equal_nan=True):raise AssertionError('Independent CI mismatch')
            checks.append({'block_days':block,'comparison':key,'method':'equal-quarter and percentile independently reconstructed','status':'PASS'})
    immutable_json(guard.RUN/'independent_review/formal_numeric.json',{'status':'PASS','checks':checks,'all_primary_attempts_verified':True})

def execute():
    # Exclusive creation is before any new test read; rerun always rejects.
    guard.claim_formal()
    try:
        bundle=guard.verify_bundle()
        for rel in bundle['canonical_code_sha256']:
            encoder.immutable_bytes(guard.RUN/'provenance/formal_source'/rel,(guard.ROOT/rel).read_bytes())
        immutable_json(guard.RUN/'provenance/formal_source_manifest.json',{'protocol_sha256':guard.CONFIG_SHA,
            'sealed_bundle_sha256':guard.sha(guard.RUN/'preflight/sealed_bundle.json'),
            'code_sha256':bundle['canonical_code_sha256'],'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=guard.ROOT,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=guard.ROOT,text=True),
            'created_at_utc':guard.now()})
        selected=fit('B')
        from research.frozen.experiment_03 import data,diagnostics
        data.collect_holdout(); five=data.load_holdout_5m();candles=inputs.load_hourly('B')
        opp=pd.concat([inputs.opportunities(candles,q['id'],'B') for q in guard.config()['roles']['test_quarters']]).set_index('opportunity_id',drop=False).sort_index()
        labels=inputs.build_labels(five,opp,'B').set_index('opportunity_id',drop=False).loc[opp.index]
        feature=inputs.build_features(candles,opp,'B'); hidden=encode_test(candles,opp)
        # Perturb all future hourly values: historical features/windows must remain identical.
        causal=[]
        for pos in (0,len(opp)//2,len(opp)-1):
            one=opp.iloc[[pos]];altered=candles.copy();mask=altered.bar_open_at>=pd.Timestamp(one.iloc[0].history_end_exclusive)
            altered.loc[mask,encoder.FIELDS]=altered.loc[mask,encoder.FIELDS]*7+11
            if not np.array_equal(inputs.build_features(altered,one,'B').to_numpy(),feature.loc[one.index].to_numpy()):raise ValueError('Future changed formal historical features')
            old=inputs.prepare_window(candles,one.iloc[0],'B');new=inputs.prepare_window(altered,one.iloc[0],'B')
            if not np.array_equal(old[0],new[0]) or old[2]!=new[2]:raise ValueError('Future changed formal frozen window')
            causal.append(str(one.index[0]))
        immutable_json(guard.RUN/'features/holdout_causality.json',{'status':'PASS','ids':causal})
        pred=predict_test(feature,hidden,selected)
        immutable_csv(guard.RUN/'labels/holdout.csv',labels,False);immutable_csv(guard.RUN/'features/holdout_ordinary_risk.csv',feature)
        frame=labels.copy();out=frame.loc[:,['opportunity_id','decision_at','decision_boundary_at','role','entry_at','label_end','labelable_at','RV_raw']].copy()
        for family,values in pred.items():out[family]=values
        immutable_csv(guard.RUN/'predictions/holdout.csv',out,False)
        thresholds=json.loads((guard.RUN/'calibration/training_thresholds.json').read_text())
        result=statistics.evaluate_quarters(frame,pred,guard.RUN/'bootstrap',guard.config(),training_thresholds={'high_RV':thresholds['train_q90']})
        immutable_json(guard.RUN/'metrics/formal_result.json',statistics._json_clean(result))
        calibration=json.loads((guard.RUN/'calibration/sealed_calibration.json').read_text())
        diagnostics.analyze(frame,pred,calibration,guard.RUN/'diagnostics/holdout',context='authorized_holdout')
        independent_numbers(frame,pred,result)
        guard.verify_bundle()  # Full closure below clears the stat cache deliberately.
        guard._BUNDLE_CACHE=None;guard.verify_bundle()
        guard.finish_formal('CONSUMED',{'status':result['status'],'numeric_review':'PASS','protocol_sha256':guard.CONFIG_SHA})
        print(json.dumps({'status':result['status'],'holdout':'CONSUMED'}),flush=True)
    except BaseException as error:
        # Terminal evidence must survive even when a source/hash change caused failure.
        guard.finish_formal('INVALIDATED',{'error':repr(error),'traceback':traceback.format_exc(),'no_retry':True})
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-once',action='store_true',required=True);p.parse_args();execute()
    # Publication consumes only already-generated artifacts after terminal CONSUMED;
    # a report failure never repeats a scientific run.
    try:
        from research.frozen.experiment_03.report import phase_b
        phase_b()
    except Exception as error:
        immutable_json(guard.RUN/'reports/publication_error.json',{'status':'PUBLICATION_PENDING','error':repr(error),
            'holdout_remains':'CONSUMED','no_scientific_retry':True})
        raise
if __name__=='__main__':main()
