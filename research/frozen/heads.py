"""Fit only downstream Ridge heads on accepted development-only frozen caches."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
from prepare_e00r import fit_candidates, validation_cutoffs, choose_candidate
from ordinary_features_v2 import FEATURE_NAMES, features_for_opportunity
from summarize_e00r import development_labels
from encoder import load_cache
import json
import shutil
import numpy as np
import pandas as pd
import yaml

FAMILIES = {'ridge_pretrained':'pretrained', **{f'ridge_random_s{s}':f'random_s{s}' for s in (17,29,43)}}

def reuse_baseline(output, config, common, labels, split):
    baseline = ROOT / common['variants']['E00R']['reuse_run']
    registry = json.loads((ROOT/'research/registry'/f'{baseline.name}.json').read_text(encoding='utf-8'))
    require(registry['status'] == 'completed_exploratory_development', 'Baseline not accepted')
    original = yaml.safe_load((baseline/'provenance/revision_config.yaml').read_text(encoding='utf-8'))
    old_config = yaml.safe_load((baseline/'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    for k in ['head','target','ordinary_features','actions','controls']:
        require(common[k] == original[k], f'Common design differs: {k}')
    for k in ['market','timing','fixed_strategy','splits','labels_and_costs','backtest_engine']:
        require(config[k] == old_config[k], f'Immutable contract differs: {k}')
    prep = {k.replace('\\','/'):v for k,v in registry['artifacts']['preparation_hashes'].items()}
    copies=['development_features.csv','training_attempts.json','selected_heads.json',
            'development_test_predictions.csv','signals_manifest.json','metrics.csv','engine_report.json']
    target=output/'baseline_reuse'; target.mkdir()
    hashes={}
    for name in copies:
        p=baseline/name
        expected=prep.get(name)
        if expected is None:
            for group in ['analysis_hashes','final_hashes','engine_hashes']:
                expected={k.replace('\\','/'):v for k,v in registry['artifacts'].get(group,{}).items()}.get(name)
                if expected:break
        require(expected is not None and sha(p)==expected, 'Baseline hash not verified: '+name)
        shutil.copy2(p,target/name); hashes[name]=sha(p)
    features=pd.read_csv(target/'development_features.csv').set_index('opportunity_id')
    require(tuple(features.columns)==FEATURE_NAMES and features.index.is_unique and set(features.index)==set(labels.index), 'Baseline features/membership differ')
    attempts=json.loads((target/'training_attempts.json').read_text(encoding='utf-8'))
    old_heads=json.loads((target/'selected_heads.json').read_text(encoding='utf-8'))
    old_pred=pd.read_csv(target/'development_test_predictions.csv')
    for fold in config['splits']['walk_forward_dates']:
        candidates=[a for a in attempts if a['fold_id']==fold['id'] and a['feature_set']=='ridge19']
        require(len(candidates)==4 and {a['lambda'] for a in candidates}==set(common['head']['lambda_candidates'])
                and all(a['status']=='completed' for a in candidates),'Baseline candidate budget/status differs')
        chosen=choose_candidate(candidates)
        head=next(h for h in old_heads if h['fold_id']==fold['id'] and h['feature_set']=='ridge19')
        require(chosen['lambda']==head['lambda'] and chosen['validation_mse_bps_squared']==head['validation_mse_bps_squared'],'Baseline selected head differs')
        test=sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq('test'),'opportunity_id'])
        saved=old_pred.loc[old_pred.fold_id.eq(fold['id']) & old_pred.feature_set.eq('ridge19')]
        require(saved.opportunity_id.is_unique and sorted(saved.opportunity_id)==test,'Baseline test predictions membership differs')
        for attempt in candidates:
            train=sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq('train'),'opportunity_id'])
            valid=sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq('validation'),'opportunity_id'])
            require(attempt['train_ids']==train and attempt['validation_ids']==valid and attempt['feature_names']==list(FEATURE_NAMES), 'Baseline fit membership differs')
            x=features.loc[train].to_numpy(float)
            require(attempt['alpha']==len(train)*attempt['lambda'] and np.allclose(x.mean(0),attempt['scaler_mean'])
                    and np.allclose(x.var(0),attempt['scaler_var']), 'Baseline scaler/head differs')
            vx=features.loc[valid].to_numpy(float)
            predicted=((vx-attempt['scaler_mean'])/attempt['scaler_scale']) @ np.asarray(attempt['coef'])+attempt['intercept']
            require(np.allclose(predicted,attempt['validation_predictions_bps']) and
                    np.isclose(np.mean((predicted-labels.loc[valid,'base_profit_ratio'].to_numpy()*10000)**2),attempt['validation_mse_bps_squared']), 'Baseline target/predictions differ')
    # Independent source-window spot checks: accepted complete prior causal audit is reused.
    bundle=ROOT/config['labels_and_costs']['verified_label_bundle']
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    source=pd.read_csv(Path(manifest['sources']['snapshot'])/'candles.csv',dtype=str,
                       usecols=['bar_open_at','open','high','low','close','volume','amount'])
    source=source.loc[pd.to_datetime(source.bar_open_at,utc=True)<pd.Timestamp('2026-04-01T00:00Z')]
    opp=pd.read_csv(bundle/'opportunities.csv').set_index('opportunity_id')
    for ident in [features.index[0],features.index[len(features)//2],features.index[-1]]:
        actual=features_for_opportunity(source,opp.loc[ident])
        require(np.allclose(features.loc[ident],list(actual.values()),rtol=1e-10,atol=1e-12), 'Ordinary features changed')
    write_json(output/'baseline_reuse_audit.json',{'status':'passed','source':str(baseline),'hashes':hashes,
                'exact_common_design':True,'train_validation_test_membership_and_target_checked':True,
                'prior_causal_audit_reused':True,'recomputed_source_windows':3,'holdout_inspected':False})
    return features

def prepare(output):
    output=output.resolve(); registry_path=ROOT/'research/registry'/f'{output.name}.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    require(registry['status']=='protocol_locked_pending_encoding','Unexpected run status')
    require(not (output/'training_attempts.json').exists(),'Do not overwrite head attempts')
    common=yaml.safe_load((output/'provenance/common_config.yaml').read_text(encoding='utf-8'))
    config=yaml.safe_load((output/'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    require(common['status']=='locked' and config['stage']=='frozen_development' and
            not config['evaluation_readiness']['final_holdout_evaluation_allowed'],'Stage/scope')
    require(sha(output/'provenance/common_config.yaml')==registry['configuration']['sha256'] and
            sha(output/'provenance/experiment_config.yaml')==registry['parent_configuration']['sha256'],'Configuration changed')
    source_dir=output/'provenance/heads_source'; source_dir.mkdir()
    paths=[Path(__file__),Path(__file__).with_name('encoder.py'),ROOT/'research/baselines/prepare_e00.py',
           ROOT/'research/baselines/prepare_e00r.py',ROOT/'research/baselines/ordinary_features.py',
           ROOT/'research/baselines/ordinary_features_v2.py',ROOT/'research/baselines/summarize_e00r.py',
           ROOT/'research/freqtrade/strategies/KronosE00.py']
    for p in paths:
        rel=p.relative_to(ROOT).as_posix(); registry['provenance']['canonical_source_sha256'][rel]=sha(p)
        dest=source_dir/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    trials=[{'fold_id':f['id'],'feature_set':family,'lambda':float(lam),'status':'planned'}
            for f in config['splits']['walk_forward_dates'] for family in FAMILIES for lam in common['head']['lambda_candidates']]
    registry['selection']['all_attempts']=trials
    registry['status']='encoding_accepted_heads_running'
    write_json(registry_path,registry); write_json(output/'training_attempts.json',trials)
    try:
        labels,split,bundle=development_labels(config)
        manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
        require(sha(bundle/'manifest.json')==registry['provenance']['label_manifest_sha256'],'Label manifest changed')
        for name in ['labels_forbidden_as_features.csv','opportunities.csv','split_index.csv']:
            require(sha(bundle/name)==manifest['artifact_sha256'][name],'Changed label bundle '+name)
        snapshot=Path(manifest['sources']['snapshot'])
        require(sha(snapshot/'manifest.json')==manifest['sources']['snapshot_manifest_sha256'] and
                sha(snapshot/'candles.csv')==manifest['sources']['snapshot_csv_sha256']['candles.csv'],'Snapshot changed')
        features=reuse_baseline(output,config,common,labels,split)
        matrices={}; contracts={}
        for family,folder in FAMILIES.items():
            cache=output/'cache'/folder
            hidden=load_cache(cache)
            contract=json.loads((cache/'contract.json').read_text(encoding='utf-8'))
            require(contract['protocol_sha256']==registry['configuration']['sha256'] and
                    contract['initial_config_sha256']==registry['parent_configuration']['sha256'],'Cache belongs to different protocol/stage')
            for key,path in {'snapshot_manifest_sha256':snapshot/'manifest.json','candles_sha256':snapshot/'candles.csv',
                             'opportunities_sha256':bundle/'opportunities.csv','split_index_sha256':bundle/'split_index.csv'}.items():
                require(contract['sources'][key]==sha(path),'Cache source changed: '+key)
            require(hidden.index.is_unique and set(hidden.index)==set(features.index) and hidden.shape[1]==512,'Hidden ID/shape mismatch')
            require(contract['model']['variant']==('pretrained' if folder=='pretrained' else 'random'),'Wrong backbone')
            require(contract['model']['seed']==(None if folder=='pretrained' else int(folder.rsplit('s',1)[1])),'Wrong seed')
            matrices[family]=features.join(hidden,validate='one_to_one').astype(float)
            require(np.isfinite(matrices[family]).all().all(),'Nonfinite combined features')
            matrices[family].to_csv(output/f'features_{family}.csv')
            contracts[family]=contract
        weights=[c['model']['model_state_sha256'] for c in contracts.values()]
        require(len(set(weights))==4 and len(set(c['model']['tokenizer_state_sha256'] for c in contracts.values()))==1,'Random pretrained weights reused/tokenizer changed')
        for field in ['sources','dtype','normalization','model_config','pooling','time_fields']:
            values=[c['sources'] if field=='sources' else c['model'][field] for c in contracts.values()]
            require(all(x==values[0] for x in values),'Representation control differs: '+field)
        write_json(output/'encoding_head_input_audit.json',{'status':'passed','rows':len(features),'hidden_dimension':512,
                   'combined_features':531,'distinct_model_states':weights,'tokenizer_state_sha256':next(iter(contracts.values()))['model']['tokenizer_state_sha256'],
                   'checks':['per_cache_byte_and_array_hashes','independent_causal_window_checks','identical_sources_and_architecture',
                             'unique_ID_one_to_one_merge','random_backbone_distinct_unchanged','same_pretrained_tokenizer'], 'holdout_inspected':False})
        opp=pd.read_csv(bundle/'opportunities.csv').set_index('opportunity_id')
        predictions=[]; selected=[]; signals=[]
        (output/'decisions').mkdir()
        for fold in config['splits']['walk_forward_dates']:
            groups={role:sorted(split.loc[split.fold_id.eq(fold['id']) & split.role.eq(role),'opportunity_id']) for role in ['train','validation','test']}
            for role,ids in groups.items():
                start,end=map(pd.Timestamp,fold[role])
                require(ids and all(start<=pd.Timestamp(x)<end and pd.Timestamp(opp.loc[x,'exit_at'])<end and pd.Timestamp(opp.loc[x,'labelable_at'])<end for x in ids),'Unpurged role')
            require(not set(groups['train']) & set(groups['validation']) and not set(groups['test']) & (set(groups['train']) | set(groups['validation'])),'Role overlap')
            for family,matrix in matrices.items():
                def record(candidate):
                    pending=next(a for a in trials if a['fold_id']==fold['id'] and a['feature_set']==family and a['lambda']==candidate['lambda'])
                    pending.update(candidate,feature_names=list(matrix.columns),train_ids=groups['train'],validation_ids=groups['validation'],
                                   train_samples=len(groups['train']),validation_samples=len(groups['validation']))
                    write_json(output/'training_attempts.json',trials)
                scaler,winner=fit_candidates(matrix.loc[groups['train']].to_numpy(), labels.loc[groups['train'],'base_profit_ratio'].to_numpy()*10000,
                                             matrix.loc[groups['validation']].to_numpy(),labels.loc[groups['validation'],'base_profit_ratio'].to_numpy()*10000,
                                             common['head']['lambda_candidates'],record)
                cuts=validation_cutoffs(winner['validation_predictions_bps'])
                scores=winner['model'].predict(scaler.transform(matrix.loc[groups['test']].to_numpy()))
                gates={key:scores>=value for key,value in cuts.items()}; gates['economic']=scores>0
                selected.append({'fold_id':fold['id'],'feature_set':family,'lambda':winner['lambda'],'alpha':winner['alpha'],
                                 'validation_mse_bps_squared':winner['validation_mse_bps_squared'],'validation_cutoffs_bps':cuts,
                                 'training_mean_bps':float(labels.loc[groups['train'],'base_profit_ratio'].mean()*10000),
                                 'test_samples':len(scores),'primary':family=='ridge_pretrained'})
                for i,(ident,score) in enumerate(zip(groups['test'],scores)):
                    predictions.append({'fold_id':fold['id'],'feature_set':family,'opportunity_id':ident,'score_bps':float(score),
                                        'gate50':int(gates['rank50'][i]),'gate25':int(gates['rank25'][i]),'gate75':int(gates['rank75'][i]),'economic':int(gates['economic'][i])})
                for gate,actions in gates.items():
                    mode=family+'_'+gate; path=Path('decisions')/f'{fold["id"]}_{mode}.json'
                    payload={'schema_version':1,'mode':'ordinary_features_gate','decisions':[
                        {'signal_at':ident,'direction':int(opp.loc[ident,'direction']),'exposure_fraction':float(action)} for ident,action in zip(groups['test'],actions)]}
                    write_json(output/path,payload)
                    signals.append({'fold_id':fold['id'],'mode':mode,'path':path.as_posix(),'selected_count':int(actions.sum()),
                                    'opportunities':len(actions),'family':family,'gate':gate,'primary':gate in ('rank50','economic'),
                                    'sha256':sha(output/path),'stress_reuses_base_signal':True})
                print(json.dumps({'fold':fold['id'],'family':family,'fits_completed':sum(a['status']=='completed' for a in trials)}),flush=True)
        require(len(trials)==64 and all(a['status']=='completed' for a in trials),'Incomplete fit budget')
        pd.DataFrame(predictions).to_csv(output/'development_test_predictions.csv',index=False)
        write_json(output/'selected_heads.json',selected)
        write_json(output/'signals_manifest.json',{'schema_version':1,'signals':signals,'final_holdout_inspected':False})
        registry['status']='heads_fitted_pending_engine_backtests'
        registry['selection']['all_attempts']=trials
        registry['artifacts']['preparation_hashes']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
        write_json(registry_path,registry)
        print(json.dumps({'status':registry['status'],'attempts':len(trials),'signals':len(signals),'holdout_inspected':False}))
    except Exception as error:
        registry.update(status='failed_preparation',result={'conclusion':'invalid_due_to_audit','failure_reason':repr(error)})
        registry['selection']['all_attempts']=trials
        write_json(registry_path,registry)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    prepare(parser.parse_args().output_dir)
