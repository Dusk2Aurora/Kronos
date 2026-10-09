"""Lock a bounded same-coordinate quantization diagnostic, or accept its immutable cache."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, git, write_json
from datetime import datetime, timezone
import argparse
import copy
import json
import shutil
import yaml

RUN_ID = 'QUANT_20261008_v1'

def lock():
    output = ROOT/'research/runs'/RUN_ID
    protocol_path = ROOT/'research/configs/frozen_quant_v1.yaml'
    config_path = ROOT/'research/configs/initial_experiment.yaml'
    prior_protocol_path = ROOT/'research/configs/frozen_comparison_v1.yaml'
    prior_registry_path = ROOT/'research/registry/FROZEN_20261008_v1.json'
    mlp_registry_path = ROOT/'research/registry/FROZEN_MLP_20261008_v2.json'
    require(not output.exists() and not protocol_path.exists(), 'Use a new immutable experiment ID')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    prior = json.loads(prior_registry_path.read_text(encoding='utf-8'))
    mlp = json.loads(mlp_registry_path.read_text(encoding='utf-8'))
    require(config['stage']=='frozen_development' and config['evaluation_readiness']['ready_to_encode'], 'Frozen prerequisites')
    require(mlp['status']=='completed_exploratory_development' and prior['status']=='completed_exploratory_development', 'Earlier probes unfinished')
    require(not config['evaluation_readiness']['final_holdout_evaluation_allowed'], 'Holdout must stay sealed')
    require(not mlp['result']['information_screen']['passed'], 'This version diagnoses failure of bounded readouts; register another question if positive')
    for run, registry in [('FROZEN_20261008_v1',prior),('FROZEN_MLP_20261008_v2',mlp)]:
        for rel, digest in registry['artifacts']['final_hashes'].items():
            require(sha(ROOT/'research/runs'/run/rel)==digest, 'Accepted artifact changed: '+run+'/'+rel)
    common = copy.deepcopy(yaml.safe_load(prior_protocol_path.read_text(encoding='utf-8')))
    old_config = yaml.safe_load((ROOT/'research/runs/FROZEN_20261008_v1/provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    for key in ('market','timing','fixed_strategy','splits','labels_and_costs','backtest_engine'):
        require(config[key]==old_config[key], 'Immutable research contract differs: '+key)
    output.mkdir(parents=True); provenance=output/'provenance'; provenance.mkdir()
    before={}
    for rel in ('research/configs/initial_experiment.yaml','research/frozen/TODO.md','AGENTS.md','research/README.md','research/frozen/README.md'):
        p=ROOT/rel; dest=provenance/'prior_stage'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest); before[rel]=sha(p)
    common.update(experiment_family='E03_same_coordinate_frozen_quantization',
        locked_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization='user_long_term_goal_first_frozen_experiment_autonomous_bounded_diagnostics_20261008',
        design_context='exploratory_seen_development_folds_checkpoint_cutoff_unproven',
        only_primary_change='continuous_L2_normalized_20_coordinate_bottleneck_versus_same_coordinate_binary_bottleneck_same_Ridge_budget',
        cache_reuse={'run':'research/runs/FROZEN_20261008_v1','registry':'research/registry/FROZEN_20261008_v1.json',
                    'parent_registry_sha256':sha(prior_registry_path)},
        quantization={'continuous':'L2_normalized_quant_embed_u_last20','quantized':'BSQuantizer_q_last20',
            'pooling':'last_valid_position','dimension':20,'dtype':'float32','batch_size':8,'chunk_size':128,
            'atol':1e-4,'rtol':1e-4,'zero_rule':'u_gt_zero_else_negative','token_ids':'explicit_s1_s2_half_true'},
        variants={'E00R':common['variants']['E00R'],
            'Continuous':{'features':'ordinary19_plus_u20','mode':'ridge_quant_continuous'},
            'Binary':{'features':'ordinary19_plus_q20','mode':'ridge_quant_binary'}},
        primary_comparisons=['continuous_vs_binary_same_coordinates','continuous_vs_ordinary19','binary_vs_ordinary19'],
        historical_reference_runs=['research/runs/E00_20261008_phase4_v1','research/runs/E00R_20261008_phase4_v1',
            'research/runs/FROZEN_20261008_v1','research/runs/FROZEN_MLP_20261008_v2'])
    common['encoding']['resource_budget']='one_frozen_tokenizer_two_same_coordinate_20d_representations_no_new_backbone_or_projection_or_precision_search'
    common['fit_budget']={'new_variants':2,'folds':4,'lambda_candidates':4,'new_Ridge_fits':32,
        'selected_heads':8,'test_predictions':2182,'gates_per_variant':4,'costs':2,'new_official_exports':64,
        'ordinary19_reused':True,'new_tokenizer_or_backbone_fits':0,'new_projection_fits':0}
    common['research_screen']['information_required_comparators']=['ordinary19','matched_noninformative_counts']
    common['quantization_screen']={
        'median_relative_MSE_improvement_continuous_vs_binary_min':.01,'positive_folds_min':3,
        'equal_fold_MSE_block_lower_95_min':0.,'mse_is_task_readability_not_general_information_loss':True,
        'economic_and_information_gates':'same_thresholds_as_prior_common_protocol_each_representation_vs_ordinary_and_matched_counts',
        'backbone_512_reference':'descriptive_only_different_dimension_and_calendar_inputs_not_a_same_coordinate_quantization_control',
        'tokenizer_rebuild_or_unfreeze':'not_automatically_authorized_even_if_diagnostic_passes_repeat_on_independent_data_required'}
    common['stop_rule']={'no_more_head_capacity_on_current_task':True,
        'after_this_diagnostic':'complete_bounded_first_frozen_experiment_synthesis_and_stop_or_recommend_separately_scoped_followup',
        'no_search_for_pooling_projection_or_tokenizer':True,'holdout_and_weight_training':'forbidden'}
    protocol_path.write_text(yaml.safe_dump(common,allow_unicode=True,sort_keys=False),encoding='utf-8')
    previous_status=config['status']; config['status']='frozen_quant_protocol_locked_holdout_sealed'
    config['frozen_readout_head']={'type':'Ridge','protocol':protocol_path.relative_to(ROOT).as_posix(),
        'training_scope':'development_downstream_quantization_diagnostic_only'}
    config['evaluation_readiness'].update(active_frozen_experiment=RUN_ID,
        active_frozen_protocol=protocol_path.relative_to(ROOT).as_posix(),active_frozen_protocol_sha256=sha(protocol_path),
        next_research_action='encode_verify_same_coordinate_quantization_fit_replay_synthesize_first_frozen_experiment',
        current_task_stop='after_first_frozen_experiment_synthesis_before_holdout_or_weight_training')
    config_path.write_text('# Same-coordinate frozen tokenizer diagnostics authorized; final holdout sealed.\n'+yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    bundle=ROOT/config['labels_and_costs']['verified_label_bundle']
    labels=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    copies={'experiment_config.yaml':config_path,'common_config.yaml':protocol_path,
        'labels_manifest.json':bundle/'manifest.json','split_index.csv':bundle/'split_index.csv',
        'dataset_manifest.json':Path(labels['sources']['snapshot'])/'manifest.json',
        'version_manifest.json':ROOT/'research/initialization/version_manifest.json',
        'kronos_requirements.lock.txt':ROOT/'research/environment/requirements.lock.txt',
        'freqtrade_requirements.lock.txt':ROOT/'research/freqtrade/environment/requirements.lock.txt',
        'parent_frozen_registry.json':prior_registry_path,'parent_mlp_registry.json':mlp_registry_path}
    for name, src in copies.items(): shutil.copy2(src,provenance/name)
    write_json(provenance/'state_transition.json',{'at_utc':common['locked_at_utc'],'from_status':previous_status,
        'to_status':config['status'],'authorization':common['authorization'],'before_sha256':before,
        'protocol_sha256':sha(protocol_path),'after_config_sha256':sha(config_path),'final_holdout_opened':False})
    dirty=[]
    for line in git('-c','core.quotePath=false','status','--porcelain','--untracked-files=all').splitlines():
        rel=line[3:]; src=ROOT/rel
        if src.is_file():
            dst=provenance/'dirty_source'/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
            dirty.append({'path':rel,'status':line[:2],'sha256':sha(src)})
    strategy=ROOT/'research/freqtrade/strategies/KronosE00.py'
    record={'schema_version':1,'experiment_id':RUN_ID,'parent_experiment_id':'FROZEN_MLP_20261008_v2',
        'stage':'frozen_development','status':'protocol_locked_pending_quant_encoding','created_at_utc':common['locked_at_utc'],
        'hypothesis':'Binary quantization may affect task-readable information in the same normalized 20-coordinate bottleneck; this diagnostic distinguishes readout failure from a candidate quantization bottleneck',
        'baseline_id':'E00R_ridge19','only_primary_change':common['only_primary_change'],
        'configuration':{'path':str(protocol_path),'sha256':sha(protocol_path),'full_text':protocol_path.read_bytes().decode('utf-8')},
        'parent_configuration':{'path':str(config_path),'sha256':sha(config_path),'full_text':config_path.read_bytes().decode('utf-8')},
        'provenance':{'repository_commit':git('rev-parse','HEAD'),'dirty_files_and_sha256':dirty,
            'canonical_source_sha256':{strategy.relative_to(ROOT).as_posix():sha(strategy)},
            'parent_frozen_registry_sha256':sha(prior_registry_path),'parent_mlp_registry_sha256':sha(mlp_registry_path),
            'dataset_snapshot_id':labels['snapshot_id'],'dataset_manifest_sha256':sha(provenance/'dataset_manifest.json'),
            'label_manifest_sha256':sha(bundle/'manifest.json'),'split_sha256':sha(bundle/'split_index.csv'),
            'version_manifest_sha256':sha(provenance/'version_manifest.json'),
            'environment_locks':{p.name:sha(p) for p in provenance.glob('*lock.txt')},
            'model_revision':config['frozen_representation']['model']['revision'],
            'tokenizer_revision':config['frozen_representation']['tokenizer']['revision']},
        'selection':{'head_family':'Ridge','hyperparameters':common['head'],'locked_test_used_for_selection':False,'all_attempts':[]},
        'final_holdout_inspected':False,'artifacts':{'root':str(output)},'result':{'conclusion':None}}
    write_json(ROOT/'research/registry'/f'{RUN_ID}.json',record)
    print(json.dumps({'status':record['status'],'run':str(output),'protocol_sha256':sha(protocol_path)}))

def accept():
    from quant_encoder import load_cache
    output=ROOT/'research/runs'/RUN_ID; regpath=ROOT/'research/registry'/f'{RUN_ID}.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    require(reg['status']=='protocol_locked_pending_quant_encoding','Unexpected encoding acceptance state')
    cache=output/'cache/quantization'; loaded=load_cache(cache)
    manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    contract=json.loads((cache/'contract.json').read_text(encoding='utf-8'))
    check=json.loads((cache/'verification.json').read_text(encoding='utf-8'))
    require(check['status']=='passed' and check['original_all_position_ids_identical'] and
        check['batch_all_position_ids_identical'] and check['tokenizer_state_unchanged'] and
        check['future_perturbation']=='unchanged' and all(x=='rejected' for x in check['negative_checks'].values()), 'Engineering acceptance failed')
    require(manifest['tokenizer_state_unchanged'] and contract['mode']=='development_encoding' and
        contract['protocol_sha256']==reg['configuration']['sha256'] and
        contract['initial_config_sha256']==reg['parent_configuration']['sha256'], 'Cache stage/source changed')
    require(all(x.shape==(2460,20) for x in loaded.values()), 'Wrong quantization scope/dimension')
    report={'status':'passed','rows':2460,'dimension':20,'same_coordinates':True,'new_weight_training':False,
        'holdout_inspected':False,'manifest_sha256':sha(cache/'manifest.json'),'contract_sha256':sha(cache/'contract.json'),
        'engineering_checks':check,'immutable_cache_checks':'Every chunk/key/array/source/ID verified by load_cache',
        'tokenizer_state_unchanged':True,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'encoding_acceptance.json',report)
    reg['status']='quant_encoding_accepted_pending_heads'
    reg['artifacts']['encoding_hashes']={p.relative_to(output).as_posix():sha(p) for p in cache.rglob('*') if p.is_file()}
    reg['artifacts']['encoding_hashes']['encoding_acceptance.json']=sha(output/'encoding_acceptance.json')
    reg['provenance']['canonical_source_sha256'].update({k:v for k,v in contract['source_sha256'].items() if k.endswith('.py')})
    write_json(regpath,reg)
    print(json.dumps({'status':reg['status'],'rows':2460,'dimension':20,'holdout_inspected':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--accept-encoding',action='store_true')
    if parser.parse_args().accept_encoding: accept()
    else: lock()
