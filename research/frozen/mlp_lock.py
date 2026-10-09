"""Register one bounded MLP readout experiment before any new fit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, git, write_json
from datetime import datetime, timezone
import copy
import json
import shutil
import yaml
import argparse

RUN_ID = 'FROZEN_MLP_20261008_v1'

def main(run_id=RUN_ID):
    require(run_id in (RUN_ID, 'FROZEN_MLP_20261008_v2'), 'Unregistered experiment ID')
    output = ROOT / 'research/runs' / run_id
    protocol_path = ROOT / 'research/configs' / ('frozen_mlp_' + run_id.rsplit('_',1)[1] + '.yaml')
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    parent_run = ROOT / 'research/runs/FROZEN_20261008_v1'
    parent_registry = ROOT / 'research/registry/FROZEN_20261008_v1.json'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    prior_protocol = ROOT / 'research/configs/frozen_comparison_v1.yaml'
    common = yaml.safe_load(prior_protocol.read_text(encoding='utf-8'))
    parent = json.loads(parent_registry.read_text(encoding='utf-8'))
    previous_status = config['status']
    failed_attempt = None
    if run_id.endswith('_v2'):
        failed_path = ROOT / 'research/registry' / (RUN_ID + '.json')
        failed_attempt = json.loads(failed_path.read_text(encoding='utf-8'))
        require(failed_attempt['status'] == 'failed_preparation' and
                all(a['status'] == 'planned' for a in failed_attempt['selection']['all_attempts']),
                'V2 only repairs a pre-fit engineering failure; no additional model search')
    require(not output.exists() and not protocol_path.exists(), 'Use a new immutable experiment ID')
    require(config['stage'] == 'frozen_development' and config['evaluation_readiness']['ready_to_encode'], 'M0 / frozen prerequisite')
    require(parent['status'] == 'completed_exploratory_development', 'Prior experiment unfinished')
    require(not config['evaluation_readiness']['final_holdout_evaluation_allowed'], 'Holdout must remain sealed')
    require(sha(prior_protocol) == parent['configuration']['sha256'], 'Prior protocol changed')
    for rel, digest in parent['artifacts']['final_hashes'].items():
        require(sha(parent_run / rel) == digest, 'Prior artifact changed: ' + rel)
    parent_config = yaml.safe_load((parent_run / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    for key in ('market','timing','fixed_strategy','splits','labels_and_costs','backtest_engine'):
        require(config[key] == parent_config[key], 'Immutable research contract changed: ' + key)
    output.mkdir(parents=True)
    provenance = output / 'provenance'; provenance.mkdir()
    before = {}
    for rel in ['research/configs/initial_experiment.yaml','research/frozen/TODO.md','AGENTS.md','research/README.md','research/frozen/README.md']:
        src = ROOT / rel; dst = provenance / 'prior_stage' / rel
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); before[rel] = sha(src)
    protocol = copy.deepcopy(common)
    protocol.update(experiment_family='E00M_E01M_E02M', locked_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization='user_continue_after_bounded_small_MLP_proposal_20261008',
        only_primary_change='replace_Ridge_with_fixed_small_nonlinear_readout_and_identical_three_head_ensemble_budget_in_all_groups',
        design_context='exploratory_previously_seen_development_folds_checkpoint_cutoff_unproven_no_test_based_tuning',
        cache_reuse={'run':'research/runs/FROZEN_20261008_v1','registry':'research/registry/FROZEN_20261008_v1.json',
                    'new_encoding':False,'parent_registry_sha256':sha(parent_registry)},
        variants={'mlp19':None,'mlp_pretrained':'pretrained','mlp_random_s17':'random_s17',
                  'mlp_random_s29':'random_s29','mlp_random_s43':'random_s43'},
        primary_comparisons=['mlp_pretrained_vs_mlp19','mlp_pretrained_vs_mean_three_random_backbones',
                             'mlp_pretrained_vs_matched_month_direction_counts_random'],
        historical_reference_runs=['research/runs/FROZEN_20261008_v1','research/runs/E00R_20261008_phase4_v1','research/runs/E00_20261008_phase4_v1'])
    protocol['ordinary_features']['ablation'] = 'historical_original13_Ridge_retained_as_secondary_reference_not_retrained_or_selected'
    protocol['head'] = {
        'type':'MLP','hidden_units':16,'activation':'Tanh','epochs':300,'learning_rate':0.003,
        'betas':[0.9,0.999],'eps':1e-8,'weight_decay_candidates':[0.001,0.01,0.1],'seeds':[17,29,43],
        'device':'cpu','dtype':'float32','threads':1,'deterministic_algorithms':True,'optimizer':'AdamW',
        'weight_decay_scope':'all_parameters_including_bias','loss':'mean_squared_error_on_training_standardized_target',
        'batch':'full_training_fold','target_scaling':'train_mean_std_ddof0','preprocessing':'StandardScaler_fit_training_fold_only',
        'selection':'lowest_validation_ensemble_MSE_bps_squared','tie_break':'larger_weight_decay',
        'ensemble':'arithmetic_mean_all_three_seeds','no_early_stopping':True,'epoch_selection':False,
        'parameter_count_note':'Same hidden width and candidate budget; input dimensions 19 versus 531 produce different parameter counts, explicitly reported',
        'training_diagnostics':'Record fixed_epoch_loss_and_gradient_finiteness; no_additional_fit_or_optimizer_tuning_after_failure'}
    protocol['fit_budget'] = {'new_variants':5,'folds':4,'weight_decay_candidates':3,'head_seeds':3,
        'new_MLP_fits':180,'selected_ensembles':20,'selected_constituent_heads':60,
        'test_ensemble_predictions':5455,'gates_per_variant':4,'costs':2,'new_official_exports':160,
        'new_backbone_or_tokenizer_fits':0,'new_encoding':False}
    protocol['stop_rule'] = {'if_information_and_economic_screens_fail':'stop_increasing_head_capacity_on_this_task',
        'quantization_diagnosis':'separately_registered_only_if_a_specific_representation_bottleneck_question_is_identified',
        'if_nonfinite_training_or_audit_failure':'retain_failure_no_hidden_budget_extension',
        'no_automatic_unfreeze_holdout_or_new_task':True}
    protocol_path.write_text(yaml.safe_dump(protocol,allow_unicode=True,sort_keys=False),encoding='utf-8')
    config['status'] = 'frozen_mlp_protocol_locked_holdout_sealed'
    config['frozen_readout_head'] = {'type':'MLP','protocol':'research/configs/frozen_mlp_v1.yaml','training_scope':'development_downstream_only'}
    config['frozen_readout_head']['protocol'] = protocol_path.relative_to(ROOT).as_posix()
    config['evaluation_readiness'].update(active_frozen_experiment=run_id,
        active_frozen_protocol=protocol_path.relative_to(ROOT).as_posix(),active_frozen_protocol_sha256=sha(protocol_path),
        next_research_action='fit_audit_replay_bounded_MLP_development_only',
        current_task_stop='after_bounded_MLP_development_comparison_before_holdout_or_backbone_training')
    config_path.write_text('# Bounded downstream MLP development authorized; frozen weights and final holdout sealed.\n'+yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    label_manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    copies = {'experiment_config.yaml':config_path,'common_config.yaml':protocol_path,
        'labels_manifest.json':bundle/'manifest.json','split_index.csv':bundle/'split_index.csv',
        'dataset_manifest.json':Path(label_manifest['sources']['snapshot'])/'manifest.json',
        'version_manifest.json':ROOT/'research/initialization/version_manifest.json',
        'kronos_requirements.lock.txt':ROOT/'research/environment/requirements.lock.txt',
        'freqtrade_requirements.lock.txt':ROOT/'research/freqtrade/environment/requirements.lock.txt',
        'parent_frozen_registry.json':parent_registry,'parent_frozen_protocol.yaml':prior_protocol}
    for name, src in copies.items(): shutil.copy2(src, provenance / name)
    if failed_attempt:
        shutil.copy2(failed_path,provenance/'failed_preflight_registry.json')
    transition = {'at_utc':protocol['locked_at_utc'],'authorization':protocol['authorization'],
        'from_status':previous_status,'to_status':config['status'],
        'stage':'frozen_development','prior_bytes':str(provenance/'prior_stage'),'before_sha256':before,
        'protocol_sha256':sha(protocol_path),'after_config_sha256':sha(config_path),'final_holdout_opened':False}
    write_json(provenance/'state_transition.json',transition)
    dirty = []
    for line in git('-c','core.quotePath=false','status','--porcelain','--untracked-files=all').splitlines():
        rel = line[3:]; src = ROOT / rel
        if src.is_file():
            dst = provenance / 'dirty_source' / rel; dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst); dirty.append({'path':rel,'status':line[:2],'sha256':sha(src)})
    strategy = ROOT / 'research/freqtrade/strategies/KronosE00.py'
    record = {'schema_version':1,'experiment_id':run_id,'parent_experiment_id':parent_run.name,
        'stage':'frozen_development','status':'protocol_locked_pending_heads','created_at_utc':protocol['locked_at_utc'],
        'hypothesis':'A bounded nonlinear readout may use frozen historical information not accessible to Ridge, beyond a matched ordinary nonlinear head and random backbone controls',
        'baseline_id':'E00M_mlp19','only_primary_change':protocol['only_primary_change'],
        'configuration':{'path':str(protocol_path),'sha256':sha(protocol_path),'full_text':protocol_path.read_bytes().decode('utf-8')},
        'parent_configuration':{'path':str(config_path),'sha256':sha(config_path),'full_text':config_path.read_bytes().decode('utf-8')},
        'provenance':{'repository_commit':git('rev-parse','HEAD'),'dirty_files_and_sha256':dirty,
            'canonical_source_sha256':{strategy.relative_to(ROOT).as_posix():sha(strategy)},
            'parent_frozen_registry_sha256':sha(parent_registry),'parent_frozen_protocol_sha256':sha(prior_protocol),
            'dataset_snapshot_id':label_manifest['snapshot_id'],'dataset_manifest_sha256':sha(provenance/'dataset_manifest.json'),
            'label_manifest_sha256':sha(bundle/'manifest.json'),'split_sha256':sha(bundle/'split_index.csv'),
            'version_manifest_sha256':sha(provenance/'version_manifest.json'),
            'environment_locks':{p.name:sha(p) for p in provenance.glob('*lock.txt')},
            'model_revision':config['frozen_representation']['model']['revision'],
            'tokenizer_revision':config['frozen_representation']['tokenizer']['revision']},
        'selection':{'head_family':'MLP','hyperparameters':protocol['head'],'seeds':[17,29,43],
            'locked_test_used_for_selection':False,'all_attempts':[]},'final_holdout_inspected':False,
        'artifacts':{'root':str(output)},'result':{'conclusion':None}}
    if failed_attempt:
        record['failure_history'] = [{'experiment_id':RUN_ID,'registry':str(failed_path),
            'registry_sha256':sha(failed_path),'formal_fits_started':0,'reason':failed_attempt['result']['failure_reason'],
            'resolution':'Repair opportunity positional-index dtype; identical head design and total 180 formal fits'}]
    write_json(ROOT/'research/registry'/f'{run_id}.json',record)
    print(json.dumps({'status':record['status'],'run':str(output),'protocol_sha256':sha(protocol_path)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-id', default=RUN_ID)
    main(parser.parse_args().run_id)
