"""Lock a common development protocol and preserve the prior stage byte-for-byte."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, git, write_json
from datetime import datetime, timezone
import shutil
import json
import yaml

RUN_ID = 'FROZEN_20261008_v1'

def main():
    output = ROOT / 'research/runs' / RUN_ID
    require(not output.exists(), 'Use a new immutable run')
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    protocol_path = ROOT / 'research/configs/frozen_comparison_v1.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    base = yaml.safe_load((ROOT / 'research/configs/baseline_revision_v2.yaml').read_text(encoding='utf-8'))
    require(config['evaluation_readiness']['ready_to_encode'] and config['stage'] == 'ready_to_encode', 'M0 prerequisite')
    accepted = json.loads((ROOT / 'research/initialization/M0_acceptance_report.json').read_text(encoding='utf-8'))
    require(accepted['ready_to_encode'] and accepted['status'] == 'accepted_ready_to_encode', 'M0 report')
    e00 = json.loads((ROOT / 'research/registry/E00R_20261008_phase4_v1.json').read_text(encoding='utf-8'))
    require(e00['status'] == 'completed_exploratory_development', 'Baseline unfinished')
    require(not protocol_path.exists(), 'Protocol already exists')
    output.mkdir(parents=True)
    provenance = output / 'provenance'
    provenance.mkdir()
    before = {}
    for rel in ['research/configs/initial_experiment.yaml', 'research/initialization/TODO.md', 'AGENTS.md', 'research/README.md', 'research/baselines/E00_baseline_roles.md']:
        p = ROOT / rel
        dest = provenance / 'prior_stage' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        before[rel] = sha(p)
    protocol = {
        'schema_version': 1, 'experiment_family': 'E00R_E01R_E02R', 'status': 'locked',
        'stage': 'frozen_development', 'locked_at_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'user_request_based_on_HANDOFF_FROZEN_REPRESENTATION_and_roadmap',
        'design_context': 'exploratory_previously_seen_development_folds_checkpoint_cutoff_unproven',
        'inherit_market_timing_costs_splits_from': 'research/configs/initial_experiment.yaml',
        'final_holdout_access': 'forbidden', 'frozen_representation_extraction': 'development_only',
        'backbone_training': 'forbidden', 'tokenizer_training': 'forbidden', 'future_path_generation': 'forbidden',
        'target': base['target'], 'ordinary_features': base['ordinary_features'], 'head': base['head'],
        'actions': base['actions'], 'controls': base['controls'], 'evaluation': base['evaluation'],
        'variants': {
            'E00R': {'features': 'ordinary19', 'reuse_run': 'research/runs/E00R_20261008_phase4_v1', 'mode': 'ridge19'},
            'E01R': {'features': 'ordinary19_plus_pretrained_last_hidden', 'mode': 'ridge_pretrained'},
            'E02R': {'features': 'ordinary19_plus_random_last_hidden', 'modes': ['ridge_random_s17', 'ridge_random_s29', 'ridge_random_s43'],
                      'seeds': [17, 29, 43], 'initialization': 'torch.manual_seed_then_official_Kronos_constructor_from_pinned_config_no_pretrained_backbone_load',
                      'tokenizer': 'same_pretrained_tokenizer', 'aggregation': 'report_every_seed_and_arithmetic_mean_no_seed_selection'}},
        'encoding': {'window_bars': 256, 'fields': ['open','high','low','close','volume','amount'],
                     'normalization': config['frozen_representation']['normalization'], 'pooling': 'decode_s1_last_valid_context',
                     'dtype': 'float32', 'autocast': False, 'tf32': False, 'batch_size': 8, 'chunk_size': 128,
                     'atol': 0.0001, 'rtol': 0.0001, 'development_end_exclusive': '2026-04-01T00:00:00Z',
                     'resource_budget': 'one_pretrained_and_three_random_backbones_no_extra_representations_or_precision_search'},
        'primary_comparisons': ['E01R_vs_E00R', 'E01R_vs_mean_E02R', 'E01R_vs_matched_month_direction_counts_random'],
        'paired_uncertainty': {'method': 'stationary_bootstrap_daily_reference_differences_within_fold_then_equal_fold_mean',
                               'draws': 5000, 'seed': 17, 'mean_block_days': 7, 'sensitivity_days': [3,14],
                               'scope': 'independent_reference_utility_and_paired_prediction_loss_not_compounded_portfolio_return',
                               'matched_null': 'exact_stratum_expected_return_plus_random_draw_distribution'},
        'research_screen': {
            'basis': 'user_accepted_proposed_screen_before_any_frozen_result',
            'automatic_promotion': False, 'information_and_economic_judged_separately': True,
            'prediction_median_relative_MSE_improvement_min': 0.01,
            'matched_random_median_lift_bps_min': 1.0,
            'positive_increment_folds_min': 3,
            'information_equal_fold_primary_block_lower_95_bound_min': 0.0,
            'information_required_comparators': ['ordinary19', 'mean_random_backbones', 'matched_noninformative_counts'],
            'economic_median_quarterly_increment_vs_ordinary_min_fraction': 0.01,
            'economic_positive_quarters_vs_cash_min': 3,
            'economic_stress_median_return_vs_cash': 'strictly_positive',
            'economic_stress_median_increment_vs_ordinary': 'strictly_positive',
            'economic_median_closed_trade_drawdown_change_max_fraction': 0.0,
            'sparse_or_wide_interval': 'insufficient_evidence_no_threshold_change',
            'scope': 'support_further_research_only_no_holdout_or_live_permission'},
        'fit_budget': {'new_variants': 4, 'folds': 4, 'lambda_candidates': 4, 'new_Ridge_fits': 64,
                       'gates_per_variant': 4, 'costs': 2, 'new_official_exports': 128,
                       'E00R_reuse_only_after_exact_design_membership_and_hash_check': True}}
    protocol_path.write_text(yaml.safe_dump(protocol, allow_unicode=True, sort_keys=False), encoding='utf-8')
    config['status'] = 'common_protocol_locked_frozen_development_authorized'
    config['stage'] = 'frozen_development'
    frozen = config['frozen_representation']
    frozen.update(enabled_in_current_stage=True, inference_in_current_stage='allowed', feature_extraction_in_current_stage='allowed',
                  inference_scope='development_history_encoding_only', backbone_training='forbidden', tokenizer_training='forbidden',
                  future_path_generation='forbidden', protocol='research/configs/frozen_comparison_v1.yaml')
    ready = config['evaluation_readiness']
    ready.update(common_protocol='research/configs/frozen_comparison_v1.yaml', common_protocol_sha256=sha(protocol_path),
                 next_research_action='encode_audit_fit_replay_E01R_E02R_development_only',
                 baseline_revision_status='E00R_completed_common_protocol_locked', current_task_stop='after_frozen_development_comparison_before_holdout',
                 development_frozen_backtests_allowed=True,
                 formal_evaluation_scope_note='final_holdout_forbidden_in_this_task_even_after_common_protocol_lock')
    config_path.write_text('# Frozen historical encoding and downstream Ridge authorized; final holdout sealed.\n' + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    todo = ROOT / 'research/initialization/TODO.md'
    text = todo.read_text(encoding='utf-8')
    text = text.replace('冻结 tokenizer／骨干的推理、表征提取及训练仍禁用，最终 holdout 封存。', '共用协议已锁定，本次开放开发期冻结历史编码与下游 Ridge；tokenizer／骨干训练、未来路径生成及最终 holdout 仍禁用。')
    text = text.replace('- [ ] **冻结表征前：锁定共用协议。**', '- [x] **冻结表征前：锁定共用协议。**')
    text = text.replace('- [ ] **协议锁定后：明确下一阶段范围。**', '- [x] **协议锁定后：明确下一阶段范围。**')
    text = text.replace('当前尚未开始冻结提取。', '当前运行 `FROZEN_20261008_v1`；开发期冻结编码、64次新Ridge拟合与128份官方回放待完成。')
    text += '\n- [ ] **E01R／E02R开发实验**：协议、阶段记录见 `frozen_comparison_v1.yaml` 与 `frozen_state_transition_v1.json`；缓存、因果验收、读出器、官方回放与结果待完成。最终holdout封存。\n'
    todo.write_text(text, encoding='utf-8')
    p = ROOT / 'AGENTS.md'
    text = p.read_text(encoding='utf-8').replace('当前研究目标止于冻结表征提取准备就绪；按配置验收 M0，不提前开展表征提取或骨干/tokenizer 训练、微调。后续范围以用户最新指令和阶段配置为准。', '当前用户授权按共用协议开展 E01R／E02R 开发期冻结历史表征、下游 Ridge 与官方回放；协议和阶段以配置为准。骨干/tokenizer 训练、微调、原生路径生成及最终 holdout 仍禁用。')
    p.write_text(text, encoding='utf-8')
    p = ROOT / 'research/README.md'
    text = p.read_text(encoding='utf-8').replace('下一步是锁定 E00R／E01R／E02R 共用比较协议；冻结表征提取仍禁用，最终 holdout 未评估。', 'E00R／E01R／E02R 共用比较协议已锁定；当前授权开发期冻结历史编码、下游 Ridge 与官方回放，最终 holdout 仍封存。')
    text = text.replace('当前协议尚未锁定，不把原 E00 参数自动沿用到 E01R／E02R。', '共用协议见 [冻结比较配置](configs/frozen_comparison_v1.yaml)，保留原 E00 与 E00R 证据。')
    p.write_text(text, encoding='utf-8')
    p = ROOT / 'research/baselines/E00_baseline_roles.md'
    p.write_text(p.read_text(encoding='utf-8').replace('当前仍止于冻结表征提取之前。', '本次按共用配置开展开发期冻结表征比较，tokenizer与骨干保持冻结。'), encoding='utf-8')
    bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    labels = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    copies = {'experiment_config.yaml': config_path, 'common_config.yaml': protocol_path,
              'labels_manifest.json': bundle/'manifest.json', 'split_index.csv': bundle/'split_index.csv',
              'dataset_manifest.json': Path(labels['sources']['snapshot'])/'manifest.json',
              'version_manifest.json': ROOT/'research/initialization/version_manifest.json',
              'kronos_requirements.lock.txt': ROOT/'research/environment/requirements.lock.txt',
              'freqtrade_requirements.lock.txt': ROOT/'research/freqtrade/environment/requirements.lock.txt',
              'baseline_revision_config.yaml': ROOT/'research/configs/baseline_revision_v2.yaml'}
    for name, path in copies.items(): shutil.copy2(path, provenance / name)
    transition = {'at_utc': datetime.now(timezone.utc).isoformat(), 'authorization': protocol['authorization'],
                  'from': 'ready_to_encode', 'to': 'frozen_development', 'before_sha256': before,
                  'prior_bytes': str(provenance/'prior_stage'), 'protocol_sha256': sha(protocol_path),
                  'after_config_sha256': sha(config_path), 'final_holdout_opened': False,
                  'M0_report_unchanged_sha256': sha(ROOT/'research/initialization/M0_acceptance_report.json')}
    write_json(ROOT/'research/initialization/frozen_state_transition_v1.json', transition)
    write_json(provenance/'state_transition.json', transition)
    dirty=[]
    for line in git('-c','core.quotePath=false','status','--porcelain','--untracked-files=all').splitlines():
        rel=line[3:]; p=ROOT/rel
        if p.is_file():
            dst=provenance/'dirty_source'/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dst)
            dirty.append({'path':rel,'status':line[:2],'sha256':sha(p)})
    record={'schema_version':1,'experiment_id':RUN_ID,'parent_experiment_id':'E00R_20261008_phase4_v1',
            'stage':'frozen_development','status':'protocol_locked_pending_encoding','created_at_utc':protocol['locked_at_utc'],
            'hypothesis':'Frozen pretrained history may add readable net-return information beyond ordinary19 and random backbone controls',
            'baseline_id':'E00R_ridge19','only_primary_change':'add_pretrained_or_random_512_context_same_Ridge_budget',
            'configuration':{'path':str(protocol_path),'sha256':sha(protocol_path),'full_text':protocol_path.read_bytes().decode('utf-8')},
            'parent_configuration':{'path':str(config_path),'sha256':sha(config_path),'full_text':config_path.read_bytes().decode('utf-8')},
            'provenance':{'repository_commit':git('rev-parse','HEAD'),'dirty_files_and_sha256':dirty,
                          'canonical_source_sha256':{},'dataset_snapshot_id':labels['snapshot_id'],
                          'dataset_manifest_sha256':sha(provenance/'dataset_manifest.json'),
                          'label_manifest_sha256':sha(bundle/'manifest.json'), 'split_sha256':sha(bundle/'split_index.csv'),
                          'version_manifest_sha256':sha(provenance/'version_manifest.json'),
                          'environment_locks':{p.name:sha(p) for p in provenance.glob('*lock.txt')},
                          'baseline_registry_sha256':sha(ROOT/'research/registry/E00R_20261008_phase4_v1.json')},
            'selection':{'head_family':'Ridge','hyperparameters':protocol['head'],'seeds':[17,29,43],
                         'locked_test_used_for_selection':False,'all_attempts':[]},
            'final_holdout_inspected':False,'artifacts':{'root':str(output)},'result':{'conclusion':None}}
    write_json(ROOT/'research/registry'/f'{RUN_ID}.json',record)
    print(json.dumps({'status':record['status'],'run':str(output),'protocol_sha256':sha(protocol_path)}))

if __name__ == '__main__': main()
