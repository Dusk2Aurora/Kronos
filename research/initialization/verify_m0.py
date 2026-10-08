"""Accept retained M0 evidence; no network, training, backtest or model inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def verify(target):
    target = target.resolve()
    require(target.is_relative_to(ROOT / 'research/initialization') and not target.exists(), 'Use new local report path')
    config_path = ROOT / 'research/configs/initial_experiment.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(config['task_id'] == 9 and config['stage'] == 'M0_acceptance', 'M0 acceptance stage required')
    evidence = {}
    paths = ['research/environment/verification.json', 'research/freqtrade/environment/verification.json',
             'research/initialization/data_integrity_audit.json', 'research/initialization/data_remote_recheck.json',
             'research/initialization/reference_labels_report.json', 'research/initialization/freqtrade_import_manifest.json',
             'research/initialization/e00_report.json', 'research/initialization/e00_causality_report.json',
             'research/registry/E00_20261008_phase4_v1.json', 'research/registry/E00_20261008_phase4_causality.json',
             'research/initialization/version_manifest.json']
    evidence = {p: {'sha256': sha(ROOT / p)} for p in paths}
    env, ft_env = read(paths[0]), read(paths[1])
    require(not env['include_system_site_packages'] and not ft_env['user_site'], 'Environment isolation evidence failed')
    require(env['checks']['pip_check'] == ft_env['checks']['pip_check'] == 'passed', 'Dependencies not accepted')
    require(sha(ROOT / 'research/environment/requirements.lock.txt') == env['dependency_lock_sha256'], 'Kronos dependency lock changed')
    require(sha(ROOT / 'research/freqtrade/environment/requirements.lock.txt') == ft_env['requirements_lock_sha256'], 'Freqtrade dependency lock changed')
    require(Path(env['base_interpreter']).exists(), 'Base interpreter removed')
    engine = Path(config['backtest_engine']['repository_path'])
    engine_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=engine, text=True).strip()
    require(engine_commit == config['backtest_engine']['repository_commit'] == ft_env['repository_commit'], 'Engine version mismatch')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=engine, text=True).strip(), 'Engine tracked core changed')
    require(not subprocess.check_output(['git', 'diff', 'HEAD', '--', 'model'], cwd=ROOT, text=True).strip(), 'Kronos core changed')
    data, remote, labels, native = (read(p) for p in paths[2:6])
    require(data['status'] == remote['status'] == 'passed', 'Source quality evidence missing')
    require(config['evaluation_readiness']['data_integrity_accepted'], 'Source scope not accepted')
    require(labels['status'] == 'official_reference_labels_verified' and labels['label_rows'] == 3011
            and labels['role_disjoint_and_holdout_isolated'] and not labels['locked_holdout_performance_inspected'], 'Label/split audit missing')
    require(native['status'] == 'imported_and_roundtrip_verified', 'Native import failed')
    snapshot = Path(data['snapshot'])
    require(sha(snapshot / 'manifest.json') == data['snapshot_manifest_sha256']
            == remote['snapshot_manifest_sha256'] == native['snapshot_manifest_sha256'], 'Snapshot evidence mismatch')
    bundle = Path(labels['bundle_path'])
    require(sha(bundle / 'manifest.json') == labels['manifest_sha256'], 'Label manifest changed')
    e00, causal, registry, causal_registry = (read(p) for p in paths[6:10])
    require(e00['status'] == 'verified_development_only' and e00['engine_runs'] == 48 and e00['training_attempts'] == 36
            and registry['status'] == 'completed' and all(a['status'] == 'verified' for a in registry['engine_attempts']), 'E00 not fully audited')
    require(not e00['final_holdout_inspected'] and not registry['final_holdout_inspected'], 'Holdout has been inspected')
    require(causal['status'] == 'accepted_scoped' and len(causal['results']) == 4
            and all(r['accepted'] and r['returncode'] == 0 for r in causal['results'])
            and not causal['final_holdout_read_or_evaluated'], 'Causality audit failed')
    run = Path(causal['run_directory'])
    for result in causal['results']:
        for relative, digest in result['evidence_sha256'].items():
            require(sha(run / relative) == digest, 'Causality evidence changed: ' + relative)
    protocol = config['labels_and_costs']['promotion_protocol']
    require(protocol['status'] == 'locked_before_development_performance_inspection'
            and config['labels_and_costs']['minimum_economic_increment_for_promotion'] == .01, 'Economic protocol not locked')
    require(config['splits']['status'] == 'locked_for_first_round'
            and config['splits']['fit_preprocessing_on'] == config['splits']['fit_model_on'] == 'training_fold_only', 'Split/fitting contract changed')
    version = read(paths[-1])
    weights = {}
    for name in ('model', 'tokenizer'):
        weight = version['weights'][name]
        require(weight['revision'] == config['frozen_representation'][name]['revision'], 'Frozen revision mismatch')
        for item in weight['files']:
            path = ROOT / weight['local_cache'] / item['name']
            require(path.exists() and sha(path) == item['sha256'], 'Pinned cache file missing or changed: ' + str(path))
        weights[name] = {'revision': weight['revision'], 'files_sha256_verified': len(weight['files'])}
    frozen = config['frozen_representation']
    require(not frozen['enabled_in_current_stage'] and frozen['inference_in_current_stage'] == 'forbidden'
            and frozen['feature_extraction_in_current_stage'] == 'forbidden', 'Frozen extraction stage prematurely enabled')
    require(frozen['normalization'] == {'context': 'historical_lookback_only', 'std_function': 'numpy.std',
                                      'ddof': 0, 'epsilon': .00001, 'clip_abs': 5.0}
            and frozen['pooling'] == 'last_valid_hidden', 'Representation contract changed')
    require(config['evaluation_readiness']['final_holdout_evaluation_allowed'] is False, 'Holdout must stay sealed')
    report = {'schema_version': 1, 'accepted_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'accepted_ready_to_encode', 'ready_to_encode': True,
              'scope': 'Research initialization and engineering prerequisites; stop before frozen representation extraction',
              'config_evaluated_sha256': sha(config_path), 'evidence': evidence, 'pinned_weights': weights,
              'checks': {name: 'passed' for name in ['isolated_environments_and_locks', 'upstream_core_unchanged',
                          'audited_official_data_and_units', 'real_amount_and_native_import', 'signed_funding',
                          'causal_timing_and_purged_labels', 'locked_splits_and_costs', 'registered_E00_all_48_verified',
                          'ordinary_train_only_and_validation_selection_review', 'official_scoped_causality_tools',
                          'pinned_weights_and_representation_contract', 'holdout_sealed_and_extraction_disabled']},
              'economic_screen_passed': e00['economic_screen_passed'],
              'information_value_over_cash_established': False,
              'final_holdout_evaluated': False, 'frozen_representation_extraction_run': False,
              'limitations': ['Published official data fidelity does not prove exchange internal ledger, historical precision or availability.',
                             'Checkpoint training cutoff unproven; early folds retrospective.',
                             'Official drawdown is closed-trade equity, not intrahour mark-to-market.',
                             'Sparse E00 gate improves losing momentum but does not establish profitable information value over cash.',
                             'Official causality tools cover stated development modes; sidecar producer has separate causal tests and train-only artifact review.'],
              'next_stage': 'Discuss E01/E02 frozen representation protocol; do not extract in this task'}
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']), 'report': str(target),
                      'frozen_extraction_run': False, 'holdout_evaluated': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
