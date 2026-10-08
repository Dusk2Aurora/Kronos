"""Official Freqtrade execution for E00R; reuse the unchanged audited adapter."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from prepare_e00 import ROOT, require, sha, write_json, git
from run_e00 import engine_run, read_result, audit


def run(output: Path, workers: int, audit_existing: bool = False) -> None:
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs'), 'Run must be in research/runs')
    registry_path = ROOT / 'research/registry' / (output.name + '.json')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    expected = 'failed_engine_or_audit' if audit_existing else 'heads_fitted_pending_engine_backtests'
    require(registry['status'] == expected, 'Invalid preparation/resumption status')
    for relative, digest in registry['artifacts']['preparation_hashes'].items():
        require(sha(output / relative) == digest, 'Preparation changed: ' + relative)
    config = yaml.safe_load((output / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    revision = yaml.safe_load((output / 'provenance/revision_config.yaml').read_text(encoding='utf-8'))
    require(revision['final_holdout_access'] == 'forbidden'
            and revision['frozen_representation_extraction'] == 'forbidden', 'Incorrect scope')
    require(config['evaluation_readiness']['development_E00_backtests_allowed'] is True, 'Development not authorized')
    for key in ('configuration', 'parent_configuration'):
        record = registry[key]
        require(sha(Path(record['path'])) == record['sha256'], 'Registered config changed')
    engine = Path(config['backtest_engine']['repository_path'])
    require(git('rev-parse', 'HEAD', cwd=engine) == config['backtest_engine']['repository_commit'], 'Engine version changed')
    require(not git('status', '--porcelain', '--untracked-files=no', cwd=engine), 'Engine core changed')
    runtime = engine / 'user_data/kronos_research'
    strategy = ROOT / 'research/freqtrade/strategies/KronosE00.py'
    strategy_key = str(strategy.relative_to(ROOT)).replace('\\', '/')
    require(sha(strategy) == registry['provenance']['canonical_source_sha256'][strategy_key], 'Adapter source changed')
    shutil.copy2(strategy, runtime / 'strategies/KronosE00.py')
    native_path = ROOT / 'research/initialization/freqtrade_import_manifest.json'
    native = json.loads(native_path.read_text(encoding='utf-8'))
    labels = json.loads((output / 'provenance/labels_manifest.json').read_text(encoding='utf-8'))
    snapshot = Path(labels['sources']['snapshot'])
    datadir = runtime / 'data' / labels['snapshot_id']
    require(native['snapshot_manifest_sha256'] == labels['sources']['snapshot_manifest_sha256'], 'Snapshot mismatch')
    for relative, digest in native['files'].items():
        require(sha(datadir / relative) == digest, 'Native input changed: ' + relative)
    for name, digest in labels['sources']['snapshot_csv_sha256'].items():
        require(sha(snapshot / name) == digest, 'Snapshot CSV changed: ' + name)
    candles = pd.read_csv(snapshot / 'candles.csv').set_index('bar_open_at')
    candles.index = pd.to_datetime(candles.index, utc=True)
    funding = pd.read_csv(snapshot / 'funding.csv').set_index('funding_at')
    funding.index = pd.to_datetime(funding.index, utc=True)
    marks = pd.read_csv(snapshot / 'mark_prices.csv').set_index('bar_open_at')
    marks.index = pd.to_datetime(marks.index, utc=True)
    base_path = ROOT / 'research/freqtrade/config.base.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))
    signals = json.loads((output / 'signals_manifest.json').read_text(encoding='utf-8'))['signals']
    folds = {f['id']: f for f in config['splits']['walk_forward_dates']}
    require(set(folds) == {'WF01', 'WF02', 'WF03', 'WF04'}, 'Unexpected development folds')
    require(len({(s['fold_id'], s['mode']) for s in signals}) == len(signals), 'Duplicate signal batch')
    require(all(s['fold_id'] in folds for s in signals), 'Non-development signals')
    if audit_existing:
        history = registry.setdefault('failure_history', [])
        snapshot_path = output / f'failed_engine_registry_{len(history) + 1}.json'
        write_json(snapshot_path, registry)
        history.append({'registry_snapshot': str(snapshot_path), 'reason': registry['result'],
                        'resolution': 'Audit existing official exports; do not refit or rerun engine'})
    sources = [Path(__file__), ROOT / 'research/baselines/run_e00.py',
               ROOT / 'research/baselines/prepare_e00.py', base_path, native_path]
    source_dir = output / 'provenance' / ('engine_audit_source_' + str(len(registry.get('failure_history', [])))
                                        if audit_existing else 'engine_execution_source')
    source_dir.mkdir()
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
    tiers = datadir / 'futures/leverage_tiers_USDT.json'
    if tiers.exists():
        shutil.copy2(tiers, source_dir / tiers.name)
    registry['provenance'].setdefault('execution_source_versions', []).append({
        'path': str(source_dir), 'sha256': {p.name: sha(p) for p in source_dir.iterdir()}})
    jobs, details = [], []
    if not audit_existing:
        registry['engine_attempts'] = []
    for signal in signals:
        fold = folds[signal['fold_id']]
        start = pd.Timestamp(fold['test'][0]).strftime('%Y%m%d')
        end = (pd.Timestamp(fold['test'][1]) - pd.Timedelta(hours=1)).strftime('%Y%m%dT%H%M')
        sidecar = (output / signal['path']).resolve()
        require(sidecar.is_relative_to(output / 'decisions'), 'Sidecar outside run')
        payload = json.loads(sidecar.read_text(encoding='utf-8'))
        require(payload['mode'] == 'ordinary_features_gate', 'Only fixed-direction binary gates')
        require(len(payload['decisions']) == signal['opportunities'], 'Opportunity count mismatch')
        require(sum(r['exposure_fraction'] > 0 for r in payload['decisions']) == signal['selected_count'], 'Gate count mismatch')
        for row in payload['decisions']:
            require(set(row) == {'signal_at', 'direction', 'exposure_fraction'}, 'Outcomes in strategy input')
            t = pd.Timestamp(row['signal_at'])
            require(pd.Timestamp(fold['test'][0]) <= t < pd.Timestamp(fold['test'][1]), 'Out-of-fold sidecar')
            require(row['exposure_fraction'] in (0, 1), 'Nonbinary gate')
        for cost, fee in (('base', .0007), ('stress', .0014)):
            folder = output / 'engine' / fold['id'] / signal['mode'] / cost
            run_config = {**base, 'strategy': 'KronosE00', 'fee': fee, 'e00_decisions_path': str(sidecar)}
            if not audit_existing:
                folder.mkdir(parents=True)
                write_json(folder / 'config.json', run_config)
            else:
                require(json.loads((folder / 'config.json').read_text(encoding='utf-8')) == run_config, 'Resumption config drift')
            command = [config['backtest_engine']['python'], '-m', 'freqtrade', 'backtesting',
                       '--config', str(folder / 'config.json'), '--strategy', 'KronosE00',
                       '--strategy-path', str(runtime / 'strategies'), '--datadir', str(datadir),
                       '--timerange', start + '-' + end, '--cache', 'none', '--export', 'trades',
                       '--backtest-directory', str(folder), '--no-color']
            details.append((signal, fold, fee, payload))
            jobs.append((folder, command, engine))
            if not audit_existing:
                registry['engine_attempts'].append({'fold_id': fold['id'], 'mode': signal['mode'], 'cost': cost,
                    'command': command, 'configuration_sha256': sha(folder / 'config.json'), 'status': 'planned'})
    require(len(registry['engine_attempts']) == len(jobs), 'Resumption attempt count mismatch')
    registry['status'] = 'engine_backtests_running'
    write_json(registry_path, registry)
    metrics = []
    try:
        def obtain(job):
            if not audit_existing:
                return engine_run(job)
            archives = list(job[0].glob('*.zip'))
            require(len(archives) == 1, 'No unique existing engine archive')
            return job[0], archives[0]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(obtain, job): i for i, job in enumerate(jobs)}
            for future in as_completed(future_map):
                i = future_map[future]
                attempt = registry['engine_attempts'][i]
                signal, fold, fee, payload = details[i]
                folder, archive = future.result()
                stats = read_result(archive)
                equity = audit(stats, payload, fold, fee, candles, funding, marks)
                pd.DataFrame(equity).to_csv(folder / 'official_closed_trade_equity.csv', index=False)
                require(stats['total_trades'] == signal['selected_count'], 'Engine lost gate selections')
                metrics.append({'fold_id': fold['id'], 'mode': signal['mode'], 'cost': attempt['cost'],
                    'family': signal.get('family'), 'gate': signal.get('gate'), 'primary': signal.get('primary', False),
                    'fee_bps_per_side': fee * 10000, 'opportunities': signal['opportunities'],
                    'trades': stats['total_trades'], 'net_return_pct': stats['profit_total'] * 100,
                    'net_profit_USDT': stats['profit_total_abs'],
                    'closed_trade_max_drawdown_pct': stats['max_drawdown_account'] * 100,
                    'funding_fees_USDT': sum(t['funding_fees'] for t in stats['trades']),
                    'archive': str(archive), 'archive_sha256': sha(archive)})
                attempt.update(status='verified', archive=str(archive), archive_sha256=sha(archive), trades=stats['total_trades'])
                write_json(registry_path, registry)
                print(json.dumps({'completed': len(metrics), 'total': len(jobs), 'fold': fold['id'],
                                  'mode': signal['mode'], 'cost': attempt['cost'], 'trades': stats['total_trades']}), flush=True)
        pd.DataFrame(metrics).sort_values(['cost', 'fold_id', 'mode']).to_csv(output / 'metrics.csv', index=False)
        result = {'status': 'verified_engine_pending_analysis', 'engine_runs': len(metrics),
                  'final_holdout_inspected': False, 'frozen_representations_extracted': False,
                  'old_E00_rerun': False, 'development_scope': 'post_E00_exploratory',
                  'audit': 'All admitted signals executed; official prices, 4h holds, directions, costs and signed original-mark funding verified'}
        write_json(output / 'engine_report.json', result)
        registry['status'] = 'engine_verified_pending_analysis'
        registry['result'] = result
        registry['artifacts']['engine_hashes'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                                                 if p.is_file() and p.suffix != '.log'}
        write_json(registry_path, registry)
        print(json.dumps(result), flush=True)
    except Exception as error:
        registry['status'] = 'failed_engine_or_audit'
        registry['result'] = {'conclusion': 'invalid_due_to_audit', 'failure_reason': repr(error)}
        write_json(registry_path, registry)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2, 3))
    parser.add_argument('--audit-existing', action='store_true')
    args = parser.parse_args()
    run(args.output_dir, args.workers, args.audit_existing)
