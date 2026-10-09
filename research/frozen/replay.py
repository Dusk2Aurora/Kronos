"""Official development replay for frozen gates; no extraction or head fitting.

Resume reuses complete official ZIPs and starts only untouched jobs. A partial
job without an export is preserved and rejected, never silently overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
import numpy as np
import pandas as pd
import yaml

from prepare_e00 import ROOT, require, sha, write_json, git
from run_e00 import engine_run, read_result, audit


def check_payload(payload, signal, fold, config, candles):
    require(set(payload) == {'schema_version', 'mode', 'decisions'}, 'Invalid sidecar schema')
    require(payload['schema_version'] == 1 and payload['mode'] == 'ordinary_features_gate',
            'Only fixed-direction binary gates')
    require(len(payload['decisions']) == signal['opportunities'], 'Opportunity count mismatch')
    require(sum(r['exposure_fraction'] > 0 for r in payload['decisions']) == signal['selected_count'],
            'Gate count mismatch')
    previous = None
    for row in payload['decisions']:
        require(set(row) == {'signal_at', 'direction', 'exposure_fraction'}, 'Outcomes in strategy input')
        t = pd.Timestamp(row['signal_at'])
        require(isinstance(row['signal_at'], str) and row['signal_at'].endswith('Z')
                and t.tzinfo is not None and t == t.floor('h'), 'Signal must be an exact UTC hour')
        require(t.hour in config['timing']['decision_hours_utc'], 'Wrong decision phase')
        require(previous is None or t > previous, 'Duplicate/nonchronological signal')
        previous = t
        require(pd.Timestamp(fold['test'][0]) <= t
                and t + pd.Timedelta(hours=5, seconds=60) < pd.Timestamp(fold['test'][1]),
                'Signal/position/label availability crosses fold boundary')
        require(type(row['exposure_fraction']) in (int, float) and row['exposure_fraction'] in (0, 1),
                'Nonbinary gate')
        require(type(row['direction']) is int and row['direction'] in (-1, 1), 'Invalid direction')
        index = pd.date_range(t - pd.Timedelta(hours=256), periods=256, freq='h')
        require(index.isin(candles.index).all(), 'Missing historical window')
        values = candles.loc[index, ['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy()
        require(np.isfinite(values).all() and (values[:, :4] > 0).all()
                and (values[:, 4:] >= 0).all(), 'Invalid true OHLCVA history')
        direction = int(np.sign(np.log(candles.loc[t - pd.Timedelta(hours=1), 'close']
                                       / candles.loc[t - pd.Timedelta(hours=25), 'close'])))
        require(row['direction'] == direction, 'Sidecar changes fixed momentum direction')


def run(output: Path, workers: int, audit_existing: bool = False, resume: bool = False) -> None:
    require(workers in (1, 2, 3), 'Workers must be 1..3')
    require(not (audit_existing and resume), 'Choose audit-existing or resume')
    output = output.resolve()
    require(output.is_relative_to(ROOT / 'research/runs'), 'Run must be in research/runs')
    registry_path = ROOT / 'research/registry' / (output.name + '.json')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    expected = ({'failed_engine_or_audit', 'engine_backtests_running'} if (audit_existing or resume)
                else {'heads_fitted_pending_engine_backtests'})
    require(registry['status'] in expected, 'Invalid preparation/resumption status')
    preparation = registry['artifacts']['preparation_hashes']
    require({'provenance/experiment_config.yaml', 'provenance/common_config.yaml',
             'provenance/labels_manifest.json', 'signals_manifest.json'} <= set(preparation),
            'Required inputs missing preparation hashes')
    for relative, digest in registry['artifacts']['preparation_hashes'].items():
        path = (output / relative).resolve()
        require(path.is_relative_to(output), 'Preparation outside run')
        require(sha(path) == digest, 'Preparation changed: ' + relative)
    config = yaml.safe_load((output / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    common = yaml.safe_load((output / 'provenance/common_config.yaml').read_text(encoding='utf-8'))
    require(common['final_holdout_access'] == 'forbidden'
            and common['frozen_representation_extraction'] == 'development_only', 'Incorrect scope')
    require(config['stage'] == 'frozen_development', 'Frozen development stage required')
    require(config['evaluation_readiness']['final_holdout_evaluation_allowed'] is False,
            'Final holdout must remain forbidden')
    require(config['evaluation_readiness']['development_E00_backtests_allowed'] is True, 'Development not authorized')
    for key in ('configuration', 'parent_configuration'):
        record = registry[key]
        require(sha(Path(record['path'])) == record['sha256'], 'Registered config changed')
    require(sha(output / 'provenance/common_config.yaml') == registry['configuration']['sha256']
            and sha(output / 'provenance/experiment_config.yaml') == registry['parent_configuration']['sha256'],
            'Saved configuration differs from registration')
    require(config['timing']['lookback_bars'] == 256 and config['timing']['holding_horizon_bars'] == 4
            and config['timing']['availability_delay_seconds'] == 60
            and config['timing']['decision_hours_utc'] == [4, 12, 20]
            and config['fixed_strategy']['momentum_lookback_bars'] == 24,
            'Audited fixed timing/direction contract changed')
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
    require(sha(snapshot / 'manifest.json') == labels['sources']['snapshot_manifest_sha256'], 'Snapshot manifest changed')
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
    require(candles.index.is_unique and funding.index.is_unique and marks.index.is_unique,
            'Duplicate raw input timestamps')
    require(funding.index.isin(marks.index).all(), 'Missing original funding mark')
    base_path = ROOT / 'research/freqtrade/config.base.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))
    require(base['dry_run_wallet'] == 10000 and base['tradable_balance_ratio'] == .99
            and base['timeframe'] == '1h', 'Portfolio config changed')
    signals = json.loads((output / 'signals_manifest.json').read_text(encoding='utf-8'))['signals']
    folds = {f['id']: f for f in config['splits']['walk_forward_dates']}
    require(set(folds) == {'WF01', 'WF02', 'WF03', 'WF04'}, 'Unexpected development folds')
    require(all(pd.Timestamp(f['test'][1]) <= pd.Timestamp(config['splits']['final_holdout'][0])
                for f in folds.values()), 'Development fold accesses final holdout')
    require(len({(s['fold_id'], s['mode']) for s in signals}) == len(signals), 'Duplicate signal batch')
    require(all(s['fold_id'] in folds for s in signals), 'Non-development signals')
    require(signals, 'No signals to replay')
    if audit_existing or resume:
        history = registry.setdefault('failure_history', [])
        snapshot_path = output / f'failed_engine_registry_{len(history) + 1}.json'
        write_json(snapshot_path, registry)
        history.append({'registry_snapshot': str(snapshot_path), 'reason': registry.get('result'),
                        'resolution': 'Reuse existing exports; no refit; run only untouched jobs on resume'})
    sources = [Path(__file__), ROOT / 'research/baselines/run_e00.py',
               ROOT / 'research/baselines/run_e00r.py', ROOT / 'research/baselines/prepare_e00.py', base_path, native_path]
    source_dir = output / 'provenance' / ('engine_audit_source_' + str(len(registry.get('failure_history', [])))
                                        if (audit_existing or resume) else 'engine_execution_source')
    source_dir.mkdir()
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
    if audit_existing or resume:
        for path in [output / 'metrics.csv', output / 'engine_report.json',
                     *output.glob('engine/*/*/*/official_closed_trade_equity.csv')]:
            if path.exists():
                preserved = source_dir / 'previous_derived' / path.relative_to(output)
                preserved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, preserved)
    tiers = datadir / 'futures/leverage_tiers_USDT.json'
    if tiers.exists():
        shutil.copy2(tiers, source_dir / tiers.name)
    registry['provenance'].setdefault('execution_source_versions', []).append({
        'path': str(source_dir), 'sha256': {str(p.relative_to(source_dir)): sha(p)
                                           for p in source_dir.rglob('*') if p.is_file()}})
    jobs, details = [], []
    if not (audit_existing or resume):
        registry['engine_attempts'] = []
    for signal in signals:
        fold = folds[signal['fold_id']]
        start = pd.Timestamp(fold['test'][0]).strftime('%Y%m%d')
        end = (pd.Timestamp(fold['test'][1]) - pd.Timedelta(hours=1)).strftime('%Y%m%dT%H%M')
        sidecar = (output / signal['path']).resolve()
        require(sidecar.is_relative_to(output / 'decisions'), 'Sidecar outside run')
        require(str(sidecar.relative_to(output)).replace('\\', '/') in preparation,
                'Sidecar missing preparation hash')
        payload = json.loads(sidecar.read_text(encoding='utf-8'))
        check_payload(payload, signal, fold, config, candles)
        require(isinstance(signal['mode'], str) and signal['mode'] not in ('', '.', '..')
                and '/' not in signal['mode'] and '\\' not in signal['mode'], 'Unsafe batch mode')
        for cost, fee in (('base', .0007), ('stress', .0014)):
            folder = output / 'engine' / fold['id'] / signal['mode'] / cost
            run_config = {**base, 'strategy': 'KronosE00', 'fee': fee, 'e00_decisions_path': str(sidecar)}
            require(folder.resolve().is_relative_to(output / 'engine'), 'Engine folder outside run')
            if not folder.exists():
                require(not audit_existing, 'Missing existing engine folder')
                folder.mkdir(parents=True)
                write_json(folder / 'config.json', run_config)
            else:
                require(audit_existing or resume, 'Existing engine output; use resume')
                require(json.loads((folder / 'config.json').read_text(encoding='utf-8')) == run_config, 'Resumption config drift')
            command = [config['backtest_engine']['python'], '-m', 'freqtrade', 'backtesting',
                       '--config', str(folder / 'config.json'), '--strategy', 'KronosE00',
                       '--strategy-path', str(runtime / 'strategies'), '--datadir', str(datadir),
                       '--timerange', start + '-' + end, '--cache', 'none', '--export', 'trades',
                       '--backtest-directory', str(folder), '--no-color']
            details.append((signal, fold, fee, payload))
            jobs.append((folder, command, engine))
            if not (audit_existing or resume):
                registry['engine_attempts'].append({'fold_id': fold['id'], 'mode': signal['mode'], 'cost': cost,
                    'command': command, 'configuration_sha256': sha(folder / 'config.json'), 'status': 'planned'})
    require(len(registry['engine_attempts']) == len(jobs), 'Resumption attempt count mismatch')
    for attempt, (folder, command, _) in zip(registry['engine_attempts'], jobs):
        require(attempt['command'] == command and attempt['configuration_sha256'] == sha(folder / 'config.json'),
                'Registered engine attempt changed')
    registry['status'] = 'engine_backtests_running'
    write_json(registry_path, registry)
    metrics = []
    try:
        def obtain(job):
            archives = list(job[0].glob('*.zip'))
            if archives:
                require(audit_existing or resume, 'Existing export cannot be overwritten')
                require(len(archives) == 1, 'No unique existing engine archive')
                return job[0], archives[0]
            require(not audit_existing, 'Missing existing engine archive')
            require(set(p.name for p in job[0].iterdir()) == {'config.json'},
                    'Partial engine output without ZIP preserved; use a new registered run')
            return engine_run(job)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(obtain, job): i for i, job in enumerate(jobs)}
            for future in as_completed(future_map):
                i = future_map[future]
                attempt = registry['engine_attempts'][i]
                signal, fold, fee, payload = details[i]
                folder, archive = future.result()
                if attempt.get('archive_sha256'):
                    require(sha(archive) == attempt['archive_sha256'], 'Existing official export changed')
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
                  'final_holdout_inspected': False, 'frozen_representations_extracted': True,
                  'old_E00_rerun': False, 'development_scope': 'frozen_development_exploratory',
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
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output_dir, args.workers, args.audit_existing, args.resume)
