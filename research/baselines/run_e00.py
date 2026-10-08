"""Run and audit official Freqtrade development portfolios; export tables/plots."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from prepare_e00 import ROOT, MODES, require, sha, write_json, git, iso


def engine_run(job):
    folder, command, engine = job
    with (folder / 'engine.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=engine, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f'Freqtrade failed ({result.returncode}): {folder}/engine.log')
    archives = list(folder.glob('*.zip'))
    require(len(archives) == 1, 'Expected unique official ZIP: ' + str(folder))
    return folder, archives[0]


def read_result(archive):
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith('.json') and not n.endswith('_config.json') and '_KronosE00' not in n)
        require(next(z.read(n) for n in z.namelist() if n.endswith('_KronosE00.py'))
                == (ROOT / 'research/freqtrade/strategies/KronosE00.py').read_bytes(), 'Embedded strategy differs')
        return json.loads(z.read(name))['strategy']['KronosE00']


def audit(stats, payload, fold, fee, candles, funding, marks):
    allowed = {iso(pd.Timestamp(row['signal_at']) + pd.Timedelta(hours=1)): row
               for row in payload['decisions'] if row['exposure_fraction'] > 0}
    trades = stats['trades']
    require(math.isclose(stats['starting_balance'], 10000), 'Wrong initial balance')
    seen, equity = set(), [{'at': fold['test'][0], 'closed_trade_equity_USDT': 10000.0}]
    total = 10000.0
    for t in sorted(trades, key=lambda t: t['close_date']):
        opened, closed = pd.Timestamp(t['open_date']), pd.Timestamp(t['close_date'])
        opened = opened.tz_localize('UTC') if opened.tzinfo is None else opened.tz_convert('UTC')
        closed = closed.tz_localize('UTC') if closed.tzinfo is None else closed.tz_convert('UTC')
        entry = iso(opened)
        require(entry in allowed and entry not in seen, 'Unexpected or duplicate entry')
        seen.add(entry)
        row = allowed[entry]
        require(t['is_short'] == (row['direction'] == -1), 'Direction reversal')
        require(float(t['leverage']) == 1 and t['fee_open'] == fee and t['fee_close'] == fee, 'Wrong leverage/cost')
        require(pd.Timestamp(fold['test'][0]) <= opened < closed < pd.Timestamp(fold['test'][1]), 'Out-of-fold position')
        if payload['mode'] == 'buy_hold':
            require(closed == pd.Timestamp(payload['hold_until']) and t['exit_reason'] == 'e00_buy_hold_end', 'Buy-hold exit interrupted')
        else:
            require(closed - opened == pd.Timedelta(hours=4) and t['exit_reason'] == 'fixed_hold_4h', 'Holding period interrupted')
        require(math.isclose(t['open_rate'], candles.loc[opened, 'open'], rel_tol=1e-12)
                and math.isclose(t['close_rate'], candles.loc[closed, 'open'], rel_tol=1e-12), 'Non-open execution price')
        events = funding.loc[(funding.index >= opened) & (funding.index <= closed)]
        expected_funding = float((events.realized_rate * marks.loc[events.index, 'open']).sum()) * float(t['amount'])
        if not t['is_short']:
            expected_funding = -expected_funding
        require(math.isclose(t['funding_fees'], expected_funding, rel_tol=1e-10, abs_tol=1e-8), 'Signed original mark funding mismatch')
        total += float(t['profit_abs'])
        equity.append({'at': iso(closed), 'closed_trade_equity_USDT': total})
    require(math.isclose(sum(t['profit_abs'] for t in trades), stats['profit_total_abs'], abs_tol=1e-7), 'Official summary differs from exports')
    if payload['mode'] == 'cash':
        require(not trades and stats['profit_total'] == 0, 'Cash traded')
    if payload['mode'] == 'buy_hold':
        require(len(trades) == 1 and len(seen) == len(allowed) == 1, 'Missing buy-hold benchmark trade')
    if payload['mode'] in ('fixed_momentum', 'constant_half_exposure', 'vol_target', 'ordinary_features_gate'):
        require(len(seen) == len(allowed), 'Eligible signals skipped by engine sizing or wallet; audit before accepting')
    equity.append({'at': iso(pd.Timestamp(fold['test'][1]) - pd.Timedelta(hours=1)), 'closed_trade_equity_USDT': total})
    return equity


def plots(output, metrics, increments):
    order = list(MODES)
    titles = ['Cash', 'Buy & hold', 'Momentum', 'Half exposure', 'Vol target', 'Ordinary gate']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, cost in zip(axes, ('base', 'stress')):
        data = metrics.loc[metrics.cost.eq(cost)].pivot(index='fold_id', columns='mode', values='net_return_pct').reindex(columns=order)
        values = data.to_numpy()
        limit = max(abs(metrics.net_return_pct.min()), abs(metrics.net_return_pct.max()), 1)
        im = ax.imshow(values, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_xticks(range(len(order)), titles, rotation=35, ha='right')
        ax.set_yticks(range(len(data)), ['2025 Q2', '2025 Q3', '2025 Q4', '2026 Q1'])
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f'{value:+.2f}%', ha='center', va='center', fontsize=10)
        ax.set_title(f'Per-side cost proxy: {7 if cost == "base" else 14} bps')
    fig.colorbar(im, ax=axes, label='Quarter net portfolio return (%)', shrink=.8)
    fig.suptitle('E00 | Official Freqtrade | 10,000 USDT reset per fold | Funding included\nDevelopment tests only; final 2026 Q2/Q3 holdout sealed', fontsize=12)
    for ext in ('png', 'svg'):
        fig.savefig(output / ('returns_comparison.' + ext), dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    x = np.arange(4)
    for offset, cost, label in [(-.18, 'base', '7 bps'), (.18, 'stress', '14 bps')]:
        values = increments.loc[increments.cost.eq(cost)].sort_values('fold_id').net_increment_pp.to_numpy()
        bars = ax.bar(x + offset, values, .34, label=label)
        ax.bar_label(bars, fmt='%+.2f', padding=3)
    ax.axhline(0, color='black', linewidth=.8)
    ax.axhline(1, color='gray', linestyle='--', linewidth=.8, label='1 pp median hurdle (not per-fold)')
    ax.set_xticks(x, ['2025 Q2', '2025 Q3', '2025 Q4', '2026 Q1'])
    ax.set_ylabel('Ordinary gate minus fixed momentum (percentage points)')
    ax.set_title('Same directions and 4h holds | Cost stress reuses the same gate')
    ax.legend(fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(output / ('gate_increment.' + ext), dpi=170)
    plt.close(fig)


def run(output, workers, audit_existing=False):
    output = output.resolve()
    registry_path = ROOT / 'research/registry' / (output.name + '.json')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    expected_status = 'failed_engine_or_audit' if audit_existing else 'heads_fitted_pending_engine_backtests'
    require(registry['status'] == expected_status, 'Preparation not complete or invalid resumption')
    if audit_existing:
        failure_path = output / ('failed_audit_' + str(len(list(output.glob('failed_audit_*.json'))) + 1) + '.json')
        write_json(failure_path, registry)
        registry.setdefault('failure_history', []).append({'registry_snapshot': str(failure_path),
                                                          'reason': registry['result'],
                                                          'resolution': 'Correct outcome-audit CSV field; reuse existing engine ZIPs without refitting or rerunning'})
    for relative, digest in registry['artifacts']['preparation_hashes'].items():
        require(sha(output / relative) == digest, 'Preparation artifact changed: ' + relative)
    config = yaml.safe_load((output / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    require(config['evaluation_readiness'].get('development_E00_backtests_allowed') is True,
            'Development E00 backtests not enabled')
    require(sha(ROOT / 'research/configs/initial_experiment.yaml') == registry['configuration']['sha256'], 'Current config drifted')
    engine = Path(config['backtest_engine']['repository_path'])
    require(git('rev-parse', 'HEAD', cwd=engine) == config['backtest_engine']['repository_commit'], 'Engine version changed')
    runtime = engine / 'user_data/kronos_research'
    strategy = ROOT / 'research/freqtrade/strategies/KronosE00.py'
    strategy_key = str(strategy.relative_to(ROOT)).replace('\\', '/')
    strategy_digest = registry['provenance'].get('canonical_source_sha256', {}).get(strategy_key)
    if strategy_digest is None:  # Compatibility with the first already-registered preparation.
        strategy_digest = sha(output / 'provenance/source' / strategy.relative_to(ROOT))
    require(sha(strategy) == strategy_digest, 'Canonical strategy drifted')
    shutil.copy2(strategy, runtime / 'strategies/KronosE00.py')
    native_manifest = json.loads((ROOT / 'research/initialization/freqtrade_import_manifest.json').read_text(encoding='utf-8'))
    label_manifest = json.loads((output / 'provenance/labels_manifest.json').read_text(encoding='utf-8'))
    snapshot = Path(label_manifest['sources']['snapshot'])
    datadir = runtime / 'data' / label_manifest['snapshot_id']
    require(native_manifest['snapshot_manifest_sha256'] == label_manifest['sources']['snapshot_manifest_sha256'],
            'Native data and label snapshot mismatch')
    for relative, digest in native_manifest['files'].items():
        require(sha(datadir / relative) == digest, 'Native input changed: ' + relative)
    for name, digest in label_manifest['sources']['snapshot_csv_sha256'].items():
        require(sha(snapshot / name) == digest, 'Snapshot CSV changed: ' + name)
    candles = pd.read_csv(snapshot / 'candles.csv').set_index('bar_open_at')
    candles.index = pd.to_datetime(candles.index, utc=True)
    funding = pd.read_csv(snapshot / 'funding.csv').set_index('funding_at')
    funding.index = pd.to_datetime(funding.index, utc=True)
    marks = pd.read_csv(snapshot / 'mark_prices.csv').set_index('bar_open_at')
    marks.index = pd.to_datetime(marks.index, utc=True)
    # Reuse audited data; do not resample or rebuild a wallet ledger.
    native_copy = output / 'provenance/native_import_manifest.json'
    if not native_copy.exists():
        shutil.copy2(ROOT / 'research/initialization/freqtrade_import_manifest.json', native_copy)
    else:
        require(sha(native_copy) == sha(ROOT / 'research/initialization/freqtrade_import_manifest.json'), 'Import provenance changed')
    tiers = datadir / 'futures/leverage_tiers_USDT.json'
    if tiers.exists() and not (output / 'provenance/leverage_tiers_USDT.json').exists():
        shutil.copy2(tiers, output / 'provenance/leverage_tiers_USDT.json')
    base = json.loads((ROOT / 'research/freqtrade/config.base.json').read_text(encoding='utf-8'))
    execution_source = output / 'provenance' / ('engine_audit_resumption_source_' + str(len(registry.get('failure_history', [])))
                                               if audit_existing else 'engine_execution_source')
    execution_source.mkdir()
    for name in ('run_e00.py', 'prepare_e00.py', 'ordinary_features.py'):
        shutil.copy2(ROOT / 'research/baselines' / name, execution_source / name)
    registry['provenance'].setdefault('execution_source_versions', []).append({
        'path': str(execution_source), 'sha256': {p.name: sha(p) for p in execution_source.iterdir()}})
    registry['provenance']['engine_source_note'] = 'Original fitting/engine source retained; audit and registration corrections copied separately; no feature/head/decision or research protocol change'
    registry['status'] = 'engine_backtests_running'
    if not audit_existing:
        registry['engine_attempts'] = []
    write_json(registry_path, registry)
    jobs = []
    for fold in config['splits']['walk_forward_dates']:
        start = pd.Timestamp(fold['test'][0]).strftime('%Y%m%d')
        end = (pd.Timestamp(fold['test'][1]) - pd.Timedelta(hours=1)).strftime('%Y%m%dT%H%M')
        for mode in MODES:
            for cost, fee in (('base', .0007), ('stress', .0014)):
                folder = output / 'engine' / fold['id'] / mode / cost
                if not audit_existing:
                    folder.mkdir(parents=True)
                run_config = {**base, 'strategy': 'KronosE00', 'fee': fee,
                              'e00_decisions_path': str(output / 'decisions' / (fold['id'] + '_' + mode + '.json'))}
                if not audit_existing:
                    write_json(folder / 'config.json', run_config)
                else:
                    require(json.loads((folder / 'config.json').read_text(encoding='utf-8')) == run_config,
                            'Resumption engine config drifted')
                command = [config['backtest_engine']['python'], '-m', 'freqtrade', 'backtesting',
                           '--config', str(folder / 'config.json'), '--strategy', 'KronosE00',
                           '--strategy-path', str(runtime / 'strategies'), '--datadir', str(datadir),
                           '--timerange', start + '-' + end, '--cache', 'none', '--export', 'trades',
                           '--backtest-directory', str(folder), '--no-color']
                if not audit_existing:
                    registry['engine_attempts'].append({'fold_id': fold['id'], 'mode': mode, 'cost': cost,
                                                    'command': command, 'configuration_sha256': sha(folder / 'config.json'),
                                                    'status': 'planned'})
                jobs.append((folder, command, engine))
    write_json(registry_path, registry)
    metrics, increments = [], []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            def obtain(job):
                if not audit_existing:
                    return engine_run(job)
                archives = list(job[0].glob('*.zip'))
                require(len(archives) == 1, 'Missing unique existing official export: ' + str(job[0]))
                return job[0], archives[0]
            future_map = {executor.submit(obtain, job): index for index, job in enumerate(jobs)}
            for future in as_completed(future_map):
                index = future_map[future]
                attempt = registry['engine_attempts'][index]
                folder, archive = future.result()
                stats = read_result(archive)
                fold = next(f for f in config['splits']['walk_forward_dates'] if f['id'] == attempt['fold_id'])
                payload = json.loads((output / 'decisions' / (fold['id'] + '_' + attempt['mode'] + '.json')).read_text(encoding='utf-8'))
                fee = .0007 if attempt['cost'] == 'base' else .0014
                equity = audit(stats, payload, fold, fee, candles, funding, marks)
                pd.DataFrame(equity).to_csv(folder / 'official_closed_trade_equity.csv', index=False)
                attempt.update(status='verified', archive=str(archive), archive_sha256=sha(archive),
                               trades=stats['total_trades'])
                metrics.append({'fold_id': fold['id'], 'mode': attempt['mode'], 'cost': attempt['cost'],
                                'fee_bps_per_side': fee * 10000, 'opportunities': 0 if attempt['mode'] == 'cash' else len(payload['decisions']),
                                'trades': stats['total_trades'], 'net_return_pct': stats['profit_total'] * 100,
                                'net_profit_USDT': stats['profit_total_abs'],
                                'closed_trade_max_drawdown_pct': stats['max_drawdown_account'] * 100,
                                'funding_fees_USDT': sum(t['funding_fees'] for t in stats['trades']),
                                'archive': str(archive), 'archive_sha256': sha(archive)})
                write_json(registry_path, registry)
                print(json.dumps({'completed': len(metrics), 'total': len(jobs), 'fold': fold['id'],
                                  'mode': attempt['mode'], 'cost': attempt['cost'], 'trades': stats['total_trades']}), flush=True)
        metrics = pd.DataFrame(metrics).sort_values(['cost', 'fold_id', 'mode'])
        metrics.to_csv(output / 'metrics.csv', index=False)
        for (fold, cost), group in metrics.groupby(['fold_id', 'cost']):
            group = group.set_index('mode')
            increments.append({'fold_id': fold, 'cost': cost,
                               'net_increment_pp': group.loc['ordinary_features_gate', 'net_return_pct'] - group.loc['fixed_momentum', 'net_return_pct'],
                               'closed_trade_drawdown_change_pp': group.loc['ordinary_features_gate', 'closed_trade_max_drawdown_pct'] - group.loc['fixed_momentum', 'closed_trade_max_drawdown_pct']})
        increments = pd.DataFrame(increments).sort_values(['cost', 'fold_id'])
        increments.to_csv(output / 'gate_increments.csv', index=False)
        b = increments.loc[increments.cost.eq('base')]
        s = increments.loc[increments.cost.eq('stress')]
        protocol = config['labels_and_costs']['promotion_protocol']
        checks = {'positive_base_increment_folds': int(b.net_increment_pp.gt(0).sum()) >= protocol['positive_base_increment_folds_minimum'],
                  'median_base_increment': float(b.net_increment_pp.median()) >= protocol['base_increment_median_minimum'] * 100,
                  'median_drawdown_change': float(b.closed_trade_drawdown_change_pp.median()) <= protocol['official_closed_trade_drawdown_median_change_maximum'] * 100,
                  'median_stress_increment': float(s.net_increment_pp.median()) > 0}
        report = {'experiment_id': output.name, 'status': 'verified_development_only',
                  'engine_runs': len(metrics), 'training_attempts': len(json.loads((output / 'training_attempts.json').read_text(encoding='utf-8'))),
                  'checks': checks, 'economic_screen_passed': all(checks.values()),
                  'positive_base_increment_folds': int(b.net_increment_pp.gt(0).sum()),
                  'median_base_increment_pp': float(b.net_increment_pp.median()),
                  'median_closed_trade_drawdown_change_pp': float(b.closed_trade_drawdown_change_pp.median()),
                  'median_stress_increment_pp': float(s.net_increment_pp.median()),
                  'final_holdout_inspected': False, 'frozen_representations_extracted': False,
                  'drawdown_basis': config['baselines']['drawdown_basis'],
                  'conclusion': 'support_further_research' if all(checks.values()) else 'no_increment',
                  'artifacts': {'metrics': str(output / 'metrics.csv'), 'increments': str(output / 'gate_increments.csv')}}
        plots(output, metrics, increments)
        write_json(output / 'report.json', report)
        registry['status'] = 'completed'
        registry['result'] = report
        registry['artifacts']['final_hashes'] = {str(p.relative_to(output)): sha(p)
                                               for p in output.rglob('*') if p.is_file() and p.suffix != '.log'}
        write_json(registry_path, registry)
        print(json.dumps(report), flush=True)
    except Exception as error:
        registry['status'] = 'failed_engine_or_audit'
        registry['result'] = {'conclusion': 'invalid_due_to_audit', 'failure_reason': repr(error)}
        write_json(registry_path, registry)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2, 3))
    parser.add_argument('--audit-existing', action='store_true', help='Resume a failed audit using existing official ZIPs; never rerun engine')
    args = parser.parse_args()
    run(args.output_dir, args.workers, args.audit_existing)
