"""Register and run official Freqtrade causality tools on sealed development ranges.

The worker wraps result-returning methods only to retain evidence; it neither
changes the strategy nor the official computations/configuration overrides.
Run with the Freqtrade interpreter from the Kronos repository root.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENGINE = Path('D:/file/freqtrade')
RUNTIME = ENGINE / 'user_data/kronos_research'
DATA = RUNTIME / 'data/20261008T011408835523Z_7bb06df6f064'
BASE = ROOT / 'research/runs/E00_20261008_phase4_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def worker(folder, argv):
    """Retain native tool results without altering arguments or returned values."""
    import numpy as np
    from freqtrade.optimize.analysis.lookahead import LookaheadAnalysis
    from freqtrade.optimize.analysis.recursive import RecursiveAnalysis
    native_result = LookaheadAnalysis.get_result
    count = 0

    def capture_result(backtesting, processed):
        nonlocal count
        result = native_result(backtesting, processed)
        columns = ['pair', 'open_date', 'close_date', 'exit_reason', 'is_short']
        records = result['results'][columns].to_dict(orient='records')
        write(folder / f'native_trades_{count:03d}.json', records)
        count += 1
        return result

    LookaheadAnalysis.get_result = staticmethod(capture_result)
    native_recursive = RecursiveAnalysis.analyze_indicators

    def capture_recursive(instance):
        pair = instance.pair_to_used

        def row(holder):
            values = holder.indicators[pair].iloc[-1].to_dict()
            return {key: value.item() if isinstance(value, np.generic) else value
                    for key, value in values.items()}

        write(folder / 'native_recursive_tail_rows.json', {
            'base': row(instance.full_varHolder),
            'partials': [{'startup_candle': part.startup_candle, 'tail': row(part)}
                         for part in instance.partial_varHolder_array],
        })
        return native_recursive(instance)

    RecursiveAnalysis.analyze_indicators = capture_recursive
    from freqtrade.main import main
    return main(argv)


def run(output):
    if output.exists():
        raise ValueError('Use a new output directory; immutable evidence cannot be overwritten')
    output.mkdir(parents=True)
    provenance = output / 'provenance'
    provenance.mkdir()
    registry_path = ROOT / 'research/registry' / (output.name + '.json')
    if registry_path.exists():
        raise ValueError('Registry already exists')
    sources = [Path(__file__), ROOT / 'research/configs/initial_experiment.yaml',
               ROOT / 'research/freqtrade/strategies/KronosE00.py',
               ROOT / 'research/baselines/ordinary_features.py',
               ROOT / 'research/baselines/prepare_e00.py',
               ROOT / 'research/initialization/freqtrade_import_manifest.json',
               ROOT / 'research/registry/E00_20261008_phase4_v1.json',
               RUNTIME / 'strategies/KronosE00.py']
    sources += [ENGINE / f'freqtrade/optimize/analysis/{name}.py' for name in
                ('lookahead', 'lookahead_helpers', 'recursive', 'recursive_helpers', 'base_analysis')]
    sources += [ENGINE / 'freqtrade/optimize/backtesting.py', ENGINE / 'freqtrade/strategy/interface.py']
    manifest = json.loads((ROOT / 'research/initialization/freqtrade_import_manifest.json').read_text(encoding='utf-8'))
    native = {}
    for relative, expected in manifest['files'].items():
        actual = sha(DATA / relative)
        if actual != expected:
            raise ValueError('Native data hash mismatch: ' + relative)
        native[relative] = actual
    if sha(sources[2]) != sha(RUNTIME / 'strategies/KronosE00.py'):
        raise ValueError('Runtime strategy differs from canonical source')
    jobs = []
    for fold, mode, timerange, target in (
        ('WF01', 'fixed_momentum', '20250401-20250630T2300', 12),
        ('WF03', 'ordinary_features_gate', '20251001-20251231T2300', 3),
    ):
        config_path = BASE / 'engine' / fold / mode / 'base/config.json'
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
        sources += [config_path, Path(cfg['e00_decisions_path'])]
        for tool in ('lookahead-analysis', 'recursive-analysis'):
            folder = output / fold / tool
            folder.mkdir(parents=True)
            args = [tool, '--config', str(config_path), '--strategy', 'KronosE00',
                    '--strategy-path', str(RUNTIME / 'strategies'), '--datadir', str(DATA),
                    '--timerange', timerange, '--pairs', 'BTC/USDT:USDT', '--no-color']
            if tool == 'lookahead-analysis':
                args += ['--minimum-trade-amount', '1', '--targeted-trade-amount', str(target),
                         '--lookahead-analysis-exportfilename', str(folder / 'lookahead.csv'),
                         '--backtest-directory', str(folder), '--export', 'none']
            else:
                args += ['--startup-candle', '256', '512', '1024']
            command = [str(ENGINE / '.venv/Scripts/python.exe'), str(Path(__file__).resolve()),
                       '--worker', str(folder), '--', *args]
            jobs.append({'fold': fold, 'mode': mode, 'tool': tool, 'timerange': timerange,
                         'targeted_trade_amount': target if tool == 'lookahead-analysis' else None,
                         'command': command, 'folder': str(folder), 'status': 'planned'})
    source_records = []
    for index, source in enumerate(sources):
        destination = provenance / f'{index:02d}_{source.name}'
        shutil.copy2(source, destination)
        source_records.append({'path': str(source), 'sha256': sha(source), 'copy': str(destination)})
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ENGINE, text=True).strip()
    registry = {'schema_version': 1, 'id': output.name, 'status': 'registered_before_execution',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'hypothesis': 'Historical adapter signals and fixed exits are stable under official truncation and startup tests',
                'selection_scope': 'WF01 fixed_momentum and WF03 ordinary_features_gate only; no model/threshold selection',
                'final_holdout_read_or_evaluated': False, 'engine_commit': commit,
                'sources': source_records, 'native_file_sha256': native, 'attempts': jobs,
                'limitations': ['Precomputed sidecar truncation cannot certify the producer training/features.',
                                'Official lookahead overrides stake, wallet, order type and max_open_trades.',
                                'Recursive tool checks indicators only and only the last row per startup size.']}
    write(registry_path, registry)
    registry['status'] = 'running'
    write(registry_path, registry)
    results = []
    for job in jobs:
        folder = Path(job['folder'])
        print('START', job['fold'], job['tool'], flush=True)
        with (folder / 'tool.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(job['command'], cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT)
        job.update(status='completed' if completed.returncode == 0 else 'failed', returncode=completed.returncode)
        log_text = (folder / 'tool.log').read_text(encoding='utf-8')
        result = {key: job[key] for key in ('fold', 'mode', 'tool', 'returncode', 'timerange')}
        if job['tool'] == 'lookahead-analysis':
            csv_path = folder / 'lookahead.csv'
            rows = list(csv.DictReader(csv_path.open(encoding='utf-8'))) if csv_path.exists() else []
            result['official_csv_rows'] = rows
            trades_path = folder / 'native_trades_000.json'
            trades = json.loads(trades_path.read_text(encoding='utf-8')) if trades_path.exists() else []
            result['baseline_native_trades'] = len(trades)
            result['baseline_exit_reasons'] = sorted({t['exit_reason'] for t in trades})
            result['baseline_directions'] = sorted({t['is_short'] for t in trades})
            result['all_baseline_exits_custom_4h'] = bool(trades) and all(
                t['exit_reason'] == 'fixed_hold_4h' and
                (datetime.fromisoformat(t['close_date']) - datetime.fromisoformat(t['open_date'])).total_seconds() == 14400
                for t in trades)
            result['accepted'] = completed.returncode == 0 and len(rows) == 1 and rows[0]['has_bias'] == 'False' and result['all_baseline_exits_custom_4h']
        else:
            result['native_no_recursive_variance_message'] = 'No variance on indicator(s) found due to recursive formula.' in log_text
            result['native_no_indicator_lookahead_message'] = 'No lookahead bias on indicators found.' in log_text
            tails_path = folder / 'native_recursive_tail_rows.json'
            tails = json.loads(tails_path.read_text(encoding='utf-8')) if tails_path.exists() else {}
            result['tail_rows_exact_match'] = bool(tails) and all(t['tail'] == tails['base'] for t in tails['partials'])
            result['accepted'] = completed.returncode == 0 and result['tail_rows_exact_match'] and result['native_no_indicator_lookahead_message']
        result['evidence_sha256'] = {str(p.relative_to(output)): sha(p) for p in folder.iterdir() if p.is_file()}
        results.append(result)
        write(folder / 'result.json', result)
        write(registry_path, registry)
        print('DONE', job['fold'], job['tool'], 'accepted=', result['accepted'], flush=True)
    report = {'schema_version': 1, 'status': 'accepted_scoped' if all(r['accepted'] for r in results) else 'requires_review',
              'run_directory': str(output), 'registry': str(registry_path), 'results': results,
              'scope': 'Official adapter causality checks on WF01 and WF03 development periods only',
              'limitations': registry['limitations'], 'final_holdout_read_or_evaluated': False,
              'ready_to_encode_changed': False}
    write(output / 'report.json', report)
    report_path = ROOT / 'research/initialization/e00_causality_report.json'
    if report_path.exists():
        report_path = output / 'report.json'
    else:
        write(report_path, report)
    registry.update(status=report['status'], report=str(report_path), report_sha256=sha(report_path))
    write(registry_path, registry)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        sys.exit(worker(Path(sys.argv[2]), sys.argv[4:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'research/runs/E00_20261008_phase4_causality')
    run(parser.parse_args().output.resolve())
