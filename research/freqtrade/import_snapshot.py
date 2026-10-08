"""Validate an immutable OKX snapshot and import it with official Freqtrade handlers.

Run with the independent Freqtrade virtual environment. This imports research data
only; it does not establish settlement schedule completeness or run a backtest.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import freqtrade
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType

HOUR = 3_600_000
PAIR = 'BTC/USDT:USDT'
TIMEFRAME = '1h'
OHLCV = ['date', 'open', 'high', 'low', 'close', 'volume']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def milliseconds(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(dt.utcoffset() == timedelta(0), f'Explicit UTC timestamp required: {value}')
    require(dt.microsecond % 1000 == 0, f'Submillisecond timestamp: {value}')
    return int(dt.timestamp() * 1000)


def number(value, name, nonnegative=False):
    result = float(value)
    require(math.isfinite(result), f'Nonfinite {name}')
    require(not nonnegative or result >= 0, f'Negative {name}')
    return result


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f'Empty CSV: {path.name}')
    return rows


def candle_frame(rows, start, end, *, mark=False, evaluation_start=None):
    timestamps, values = [], []
    for row in rows:
        ts = milliseconds(row['bar_open_at'])
        require(ts % HOUR == 0, f'Off-grid candle: {row["bar_open_at"]}')
        require(row['confirm'] == '1', f'Unconfirmed candle at {ts}')
        o, h, low, close = [number(row[key], key) for key in ['open', 'high', 'low', 'close']]
        require(min(o, h, low, close) > 0 and low <= min(o, close)
                and h >= max(o, close) and low <= h, f'Invalid OHLC at {ts}')
        if mark:
            volume = 0.0  # Native OHLCV schema placeholder, not measured volume.
        else:
            volume = number(row['volume'], 'volume', True)
            number(row['amount'], 'actual quote turnover', True)
            number(row['raw_contract_volume'], 'raw contract volume', True)
            require(milliseconds(row['bar_close_at']) == ts + HOUR, f'Invalid bar close at {ts}')
            require(row['is_warmup'] == str(ts < evaluation_start), f'Invalid warmup flag at {ts}')
        timestamps.append(ts)
        values.append([o, h, low, close, volume])
    require(timestamps == list(range(start, end, HOUR)),
            f'{"Mark" if mark else "Trade"} candles must be ordered, unique and continuous over their exact range')
    frame = pd.DataFrame(values, columns=OHLCV[1:], dtype='float64')
    frame.insert(0, 'date', pd.to_datetime(timestamps, unit='ms', utc=True))
    return frame


def funding_frame(rows, start, end, mark_frame):
    timestamps, rates = [], []
    marks = set(mark_frame['date'].dt.as_unit('ms').astype('int64'))
    for row in rows:
        require(row['instrument'] == 'BTC-USDT-SWAP', 'Unexpected funding instrument')
        ts = milliseconds(row['funding_at'])
        require(start <= ts < end and ts % HOUR == 0, f'Funding outside evaluation or hourly grid: {ts}')
        require(ts in marks, f'Funding has no original settlement mark: {ts}')
        timestamps.append(ts)
        rates.append(number(row['realized_rate'], 'realized funding rate'))
    require(timestamps == sorted(set(timestamps)), 'Funding events must be ordered and unique')
    return pd.DataFrame({'date': pd.to_datetime(timestamps, unit='ms', utc=True),
                         'funding_rate': np.asarray(rates, dtype='float64')})


def verify_roundtrip(expected, actual, columns):
    require(len(expected) == len(actual), 'Native roundtrip row count changed')
    expected_dates = expected['date'].dt.as_unit('ns').astype('int64').to_numpy()
    actual_dates = actual['date'].dt.as_unit('ns').astype('int64').to_numpy()
    require(np.array_equal(expected_dates, actual_dates), 'Native roundtrip timestamps changed')
    expected_values = expected[columns].to_numpy(dtype='float64')
    actual_values = actual[columns].to_numpy(dtype='float64')
    exact = bool(np.array_equal(expected_values, actual_values))
    require(np.allclose(expected_values, actual_values, rtol=1e-12, atol=0, equal_nan=False),
            'Native roundtrip values changed beyond tolerance')
    return {'passed': True, 'rows': len(expected), 'timestamps_exact': True,
            'numeric_values_exact': exact, 'relative_tolerance': 1e-12,
            'absolute_tolerance': 0, 'fill_missing': False, 'drop_incomplete': False}


def import_snapshot(snapshot, datadir):
    snapshot, datadir = snapshot.resolve(), datadir.resolve()
    require(snapshot.is_dir(), 'Snapshot directory does not exist')
    require(datadir != snapshot and not datadir.is_relative_to(snapshot),
            'Output must not modify the immutable snapshot')
    require(not datadir.exists() or (datadir.is_dir() and not any(datadir.iterdir())),
            'Output datadir must be new or empty; existing data will not be overwritten')
    manifest_path = snapshot / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['status'] == 'collected', 'Only successful collected snapshots may be imported')
    files = manifest['files']
    for filename in ['candles.csv', 'mark_prices.csv', 'funding.csv', 'quality_report.json']:
        require(filename in files, f'Missing snapshot hash: {filename}')
    for filename, expected_hash in files.items():
        path = (snapshot / filename).resolve()
        require(path.is_relative_to(snapshot) and path != manifest_path,
                f'Invalid manifest file path: {filename}')
        require(path.is_file() and sha256(path) == expected_hash, f'Snapshot hash mismatch: {filename}')
    report = json.loads((snapshot / 'quality_report.json').read_text(encoding='utf-8'))
    require(report['ready_for_labeling'] is True, 'Snapshot structural audit is not ready')
    require(report['snapshot_id'] == manifest['snapshot_id'], 'Snapshot identifier mismatch')
    start = milliseconds(report['evaluation_start'])
    end = milliseconds(report['evaluation_end_exclusive'])
    data_start = milliseconds(report['data_start'])
    require(start < end and start % HOUR == end % HOUR == data_start % HOUR == 0,
            'Invalid UTC hourly range boundaries')
    require(report['warmup_bars'] == 256 and data_start == start - 256 * HOUR,
            'Expected exactly 256 hours of pre-evaluation warmup')
    candles = candle_frame(read_csv(snapshot / 'candles.csv'), data_start, end, evaluation_start=start)
    marks = candle_frame(read_csv(snapshot / 'mark_prices.csv'), start, end, mark=True)
    funding = funding_frame(read_csv(snapshot / 'funding.csv'), start, end, marks)
    for name, frame in [('candles', candles), ('mark_prices', marks), ('funding', funding)]:
        require(report['row_counts'][name] == len(frame), f'Reported row count mismatch: {name}')

    # All input validation finishes before creating any native data files.
    datadir.mkdir(parents=True, exist_ok=True)
    handler = get_datahandler(datadir, 'feather')
    checks = {}
    for name, candle_type, frame, numeric in [
        ('futures', CandleType.FUTURES, candles, OHLCV[1:]),
        ('mark', CandleType.MARK, marks, OHLCV[1:]),
        ('funding_rate', CandleType.FUNDING_RATE, funding, ['funding_rate']),
    ]:
        handler.ohlcv_store(PAIR, TIMEFRAME, frame, candle_type)
        actual = handler.ohlcv_load(PAIR, TIMEFRAME, candle_type,
                                    fill_missing=False, drop_incomplete=False)
        checks[name] = verify_roundtrip(frame, actual, numeric)
    repo = Path(freqtrade.__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    result = {'schema_version': 1, 'status': 'imported_and_roundtrip_verified',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'pair': PAIR, 'timeframe': TIMEFRAME, 'format': 'feather',
              'freqtrade_repository': str(repo), 'freqtrade_commit': commit,
              'snapshot': str(snapshot), 'snapshot_id': manifest['snapshot_id'],
              'snapshot_manifest_sha256': sha256(manifest_path),
              'snapshot_csv_sha256': {name: files[name] for name in ['candles.csv', 'mark_prices.csv', 'funding.csv']},
              'quality_report_sha256': files['quality_report.json'],
              'data_start': report['data_start'], 'evaluation_start': report['evaluation_start'],
              'evaluation_end_exclusive': report['evaluation_end_exclusive'], 'warmup_bars': 256,
              'actual_amount_sidecar': str(snapshot / 'candles.csv'),
              'mark_volume': 'zero_native_schema_placeholder_not_measured_volume',
              'funding_encoding': 'original_signed_realized_events_only_no_zero_expansion',
              'formal_evaluation_allowed': False,
              'formal_evaluation_blockers': ['historical_funding_settlement_schedule_unproven'],
              'backtest_boundary_caveat': 'Freqtrade stopdt is inclusive; outer labels and trades must enforce evaluation_start <= time < evaluation_end_exclusive and reject boundary-crossing trades',
              'roundtrip': checks,
              'files': {str(path.relative_to(datadir)): sha256(path)
                        for path in sorted(datadir.rglob('*.feather'))}}
    (datadir / 'import_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--datadir', required=True, type=Path)
    args = parser.parse_args()
    result = import_snapshot(args.snapshot, args.datadir)
    print(f'Imported {result["snapshot_id"]}: {args.datadir.resolve()} (roundtrip verified)')


if __name__ == '__main__':
    main()
