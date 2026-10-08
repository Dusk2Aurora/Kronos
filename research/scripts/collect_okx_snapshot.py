"""Collect public OKX candles and realized funding into an immutable research snapshot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from uuid import uuid4
import zipfile

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
LOCAL = timezone(timedelta(hours=8))
HOUR = 3_600_000
MAX_ARCHIVE = 2 * 1024 * 1024


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat().replace('+00:00', 'Z')


def stamp():
    return datetime.now(UTC).isoformat().replace('+00:00', 'Z')


def millis(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.utcoffset() != timedelta(0):
        raise ValueError('Research boundaries must explicitly use UTC')
    return int(dt.timestamp() * 1000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def next_month(dt):
    return dt.replace(year=dt.year + (dt.month == 12), month=dt.month % 12 + 1)


class Collector:
    def __init__(self, directory):
        self.directory = directory
        self.raw = directory / 'raw'
        self.raw.mkdir()
        self.session = requests.Session()
        self.sources = []

    def get(self, url, params=None, archive=False):
        parsed = urlparse(url)
        allowed = {'static.okx.com'} if archive else {'www.okx.com'}
        if parsed.scheme != 'https' or parsed.hostname not in allowed or parsed.port not in (None, 443):
            raise ValueError(f'Unapproved public source: {url}')
        for attempt in range(4):
            try:
                with self.session.get(url, params=params, timeout=20, stream=True,
                                      allow_redirects=False) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        response.raise_for_status()
                    if response.status_code != 200:
                        raise ValueError(f'HTTP {response.status_code}: {response.url}')
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > (MAX_ARCHIVE if archive else 8 * MAX_ARCHIVE):
                            raise ValueError('Response exceeds size limit')
                        chunks.append(chunk)
                    body = b''.join(chunks)
                    path = self.raw / f'{len(self.sources):05d}.{"zip" if archive else "json"}'
                    path.write_bytes(body)
                    self.sources.append({'url': response.url, 'collected_at': stamp(),
                                         'path': str(path.relative_to(self.directory)),
                                         'sha256': digest(path)})
                    if archive:
                        return body
                    value = json.loads(body)
                    if value.get('code') != '0':
                        raise ValueError(f'OKX API error: {value}')
                    return value['data']
            except (requests.RequestException, json.JSONDecodeError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError('unreachable')

    def api(self, endpoint, **params):
        return self.get('https://www.okx.com/api/v5/' + endpoint, params)


def finite(value, nonnegative=False):
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f'Invalid numeric value: {value}')
    return result


def candles(collector, instrument, start, end, delay, report, snapshot_id, mark=False,
            evaluation_start=None):
    rows, cursor = {}, end
    prefix = 'mark' if mark else 'candle'
    endpoint = 'market/history-mark-price-candles' if mark else 'market/history-candles'
    while True:
        page = collector.api(endpoint, instId=instrument, bar='1H', limit=100, after=cursor)
        if not page:
            # Missing historical mark coverage is an audit blocker, never filled.
            if mark:
                report['mark_pagination_stopped_on_empty_page'] = True
                break
            raise ValueError('Empty candle page before reaching start boundary')
        minimum = min(int(row[0]) for row in page)
        if minimum >= cursor:
            raise ValueError('Candle pagination failed to advance')
        for row in page:
            ts = int(row[0])
            if not start <= ts < end:
                continue
            if len(row) != (6 if mark else 9):
                raise ValueError('Unexpected OKX candle field count')
            if row[-1] != '1':
                report[f'incomplete_{prefix}s_excluded'] += 1
                continue
            numeric = tuple(finite(v, True) for v in row[1:-1])
            o, h, low, close = numeric[:4]
            if min(o, h, low, close) <= 0 or low > min(o, close) or h < max(o, close) or low > h:
                raise ValueError(f'Invalid OHLC at {iso(ts)}')
            if ts % HOUR:
                raise ValueError(f'Candle off UTC hourly grid: {ts}')
            if ts in rows and rows[ts] != numeric:
                report['conflicting_duplicates'].append({'kind': prefix, 'timestamp': iso(ts)})
                raise ValueError('Conflicting duplicate candle')
            rows[ts] = numeric
        if minimum <= start:
            break
        cursor = minimum
        time.sleep(0.12)
    expected = set(range(start, end, HOUR))
    missing = sorted(expected - rows.keys())
    report[f'{prefix}s_count'] = len(rows)
    report[f'{prefix}s_expected_count'] = len(expected)
    report[f'{prefix}_grid_valid'] = True
    report[f'{prefix}_gaps'] = [iso(ts) for ts in missing]
    output = []
    for ts, values in sorted(rows.items()):
        o, h, low, close = values[:4]
        if mark:
            output.append({'bar_open_at': iso(ts), 'open': o, 'high': h,
                           'low': low, 'close': close, 'confirm': 1})
            continue
        contracts, volume, amount = values[4:]
        output.append({'bar_open_at': iso(ts), 'bar_close_at': iso(ts + HOUR),
                       'open': o, 'high': h, 'low': low, 'close': close, 'volume': volume,
                       'amount': amount, 'raw_contract_volume': contracts, 'confirm': 1,
                       'available_at': iso(ts + HOUR + delay * 1000),
                       'is_warmup': evaluation_start is not None and ts < evaluation_start,
                       'availability_is_measured': False, 'snapshot_id': snapshot_id})
    return output


def funding(collector, instrument, start, end, report):
    rates = {}
    overlap = 0

    def add(ts, rate, kind):
        nonlocal overlap
        rate = finite(rate)
        if ts in rates:
            old = rates[ts]
            if abs(old['realized_rate'] - rate) > 1e-14:
                report['conflicting_duplicates'].append({'kind': 'funding', 'timestamp': iso(ts)})
                raise ValueError('Conflicting realized funding rates')
            if kind == 'REST_realizedRate' and old['source_kind'] == 'archive_realized':
                overlap += 1
                old['source_kind'] = 'archive_and_REST_realized'
            return
        rates[ts] = {'instrument': instrument, 'funding_at': iso(ts),
                     'realized_rate': rate, 'source_kind': kind}

    month = datetime.fromtimestamp(start / 1000, LOCAL).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = []
    while int(month.timestamp() * 1000) < end:
        months.append(month)
        month = next_month(month)
    files = {}
    for offset in range(0, len(months), 9):
        batch = months[offset:offset + 9]
        listing = collector.api('public/market-data-history', module=3, instType='SWAP',
                                instFamilyList='BTC-USDT', dateAggrType='monthly',
                                begin=int(batch[0].timestamp() * 1000),
                                end=int(batch[-1].timestamp() * 1000))
        for group in listing:
            for detail in group.get('details', []):
                for entry in detail.get('groupDetails', []):
                    filename = entry['filename']
                    if filename.startswith(instrument + '-fundingrates-'):
                        if filename in files and files[filename] != entry['url']:
                            raise ValueError('Conflicting archive URLs')
                        files[filename] = entry['url']
    missing, bridged = [], []
    for month in months:
        filename = f'{instrument}-fundingrates-{month:%Y-%m}.zip'
        if filename not in files:
            # The final partial UTC+8 month may not yet be published; REST bridges it.
            if int(next_month(month).timestamp() * 1000) > end:
                bridged.append(filename)
            else:
                missing.append(filename)
            continue
        body = collector.get(files[filename], archive=True)
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if not members or sum(item.file_size for item in members) > MAX_ARCHIVE:
                raise ValueError('Archive expanded size invalid')
            for member in members:
                text = archive.read(member).decode('utf-8-sig')
                reader = csv.DictReader(io.StringIO(text))
                if not {'instrument_name', 'funding_rate', 'funding_time'} <= set(reader.fieldnames or []):
                    raise ValueError('Unexpected funding archive columns')
                for row in reader:
                    if row['instrument_name'] == instrument:
                        add(int(row['funding_time']), row['funding_rate'], 'archive_realized')
    recent = collector.api('public/funding-rate-history', instId=instrument, limit=400)
    for row in recent:
        if row['instId'] == instrument:
            if not row.get('realizedRate'):
                raise ValueError('REST funding lacks realizedRate')
            add(int(row['fundingTime']), row['realizedRate'], 'REST_realizedRate')
    times = sorted(ts for ts in rates if start <= ts < end)
    intervals, suspicious = {}, []
    for left, right in zip(times, times[1:]):
        duration = right - left
        key = str(duration / HOUR)
        intervals[key] = intervals.get(key, 0) + 1
        if duration > 8 * HOUR:
            suspicious.append({'after': iso(left), 'before': iso(right), 'hours': duration / HOUR})
    boundaries_ok = bool(times) and times[0] < start + 8 * HOUR and times[-1] >= end - 8 * HOUR
    report['funding_coverage'] = {'count': len(times), 'first': iso(times[0]) if times else None,
        'last': iso(times[-1]) if times else None, 'missing_archive_files': missing,
        'partial_month_REST_bridge_candidates': bridged, 'boundary_8h_audit_passed': boundaries_ok,
        'suspected_missing_intervals': suspicious,
        'historical_schedule_completeness_proven': False}
    report['interval_counts_hours'] = intervals
    report['funding_overlap_match'] = {'matched_events': overlap, 'absolute_tolerance': 1e-14,
                                      'passed': overlap > 0}
    return [rates[ts] for ts in times]


def audit_funding_marks(funding_rows, mark_rows):
    """Require each event to match exactly one original hourly mark candle."""
    mark_counts = {}
    for row in mark_rows:
        ts = millis(row['bar_open_at'])
        mark_counts[ts] = mark_counts.get(ts, 0) + 1
    event_counts = {}
    off_grid, unmatched = [], []
    matched = 0
    for row in funding_rows:
        ts = millis(row['funding_at'])
        event_counts[ts] = event_counts.get(ts, 0) + 1
        if ts % HOUR:
            off_grid.append(iso(ts))
        if mark_counts.get(ts, 0) != 1:
            unmatched.append(iso(ts))
        else:
            matched += 1
    duplicates = [iso(ts) for ts, count in sorted(event_counts.items()) if count != 1]
    return {'funding_event_count': len(funding_rows), 'matched_event_count': matched,
            'off_hourly_grid_events': off_grid, 'unmatched_or_ambiguous_events': unmatched,
            'duplicate_funding_events': duplicates,
            'passed': bool(funding_rows) and not off_grid and not unmatched and not duplicates,
            'basis': 'exact_UTC_timestamp_join_to_original_unfilled_marks'}


def write_csv(path, rows, fieldnames=None):
    if not rows and not fieldnames:
        raise ValueError(f'No rows for {path.name}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='research/configs/initial_experiment.yaml')
    parser.add_argument('--output-root', default='research/data/snapshots')
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    market = config['market']
    start, end = millis(market['history_start']), millis(market['history_end_exclusive'])
    if market['instrument'] != 'BTC-USDT-SWAP' or market['timeframe'] != '1h':
        raise ValueError('This collector is scoped to BTC-USDT-SWAP 1h')
    if start >= end or start % HOUR or end % HOUR:
        raise ValueError('Expected increasing UTC hourly boundaries')
    warmup_bars = config['timing']['lookback_bars']
    if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int) or warmup_bars <= 0:
        raise ValueError('Expected positive integer lookback_bars')
    data_start = start - warmup_bars * HOUR
    snapshot_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:12]
    directory = resolve_path(args.output_root) / snapshot_id
    directory.mkdir(parents=True, exist_ok=False)
    collector = Collector(directory)
    report = {'schema_version': 2, 'snapshot_id': snapshot_id,
              'data_start': iso(data_start), 'evaluation_start': iso(start),
              'evaluation_end_exclusive': iso(end), 'ready_for_labeling': False,
              'readiness_basis': 'structural_data_quality_only_not_settlement_schedule_proof',
              'formal_evaluation_allowed': False,
              'formal_evaluation_blockers': ['historical_funding_settlement_schedule_unproven'],
              'warmup_bars': warmup_bars, 'warmup_range': [iso(data_start), iso(start)],
              'candles_expected_count': (end - data_start) // HOUR,
              'row_counts': {'candles': 0, 'funding': 0, 'mark_prices': 0},
              'marks_expected_count': (end - start) // HOUR,
              'mark_grid_valid': None, 'mark_gaps': None,
              'candle_grid_valid': None, 'candle_gaps': None,
              'funding_coverage': None, 'interval_counts_hours': {}, 'funding_overlap_match': None,
              'funding_mark_join': None,
              'conflicting_duplicates': [], 'incomplete_candles_excluded': 0, 'incomplete_marks_excluded': 0,
              'assumptions': {'availability_is_measured': False,
                              'availability_delay_seconds': config['timing']['availability_delay_seconds'],
                              'funding_interval_audit_reference_hours': 8,
                              'settlement_mark_proxy': 'hourly_open_not_tick_exact_or_account_debit',
                              'collected_at_is_point_in_time_evidence': False}}
    manifest = {'schema_version': 1, 'snapshot_id': snapshot_id, 'status': 'collecting',
                'created_at': stamp(), 'sources': collector.sources}
    for name, path in [('config', config_path), ('script', Path(__file__)),
                       ('data_contract', ROOT / 'research/data_contracts/okx_usdt_perp.yaml'),
                       ('version_manifest', ROOT / 'research/initialization/version_manifest.json')]:
        manifest[name] = {'path': str(path), 'sha256': digest(path) if path.exists() else None}
    write_json(directory / 'manifest.json', manifest)
    try:
        manifest['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        metadata = collector.api('public/instruments', instType='SWAP', instId=market['instrument'])
        if len(metadata) != 1 or metadata[0]['instId'] != market['instrument']:
            raise ValueError('Instrument metadata absent or ambiguous')
        manifest['contract_metadata'] = metadata[0]
        if metadata[0]['ctValCcy'] != 'BTC' or metadata[0]['settleCcy'] != 'USDT':
            raise ValueError('Unexpected contract currency metadata')
        candle_rows = candles(collector, market['instrument'], data_start, end,
                              config['timing']['availability_delay_seconds'], report, snapshot_id,
                              evaluation_start=start)
        write_csv(directory / 'candles.csv', candle_rows)
        report['row_counts']['candles'] = len(candle_rows)
        report['row_counts']['warmup_candles'] = sum(row['is_warmup'] for row in candle_rows)
        report['row_counts']['evaluation_candles'] = len(candle_rows) - report['row_counts']['warmup_candles']
        mark_rows = candles(collector, market['instrument'], start, end,
                            config['timing']['availability_delay_seconds'], report, snapshot_id, mark=True)
        write_csv(directory / 'mark_prices.csv', mark_rows,
                  ['bar_open_at', 'open', 'high', 'low', 'close', 'confirm'])
        report['row_counts']['mark_prices'] = len(mark_rows)
        funding_rows = funding(collector, market['instrument'], start, end, report)
        write_csv(directory / 'funding.csv', funding_rows)
        coverage = report['funding_coverage']
        report['row_counts']['funding'] = len(funding_rows)
        report['funding_mark_join'] = audit_funding_marks(funding_rows, mark_rows)
        report['ready_for_labeling'] = (not report['candle_gaps'] and not report['mark_gaps']
            and not coverage['missing_archive_files']
            and not coverage['suspected_missing_intervals'] and coverage['boundary_8h_audit_passed']
            and report['funding_overlap_match']['passed'] and report['funding_mark_join']['passed'])
        manifest['status'] = 'collected' if report['ready_for_labeling'] else 'audit_not_ready'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        write_json(directory / 'error.json', {'error': manifest['error'], 'at': stamp()})
        raise
    finally:
        collector.session.close()
        write_json(directory / 'quality_report.json', report)
        manifest['finished_at'] = stamp()
        manifest['files'] = {str(path.relative_to(directory)): digest(path)
                             for path in sorted(directory.rglob('*'))
                             if path.is_file() and path.name != 'manifest.json'}
        write_json(directory / 'manifest.json', manifest)
        print(f'Snapshot: {directory} ({manifest["status"]})', flush=True)
    if not report['ready_for_labeling']:
        raise SystemExit('Snapshot structural audit requires review; see quality_report.json')


if __name__ == '__main__':
    main()
