"""Read-only reconstruction audit of an OKX BTC-USDT-SWAP snapshot.

No network, imports, backtests, training, or changes to the snapshot. The optional
report must be outside the snapshot. Funding grid completeness is conditional on
the observed eight-hour schedule, not proof of every historical exchange debit.
"""
from __future__ import annotations

import argparse
import calendar
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from urllib.parse import urlparse
import zipfile

HOUR = 3_600_000
INSTRUMENT = 'BTC-USDT-SWAP'
LOCAL = timezone(timedelta(hours=8))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ms(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(dt.utcoffset() == timedelta(0), 'Expected explicit UTC timestamp')
    require(dt.microsecond % 1000 == 0, 'Submillisecond timestamp')
    return int(dt.timestamp() * 1000)


def iso(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace('+00:00', 'Z')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    result = float(value)
    require(math.isfinite(result), f'Nonfinite value: {value}')
    return result


def safe_path(root, value):
    # Snapshot manifest was collected on Windows; support auditing on POSIX too.
    path = (root / value.replace('\\', '/')).resolve()
    require(path.is_relative_to(root) and path != root / 'manifest.json', f'Unsafe path: {value}')
    return path


def audit_snapshot(snapshot):
    snapshot = snapshot.resolve()
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8'))
    quality = json.loads((snapshot / 'quality_report.json').read_text(encoding='utf-8'))
    require(manifest['snapshot_id'] == quality['snapshot_id'] == snapshot.name, 'Snapshot ID mismatch')
    start, end, data_start = map(ms, (quality['evaluation_start'], quality['evaluation_end_exclusive'], quality['data_start']))
    require(data_start < start < end and all(t % HOUR == 0 for t in (start, end, data_start)), 'Invalid range')
    require(start - data_start == quality['warmup_bars'] * HOUR, 'Warmup range mismatch')
    declared = {name.replace('\\', '/'): value for name, value in manifest['files'].items()}
    require(len(declared) == len(manifest['files']), 'Equivalent manifest paths duplicated')
    actual = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob('*')
              if path.is_file() and path != snapshot / 'manifest.json'}
    require(set(declared) == actual, f'Manifest file set differs: {set(declared) ^ actual}')
    for name, expected in declared.items():
        require(sha(safe_path(snapshot, name)) == expected, f'Hash mismatch: {name}')
    sources = manifest['sources']
    source_paths = [source['path'].replace('\\', '/') for source in sources]
    require(len(source_paths) == len(set(source_paths)), 'Duplicate raw source paths')
    require(set(source_paths) == {name for name in actual if name.startswith('raw/')}, 'Raw source set mismatch')

    trade, mark, archive_rates, rest_rates = {}, {}, {}, {}
    page_counts, duplicate_counts = Counter(), Counter()
    archives, listings = [], {}

    def add(target, timestamp, value, kind):
        if timestamp in target:
            duplicate_counts[kind] += 1
            require(target[timestamp] == value, f'Conflicting {kind} duplicate at {timestamp}')
        target[timestamp] = value

    for source in sources:
        name = source['path'].replace('\\', '/')
        path = safe_path(snapshot, name)
        require(source['sha256'] == declared[name] == sha(path), f'Source hash mismatch: {name}')
        datetime.fromisoformat(source['collected_at'].replace('Z', '+00:00'))
        url = urlparse(source['url'])
        require(url.scheme == 'https' and url.port in (None, 443), 'Invalid source URL')
        endpoint = url.path.rsplit('/', 1)[-1]
        page_counts[endpoint] += 1
        if path.suffix == '.zip':
            require(url.hostname == 'static.okx.com', 'Archive host mismatch')
            require(endpoint.startswith(INSTRUMENT + '-fundingrates-') and endpoint.endswith('.zip'), 'Archive filename mismatch')
            year, month = map(int, endpoint.rsplit('fundingrates-', 1)[1][:-4].split('-'))
            left = datetime(year, month, 1, tzinfo=LOCAL)
            right = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=LOCAL)
            with zipfile.ZipFile(path) as zipped:
                members = [item for item in zipped.infolist() if not item.is_dir()]
                require(len(members) == 1 and members[0].file_size <= 2 * 1024 * 1024, 'Unexpected archive members/size')
                rows = list(csv.DictReader(io.StringIO(zipped.read(members[0]).decode('utf-8-sig'))))
            require(bool(rows) and all(row['instrument_name'] == INSTRUMENT for row in rows), 'Archive instrument mismatch')
            timestamps = [int(row['funding_time']) for row in rows]
            expected = list(range(int(left.timestamp() * 1000), int(right.timestamp() * 1000), 8 * HOUR))
            require(timestamps == expected, f'Month archive not exact observed eight-hour grid: {endpoint}')
            for row, timestamp in zip(rows, timestamps):
                add(archive_rates, timestamp, number(row['funding_rate']), 'archive')
            archives.append({'filename': endpoint, 'url': source['url'], 'raw_path': name,
                             'rows': len(rows), 'expected_rows': calendar.monthrange(year, month)[1] * 3,
                             'first': iso(timestamps[0]), 'last': iso(timestamps[-1]),
                             'utc_plus_8_month_grid_exact': True})
            continue
        require(url.hostname == 'www.okx.com', 'API host mismatch')
        response = json.loads(path.read_text(encoding='utf-8'))
        require(response['code'] == '0', f'API unsuccessful: {name}')
        data = response['data']
        if endpoint in ('history-candles', 'history-mark-price-candles'):
            is_mark = endpoint == 'history-mark-price-candles'
            target = mark if is_mark else trade
            for row in data:
                require(len(row) == (6 if is_mark else 9), 'Candle field count mismatch')
                if row[-1] != '1':
                    duplicate_counts['unconfirmed_excluded'] += 1
                    continue
                timestamp, values = int(row[0]), tuple(number(value) for value in row[1:-1])
                o, high, low, close = values[:4]
                require(timestamp % HOUR == 0 and min(o, high, low, close) > 0
                        and low <= min(o, close) and high >= max(o, close) and low <= high
                        and all(value >= 0 for value in values[4:]), 'Invalid raw candle')
                add(target, timestamp, values, endpoint)
        elif endpoint == 'funding-rate-history':
            for row in data:
                require(row['instId'] == INSTRUMENT and row.get('realizedRate') not in (None, ''), 'Missing realizedRate')
                add(rest_rates, int(row['fundingTime']), number(row['realizedRate']), 'REST')
        elif endpoint == 'market-data-history':
            for group in data:
                for detail in group.get('details', []):
                    for entry in detail.get('groupDetails', []):
                        if entry['filename'].startswith(INSTRUMENT + '-fundingrates-'):
                            previous = listings.get(entry['filename'])
                            require(previous in (None, entry['url']), 'Conflicting listing URLs')
                            listings[entry['filename']] = entry['url']
        elif endpoint == 'instruments':
            require(len(data) == 1 and data[0] == manifest['contract_metadata'], 'Instrument metadata mismatch')
        else:
            raise ValueError(f'Unexpected raw endpoint: {endpoint}')

    for archive in archives:
        require(listings.get(archive['filename']) == archive['url'], 'Downloaded archive not tied to listing')
    archive_names = {archive['filename'] for archive in archives}
    require(len(archive_names) == len(archives), 'Duplicate monthly archive')
    month = datetime.fromtimestamp(start / 1000, LOCAL).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    required_months, permitted_months = set(), set()
    while int(month.timestamp() * 1000) < end:
        following = month.replace(year=month.year + (month.month == 12), month=month.month % 12 + 1)
        name = f'{INSTRUMENT}-fundingrates-{month:%Y-%m}.zip'
        permitted_months.add(name)
        if int(following.timestamp() * 1000) <= end:
            required_months.add(name)
        month = following
    require(required_months <= archive_names <= permitted_months, 'Missing full monthly archive or archive outside requested range')
    candle_checks = {}
    for filename, raw, fields, begin in (
        ('candles.csv', trade, ['open', 'high', 'low', 'close', 'raw_contract_volume', 'volume', 'amount'], data_start),
        ('mark_prices.csv', mark, ['open', 'high', 'low', 'close'], start),
    ):
        rows = read_csv(snapshot / filename)
        times = [ms(row['bar_open_at']) for row in rows]
        require(times == list(range(begin, end, HOUR)), f'{filename} range/grid/order mismatch')
        require({t for t in raw if begin <= t < end} == set(times), f'{filename} raw timestamp set mismatch')
        for row, timestamp in zip(rows, times):
            require(row['confirm'] == '1' and tuple(number(row[key]) for key in fields) == raw[timestamp], f'{filename} raw values mismatch at {timestamp}')
            if filename == 'candles.csv':
                require(ms(row['bar_close_at']) == timestamp + HOUR, 'Close timestamp mismatch')
                require(ms(row['available_at']) == timestamp + HOUR + quality['assumptions']['availability_delay_seconds'] * 1000, 'Availability timestamp mismatch')
                require(row['is_warmup'] == str(timestamp < start) and row['snapshot_id'] == manifest['snapshot_id']
                        and row['availability_is_measured'] == 'False', 'Candle auxiliary fields mismatch')
        key = 'candles' if filename == 'candles.csv' else 'mark_prices'
        require(quality['row_counts'][key] == len(rows), 'Quality row count mismatch')
        candle_checks[key] = {'rows': len(rows), 'expected_rows': (end - begin) // HOUR,
                              'hourly_grid_exact': True, 'raw_all_numeric_fields_exact': True,
                              'raw_confirmed_only': True, 'first': iso(times[0]), 'last': iso(times[-1])}

    overlap = set(archive_rates) & set(rest_rates)
    require(bool(overlap), 'No independent archive/REST overlap')
    max_difference = max(abs(archive_rates[t] - rest_rates[t]) for t in overlap)
    require(max_difference <= 1e-14, 'Archive/REST rates conflict')
    combined = dict(archive_rates)
    combined.update(rest_rates)
    rows = read_csv(snapshot / 'funding.csv')
    times = [ms(row['funding_at']) for row in rows]
    require(times == list(range(start, end, 8 * HOUR)), 'Funding not complete observed eight-hour UTC grid')
    require(set(times) == {t for t in combined if start <= t < end}, 'Funding raw timestamp set mismatch')
    for row, timestamp in zip(rows, times):
        expected_kind = ('archive_and_REST_realized' if timestamp in overlap else
                         'archive_realized' if timestamp in archive_rates else 'REST_realizedRate')
        require(row['instrument'] == INSTRUMENT and row['source_kind'] == expected_kind
                and number(row['realized_rate']) == combined[timestamp], 'Funding source/rate mismatch')
        require(timestamp in mark, 'Funding missing original raw settlement mark')
    require(quality['row_counts']['funding'] == len(rows), 'Funding reported row count mismatch')
    bridge = [row for row in rows if row['source_kind'] == 'REST_realizedRate']
    provenance = {}
    for key in ('config', 'script', 'data_contract', 'version_manifest'):
        source = Path(manifest[key]['path'])
        provenance[key] = {'collected_sha256': manifest[key]['sha256'], 'current_file_exists': source.is_file(),
                           'current_hash_matches_collection': sha(source) == manifest[key]['sha256'] if source.is_file() else None}
    return {
        'schema_version': 1, 'status': 'passed', 'audited_at': datetime.now(timezone.utc).isoformat(),
        'snapshot_id': manifest['snapshot_id'], 'snapshot': str(snapshot), 'snapshot_manifest_sha256': sha(snapshot / 'manifest.json'),
        'method': 'independent_local_raw_reconstruction_no_network_or_snapshot_mutation',
        'ranges': {'data_start': iso(data_start), 'evaluation_start': iso(start), 'evaluation_end_exclusive': iso(end), 'warmup_bars': quality['warmup_bars']},
        'files': {'manifest_exact_file_set': True, 'hashed_files': len(actual), 'raw_sources': len(sources),
                  'all_manifest_hashes_passed': True, 'raw_source_set_exact': True, 'source_hashes_passed': True},
        'raw_pages': dict(page_counts), 'duplicates': dict(duplicate_counts),
        'candles': candle_checks, 'monthly_archives': sorted(archives, key=lambda value: value['filename']),
        'funding': {'rows': len(rows), 'observed_8h_grid_exact': True, 'first': iso(times[0]), 'last': iso(times[-1]),
                    'archive_events_in_evaluation': sum(start <= t < end for t in archive_rates),
                    'REST_events_in_evaluation': sum(start <= t < end for t in rest_rates),
                    'source_kind_counts': dict(Counter(row['source_kind'] for row in rows)), 'REST_only_events': bridge,
                    'archive_REST_overlap_events': len(overlap), 'overlap_max_absolute_difference': max_difference,
                    'normalized_rates_match_raw': True, 'nonzero_events': sum(number(row['realized_rate']) != 0 for row in rows),
                    'negative_events': sum(number(row['realized_rate']) < 0 for row in rows),
                    'original_mark_exact_timestamp_join_events': len(rows)},
        'external_provenance': provenance,
        'conclusions': {'official_source_complete': True,
                        'published_official_source_coverage_verified': True,
                        'official_source_complete_scope': 'Preserved official monthly files and realized REST events exactly reconstruct this range; complete conditional on observed 8h schedule.',
                        'structural_data_ready': True, 'historical_actual_exchange_settlement_schedule_independently_proven': False,
                        'point_in_time_availability_proven': False, 'historical_order_execution_or_tick_exact_mark_proven': False,
                        'research_protocol_or_formal_evaluation_approval': False},
        'limitations': ['Local hashes establish internal integrity, not independent authenticity of every historical exchange record.',
                        'Observed complete 8h grid cannot independently rule out an exchange omission or an undocumented extra settlement.',
                        'Collection is retrospective; available_at is a configured assumption, not measured historical availability.',
                        'Hourly mark open is a settlement proxy; no tick-exact debit or historical order fill precision is established.',
                        'Config/version manifest collection hashes are preserved, but original full files were not copied into the snapshot.'],
        'code_review_observations': [
            {'file': 'research/scripts/collect_okx_snapshot.py', 'line': 252,
             'observation': 'Collector rejects funding gaps greater than 8h; the independent audit additionally checks exact observed UTC8h phase/grid and each UTC+8 monthly archive grid.'},
            {'file': 'research/freqtrade/import_snapshot.py', 'line': 130,
             'observation': 'Importer verifies listed hashes and required normalized files; the independent audit additionally checks exact manifest/raw source sets and reconstructs all normalized raw numeric fields.'},
            {'file': 'research/freqtrade/import_snapshot.py', 'line': 89,
             'observation': 'Importer requires unique ordered hourly funding and original mark joins; observed schedule completeness is independently checked by this audit.'},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report:
        require(not args.report.resolve().is_relative_to(args.snapshot.resolve()), 'Report must be outside immutable snapshot')
    result = audit_snapshot(args.snapshot)
    body = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(body, encoding='utf-8')
    print(body, end='')


if __name__ == '__main__':
    main()
