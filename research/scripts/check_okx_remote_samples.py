"""Re-fetch deterministic public samples without modifying the audited snapshot."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import zipfile

import requests

UTC = timezone.utc
HOUR = 3_600_000


def ms(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def archive_rows(body):
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        return sorted((row['instrument_name'], int(row['funding_time']), float(row['funding_rate']))
                      for name in archive.namelist() if not name.endswith('/')
                      for row in csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--native-datadir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    output = args.output_dir.resolve()
    if output == snapshot or output.is_relative_to(snapshot):
        raise ValueError('Remote checks must not modify the original snapshot')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8'))
    quality = json.loads((snapshot / 'quality_report.json').read_text(encoding='utf-8'))
    trades = {ms(row['bar_open_at']): row for row in read_rows(snapshot / 'candles.csv')}
    marks = {ms(row['bar_open_at']): row for row in read_rows(snapshot / 'mark_prices.csv')}
    funding = {ms(row['funding_at']): float(row['realized_rate'])
               for row in read_rows(snapshot / 'funding.csv')}
    sources, checks = [], []

    def get(url, params=None, binary=False):
        host = urlparse(url).hostname
        if host not in {'www.okx.com', 'static.okx.com'} or urlparse(url).scheme != 'https':
            raise ValueError('Only recorded official public OKX endpoints are allowed')
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=(10, 20), allow_redirects=False)
                response.raise_for_status()
                if response.status_code != 200 or len(response.content) > 4 * 1024 * 1024:
                    raise ValueError('Unexpected response status or size')
                path = output / f'{len(sources):03d}.{"zip" if binary else "json"}'
                path.write_bytes(response.content)
                sources.append({'url': response.url, 'collected_at': datetime.now(UTC).isoformat(),
                                'path': str(path), 'sha256': digest(path)})
                if binary:
                    return response.content
                value = response.json()
                if value.get('code') != '0':
                    raise ValueError(f'OKX API code {value.get("code")}')
                return value['data']
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError('Unreachable')

    start, end = ms(quality['evaluation_start']), ms(quality['evaluation_end_exclusive'])
    month = datetime.fromtimestamp(start / 1000, UTC).replace(day=15, hour=12)
    samples = {start, end - HOUR, ms('2024-02-29T12:00:00Z')}
    while int(month.timestamp() * 1000) < end:
        samples.add(int(month.timestamp() * 1000))
        month = month.replace(year=month.year + (month.month == 12), month=month.month % 12 + 1)
    for ts in sorted(samples):
        for mark, local in ((False, trades), (True, marks)):
            endpoint = 'history-mark-price-candles' if mark else 'history-candles'
            page = get('https://www.okx.com/api/v5/market/' + endpoint,
                       {'instId': 'BTC-USDT-SWAP', 'bar': '1H', 'limit': 1, 'after': ts + HOUR})
            if len(page) != 1 or int(page[0][0]) != ts or page[0][-1] != '1':
                raise ValueError(f'Remote sample timestamp/confirmation mismatch at {ts}')
            keys = ['open', 'high', 'low', 'close']
            if not mark:
                keys += ['raw_contract_volume', 'volume', 'amount']
            if [float(value) for value in page[0][1:-1]] != [float(local[ts][key]) for key in keys]:
                raise ValueError(f'Remote sample numeric mismatch at {ts}, mark={mark}')
            checks.append({'kind': 'mark' if mark else 'trade', 'timestamp_ms': ts,
                           'all_numeric_fields_exact': True})
            time.sleep(0.12)
    # Warmup is outside the model evaluation interval and has its own boundary check.
    ts = ms(quality['data_start'])
    page = get('https://www.okx.com/api/v5/market/history-candles',
               {'instId': 'BTC-USDT-SWAP', 'bar': '1H', 'limit': 1, 'after': ts + HOUR})
    keys = ['open', 'high', 'low', 'close', 'raw_contract_volume', 'volume', 'amount']
    if len(page) != 1 or int(page[0][0]) != ts or page[0][-1] != '1' or (
            [float(v) for v in page[0][1:-1]] != [float(trades[ts][key]) for key in keys]):
        raise ValueError('Warmup boundary sample mismatch')
    archives = [source for source in manifest['sources'] if source['path'].endswith('.zip')]
    archive_checks = []
    for index in sorted({0, len(archives) // 2, len(archives) - 1}):
        source = archives[index]
        original = (snapshot / source['path']).read_bytes()
        refreshed = get(source['url'], binary=True)
        if archive_rows(original) != archive_rows(refreshed):
            raise ValueError('Refreshed funding archive semantic mismatch')
        archive_checks.append({'url': source['url'], 'events': len(archive_rows(original)),
                               'semantic_rows_exact': True, 'bytes_identical': original == refreshed})
    recent = get('https://www.okx.com/api/v5/public/funding-rate-history',
                 {'instId': 'BTC-USDT-SWAP', 'limit': 400})
    overlap = 0
    for row in recent:
        ts = int(row['fundingTime'])
        if ts in funding:
            if row['instId'] != 'BTC-USDT-SWAP' or float(row['realizedRate']) != funding[ts]:
                raise ValueError('Refreshed REST realized funding mismatch')
            overlap += 1
    if not overlap:
        raise ValueError('No fresh realized funding overlap to audit')
    native_manifest = json.loads((args.native_datadir / 'import_manifest.json').read_text(encoding='utf-8'))
    if native_manifest['snapshot_manifest_sha256'] != digest(snapshot / 'manifest.json'):
        raise ValueError('Native import snapshot identity mismatch')
    for name, expected in native_manifest['files'].items():
        path = (args.native_datadir / name).resolve()
        if not path.is_relative_to(args.native_datadir.resolve()) or digest(path) != expected:
            raise ValueError('Native imported file changed after roundtrip verification')
    result = {'status': 'passed', 'snapshot_id': manifest['snapshot_id'],
              'checked_at': datetime.now(UTC).isoformat(),
              'script_sha256': digest(Path(__file__)), 'snapshot_manifest_sha256': digest(snapshot / 'manifest.json'),
              'candle_samples': checks, 'warmup_boundary_refetch_exact': True,
              'funding_archive_refetch': archive_checks, 'fresh_REST_matched_events': overlap,
              'native_imported_files_hash_matched': len(native_manifest['files']), 'sources': sources,
              'scope': 'independent_refetch_samples_same_official_publisher_not_second_exchange_source'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(checks)} candle samples, warmup boundary, {len(archive_checks)} archives, '
          f'{overlap} fresh funding events, {len(native_manifest["files"])} native file hashes')


if __name__ == '__main__':
    main()
