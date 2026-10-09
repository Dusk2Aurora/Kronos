"""One-shot, explicitly authorized official holdout 5m acquisition.

No public I/O entry point can operate without the separate Phase B claim. A
failed request aborts: partial artifacts are retained and never resumed here.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import requests
import pandas as pd
from research.frozen.experiment_03 import guard, inputs

START_ISO = '2026-04-01T00:00:00Z'
END_ISO = '2026-10-01T00:00:00Z'
START = int(pd.Timestamp(START_ISO).value // 1000000)
END = int(pd.Timestamp(END_ISO).value // 1000000)
STEP = 300000
EXPECTED_ROWS = (END - START) // STEP
ENDPOINT = 'https://www.okx.com/api/v5/market/history-candles'
FIELDS = ['bar_open_at', 'bar_close_at', 'open', 'high', 'low', 'close',
          'raw_contract_volume', 'volume', 'amount', 'confirm',
          'raw_page', 'raw_row', 'raw_sha256']


def _iso(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace('+00:00', 'Z')


def _cfg():
    cfg = guard.config()
    if (cfg['roles']['holdout'] != [START_ISO, END_ISO]
            or cfg['sources']['five_minute_endpoint'] != ENDPOINT
            or cfg['sources']['instrument'] != 'BTC-USDT-SWAP'
            or cfg['sources']['page_limit'] != 300):
        raise ValueError('Fixed formal 5m contract mismatch')
    return cfg


def _request(cursor):
    params = {'instId': 'BTC-USDT-SWAP', 'bar': '5m', 'limit': '300',
              'after': str(cursor), 'before': str(START - 1)}
    return params, requests.Request('GET', ENDPOINT, params=params).prepare().url


def validate_page(value, cursor):
    """Pure page check: validate ALL timestamps before interpreting prices."""
    if not START < cursor <= END:
        raise ValueError('Cursor outside formal range')
    if not isinstance(value, dict) or value.get('code') != '0':
        raise ValueError('Official API failure code')
    page = value.get('data')
    if not isinstance(page, list) or not 1 <= len(page) <= 300:
        raise ValueError('Empty or oversized official page')
    if any(not isinstance(row, list) or len(row) != 9 for row in page):
        raise ValueError('Invalid official field count')
    timestamps = []
    for row in page:
        if not isinstance(row[0], str) or not row[0].isdigit():
            raise ValueError('Noninteger official timestamp')
        timestamps.append(int(row[0]))
    if any(not START <= t < cursor or t % STEP for t in timestamps):
        raise ValueError('Official timestamp outside request bounds/grid')
    if timestamps != sorted(timestamps, reverse=True):
        raise ValueError('Official page must be descending in UTC')
    for row in page:
        if row[-1] != '1' or any(not isinstance(v, str) for v in row):
            raise ValueError('Unconfirmed or noncanonical official row')
        nums = [float(v) for v in row[1:8]]
        if not all(math.isfinite(v) and v >= 0 for v in nums):
            raise ValueError('Invalid nonnegative official OHLCVA')
        o, h, l, c = nums[:4]
        if min(o, h, l, c) <= 0 or l > min(o, c) or h < max(o, c) or l > h:
            raise ValueError('Invalid official OHLC bounds')
    return page


def _canonical(t, row, source, index):
    return [_iso(t), _iso(t + STEP), *row[1:8], row[8], source['path'],
            str(index), source['sha256']]


def _failure(directory, page_no, cursor, url, response, error):
    # Metadata only: a rejected response may contain out-of-scope market bytes.
    record = {'page': page_no, 'cursor': cursor, 'expected_url': url,
              'at_utc': guard.now(), 'error_type': type(error).__name__,
              'response_url': getattr(response, 'url', None),
              'http_status': getattr(response, 'status_code', None),
              'protocol_sha256': guard.CONFIG_SHA, 'retry_permitted': False}
    if response is not None:
        body = response.content
        record.update(response_size_bytes=len(body),
                      response_sha256=hashlib.sha256(body).hexdigest())
    guard.new_json(directory / 'failure_ledger.json', [record])


def collect_holdout():
    guard.require_scope(END_ISO, phase='B')
    cfg = _cfg()
    directory = guard.RUN / 'data_5m'
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Formal collection already attempted; no overwrite/resume/retry')
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / 'raw'
    raw.mkdir()
    records, sources, cursor, page_no, duplicates = {}, [], END, 0, 0
    with requests.Session() as session:
        while cursor > START:
            params, url = _request(cursor)
            response = None
            try:
                response = session.get(ENDPOINT, params=params, timeout=(10, 30), allow_redirects=False)
                response.raise_for_status()
                body = response.content
                if response.status_code != 200 or len(body) > 4 * 1024 * 1024 or response.url != url:
                    raise ValueError('Unexpected official response status/size/URL')
                page = validate_page(json.loads(body), cursor)
            except Exception as error:
                _failure(directory, page_no, cursor, url, response, error)
                raise
            relative = f'raw/{page_no:05d}.json'
            source = {'url': url, 'collected_at': guard.now(), 'path': relative,
                      'sha256': hashlib.sha256(body).hexdigest(),
                      'protocol_sha256': guard.CONFIG_SHA, 'rows': len(page),
                      'after': cursor, 'before': START - 1,
                      'range': [_iso(START), _iso(END)],
                      'payload_range': [_iso(min(int(r[0]) for r in page)),
                                        _iso(max(int(r[0]) for r in page) + STEP)]}
            with (directory / relative).open('xb') as f:
                f.write(body)
            guard.new_json(raw / f'{page_no:05d}.source.json', source)
            sources.append(source)
            for index, row in enumerate(page):
                t = int(row[0])
                if t in records:
                    duplicates += 1
                    if records[t][0] != row:
                        error = ValueError('Conflicting duplicate official candle')
                        _failure(directory, page_no, cursor, url, response, error)
                        raise error
                else:
                    records[t] = (row, source, index)
            cursor = min(int(row[0]) for row in page)
            page_no += 1
    missing = [_iso(t) for t in range(START, END, STEP) if t not in records]
    audit = {'status': 'PASS' if not missing and len(records) == EXPECTED_ROWS else 'FAIL',
             'range': [_iso(START), _iso(END)], 'rows': len(records),
             'expected': EXPECTED_ROWS, 'missing': missing, 'exact_duplicates': duplicates,
             'conflicting_duplicates': 0, 'confirm1': True, 'UTC_5min_grid': True,
             'valid_OHLC_nonnegative_volumes': True, 'protocol_sha256': guard.CONFIG_SHA,
             'evidence_limit': 'Published official coverage/consistency; not historical first-publication timing or exchange internal ledger'}
    guard.new_json(directory / 'audit.json', audit)
    if audit['status'] != 'PASS':
        raise ValueError('Incomplete formal official grid; no interpolation/exclusions/1h fallback')
    csv_path = directory / 'candles.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for t, (row, source, index) in sorted(records.items()):
            writer.writerow(_canonical(t, row, source, index))
    manifest = {'protocol_sha256': guard.CONFIG_SHA, 'instrument': 'BTC-USDT-SWAP', 'bar': '5m',
                'range': [_iso(START), _iso(END)], 'rows': len(records),
                'units': {'volume': 'BTC=volCcy', 'amount': 'USDT=volCcyQuote', 'raw_contract_volume': 'contracts=vol'},
                'price_contract': cfg['timing']['price_contract'], 'sources': sources,
                'candles_sha256': guard.sha(csv_path), 'no_resume_or_request_retry': True}
    guard.new_json(directory / 'manifest.json', manifest)
    independent = audit_snapshot()
    guard.new_json(directory / 'raw_csv_audit.json', independent)
    return manifest


def audit_snapshot():
    guard.require_scope(END_ISO, phase='B')
    cfg = _cfg()
    directory = guard.RUN / 'data_5m'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['protocol_sha256'] != guard.CONFIG_SHA or manifest['range'] != [_iso(START), _iso(END)]
            or manifest['instrument'] != 'BTC-USDT-SWAP' or manifest['bar'] != '5m'
            or manifest['rows'] != EXPECTED_ROWS
            or manifest['price_contract'] != cfg['timing']['price_contract']
            or manifest['candles_sha256'] != guard.sha(directory / 'candles.csv')):
        raise ValueError('Formal source manifest mismatch')
    metadata = sorted((directory / 'raw').glob('*.source.json'))
    expected_names = [f'{i:05d}.source.json' for i in range(len(metadata))]
    if [p.name for p in metadata] != expected_names or len(metadata) != len(manifest['sources']):
        raise ValueError('Formal source page inventory mismatch')
    raw_names = sorted(p.name for p in (directory / 'raw').glob('*.json') if not p.name.endswith('.source.json'))
    if raw_names != [f'{i:05d}.json' for i in range(len(metadata))] or (directory / 'failure_ledger.json').exists():
        raise ValueError('Unexpected raw pages or failed formal acquisition')
    rows, cursor, duplicates = {}, END, 0
    for page_no, metadata_path in enumerate(metadata):
        source = json.loads(metadata_path.read_text(encoding='utf-8'))
        _, expected_url = _request(cursor)
        expected_relative = f'raw/{page_no:05d}.json'
        if (source != manifest['sources'][page_no] or source['path'] != expected_relative
                or source['url'] != expected_url or source['protocol_sha256'] != guard.CONFIG_SHA
                or source['after'] != cursor or source['before'] != START - 1
                or source['range'] != [_iso(START), _iso(END)]):
            raise ValueError('Formal raw request provenance mismatch')
        collected = pd.Timestamp(source['collected_at'])
        if collected.tzinfo is None or collected.utcoffset().total_seconds() != 0:
            raise ValueError('Source collection UTC required')
        path = directory / expected_relative
        if guard.sha(path) != source['sha256'] or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Changed official raw bytes')
        page = validate_page(json.loads(path.read_bytes()), cursor)
        payload_range = [_iso(min(int(r[0]) for r in page)), _iso(max(int(r[0]) for r in page) + STEP)]
        if source['rows'] != len(page) or source['payload_range'] != payload_range:
            raise ValueError('Raw source page metadata mismatch')
        for index, row in enumerate(page):
            t = int(row[0])
            canonical = _canonical(t, row, source, index)
            if t in rows:
                duplicates += 1
                if rows[t][:10] != canonical[:10]:
                    raise ValueError('Conflicting duplicate raw candle')
            else:
                rows[t] = canonical
        cursor = min(int(row[0]) for row in page)
    with (directory / 'candles.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        if next(reader) != FIELDS:
            raise ValueError('Formal canonical CSV schema mismatch')
        count = 0
        for t, row in zip(range(START, END, STEP), reader):
            if row != rows.get(t):
                raise ValueError('Canonical CSV differs from raw official strings')
            count += 1
        if next(reader, None) is not None or count != EXPECTED_ROWS or len(rows) != count or cursor != START:
            raise ValueError('Incomplete formal grid audit')
    audit = json.loads((directory / 'audit.json').read_text(encoding='utf-8'))
    if (audit['status'] != 'PASS' or audit['rows'] != count or audit['expected'] != EXPECTED_ROWS
            or audit['missing'] or audit['exact_duplicates'] != duplicates
            or audit['protocol_sha256'] != guard.CONFIG_SHA):
        raise ValueError('Collection audit mismatch')
    return {'status': 'PASS', 'rows': count, 'raw_pages': len(metadata),
            'range': [_iso(START), _iso(END)], 'protocol_sha256': guard.CONFIG_SHA,
            'exact_raw_csv_match': True, 'exact_duplicates': duplicates}


def load_holdout_5m():
    guard.require_scope(END_ISO, phase='B')
    audit = audit_snapshot()
    directory = guard.RUN / 'data_5m'
    preserved = json.loads((directory / 'raw_csv_audit.json').read_text(encoding='utf-8'))
    if preserved != audit:
        raise ValueError('Formal independent audit changed')
    return inputs._read_bounded(directory / 'candles.csv', pd.Timestamp(START_ISO),
                                pd.Timestamp(END_ISO), 'B', pd.Timedelta(minutes=5))
