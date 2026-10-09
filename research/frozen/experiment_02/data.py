"""Locked DEV-only official 5m snapshot and traded-price RV label construction."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'research/configs/frozen_risk_02_v1.yaml'
CONFIG_SHA = 'bc488c5bb9a3296340595e367c1c474d64f96c45e49c5249efc9f12a52d06425'
START = 1704067200000
END = 1775001600000
STEP = 300000
ENDPOINT = 'https://www.okx.com/api/v5/market/history-candles'
OUTPUT = ROOT / 'research/runs/FROZEN_RISK_02_v1/data_5m'
FIELDS = ['bar_open_at', 'bar_close_at', 'open', 'high', 'low', 'close', 'raw_contract_volume', 'volume', 'amount', 'confirm', 'raw_page', 'raw_row', 'raw_sha256']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def iso(ts):
    return datetime.fromtimestamp(ts/1000, timezone.utc).isoformat().replace('+00:00', 'Z')


def milliseconds(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None or t.utcoffset().total_seconds() != 0:
        raise ValueError('Explicit UTC timestamp required')
    return int(t.value // 1000000)


def locked_config():
    if sha(CONFIG) != CONFIG_SHA:
        raise ValueError('Locked protocol SHA256 mismatch')
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if (milliseconds(cfg['development']['start']), milliseconds(cfg['development']['end_exclusive'])) != (START, END):
        raise ValueError('Development range mismatch')
    return cfg


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def validate_page(value, cursor):
    if value.get('code') != '0' or not value.get('data'):
        raise ValueError('Official API returned error or empty page before START')
    page = value['data']
    # Examine only timestamps before interpreting any price fields.
    timestamps = [int(r[0]) for r in page]
    if any(not START <= t < cursor <= END for t in timestamps):
        raise ValueError('Response timestamp outside locked DEV request bounds')
    if min(timestamps) >= cursor:
        raise ValueError('Pagination did not advance')
    for r,t in zip(page, timestamps):
        if len(r) != 9 or r[-1] != '1' or t % STEP:
            raise ValueError('Invalid field count/confirmation/UTC grid')
        nums = [float(v) for v in r[1:8]]
        if not all(math.isfinite(v) and v >= 0 for v in nums):
            raise ValueError('Invalid nonnegative numeric value')
        o,h,l,c = nums[:4]
        if min(o,h,l,c) <= 0 or l > min(o,c) or h < max(o,c) or l > h:
            raise ValueError('Invalid official OHLC')
    return page


def collect_5m():
    """Resume verified immutable pages, then audit exact full DEV grid; no overrides."""
    cfg = locked_config()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = OUTPUT / 'raw'
    raw.mkdir(exist_ok=True)
    session = requests.Session()
    records, sources, cursor, page_no, duplicates = {}, [], END, 0, 0
    while cursor > START:
        body_path = raw / f'{page_no:05d}.json'
        meta_path = raw / f'{page_no:05d}.source.json'
        params = {'instId':'BTC-USDT-SWAP','bar':'5m','limit':'300','after':str(cursor),'before':str(START-1)}
        expected_url = requests.Request('GET', ENDPOINT, params=params).prepare().url
        if body_path.exists() or meta_path.exists():
            if not (body_path.exists() and meta_path.exists()):
                raise ValueError('Partial immutable page: preserve and investigate, never overwrite')
            source = json.loads(meta_path.read_text(encoding='utf-8'))
            if source['url'] != expected_url or source['sha256'] != sha(body_path) or source['protocol_sha256'] != CONFIG_SHA:
                raise ValueError('Immutable page provenance mismatch')
            value = json.loads(body_path.read_bytes())
            page = validate_page(value, cursor)
        else:
            for attempt in range(4):
                try:
                    response = session.get(ENDPOINT, params=params, timeout=(10,30), allow_redirects=False)
                    response.raise_for_status()
                    if response.status_code != 200 or len(response.content) > 4*1024*1024 or response.url != expected_url:
                        raise ValueError('Unexpected official response status/size/URL')
                    value = response.json()
                    page = validate_page(value, cursor)
                    # Preserve raw official bytes, including only bounded DEV rows.
                    with body_path.open('xb') as f:
                        f.write(response.content)
                    source = {'url':response.url,'collected_at':datetime.now(timezone.utc).isoformat(),
                              'path':str(body_path.relative_to(OUTPUT)), 'sha256':sha(body_path),
                              'protocol_sha256':CONFIG_SHA, 'rows':len(page),'after':cursor,'before':START-1}
                    write_json_new(meta_path, source)
                    break
                except (requests.RequestException, json.JSONDecodeError):
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
        sources.append(source)
        for idx,r in enumerate(page):
            t = int(r[0])
            if t in records:
                duplicates += 1
                if records[t][0] != r:
                    raise ValueError('Conflicting duplicate official candle')
            else:
                records[t] = (r, source['path'], idx, source['sha256'])
        cursor = min(int(r[0]) for r in page)
        page_no += 1
        if page_no % 50 == 0:
            print(json.dumps({'pages':page_no,'rows':len(records),'reached':iso(cursor)}), flush=True)
        time.sleep(.15)
    expected = (END-START)//STEP
    missing = [iso(t) for t in range(START,END,STEP) if t not in records]
    audit = {'status':'PASS' if not missing and len(records)==expected else 'FAIL',
             'range':[iso(START),iso(END)],'rows':len(records),'expected':expected,
             'missing':missing,'exact_duplicates':duplicates,'conflicting_duplicates':0,
             'confirm1':True,'UTC_5min_grid':True,'valid_OHLC_nonnegative_volumes':True,
             'holdout_rows':0,'protocol_sha256':CONFIG_SHA,
             'evidence_limit':'Published official OHLCV coverage/consistency, not historical first-publication timing or internal exchange ledger'}
    if missing or len(records) != expected:
        if not (OUTPUT/'failed_audit.json').exists():
            write_json_new(OUTPUT/'failed_audit.json',audit)
        raise ValueError('Incomplete trustworthy 5m coverage: stop formal experiment, no 1h fallback')
    csv_path = OUTPUT/'candles.csv'
    if not csv_path.exists():
        with csv_path.open('x',encoding='utf-8',newline='') as f:
            w = csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
            for t,(r,path,idx,digest) in sorted(records.items()):
                w.writerow(dict(zip(FIELDS,[iso(t),iso(t+STEP),*r[1:8],r[8],path,idx,digest])))
    else:
        audit_snapshot(OUTPUT)
    manifest = {'protocol_sha256':CONFIG_SHA,'instrument':'BTC-USDT-SWAP','bar':'5m',
                'range':[iso(START),iso(END)],'units':{'volume':'BTC=volCcy','amount':'USDT=volCcyQuote','raw_contract_volume':'contracts=vol'},
                'price_contract':cfg['timing']['price_contract'],'sources':sources,
                'candles_sha256':sha(csv_path),'rows':len(records)}
    for name,value in [('audit.json',audit),('manifest.json',manifest)]:
        if not (OUTPUT/name).exists():
            write_json_new(OUTPUT/name,value)
    independent = audit_snapshot(OUTPUT)
    if not (OUTPUT/'raw_csv_audit.json').exists():
        write_json_new(OUTPUT/'raw_csv_audit.json',independent)
    print(json.dumps({'status':'PASS','rows':len(records),'pages':page_no,'output':str(OUTPUT)}),flush=True)
    return manifest


def audit_snapshot(directory=OUTPUT):
    """Reconstruct canonical CSV from every official source and verify provenance."""
    locked_config()
    directory = Path(directory)
    sources = sorted((directory/'raw').glob('*.source.json'))
    raw_rows = {}
    cursor = END
    for m in sources:
        source = json.loads(m.read_text(encoding='utf-8'))
        path = directory/source['path']
        params = {'instId':'BTC-USDT-SWAP','bar':'5m','limit':'300','after':str(cursor),'before':str(START-1)}
        if source['url'] != requests.Request('GET', ENDPOINT,params=params).prepare().url or source['sha256'] != sha(path) or source['protocol_sha256'] != CONFIG_SHA:
            raise ValueError('Raw provenance failed independent audit')
        page = validate_page(json.loads(path.read_bytes()),cursor)
        for i,r in enumerate(page):
            t=int(r[0]); expected=[iso(t),iso(t+STEP),*r[1:8],r[8],source['path'],str(i),source['sha256']]
            if t in raw_rows and raw_rows[t][:10] != expected[:10]:
                raise ValueError('Conflicting raw duplicate')
            raw_rows.setdefault(t,expected)
        cursor=min(int(r[0]) for r in page)
    with (directory/'candles.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.reader(f)
        if next(reader) != FIELDS:
            raise ValueError('CSV schema mismatch')
        count=0
        for t,row in zip(range(START,END,STEP),reader):
            if row != raw_rows.get(t):
                raise ValueError('Canonical CSV differs from raw official data')
            count+=1
        if next(reader,None) is not None or count != (END-START)//STEP or len(raw_rows) != count or cursor != START:
            raise ValueError('Coverage audit failed')
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['candles_sha256'] != sha(directory/'candles.csv') or manifest['protocol_sha256'] != CONFIG_SHA:
            raise ValueError('Final manifest mismatch')
    return {'status':'PASS','rows':count,'raw_pages':len(sources),'exact_raw_csv_match':True,'holdout_rows':0}


def read_dev_1h(path, start=None):
    """Timestamp-first streaming barrier; never convert holdout price/label fields."""
    rows=[]
    lower=milliseconds(start) if start is not None else START-256*3600000
    with Path(path).open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            ts=milliseconds(row['bar_open_at'])
            if ts >= END:
                break
            if ts < lower:
                continue
            rows.append(row)
    result=pd.DataFrame(rows)
    for field in ['open','high','low','close','volume','amount']:
        if field in result:
            result[field]=pd.to_numeric(result[field],errors='raise')
    return result


def build_labels(candles, opportunities, delay=60):
    """No remote IO. P0 entry open + 48 completed closes; reject missing DEV bars."""
    cfg=locked_config()
    if delay != 60:
        raise ValueError('Locked label availability delay is 60s')
    times=np.array([milliseconds(t) for t in candles['bar_open_at']],dtype=np.int64)
    if np.any(times < START) or np.any(times >= END) or len(np.unique(times)) != len(times):
        raise ValueError('5m label source must be unique and strictly DEV-only')
    lookup={t:i for i,t in enumerate(times)}
    out=[]
    for _,opp in opportunities.iterrows():
        entry=milliseconds(opp['entry_at']); end=entry+4*3600000
        if not START <= entry < END:
            raise ValueError('Opportunity outside DEV')
        # Sealed range and strict availability purge, before reading future prices.
        if end+delay*1000 >= END:
            continue
        indices=[lookup.get(entry+j*STEP) for j in range(48)]
        if any(i is None for i in indices):
            raise ValueError('Incomplete 48-return label window; no interpolation')
        window=candles.iloc[indices]
        if 'confirm' in window and not all(str(v)=='1' for v in window['confirm']):
            raise ValueError('Unconfirmed label bar')
        prices=np.r_[float(window.iloc[0]['open']),window['close'].to_numpy(dtype=np.float64)]
        if not np.all(np.isfinite(prices)&(prices>0)):
            raise ValueError('Invalid label price')
        rv=float(np.sum(np.diff(np.log(prices))**2))
        row=opp.to_dict()
        row.update({'RV_raw':rv,'RV_effective':max(rv,cfg['label']['epsilon']),
                    'label_start':iso(entry),'label_end':iso(end),'labelable_at':iso(end+delay*1000),
                    'label_price_count':49,'label_return_count':48,
                    'label_prices_json':json.dumps(prices.tolist(),separators=(',',':')),
                    'label_bar_open_at_json':json.dumps([iso(milliseconds(t)) for t in window['bar_open_at']],separators=(',',':'))})
        for field in ['raw_page','raw_row','raw_sha256']:
            if field in window:
                row['label_'+field+'_json']=json.dumps(window[field].tolist(),separators=(',',':'))
        out.append(row)
    return pd.DataFrame(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['collect','audit'])
    args=parser.parse_args()
    if args.action=='collect':
        collect_5m()
    else:
        print(json.dumps(audit_snapshot()))

if __name__=='__main__':
    main()
