"""Synthetic-only official collection checks; never request market data."""
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import requests
from research.frozen.experiment_03 import data

CFG = {'roles': {'holdout': [data.START_ISO, data.END_ISO]},
       'scope_guards': {'phase_b_max_time_exclusive': data.END_ISO},
       'sources': {'five_minute_endpoint': data.ENDPOINT, 'instrument': 'BTC-USDT-SWAP', 'page_limit': 300},
       'timing': {'price_contract': 'synthetic_contract'}}


def row(t):
    return [str(t), '100', '101', '99', '100.5', '4', '0.02', '2', '1']


@pytest.mark.parametrize('name', ['collect_holdout', 'audit_snapshot', 'load_holdout_5m'])
def test_authorization_precedes_any_request_or_file_read(name):
    with patch.object(data.guard, 'require_scope', side_effect=PermissionError('SEALED')) as check, \
         patch.object(data.guard, 'config') as cfg, \
         patch.object(Path, 'read_text') as read, patch.object(Path, 'exists') as exists, \
         patch.object(requests, 'Session') as session:
        with pytest.raises(PermissionError):
            getattr(data, name)()
        check.assert_called_once_with(data.END_ISO, phase='B')
        cfg.assert_not_called()
        read.assert_not_called()
        exists.assert_not_called()
        session.assert_not_called()


def test_expected_full_range_and_real_volume_contract():
    assert data.EXPECTED_ROWS == 183*288 == 52704
    assert data.FIELDS[6:9] == ['raw_contract_volume', 'volume', 'amount']
    params, url = data._request(data.END)
    assert params['before'] == str(data.START-1)
    assert params['after'] == str(data.END)
    assert url == requests.Request('GET',data.ENDPOINT,params=params).prepare().url


def test_page_timestamp_barrier_before_any_numeric_conversion():
    bad = row(data.END)
    bad[1] = 'MUST_NOT_PARSE'
    with pytest.raises(ValueError, match='timestamp outside'):
        data.validate_page({'code':'0','data':[bad]},data.END)
    bad = row(data.START-1)
    bad[1] = 'MUST_NOT_PARSE'
    first = row(data.START)
    first[1] = 'MUST_NOT_PARSE_BEFORE_ALL_TIMESTAMPS'
    with pytest.raises(ValueError, match='timestamp outside'):
        data.validate_page({'code':'0','data':[first,bad]},data.END)


@pytest.mark.parametrize('defect', ['confirmation','bounds','nan','grid','order','api','empty','columns'])
def test_reject_bad_official_page(defect):
    page = [row(data.START+data.STEP),row(data.START)]
    value={'code':'0','data':page}
    if defect=='confirmation': page[0][-1]='0'
    if defect=='bounds': page[0][2]='98'
    if defect=='nan': page[0][7]='NaN'
    if defect=='grid': page[0][0]=str(data.START+1)
    if defect=='order': page.reverse()
    if defect=='api': value['code']='500'
    if defect=='empty': value['data']=[]
    if defect=='columns': page[0].pop()
    with pytest.raises(ValueError): data.validate_page(value,data.END)


class SyntheticSession:
    def __init__(self, omit=None, failure=False, outside=False):
        self.omit, self.failure, self.outside = omit, failure, outside
        self.calls = 0
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def get(self,url,params,**kwargs):
        self.calls += 1
        assert kwargs['allow_redirects'] is False
        if self.failure: raise requests.ConnectionError('synthetic failure')
        cursor=int(params['after'])
        times=list(range(cursor-data.STEP,max(data.START-data.STEP,cursor-301*data.STEP),-data.STEP))
        page=[row(t) for t in times if t!=self.omit]
        if self.outside: page=[row(data.END)]
        response=Mock()
        response.url=requests.Request('GET',url,params=params).prepare().url
        response.status_code=200
        response.content=json.dumps({'code':'0','data':page}).encode()
        return response


@pytest.fixture
def synthetic_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(data.guard,'RUN',tmp_path)
    monkeypatch.setattr(data.guard,'require_scope',lambda end,phase='A': None)
    monkeypatch.setattr(data.guard,'config',lambda: CFG)
    return tmp_path


def test_complete_synthetic_raw_canonical_audit_and_no_resume(synthetic_scope,monkeypatch):
    session=SyntheticSession()
    monkeypatch.setattr(requests,'Session',lambda:session)
    result=data.collect_holdout()
    assert result['rows']==52704
    assert result['units']['amount']=='USDT=volCcyQuote'
    assert session.calls==176
    audit=data.audit_snapshot()
    assert audit['exact_raw_csv_match'] and audit['rows']==52704
    loaded=data.load_holdout_5m()
    assert len(loaded)==52704 and str(loaded.amount.dtype)=='float64'
    assert loaded.amount.iloc[0]==2 and loaded.volume.iloc[0]==.02
    with pytest.raises(FileExistsError): data.collect_holdout()
    assert session.calls==176
    # Detect canonical tampering without interpreting any market fields anew.
    canonical=synthetic_scope/'data_5m/candles.csv'
    canonical.write_text(canonical.read_text().replace('100.5','100.6',1))
    with pytest.raises(ValueError,match='manifest mismatch'): data.audit_snapshot()


@pytest.mark.parametrize('outside',[False,True])
def test_failed_request_no_retry_metadata_only(synthetic_scope,monkeypatch,outside):
    session=SyntheticSession(failure=not outside,outside=outside)
    monkeypatch.setattr(requests,'Session',lambda:session)
    with pytest.raises((requests.ConnectionError,ValueError)): data.collect_holdout()
    assert session.calls==1
    directory=synthetic_scope/'data_5m'
    ledger=json.loads((directory/'failure_ledger.json').read_text())
    assert ledger[0]['retry_permitted'] is False
    assert not list((directory/'raw').iterdir())
    if outside: assert 'response_sha256' in ledger[0]
    with pytest.raises(FileExistsError): data.collect_holdout()
    assert session.calls==1


def test_missing_candle_fails_uniform_coverage(synthetic_scope,monkeypatch):
    session=SyntheticSession(omit=data.START+data.STEP)
    monkeypatch.setattr(requests,'Session',lambda:session)
    with pytest.raises(ValueError,match='Incomplete formal official grid'): data.collect_holdout()
    directory=synthetic_scope/'data_5m'
    audit=json.loads((directory/'audit.json').read_text())
    assert audit['status']=='FAIL' and audit['rows']==52703
    assert audit['missing']==[data._iso(data.START+data.STEP)]
    assert not (directory/'candles.csv').exists()


def test_conflicting_duplicate_aborts_and_preserves_failure(synthetic_scope,monkeypatch):
    class ConflictSession(SyntheticSession):
        def get(self,*args,**kwargs):
            response=super().get(*args,**kwargs)
            value=json.loads(response.content)
            value['data'][1]=value['data'][0].copy()
            value['data'][1][4]='100.6'
            response.content=json.dumps(value).encode()
            return response
    session=ConflictSession()
    monkeypatch.setattr(requests,'Session',lambda:session)
    with pytest.raises(ValueError,match='Conflicting duplicate'):
        data.collect_holdout()
    assert session.calls==1
    directory=synthetic_scope/'data_5m'
    assert (directory/'failure_ledger.json').exists()
    assert (directory/'raw/00000.json').exists()
    assert not (directory/'manifest.json').exists()
