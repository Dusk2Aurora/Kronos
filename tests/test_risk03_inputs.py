import copy
import csv
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from research.frozen import encoder
from research.frozen.experiment_03 import inputs

CFG = {
    'scope_guards': {'phase_a_max_time_exclusive': '2026-04-01T00:00:00Z', 'phase_b_max_time_exclusive': '2026-10-01T00:00:00Z'},
    'roles': {'train': ['2024-01-01T00:00:00Z', '2026-03-01T00:00:00Z'], 'validation': ['2026-03-01T00:00:00Z', '2026-04-01T00:00:00Z'],
              'test_quarters': [{'id':'TEST_Q2','start':'2026-04-01T00:00:00Z','end_exclusive':'2026-07-01T00:00:00Z'}, {'id':'TEST_Q3','start':'2026-07-01T00:00:00Z','end_exclusive':'2026-10-01T00:00:00Z'}]},
    'timing': {'availability_delay_seconds':60,'decision_hours_utc':[4,12,20], 'entry_after_boundary_hours':1, 'horizon_hours':4, 'lookback_1h_bars':256},
    'label': {'epsilon':1e-12},
}

@pytest.fixture(autouse=True)
def guarded_synthetic():
    def check(end, phase='A'):
        if phase != 'A':
            raise PermissionError('Synthetic Phase B authorization absent')
        if pd.Timestamp(end) > pd.Timestamp(CFG['scope_guards']['phase_a_max_time_exclusive']):
            raise PermissionError('Sealed')
    with patch.object(inputs.guard, 'config', return_value=copy.deepcopy(CFG)), patch.object(inputs.guard, 'require_scope', side_effect=check):
        yield

def bars(start, periods, freq='h'):
    times = pd.date_range(start, periods=periods, freq=freq)
    step = pd.Timedelta(hours=1) if freq == 'h' else pd.Timedelta(minutes=5)
    close = 100*np.exp(np.arange(periods)*.001)
    return pd.DataFrame(dict(bar_open_at=times, bar_close_at=times+step, available_at=times+step+pd.Timedelta(seconds=60),
                             open=close*.9999, high=close*1.01, low=close*.99, close=close, volume=np.ones(periods)*10,
                             amount=np.ones(periods)*1000, confirm=['1']*periods))

def opportunity(boundary='2026-03-01T04:00:00Z', role='validation'):
    b = pd.Timestamp(boundary)
    return dict(opportunity_id=b.strftime('%Y-%m-%dT%H:%M:%SZ'), role=role, decision_boundary_at=b.isoformat(), decision_at=(b+pd.Timedelta(seconds=60)).isoformat(),
                history_start_at=(b-pd.Timedelta(hours=256)).isoformat(), history_end_exclusive=b.isoformat(), entry_at=(b+pd.Timedelta(hours=1)).isoformat(), history_start_row=0, history_end_row_exclusive=256)

def history(opp):
    return bars(opp['history_start_at'], 256)

def test_exact_old_window_and_features_identity():
    opp = opportunity()
    frame = history(opp)
    actual = inputs.prepare_window(frame, opp)
    expected = encoder.prepare_window(frame, opp)
    for a, e in zip(actual[:2], expected[:2]):
        assert np.array_equal(a, e)
        assert a.dtype == np.float32
    assert actual[2] == expected[2]
    from research.frozen.experiment_02.features import build_features
    assert np.array_equal(inputs.build_features(frame,pd.DataFrame([opp])).to_numpy(), build_features(frame,pd.DataFrame([opp])).to_numpy())
    assert len(inputs.FEATURE_NAMES) == 33

@pytest.mark.parametrize('defect', ['amount', 'available', 'confirm', 'shift', 'gap'])
def test_reject_bad_history(defect):
    opp = opportunity()
    frame = history(opp)
    if defect == 'amount': frame.loc[0,'amount'] = np.nan
    if defect == 'available': frame.loc[255,'available_at'] = pd.Timestamp(opp['decision_at'])+pd.Timedelta(seconds=1)
    if defect == 'confirm': frame.loc[0,'confirm'] = '0'
    if defect == 'shift': frame.loc[0,'bar_open_at'] += pd.Timedelta(seconds=1)
    if defect == 'gap': frame = frame.drop(index=20)
    with pytest.raises(ValueError): inputs.prepare_window(frame, opp)

def test_timestamp_barrier_precedes_numeric_parse(tmp_path):
    frame = bars('2026-03-31T23:00:00Z', 1)
    future = frame.iloc[0].to_dict()
    future.update(bar_open_at='2026-04-01T00:00:00Z', open='FORBIDDEN_NOT_A_NUMBER', amount='FORBIDDEN')
    path = tmp_path/'mixed.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(frame.columns)); w.writeheader(); w.writerow(frame.iloc[0].to_dict()); w.writerow(future)
    got = inputs._read_bounded(path,pd.Timestamp('2026-03-31T23:00:00Z'),pd.Timestamp('2026-04-01T00:00:00Z'),'A',pd.Timedelta(hours=1))
    assert len(got)==1 and got.open.dtype==np.float64

def test_48_segments_and_first_entry_open():
    opp = opportunity()
    frame = bars(opp['entry_at'],48,'5min')
    got = inputs.build_labels(frame,pd.DataFrame([opp])).iloc[0]
    prices = np.r_[frame.open.iloc[0],frame.close.to_numpy()]
    assert got.RV_raw == np.sum(np.diff(np.log(prices))**2)
    assert got.label_price_count==49 and got.label_return_count==48
    assert len(json.loads(got.label_prices_json))==49
    with pytest.raises(ValueError): inputs.build_labels(frame.drop(index=20),pd.DataFrame([opp]))

def test_role_strict_purge_schedule():
    frame = bars('2026-02-18T08:00:00Z', 1000)
    frame = frame.loc[frame.bar_open_at < pd.Timestamp('2026-04-01T00:00:00Z')].reset_index(drop=True)
    result = inputs.opportunities(frame,'validation')
    assert len(result)==92
    assert result.iloc[-1].opportunity_id=='2026-03-31T12:00:00Z'
    assert '2026-03-31T20:00:00Z' not in set(result.opportunity_id)
    bad = opportunity('2026-03-31T20:00:00Z')
    with pytest.raises(ValueError): inputs.prepare_window(history(bad),bad)

def test_phase_b_cannot_bypass_authorization():
    opp=opportunity()
    with pytest.raises(PermissionError): inputs.prepare_window(history(opp),opp,phase='B')
    with pytest.raises(PermissionError): inputs.opportunities(history(opp),'TEST_Q2')
    source=history(opp)
    source.loc[len(source)] = source.iloc[-1]
    source.loc[len(source)-1,'bar_open_at'] = pd.Timestamp('2026-04-01T00:00:00Z')
    with pytest.raises(ValueError): inputs.prepare_window(source,opp)
