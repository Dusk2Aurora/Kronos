"""Scope guarded causal inputs for the independently reserved risk study.

Only timestamp-selected authorized records are numerically interpreted. Existing
DEV 5m prices are reused; new schedules do not inherit the old fold purge gaps.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.frozen import encoder
from research.frozen.experiment_02.features import (
    FEATURE_NAMES, HAR_NAMES, HOURS, R0_NAMES, EPSILON, oldfeatures,
)
from research.frozen.experiment_03 import guard

FIELDS = tuple(encoder.FIELDS)
HOUR = pd.Timedelta(hours=1)
STEP = pd.Timedelta(minutes=5)


def _utc(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None or t.utcoffset().total_seconds() != 0:
        raise ValueError('Explicit UTC timestamp required')
    return t.tz_convert('UTC')


def _limit(phase):
    cfg = guard.config()
    if phase not in ('A', 'B'):
        raise ValueError('Unknown execution phase')
    end = _utc(cfg['scope_guards'][f'phase_{phase.lower()}_max_time_exclusive'])
    guard.require_scope(end, phase=phase)
    return cfg, end


def _role(cfg, role, phase, scoped=True):
    if role in ('train', 'validation'):
        start, end = cfg['roles'][role]
    else:
        matches = [q for q in cfg['roles']['test_quarters'] if q['id'] == role]
        if len(matches) != 1:
            raise ValueError('Unknown opportunity role')
        start, end = matches[0]['start'], matches[0]['end_exclusive']
    start, end = _utc(start), _utc(end)
    if scoped:
        guard.require_scope(end, phase=phase)
    elif end > _utc(cfg['scope_guards'][f'phase_{phase.lower()}_max_time_exclusive']):
        raise PermissionError('Role exceeds authorized scope')
    return start, end


def _source_times(frame, phase, step):
    _, limit = _limit(phase)
    column = frame['bar_open_at']
    if isinstance(column.dtype, pd.DatetimeTZDtype) and str(column.dtype.tz) == 'UTC':
        times = pd.DatetimeIndex(column)
    else:
        times = pd.DatetimeIndex([_utc(v) for v in column])
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Source timestamps must be unique and ordered')
    if (times >= limit).any():
        raise ValueError('Source contains unauthorized timestamps')
    if len(times) and ((times.asi8 % step.value) != 0).any():
        raise ValueError('Shifted source timestamp grid')
    return times


def _read_bounded(path, start, end, phase, step):
    guard.require_scope(end, phase=phase)
    rows = []
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            t = _utc(row['bar_open_at'])
            # Never parse any market value until the timestamp passes this barrier.
            if not start <= t < end:
                continue
            rows.append(row)
    if not rows:
        raise ValueError('Empty authorized source')
    frame = pd.DataFrame(rows)
    frame['bar_open_at'] = pd.to_datetime(frame['bar_open_at'], utc=True)
    times = _source_times(frame, phase, step)
    expected = pd.date_range(start, end, freq=step, inclusive='left')
    if not times.equals(expected):
        raise ValueError('Incomplete authorized source grid')
    for field in FIELDS:
        if field not in frame:
            raise ValueError('Missing actual OHLCVA field: ' + field)
        frame[field] = pd.to_numeric(frame[field], errors='raise').astype(np.float64)
    _validate_values(frame)
    if 'confirm' not in frame or not frame['confirm'].astype(str).eq('1').all():
        raise ValueError('Unconfirmed source bar')
    close = pd.DatetimeIndex([_utc(v) for v in frame['bar_close_at']])
    if not close.equals(times + step):
        raise ValueError('Wrong source bar close')
    frame['bar_close_at'] = close
    if 'available_at' in frame:
        frame['available_at'] = pd.DatetimeIndex([_utc(v) for v in frame['available_at']])
    return frame


def _validate_values(frame):
    if any(field not in frame for field in FIELDS):
        raise ValueError('Missing actual OHLCVA field')
    values = frame.loc[:, FIELDS].to_numpy(dtype=np.float64)
    if not (np.isfinite(values).all() and (values[:, :4] > 0).all()
            and (values[:, 4:] >= 0).all()
            and (values[:, 1] >= np.maximum(values[:, 0], values[:, 3])).all()
            and (values[:, 2] <= np.minimum(values[:, 0], values[:, 3])).all()):
        raise ValueError('Invalid real OHLCVA values')
    return values


def load_hourly(phase='A'):
    cfg, end = _limit(phase)
    start = _utc(cfg['roles']['train'][0]) - 256 * HOUR
    directory = guard.ROOT / cfg['sources']['one_hour_snapshot']
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if encoder.sha(directory / 'candles.csv') != manifest['files']['candles.csv']:
        raise ValueError('Changed official hourly source')
    frame = _read_bounded(directory / 'candles.csv', start, end, phase, HOUR)
    if 'available_at' not in frame:
        raise ValueError('Missing hourly availability evidence')
    if (frame.available_at < frame.bar_close_at).any():
        raise ValueError('Availability precedes hourly close')
    return frame


def load_development_5m():
    cfg, end = _limit('A')
    start = _utc(cfg['roles']['train'][0])
    directory = guard.ROOT / cfg['sources']['development_5m']
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['protocol_sha256'] != cfg['parent_protocol_sha256']
            or [_utc(v) for v in manifest['range']] != [start, end]
            or manifest['instrument'] != cfg['sources']['instrument']
            or manifest['bar'] != '5m'
            or manifest['candles_sha256'] != encoder.sha(directory / 'candles.csv')):
        raise ValueError('Changed verified DEV 5m manifest/source')
    audit = json.loads((directory / 'raw_csv_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'PASS' or not audit['exact_raw_csv_match'] or audit['holdout_rows'] != 0:
        raise ValueError('DEV 5m verification missing')
    frame = _read_bounded(directory / 'candles.csv', start, end, 'A', STEP)
    if len(frame) != manifest['rows'] or len(frame) != audit['rows']:
        raise ValueError('DEV 5m row-count evidence mismatch')
    return frame


def opportunities(candles, role='train', phase='A'):
    cfg, _ = _limit(phase)
    start, end = _role(cfg, role, phase)
    times = _source_times(candles, phase, HOUR)
    rows = []
    delay = pd.Timedelta(seconds=cfg['timing']['availability_delay_seconds'])
    for day in pd.date_range(start.normalize(), end, freq='D', inclusive='left'):
        for hour in cfg['timing']['decision_hours_utc']:
            boundary = day + hour * HOUR
            entry = boundary + cfg['timing']['entry_after_boundary_hours'] * HOUR
            label_end = entry + cfg['timing']['horizon_hours'] * HOUR
            labelable = label_end + delay
            if not (start <= boundary < end and label_end < end and labelable < end):
                continue
            history_start = boundary - cfg['timing']['lookback_1h_bars'] * HOUR
            left = int(times.searchsorted(history_start))
            right = int(times.searchsorted(boundary))
            row = dict(opportunity_id=boundary.strftime('%Y-%m-%dT%H:%M:%SZ'), role=role,
                       segment_start=start.isoformat(), segment_end_exclusive=end.isoformat(),
                       decision_boundary_at=boundary.isoformat(), decision_at=(boundary+delay).isoformat(),
                       entry_at=entry.isoformat(), history_start_at=history_start.isoformat(),
                       history_end_exclusive=boundary.isoformat(), history_start_row=left,
                       history_end_row_exclusive=right, label_end=label_end.isoformat(),
                       labelable_at=labelable.isoformat())
            # Schedule rejects unavailable/missing history instead of dropping opportunities.
            _prepare_window(candles, row, times)
            rows.append(row)
    return pd.DataFrame(rows)


def _opportunity_scope(opportunity, cfg, phase, scoped=True):
    boundary = _utc(opportunity['decision_boundary_at'])
    if scoped:
        guard.require_scope(boundary + HOUR, phase=phase)
    elif boundary + HOUR > _utc(cfg['scope_guards'][f'phase_{phase.lower()}_max_time_exclusive']):
        raise PermissionError('Opportunity exceeds authorized scope')
    if 'role' in opportunity:
        start, end = _role(cfg, opportunity['role'], phase, scoped=scoped)
        entry = _utc(opportunity['entry_at'])
        expected_entry = boundary + cfg['timing']['entry_after_boundary_hours'] * HOUR
        label_end = entry + cfg['timing']['horizon_hours'] * HOUR
        available = label_end + pd.Timedelta(seconds=cfg['timing']['availability_delay_seconds'])
        if not (start <= boundary < end and entry == expected_entry
                and label_end < end and available < end):
            raise ValueError('Opportunity role boundary/availability purge violation')
    return boundary


def prepare_window(candles, opportunity, phase='A'):
    cfg, _ = _limit(phase)
    _opportunity_scope(opportunity, cfg, phase)
    times = _source_times(candles, phase, HOUR)
    return _prepare_window(candles, opportunity, times)


def _prepare_window(candles, opportunity, times):
    # Pure internal validator: every caller has already checked authorization.
    boundary = _utc(opportunity['decision_boundary_at'])
    start, end = _utc(opportunity['history_start_at']), _utc(opportunity['history_end_exclusive'])
    decision = _utc(opportunity['decision_at'])
    if (end != boundary or start != end-256*HOUR
            or str(opportunity['opportunity_id']) != boundary.strftime('%Y-%m-%dT%H:%M:%SZ')
            or boundary.hour not in (4, 12, 20) or boundary != boundary.floor('h')
            or decision != boundary+pd.Timedelta(seconds=60)):
        raise ValueError('Wrong causal window or decision identity')
    frame = candles.loc[(times >= start) & (times < end)].copy()
    selected = pd.DatetimeIndex(frame['bar_open_at'])
    if not selected.equals(pd.date_range(start, periods=256, freq='h')):
        raise ValueError('Missing, duplicated, unsorted or shifted historical bars')
    if not pd.DatetimeIndex([_utc(v) for v in frame['bar_close_at']]).equals(selected+HOUR):
        raise ValueError('Wrong bar close boundary')
    if 'available_at' not in frame or any(_utc(v) > decision for v in frame['available_at']):
        raise ValueError('Unavailable historical bar')
    if any(_utc(a) < _utc(c) for a, c in zip(frame['available_at'], frame['bar_close_at'])):
        raise ValueError('Availability precedes historical bar close')
    if 'confirm' not in frame or not frame['confirm'].astype(str).eq('1').all():
        raise ValueError('Unconfirmed historical bar')
    values = _validate_values(frame)
    normalized = np.clip((values-values.mean(axis=0))/(values.std(axis=0, ddof=0)+1e-5), -5, 5)
    stamps = encoder.calc_time_stamps(pd.Series(selected)).to_numpy(dtype=np.float32)
    identity = dict(opportunity_id=str(opportunity['opportunity_id']), asset='BTC-USDT-SWAP',
                    window_start=start.isoformat(), window_end_exclusive=end.isoformat(),
                    history_values_sha256=encoder.array_sha(values),
                    history_timestamps_sha256=encoder.digest(selected.astype(str).tolist()))
    return normalized.astype(np.float32), stamps, identity


def build_features(candles, opps, phase='A'):
    cfg, _ = _limit(phase)
    times = _source_times(candles, phase, HOUR)
    source = candles.loc[:, ['bar_open_at', *FIELDS]]
    rows, ids = [], []
    for _, opportunity in opps.iterrows():
        _opportunity_scope(opportunity, cfg, phase, scoped=False)
        _prepare_window(candles, opportunity, times)
        start, end = _utc(opportunity['history_start_at']), _utc(opportunity['history_end_exclusive'])
        window = source.loc[(source.bar_open_at >= start) & (source.bar_open_at < end)]
        # Timestamp-selected local window makes stored source offsets irrelevant.
        local = opportunity.to_dict()
        local.update(history_start_row=0, history_end_row_exclusive=256)
        values = oldfeatures(window, local)
        returns = np.diff(np.log(window.close.to_numpy(dtype=np.float64)))
        squared = returns**2
        parkinson = np.log(window.high.to_numpy(dtype=np.float64)/window.low.to_numpy(dtype=np.float64))**2/(4*np.log(2))
        for h in HOURS:
            values[f'log_rv4scaled_{h}h'] = float(np.log(max(4*squared[-h:].mean(), EPSILON)))
            values[f'log_mean_abs_return_{h}h'] = float(np.log(max(np.abs(returns[-h:]).mean(), EPSILON)))
            values[f'log_parkinson4scaled_{h}h'] = float(np.log(max(4*parkinson[-h:].mean(), EPSILON)))
        values['log_rv_ratio_4h_24h'] = values[HAR_NAMES[0]]-values[HAR_NAMES[1]]
        values['log_rv_ratio_24h_168h'] = values[HAR_NAMES[1]]-values[HAR_NAMES[2]]
        values['tail_ratio_24h'] = float(np.max(np.abs(returns[-24:]))/np.sqrt(max(squared[-24:].mean(), EPSILON)))
        for field in ('volume', 'amount'):
            history = window[field].to_numpy(dtype=np.float64)
            values[f'log_{field}_ratio_4h_24h'] = float(np.log((history[-4:].mean()+EPSILON)/(history[-24:].mean()+EPSILON)))
        weights = .97**np.arange(254, -1, -1, dtype=np.float64)
        values['r0_persistence'] = float(squared[-4:].sum())
        values['r0_ewma'] = float(4*np.dot(weights/weights.sum(), squared))
        rows.append(values)
        ids.append(str(opportunity['opportunity_id']))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate opportunity IDs')
    frame = pd.DataFrame(rows, index=pd.Index(ids, name='opportunity_id'), columns=(*FEATURE_NAMES, *R0_NAMES), dtype=np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Nonfinite risk features')
    return frame


def build_labels(five_minute, opps, phase='A'):
    cfg, limit = _limit(phase)
    times = _source_times(five_minute, phase, STEP)
    lookup = {t: i for i, t in enumerate(times)}
    out = []
    for _, opp in opps.iterrows():
        if 'role' not in opp:
            raise ValueError('Labels require an explicit purged opportunity role')
        boundary = _opportunity_scope(opp, cfg, phase, scoped=False)
        entry = _utc(opp['entry_at'])
        end = entry + 4*HOUR
        available = end + pd.Timedelta(seconds=60)
        if entry != boundary+HOUR or end >= limit or available >= limit:
            raise ValueError('Label boundary/availability scope violation')
        indices = [lookup.get(entry+j*STEP) for j in range(48)]
        if any(i is None for i in indices):
            raise ValueError('Incomplete 48-return label window')
        window = five_minute.iloc[indices]
        _validate_values(window)
        if not window['confirm'].astype(str).eq('1').all():
            raise ValueError('Unconfirmed label bar')
        selected = pd.DatetimeIndex(window['bar_open_at'])
        if not pd.DatetimeIndex([_utc(v) for v in window['bar_close_at']]).equals(selected+STEP):
            raise ValueError('Wrong label bar close boundary')
        if 'available_at' in window and any(_utc(v) > available for v in window['available_at']):
            raise ValueError('Unavailable label bar')
        prices = np.r_[float(window.iloc[0]['open']), window.close.to_numpy(dtype=np.float64)]
        rv = float(np.sum(np.diff(np.log(prices))**2))
        row = opp.to_dict()
        row.update(RV_raw=rv, RV_effective=max(rv, cfg['label']['epsilon']),
                   label_start=entry.isoformat(), label_end=end.isoformat(), labelable_at=available.isoformat(),
                   label_price_count=49, label_return_count=48,
                   label_prices_json=json.dumps(prices.tolist(), separators=(',', ':')),
                   label_bar_open_at_json=json.dumps([t.isoformat() for t in selected], separators=(',', ':')))
        for field in ('raw_page', 'raw_row', 'raw_sha256'):
            if field in window:
                row['label_'+field+'_json'] = json.dumps(window[field].tolist(), separators=(',', ':'))
        out.append(row)
    return pd.DataFrame(out)
