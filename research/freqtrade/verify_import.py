"""Small offline synthetic roundtrip check; run in the Freqtrade virtualenv."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from import_snapshot import HOUR, import_snapshot, milliseconds, sha256


def iso(ts):
    return datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat().replace('+00:00', 'Z')


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reject(call, reason):
    try:
        call()
    except ValueError as exc:
        if reason not in str(exc):
            raise AssertionError(f'Wrong rejection: {exc}') from exc
    else:
        raise AssertionError(f'Expected rejection: {reason}')


def main():
    start = milliseconds('2024-01-01T00:00:00Z')
    end, data_start = start + 8 * HOUR, start - 256 * HOUR
    # Keep temporary writes within the project workspace and remove them afterwards.
    with tempfile.TemporaryDirectory(prefix='offline_import_', dir=Path(__file__).resolve().parents[2]) as tmp:
        base = Path(tmp)
        snapshot = base / 'snapshot'
        snapshot.mkdir()
        candles = []
        for ts in range(data_start, end, HOUR):
            candles.append({'bar_open_at': iso(ts), 'bar_close_at': iso(ts + HOUR),
                            'open': 100.125, 'high': 110.5, 'low': 90.25, 'close': 101.75,
                            'volume': 0.123, 'amount': 12.4321, 'raw_contract_volume': 12.3,
                            'confirm': 1, 'is_warmup': ts < start})
        marks = [{key: value for key, value in row.items()
                  if key in ['bar_open_at', 'open', 'high', 'low', 'close', 'confirm']}
                 for row in candles if not row['is_warmup']]
        funding = [{'instrument': 'BTC-USDT-SWAP', 'funding_at': iso(start + offset * HOUR),
                    'realized_rate': rate, 'source_kind': 'offline_synthetic'}
                   for offset, rate in [(0, 0.0001), (4, -0.0002)]]
        save_csv(snapshot / 'candles.csv', candles)
        save_csv(snapshot / 'mark_prices.csv', marks)
        save_csv(snapshot / 'funding.csv', funding)
        report = {'snapshot_id': 'offline', 'ready_for_labeling': True,
                  'data_start': iso(data_start), 'evaluation_start': iso(start),
                  'evaluation_end_exclusive': iso(end), 'warmup_bars': 256,
                  'row_counts': {'candles': len(candles), 'mark_prices': len(marks), 'funding': len(funding)}}
        (snapshot / 'quality_report.json').write_text(json.dumps(report), encoding='utf-8')

        def manifest():
            (snapshot / 'manifest.json').write_text(json.dumps({
                'status': 'collected', 'snapshot_id': 'offline',
                'files': {p.name: sha256(p) for p in snapshot.iterdir() if p.name != 'manifest.json'}
            }), encoding='utf-8')

        manifest()
        result = import_snapshot(snapshot, base / 'native')
        assert all(c['passed'] and c['numeric_values_exact'] for c in result['roundtrip'].values())
        assert result['roundtrip']['futures']['rows'] == 264
        assert result['roundtrip']['funding_rate']['rows'] == 2
        assert len(result['files']) == 3 and result['formal_evaluation_allowed'] is False
        reject(lambda: import_snapshot(snapshot, base / 'native'), 'new or empty')
        # A modified source cannot be imported under its previous immutable hash.
        with (snapshot / 'funding.csv').open('a', encoding='utf-8') as handle:
            handle.write('\n')
        reject(lambda: import_snapshot(snapshot, base / 'tampered'), 'hash mismatch')
        save_csv(snapshot / 'funding.csv', funding)
        # Rehash a synthetic malformed source to exercise semantic validation too.
        save_csv(snapshot / 'mark_prices.csv', marks[:-1])
        manifest()
        reject(lambda: import_snapshot(snapshot, base / 'mark_gap'), 'continuous')
        save_csv(snapshot / 'mark_prices.csv', marks)
        funding[0]['funding_at'] = iso(start + 60_000)
        save_csv(snapshot / 'funding.csv', funding)
        manifest()
        reject(lambda: import_snapshot(snapshot, base / 'off_grid'), 'hourly grid')
        assert not (base / 'tampered').exists() and not (base / 'mark_gap').exists()
        assert not (base / 'off_grid').exists()
    print('PASS: official feather roundtrip preserves UTC/OHLCV/signed sparse funding; rejects overwrite, hash tampering, mark gaps and off-grid events')


if __name__ == '__main__':
    main()
