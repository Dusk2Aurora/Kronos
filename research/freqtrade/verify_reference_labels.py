"""Offline synthetic contract checks; never a substitute for official exports."""
from __future__ import annotations

import copy
import json

import pandas as pd

from build_reference_labels import HOUR, PAIR, iso, match_trades, opportunities, split_rows, stamp


def must_reject(function, text):
    try:
        function()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"Expected rejection: {text}")


def main():
    start = stamp("2024-01-01T00:00:00Z")
    end = start + 48 * HOUR
    dates = pd.date_range(start - 256 * HOUR, end, freq="h", inclusive="left")
    candles = pd.DataFrame({"date": dates, "close": [100 + i * .1 for i in range(len(dates))],
                            "open": [100 + i * .1 for i in range(len(dates))],
                            "available_at": [iso(date + HOUR + pd.Timedelta(seconds=60)) for date in dates]})
    marks = pd.DataFrame({"date": pd.date_range(start, end, freq="h", inclusive="left"), "open": 110.0})
    funding = pd.DataFrame({"date": pd.date_range(start, end, freq="8h", inclusive="left"), "realized_rate": .0001})
    records = opportunities(candles, start, end)
    assert len(records) == 6 and records[-1]["reason"] == "research_right_boundary"
    assert records[0]["history_start_row"] == 4 and records[0]["history_end_row_exclusive"] == 260
    assert records[0]["entry_at"] == "2024-01-01T05:00:00Z"
    assert records[0]["exit_at"] == "2024-01-01T09:00:00Z"
    assert records[0]["labelable_at"] == "2024-01-01T09:01:00Z"
    zero = candles.copy()
    zero.loc[259, "close"] = zero.loc[235, "close"]
    assert opportunities(zero, start, end)[0]["reason"] == "zero_momentum"
    future = candles.copy()
    future.loc[260:, "close"] = 999999.0
    assert opportunities(future, start, end)[0]["momentum_24h"] == records[0]["momentum_24h"]
    late = candles.copy()
    late.loc[259, "available_at"] = "2024-01-01T04:01:01Z"
    must_reject(lambda: opportunities(late, start, end), "Unavailable history")
    prices = candles.set_index("date").open
    trades = []
    for row in records:
        if row["reason"] != "labelable":
            continue
        entry, exit_at = stamp(row["entry_at"]), stamp(row["exit_at"])
        trades.append({"pair": PAIR, "open_date": iso(entry), "close_date": iso(exit_at),
                       "open_timestamp": int(entry.timestamp() * 1000), "close_timestamp": int(exit_at.timestamp() * 1000),
                       "open_rate": float(prices[entry]), "close_rate": float(prices[exit_at]),
                       "fee_open": .0007, "fee_close": .0007, "leverage": 1.0,
                       "is_short": False, "enter_tag": "momentum_24h_long", "trade_duration": 240,
                       "exit_reason": "fixed_hold_4h", "is_open": False, "amount": 2.0,
                       "funding_fees": -.022, "profit_ratio": .001, "profit_abs": 1.0})
    assert len(match_trades(records, trades, .0007, candles, marks, funding)) == 5
    must_reject(lambda: match_trades(records, trades[:-1], .0007, candles, marks, funding), "count mismatch")
    duplicate = copy.deepcopy(trades)
    duplicate[1] = copy.deepcopy(duplicate[0])
    must_reject(lambda: match_trades(records, duplicate, .0007, candles, marks, funding), "duplicate")
    must_reject(lambda: match_trades(records, trades, .0014, candles, marks, funding), "fee_open")
    wrong = copy.deepcopy(trades)
    wrong[0]["funding_fees"] *= -1
    must_reject(lambda: match_trades(records, wrong, .0007, candles, marks, funding), "signed payment")
    wrong = copy.deepcopy(trades)
    wrong[0]["exit_reason"] = "force_exit"
    must_reject(lambda: match_trades(records, wrong, .0007, candles, marks, funding), "Unexpected exit")
    wrong = copy.deepcopy(trades)
    wrong[0]["is_short"] = True
    must_reject(lambda: match_trades(records, wrong, .0007, candles, marks, funding), "direction")
    # Equality at label availability is purged, even when the physical exit is earlier.
    boundary = [iso(start), records[0]["labelable_at"]]
    config = {"splits": {"walk_forward_dates": [{"id": "SYNTHETIC", "train": boundary,
                        "validation": boundary, "test": boundary}],
                        "final_fit": {"train": boundary, "validation": boundary}, "final_holdout": boundary}}
    indexed = split_rows(records, config)
    assert len(indexed) == 6 and all(row["reason"] == "purged_segment_right_boundary" for row in indexed)
    assert not any(row["included"] for row in indexed)
    print(json.dumps({"passed": True, "synthetic_only": True,
                      "checks": ["256_history", "60_second_availability", "future_candle_invariance",
                                 "zero_momentum", "research_right_boundary", "labelable_at_equality_purge",
                                 "unique_matching", "missing_trade_rejection", "fee_rejection",
                                 "funding_sign_rejection", "forced_exit_rejection", "direction_rejection"]}, indent=2))


if __name__ == "__main__":
    main()
