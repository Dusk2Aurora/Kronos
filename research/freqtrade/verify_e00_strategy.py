"""Offline E00 adapter checks using installed Freqtrade conversion, not a backtest."""

import copy
import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent / "strategies"))
from KronosE00 import KronosE00
from freqtrade.configuration import TimeRange
from freqtrade.optimize.backtesting import Backtesting, HEADERS

PAIR = "BTC/USDT:USDT"


def candles(slope=1):
    close = 1000 + slope * np.arange(337, dtype=float)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(close), freq="h", tz="UTC"),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.ones(len(close)),
    })


def signals(strategy, frame):
    result = strategy.populate_indicators(frame.copy(), {"pair": PAIR})
    result = strategy.populate_entry_trend(result, {"pair": PAIR})
    return strategy.populate_exit_trend(result, {"pair": PAIR})


def main():
    boundary = candles().loc[268, "date"]
    base = {"schema_version": 1, "mode": "fixed_momentum", "decisions": [
        {"signal_at": boundary.isoformat().replace("+00:00", "Z"),
         "direction": 1, "exposure_fraction": 1}
    ]}
    with tempfile.TemporaryDirectory(prefix="kronos_e00_check_") as temporary:
        path = Path(temporary) / "decisions.json"

        def create(payload):
            path.write_text(json.dumps(payload), encoding="utf-8")
            return KronosE00({"e00_decisions_path": str(path)})

        strategy = create(base)
        up = signals(strategy, candles())
        assert up.loc[268, "enter_long"] == 1
        assert up[["enter_long", "enter_short"]].to_numpy().sum() == 1
        assert not up.loc[:255, "history_ready"].any()
        changed = candles()
        changed.loc[268:, ["open", "high", "low", "close", "volume"]] *= 17
        columns = ["history_ready", "momentum_24h", "enter_long", "enter_short", "e00_exposure"]
        assert_frame_equal(up.loc[:268, columns], signals(strategy, changed).loc[:268, columns])
        gap = candles().drop(index=200).reset_index(drop=True)
        assert not signals(strategy, gap)[["enter_long", "enter_short"]].to_numpy().sum()
        invalid_bar = candles()
        invalid_bar.loc[200, "close"] = np.nan
        assert not signals(strategy, invalid_bar).loc[268, "enter_long"]

        # Real official engine shifts only signal columns once; no exchange calls.
        engine = Backtesting.__new__(Backtesting)
        engine.strategy = SimpleNamespace(ft_advise_signals=lambda frame, metadata: signals(strategy, frame))
        engine.dataprovider = SimpleNamespace(_set_cached_df=lambda *args: None)
        engine.config = {"candle_type_def": "futures"}
        engine.timeframe = "1h"
        engine.timerange = TimeRange()
        engine.required_startup = 0
        engine._set_progress_step = lambda *args: None
        engine._increment_progress = lambda *args: None
        engine.check_abort = lambda: None
        converted = pd.DataFrame(engine._get_ohlcv_as_lists({PAIR: candles()})[PAIR], columns=HEADERS)
        entry = boundary + timedelta(hours=1)
        assert converted.loc[converted["date"].eq(boundary), "enter_long"].iloc[0] == 0
        assert converted.loc[converted["date"].eq(entry), "enter_long"].iloc[0] == 1

        def stake(adapter, minimum=10, current=entry, side="long", maximum=1000):
            return adapter.custom_stake_amount(PAIR, current, 1000, 1000, minimum,
                                               maximum, 1, None, side)

        assert stake(strategy) == 1000
        assert stake(strategy, current=boundary) == 0
        assert stake(strategy, side="short") == 0
        assert stake(strategy, maximum=700) == 700
        short_payload = copy.deepcopy(base)
        short_payload["decisions"][0]["direction"] = -1
        short_strategy = create(short_payload)
        assert signals(short_strategy, candles(-1)).loc[268, "enter_short"] == 1
        assert stake(short_strategy, side="short") == 1000
        half = copy.deepcopy(base)
        half["mode"] = "constant_half_exposure"
        half["decisions"][0]["exposure_fraction"] = 0.5
        assert stake(create(half)) == 500
        assert stake(create(half), minimum=501) == 0
        vol = copy.deepcopy(base)
        vol["mode"] = "vol_target"
        vol["decisions"][0]["exposure_fraction"] = 0.3
        assert stake(create(vol)) == 300
        gated = copy.deepcopy(base)
        gated["mode"] = "ordinary_features_gate"
        gated["decisions"][0]["exposure_fraction"] = 0
        gate = create(gated)
        assert stake(gate) == 0
        assert signals(gate, candles())[["enter_long", "enter_short"]].to_numpy().sum() == 0
        cash = create({"schema_version": 1, "mode": "cash", "decisions": []})
        assert signals(cash, candles())[["enter_long", "enter_short"]].to_numpy().sum() == 0

        trade = SimpleNamespace(open_date_utc=entry)
        assert strategy.custom_exit(PAIR, trade, entry + timedelta(hours=4, seconds=-1), 1, 0) is None
        assert strategy.custom_exit(PAIR, trade, entry + timedelta(hours=4), 1, 0) == "fixed_hold_4h"
        buy = copy.deepcopy(base)
        buy["mode"] = "buy_hold"
        buy["hold_until"] = "2026-01-14T23:00:00Z"
        buy_hold = create(buy)
        assert signals(buy_hold, candles(-1)).loc[268, "enter_long"] == 1
        assert buy_hold.custom_exit(PAIR, trade, entry + timedelta(hours=4), 1, 0) is None
        assert buy_hold.custom_exit(PAIR, trade, pd.Timestamp(buy["hold_until"]), 1, 0) == "e00_buy_hold_end"
        assert strategy.leverage(PAIR, entry, 1, 4, 10, None, "long") == 1
        assert strategy.minimal_roi == {} and strategy.use_exit_signal and not strategy.trailing_stop

        invalid = []
        for key, value in (("profit", 1), ("funding", 0), ("label", 1)):
            case = copy.deepcopy(base)
            case["decisions"][0][key] = value
            invalid.append(case)
        for fraction in (-0.1, 1.1, float("nan"), True):
            case = copy.deepcopy(vol)
            case["decisions"][0]["exposure_fraction"] = fraction
            invalid.append(case)
        for direction in (0, True):
            case = copy.deepcopy(base)
            case["decisions"][0]["direction"] = direction
            invalid.append(case)
        duplicate = copy.deepcopy(base)
        duplicate["decisions"] *= 2
        invalid.append(duplicate)
        for timestamp in ("2026-01-12Z", "2026-01-12T04:00:00", "2026-01-12T04:01:00Z",
                          "2026-01-12T05:00:00Z"):
            case = copy.deepcopy(base)
            case["decisions"][0]["signal_at"] = timestamp
            invalid.append(case)
        invalid.extend([{}, {**base, "schema_version": True}, {**base, "mode": "unknown"},
                        {**base, "future_profit": 1}, {**base, "hold_until": buy["hold_until"]},
                        {**buy, "hold_until": "2026-01-12T05:00:00Z"}])
        for payload in invalid:
            try:
                create(payload)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid sidecar accepted")
        try:
            KronosE00({})
        except ValueError:
            pass
        else:
            raise AssertionError("Missing configuration accepted")
        path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        try:
            KronosE00({"e00_decisions_path": str(path)})
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate JSON keys accepted")
        reversed_direction = copy.deepcopy(base)
        reversed_direction["decisions"][0]["direction"] = -1
        try:
            signals(create(reversed_direction), candles())
        except ValueError:
            pass
        else:
            raise AssertionError("Direction reversal accepted")
    print("PASS: six E00 adapters, strict outcome-free JSON, official 04->05 signal shift,")
    print("      causal history and future perturbation, sizing/minimum/zero gate, 4h and buy-hold exits.")
    print("Offline synthetic verification only; no market performance or holdout evaluated.")


if __name__ == "__main__":
    main()
