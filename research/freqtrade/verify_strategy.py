"""Small offline checks against the installed Freqtrade engine; no market backtest."""

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent / "strategies"))
from KronosFixedMomentum import KronosFixedMomentum
from freqtrade.configuration import TimeRange
from freqtrade.optimize.backtesting import Backtesting, HEADERS


PAIR = "BTC/USDT:USDT"
strategy = KronosFixedMomentum({})


def candles(slope=1.0):
    close = 1000.0 + slope * np.arange(337)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(close), freq="h", tz="UTC"),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.ones(len(close)),
    })


def signals(frame):
    result = strategy.populate_indicators(frame.copy(), {"pair": PAIR})
    result = strategy.populate_entry_trend(result, {"pair": PAIR})
    return strategy.populate_exit_trend(result, {"pair": PAIR})


up = signals(candles())
assert not up.loc[:255, "history_ready"].any()
assert up.loc[256, "history_ready"]
assert set(up.loc[up["decision_slot"], "date"].dt.hour) == {4, 12, 20}
assert up.loc[268, "enter_long"] == 1  # Jan 12 04:00 UTC.
assert up.loc[268, "enter_short"] == 0
assert up.loc[268, "momentum_24h"] == 1267 / 1243 - 1
assert signals(candles(-1)).loc[268, "enter_short"] == 1
assert signals(candles(0))[["enter_long", "enter_short"]].to_numpy().sum() == 0
assert up.loc[~up["decision_slot"], ["enter_long", "enter_short"]].to_numpy().sum() == 0

# The decision at row 268 uses bars 12..267; its own and future prices cannot affect it.
changed = candles()
changed.loc[268:, ["open", "high", "low", "close", "volume"]] *= 17
assert_frame_equal(up.loc[:268, ["history_ready", "momentum_24h", "enter_long", "enter_short"]],
                   signals(changed).loc[:268, ["history_ready", "momentum_24h", "enter_long", "enter_short"]])
gap = candles().drop(index=200).reset_index(drop=True)
gap_signals = signals(gap)
assert not gap_signals.loc[gap_signals["date"].eq(up.loc[268, "date"]), "history_ready"].any()
invalid = candles()
invalid.loc[200, "close"] = np.nan
assert not signals(invalid).loc[268, "history_ready"]

# Invoke the official conversion function, which shifts signal columns once.
# Stub only cache/progress plumbing; timing and conversion are the actual engine.
engine = Backtesting.__new__(Backtesting)
engine.strategy = SimpleNamespace(ft_advise_signals=lambda frame, metadata: signals(frame))
engine.dataprovider = SimpleNamespace(_set_cached_df=lambda *args: None)
engine.config = {"candle_type_def": "futures"}
engine.timeframe = "1h"
engine.timerange = TimeRange()
engine.required_startup = 0
engine._set_progress_step = lambda *args: None
engine._increment_progress = lambda *args: None
engine.check_abort = lambda: None
rows = engine._get_ohlcv_as_lists({PAIR: candles()})[PAIR]
converted = pd.DataFrame(rows, columns=HEADERS)
boundary = up.loc[268, "date"]
assert converted.loc[converted["date"].eq(boundary), "enter_long"].iloc[0] == 0
assert converted.loc[converted["date"].eq(boundary + timedelta(hours=1)), "enter_long"].iloc[0] == 1

trade = SimpleNamespace(open_date_utc=boundary + timedelta(hours=1))
assert strategy.custom_exit(PAIR, trade, boundary + timedelta(hours=5, seconds=-1), 1, -0.1) is None
assert strategy.custom_exit(PAIR, trade, boundary + timedelta(hours=5), 1, -0.1) == "fixed_hold_4h"
assert strategy.use_exit_signal and strategy.minimal_roi == {} and not strategy.trailing_stop
assert up[["exit_long", "exit_short"]].to_numpy().sum() == 0
assert strategy.leverage(PAIR, boundary, 1, 5, 10, None, "short") == 1.0
print("PASS: causal 256-bar history, gaps/invalid data guard, long/short/flat, UTC slots,")
print("      official engine 04:00 -> 05:00 shift, 09:00 timed exit, ROI/trailing off, leverage 1.")
print("These are offline strategy checks, not a completed market backtest.")
