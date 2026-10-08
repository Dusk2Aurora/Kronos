"""Small deterministic causality/contract checks, using synthetic OHLCVA only.

Run with the Kronos interpreter from the project root. No label, holdout,
training or backtest data is opened.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ordinary_features import (
    FEATURE_NAMES, features_for_opportunity, historical_window, ordinary_features,
)


def rejects(call, description):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f"Expected rejection: {description}")


def main():
    dates = pd.date_range("2024-01-01", periods=340, freq="h", tz="UTC")
    index = np.arange(len(dates), dtype=float)
    close = 100 * np.exp(0.001 * index + 0.004 * np.sin(index / 5))
    candles = pd.DataFrame({"date": dates, "open": close, "high": close * 1.01,
                            "low": close * 0.99, "close": close,
                            "volume": 10 + index / 100, "amount": 1000 + index ** 1.5})
    boundary = dates[300]
    opportunity = {"decision_boundary_at": boundary, "history_start_row": 44,
                   "history_end_row_exclusive": 300}
    baseline = features_for_opportunity(candles, opportunity)
    assert tuple(baseline) == FEATURE_NAMES and len(baseline) == 13
    assert baseline == ordinary_features(candles, decision_boundary=boundary)
    relabeled = candles.copy()
    relabeled.index = np.arange(len(candles)) + 999
    assert baseline == features_for_opportunity(relabeled, opportunity)
    future = candles.copy()
    future.loc[300:, ["open", "high", "low", "close", "volume", "amount"]] = np.nan
    assert baseline == features_for_opportunity(future, opportunity)
    assert baseline == ordinary_features(future, decision_boundary=boundary)
    changed = candles.copy()
    changed.loc[276:299, "amount"] *= 3
    changed_features = features_for_opportunity(changed, opportunity)
    assert changed_features["amount_mean_ratio_24h_168h"] != baseline["amount_mean_ratio_24h_168h"]
    assert all(changed_features[name] == value for name, value in baseline.items()
               if name != "amount_mean_ratio_24h_168h")
    window = historical_window(candles, decision_boundary=boundary,
                               history_start_row=44, history_end_row_exclusive=300)
    assert len(window) == 256 and window.date.iloc[0] == dates[44]
    assert window.date.iloc[-1] == dates[299]
    assert np.isclose(baseline["log_return_24h"], np.log(close[299] / close[275]))
    assert np.isclose(baseline["hourly_log_return_std_24h"],
                      np.std(np.diff(np.log(close[275:300])), ddof=0))
    alias = candles.rename(columns={"date": "bar_open_at", "volume": "volCcy",
                                    "amount": "volCcyQuote"})
    assert baseline == features_for_opportunity(alias, opportunity)
    rejects(lambda: ordinary_features(candles.drop(index=120), decision_boundary=boundary), "gap")
    rejects(lambda: features_for_opportunity(candles, {**opportunity, "history_start_row": 43,
                                                      "history_end_row_exclusive": 299}), "off-by-one")
    rejects(lambda: features_for_opportunity(candles.assign(profit=0), opportunity), "outcome input")
    invalid = candles.copy()
    invalid.loc[200, "high"] = invalid.loc[200, "low"] / 2
    rejects(lambda: features_for_opportunity(invalid, opportunity), "OHLC bounds")
    invalid = candles.copy()
    invalid.loc[200, "amount"] = np.nan
    rejects(lambda: features_for_opportunity(invalid, opportunity), "missing actual amount")
    zero = candles.copy()
    zero.loc[132:299, "volume"] = 0
    rejects(lambda: features_for_opportunity(zero, opportunity), "zero ratio denominator")
    rejects(lambda: ordinary_features(candles, decision_boundary=boundary.tz_localize(None)), "naive UTC")
    print(json.dumps({"status": "passed", "feature_count": len(FEATURE_NAMES),
                      "checks": ["future_value_perturbation", "amount_isolation", "gap_rejection",
                                 "positional_window_and_boundary", "fixed_formula_and_ddof0",
                                 "snapshot_aliases", "invalid_input_rejection"],
                      "data": "synthetic_only_no_labels_or_holdout"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
