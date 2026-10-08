"""Fixed causal OHLCVA features; no fitting, labels or funding inputs.

Select only the seven input columns before calling this module. Snapshot aliases
are accepted explicitly, but raw contract volume is never treated as BTC volume.
Dates are UTC bar-open timestamps. A boundary at 04:00 consumes bars ending at
04:00; its last input bar opens at 03:00. The caller schedules the decision at
boundary + the configured 60-second availability delay.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd

LOOKBACK_BARS = 256
HOUR = pd.Timedelta(hours=1)
RETURN_HOURS = (1, 4, 12, 24, 72, 168)
VOLATILITY_HOURS = (24, 72, 168)
INPUT_FIELDS = ("date", "open", "high", "low", "close", "volume", "amount")
ALIASES = {"bar_open_at": "date", "volCcy": "volume", "volCcyQuote": "amount"}
# Insertion order is the model-column contract. No annualization or percentages.
FEATURE_UNITS = {
    **{f"log_return_{hours}h": "natural_log_ratio" for hours in RETURN_HOURS},
    **{f"hourly_log_return_std_{hours}h": "natural_log_ratio_ddof0"
       for hours in VOLATILITY_HOURS},
    "mean_range_fraction_24h": "fraction_of_bar_close",
    "volume_mean_ratio_24h_168h": "dimensionless_mean_ratio",
    "amount_mean_ratio_24h_168h": "dimensionless_mean_ratio",
    "momentum_direction_24h": "sign_minus1_zero_plus1",
}
FEATURE_NAMES = tuple(FEATURE_UNITS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    _require(not pd.isna(result) and result.tzinfo is not None
             and result.utcoffset().total_seconds() == 0,
             f"Explicit UTC timestamp required: {value}")
    return result.tz_convert("UTC")


def _columns(candles: pd.DataFrame) -> pd.DataFrame:
    _require(candles.columns.is_unique, "Duplicate input columns")
    renamed = candles.rename(columns=ALIASES)
    _require(renamed.columns.is_unique, "Ambiguous canonical/alias columns")
    _require(set(renamed.columns) == set(INPUT_FIELDS),
             "Input must contain only date/open/high/low/close/volume/amount "
             "(or documented aliases); select source columns explicitly")
    return renamed.loc[:, INPUT_FIELDS]


def historical_window(
    candles: pd.DataFrame,
    *,
    decision_boundary: object,
    history_start_row: int | None = None,
    history_end_row_exclusive: int | None = None,
) -> pd.DataFrame:
    """Return a validated copy of exactly 256 completed hourly bars.

    Positional indices refer to the supplied original snapshot ordering, even
    when its pandas index is nondefault. Supply both indices or neither. Without
    indices, find the 256 timestamps preceding the explicit boundary; future
    OHLCVA values are never validated or consumed. With indices, timestamps must
    match the boundary exactly, so a wrong slice cannot silently pass.
    """
    source = _columns(candles)
    boundary = _utc(decision_boundary)
    _require(boundary == boundary.floor("h"), "Decision boundary must be on the hour")
    expected = pd.date_range(boundary - LOOKBACK_BARS * HOUR,
                             periods=LOOKBACK_BARS, freq="h")
    indexed = history_start_row is not None or history_end_row_exclusive is not None
    if indexed:
        _require(isinstance(history_start_row, Integral)
                 and not isinstance(history_start_row, bool)
                 and isinstance(history_end_row_exclusive, Integral)
                 and not isinstance(history_end_row_exclusive, bool),
                 "Both positional indices must be integers")
        _require(0 <= history_start_row < history_end_row_exclusive <= len(source)
                 and history_end_row_exclusive - history_start_row == LOOKBACK_BARS,
                 "Window indices must select exactly 256 existing rows")
        window = source.iloc[history_start_row:history_end_row_exclusive].copy()
        dates = pd.DatetimeIndex([_utc(value) for value in window.date])
    else:
        dates = pd.DatetimeIndex([_utc(value) for value in source.date])
        _require(dates.is_unique and dates.is_monotonic_increasing,
                 "Source timestamps must be unique and ordered")
        selected = (dates >= expected[0]) & (dates < boundary)
        window = source.loc[selected].copy()
        dates = dates[selected]
    _require(dates.equals(expected), "Missing, unordered, or noncontinuous historical window")
    window["date"] = dates
    try:
        values = window.loc[:, INPUT_FIELDS[1:]].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("OHLCVA must be numeric") from error
    _require(np.isfinite(values).all(), "Nonfinite historical OHLCVA")
    _require((values[:, :4] > 0).all(), "Historical prices must be positive")
    _require((values[:, 4:] >= 0).all(), "Historical volume/amount must be nonnegative")
    opened, high, low, close = values[:, :4].T
    _require(((high >= np.maximum(opened, close))
              & (low <= np.minimum(opened, close)) & (high >= low)).all(),
             "Invalid historical OHLC bounds")
    window.loc[:, INPUT_FIELDS[1:]] = values
    return window.reset_index(drop=True)


def ordinary_features(
    candles: pd.DataFrame,
    *,
    decision_boundary: object,
    history_start_row: int | None = None,
    history_end_row_exclusive: int | None = None,
) -> dict[str, float]:
    """Compute the fixed 13-feature contract from a validated causal window.

    Return(h) = log(last close / close h bars earlier). Std(h) uses exactly h
    hourly log returns, hence h+1 historical closes, with numpy.std(ddof=0).
    Volume/amount ratios divide the last-24-bar mean by last-168-bar mean; a zero
    denominator rejects the window rather than inventing an epsilon or filling.
    Amount must be actual traded USDT turnover, never price times volume.
    """
    window = historical_window(
        candles, decision_boundary=decision_boundary,
        history_start_row=history_start_row,
        history_end_row_exclusive=history_end_row_exclusive,
    )
    close = window.close.to_numpy(dtype=float)
    log_close = np.log(close)
    returns = np.diff(log_close)
    result = {f"log_return_{hours}h": float(log_close[-1] - log_close[-1-hours])
              for hours in RETURN_HOURS}
    result.update({f"hourly_log_return_std_{hours}h": float(np.std(returns[-hours:], ddof=0))
                   for hours in VOLATILITY_HOURS})
    result["mean_range_fraction_24h"] = float(np.mean(
        (window.high.to_numpy(dtype=float)[-24:] - window.low.to_numpy(dtype=float)[-24:])
        / close[-24:]))
    for field in ("volume", "amount"):
        values = window[field].to_numpy(dtype=float)
        denominator = float(np.mean(values[-168:]))
        _require(denominator > 0, f"Zero {field} 168h mean cannot define ratio")
        result[f"{field}_mean_ratio_24h_168h"] = float(np.mean(values[-24:]) / denominator)
    result["momentum_direction_24h"] = float(np.sign(result["log_return_24h"]))
    _require(tuple(result) == FEATURE_NAMES and np.isfinite(list(result.values())).all(),
             "Nonfinite or unexpected feature output")
    return result


def features_for_opportunity(candles: pd.DataFrame, opportunity: Mapping) -> dict[str, float]:
    """Consume only the three opportunity window-reference fields, never outcomes."""
    return ordinary_features(
        candles, decision_boundary=opportunity["decision_boundary_at"],
        history_start_row=opportunity["history_start_row"],
        history_end_row_exclusive=opportunity["history_end_row_exclusive"],
    )
