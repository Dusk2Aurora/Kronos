"""Causal 33-feature risk contract plus two non-model R0 forecasts.

The caller supplies development-only candles. No source loader or label access
is provided here. R0 columns must not be included in a fitted feature family.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASELINES = ROOT / "research/baselines"
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ordinary_features import historical_window as oldhistorical_window
from ordinary_features_v2 import FEATURE_NAMES as ORDINARY_NAMES
from ordinary_features_v2 import features_for_opportunity as oldfeatures
from research.frozen.encoder import CUTOFF, prepare_window

HOURS = (4, 24, 168)
HAR_NAMES = tuple(f"log_rv4scaled_{h}h" for h in HOURS)
RISK_NAMES = (
    *HAR_NAMES,
    *(f"log_mean_abs_return_{h}h" for h in HOURS),
    *(f"log_parkinson4scaled_{h}h" for h in HOURS),
    "log_rv_ratio_4h_24h", "log_rv_ratio_24h_168h", "tail_ratio_24h",
    "log_volume_ratio_4h_24h", "log_amount_ratio_4h_24h",
)
FEATURE_NAMES = (*ORDINARY_NAMES, *RISK_NAMES)
R0_NAMES = ("r0_persistence", "r0_ewma")
EPSILON = 1e-12


def _integer(value):
    """Accept CSV integer strings without silently truncating bad offsets."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean positional offset")
    result = int(value)
    if str(value).strip() != str(result) and float(value) != result:
        raise ValueError("Noninteger positional offset")
    return result


def build_features(candles, opportunities):
    """Return float64 [ordinary19, risk14, R0x2], indexed by opportunity_id.

    FEATURE_NAMES explicitly excludes R0 columns. Stored CSV offsets are checked
    by the original window implementation; encoder validation independently
    checks completed bars, availability, amount, and timestamp selection.
    """
    times = pd.to_datetime(candles["bar_open_at"], utc=True)
    if (times >= CUTOFF).any():
        raise ValueError("Caller must supply development-only candles")
    source = candles.copy()
    source["bar_open_at"] = times
    ordinary_source = source.loc[:, ["bar_open_at", "open", "high", "low",
                                     "close", "volume", "amount"]]
    rows, ids = [], []
    for _, opportunity in opportunities.iterrows():
        opportunity = opportunity.to_dict()
        for field in ("history_start_row", "history_end_row_exclusive"):
            opportunity[field] = _integer(opportunity[field])
        prepare_window(source, opportunity)
        window = oldhistorical_window(
            ordinary_source, decision_boundary=opportunity["decision_boundary_at"],
            history_start_row=opportunity["history_start_row"],
            history_end_row_exclusive=opportunity["history_end_row_exclusive"],
        )
        values = oldfeatures(ordinary_source, opportunity)
        returns = np.diff(np.log(window.close.to_numpy(dtype=np.float64)))
        squared = returns ** 2
        parkinson = np.log(window.high.to_numpy(dtype=np.float64)
                           / window.low.to_numpy(dtype=np.float64)) ** 2 / (4 * np.log(2))
        for h in HOURS:
            values[f"log_rv4scaled_{h}h"] = float(np.log(max(4 * squared[-h:].mean(), EPSILON)))
            values[f"log_mean_abs_return_{h}h"] = float(np.log(max(np.abs(returns[-h:]).mean(), EPSILON)))
            values[f"log_parkinson4scaled_{h}h"] = float(np.log(max(4 * parkinson[-h:].mean(), EPSILON)))
        values["log_rv_ratio_4h_24h"] = values[HAR_NAMES[0]] - values[HAR_NAMES[1]]
        values["log_rv_ratio_24h_168h"] = values[HAR_NAMES[1]] - values[HAR_NAMES[2]]
        values["tail_ratio_24h"] = float(np.max(np.abs(returns[-24:]))
                                          / np.sqrt(max(squared[-24:].mean(), EPSILON)))
        for field in ("volume", "amount"):
            history = window[field].to_numpy(dtype=np.float64)
            values[f"log_{field}_ratio_4h_24h"] = float(np.log(
                (history[-4:].mean() + EPSILON) / (history[-24:].mean() + EPSILON)))
        weights = .97 ** np.arange(254, -1, -1, dtype=np.float64)
        values["r0_persistence"] = float(squared[-4:].sum())
        values["r0_ewma"] = float(4 * np.dot(weights / weights.sum(), squared))
        rows.append(values)
        ids.append(str(opportunity["opportunity_id"]))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate opportunity IDs")
    frame = pd.DataFrame(rows, index=pd.Index(ids, name="opportunity_id"),
                         columns=(*FEATURE_NAMES, *R0_NAMES), dtype=np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite risk features")
    return frame
