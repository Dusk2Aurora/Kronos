"""E00R's additive causal 19-feature contract; preserve original 13 unchanged."""
from ordinary_features import FEATURE_NAMES as ORIGINAL_FEATURE_NAMES
from ordinary_features import FEATURE_UNITS as ORIGINAL_FEATURE_UNITS
from ordinary_features import RETURN_HOURS, features_for_opportunity as original_features

FEATURE_UNITS = {**ORIGINAL_FEATURE_UNITS,
                 **{f"direction_aligned_log_return_{h}h": "direction_times_natural_log_ratio"
                    for h in RETURN_HOURS}}
FEATURE_NAMES = tuple(FEATURE_UNITS)
FEATURE_SETS = {"ridge13": ORIGINAL_FEATURE_NAMES, "ridge19": FEATURE_NAMES}


def features_for_opportunity(candles, opportunity):
    values = original_features(candles, opportunity)
    direction = values["momentum_direction_24h"]
    values.update({f"direction_aligned_log_return_{h}h": direction * values[f"log_return_{h}h"]
                   for h in RETURN_HOURS})
    return values
