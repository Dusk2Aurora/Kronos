"""Offline E00 portfolio adapter consuming outcome-free, precomputed decisions.

signal_at is the candle OPEN boundary (04/12/20 UTC), not the 60-second
availability timestamp. Freqtrade shifts entry signals once to the next open.
The producer owns training-fold isolation and decision provenance; this adapter
never consumes labels, profit, funding, or precomputed entry prices.
"""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


MODES = {"cash", "buy_hold", "fixed_momentum", "constant_half_exposure",
         "vol_target", "ordinary_features_gate"}


def _utc_hour(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("E00 timestamps must be UTC strings ending in Z")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("Invalid E00 timestamp") from exc
    if result.tzinfo != timezone.utc:
        raise ValueError("E00 timestamps require an explicit UTC time")
    if result.minute or result.second or result.microsecond:
        raise ValueError("E00 timestamps must be exact hour boundaries")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in E00 decisions")
        result[key] = value
    return result


class KronosE00(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    startup_candle_count = 256
    process_only_new_candles = True
    minimal_roi = {}
    stoploss = -0.99
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    position_adjustment_enable = False

    def __init__(self, config: dict):
        super().__init__(config)
        path = config.get("e00_decisions_path")
        if not isinstance(path, str) or not path:
            raise ValueError("e00_decisions_path is required")
        payload = json.loads(Path(path).read_text(encoding="utf-8"),
                             object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or set(payload) - {
            "schema_version", "mode", "decisions", "hold_until"
        } or not {"schema_version", "mode", "decisions"} <= set(payload):
            raise ValueError("Invalid E00 top-level schema")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("Unsupported E00 schema_version")
        self.e00_mode = payload["mode"]
        if not isinstance(self.e00_mode, str) or self.e00_mode not in MODES:
            raise ValueError("Unsupported E00 mode")
        if not isinstance(payload["decisions"], list):
            raise ValueError("E00 decisions must be a list")
        self.e00_decisions = {}
        previous = None
        for row in payload["decisions"]:
            if not isinstance(row, dict) or set(row) != {
                "signal_at", "direction", "exposure_fraction"
            }:
                raise ValueError("Invalid E00 decision schema")
            date = _utc_hour(row["signal_at"])
            direction, fraction = row["direction"], row["exposure_fraction"]
            if type(direction) is not int or direction not in (-1, 1):
                raise ValueError("E00 direction must be -1 or 1")
            if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("E00 exposure_fraction must be finite in [0, 1]")
            if previous is not None and date <= previous:
                raise ValueError("E00 decisions must be unique and chronological")
            previous = date
            if self.e00_mode != "buy_hold" and date.hour not in (4, 12, 20):
                raise ValueError("E00 decisions must use configured UTC slots")
            required = {"fixed_momentum": 1, "constant_half_exposure": 0.5,
                        "buy_hold": 1}.get(self.e00_mode)
            if required is not None and fraction != required:
                raise ValueError("Exposure does not match E00 baseline")
            if self.e00_mode == "ordinary_features_gate" and fraction not in (0, 1):
                raise ValueError("Ordinary gate must participate or skip")
            self.e00_decisions[date] = (direction, float(fraction))
        self.e00_hold_until = None
        if self.e00_mode == "cash":
            if self.e00_decisions:
                raise ValueError("Cash must have no entry decisions")
        if self.e00_mode == "buy_hold":
            if len(self.e00_decisions) != 1 or next(iter(self.e00_decisions.values()))[0] != 1:
                raise ValueError("Buy-hold requires one long decision")
            self.e00_hold_until = _utc_hour(payload.get("hold_until"))
            if self.e00_hold_until <= next(iter(self.e00_decisions)) + timedelta(hours=1):
                raise ValueError("Buy-hold exit must follow entry")
        elif payload.get("hold_until") is not None:
            raise ValueError("hold_until is only valid for buy_hold")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dates = pd.to_datetime(dataframe["date"], utc=True)
        values = dataframe[["open", "high", "low", "close", "volume"]]
        valid = (np.isfinite(values).all(axis=1)
                 & values[["open", "high", "low", "close"]].gt(0).all(axis=1)
                 & values["volume"].ge(0))
        gap = dates.diff().eq(pd.Timedelta(hours=1))
        dataframe["history_ready"] = (
            valid.shift(1).rolling(256, min_periods=256).sum().eq(256)
            & gap.shift(1).rolling(255, min_periods=255).sum().eq(255) & gap
        )
        dataframe["momentum_24h"] = dataframe["close"].shift(1) / dataframe["close"].shift(25) - 1
        dataframe["e00_direction"] = dates.map(lambda date: self.e00_decisions.get(date, (0, 0))[0])
        dataframe["e00_exposure"] = dates.map(lambda date: self.e00_decisions.get(date, (0, 0))[1])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        eligible = dataframe["history_ready"] & dataframe["e00_exposure"].gt(0)
        if self.e00_mode not in ("buy_hold", "cash"):
            expected = np.sign(dataframe["momentum_24h"])
            if (eligible & dataframe["e00_direction"].ne(expected)).any():
                raise ValueError("E00 decision reverses the fixed momentum direction")
        for direction, column in ((1, "enter_long"), (-1, "enter_short")):
            mask = eligible & dataframe["e00_direction"].eq(direction)
            dataframe.loc[mask, [column, "enter_tag"]] = [1, "e00_" + self.e00_mode]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None,
                            max_stake: float, leverage: float, entry_tag: str | None,
                            side: str, **kwargs) -> float:
        signal = current_time.astimezone(timezone.utc) - timedelta(hours=1)
        direction, fraction = self.e00_decisions.get(signal, (0, 0))
        if direction != (1 if side == "long" else -1) or leverage != 1:
            return 0.0
        stake = min(proposed_stake * fraction, max_stake)
        if not math.isfinite(stake) or stake <= 0 or (min_stake is not None and stake < min_stake):
            return 0.0
        return stake

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        if self.e00_mode == "buy_hold":
            return "e00_buy_hold_end" if current_time >= self.e00_hold_until else None
        if current_time >= trade.open_date_utc + timedelta(hours=4):
            return "fixed_hold_4h"
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str | None, side: str, **kwargs) -> float:
        return 1.0
