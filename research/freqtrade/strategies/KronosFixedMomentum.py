"""Fixed, causal momentum baseline for the hourly futures research backtest.

Rows are candle OPEN times. At UTC 04/12/20, only the preceding 256
completed candles are used. The latest close is available 60 seconds after
the decision boundary. Freqtrade shifts this row's signal once, so entry is
at UTC 05/13/21 (strictly after that decision), never at the boundary.
The scheduled 4h exits are UTC 09/17/01 on the corresponding or next day.
This alignment is designed for backtesting, not a live execution scheduler.
Configure max_open_trades=1 and futures mode in the backtest configuration.
The emergency 99% stop and exchange liquidation may interrupt the 4h hold;
report those exits separately instead of treating them as scheduled exits.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class KronosFixedMomentum(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    startup_candle_count = 256
    process_only_new_candles = True
    minimal_roi = {}
    stoploss = -0.99
    trailing_stop = False
    use_exit_signal = True  # Required by Freqtrade to invoke custom_exit.
    exit_profit_only = False
    position_adjustment_enable = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dates = pd.to_datetime(dataframe["date"], utc=True)
        values = dataframe[["open", "high", "low", "close", "volume"]]
        valid_bar = (
            np.isfinite(values).all(axis=1)
            & values[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & values["volume"].ge(0)
        )
        valid_history = valid_bar.shift(1).rolling(256, min_periods=256).sum().eq(256)
        hourly_gap = dates.diff().eq(pd.Timedelta(hours=1))
        continuous_history = (
            hourly_gap.shift(1).rolling(255, min_periods=255).sum().eq(255)
            & hourly_gap
        )
        dataframe["history_ready"] = valid_history & continuous_history
        dataframe["momentum_24h"] = dataframe["close"].shift(1) / dataframe["close"].shift(25) - 1.0
        dataframe["decision_slot"] = (
            dates.dt.hour.isin([4, 12, 20])
            & dates.dt.minute.eq(0)
            & dates.dt.second.eq(0)
            & dates.dt.microsecond.eq(0)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        eligible = dataframe["history_ready"] & dataframe["decision_slot"]
        long_signal = eligible & dataframe["momentum_24h"].gt(0)
        short_signal = eligible & dataframe["momentum_24h"].lt(0)
        dataframe.loc[long_signal, ["enter_long", "enter_tag"]] = [1, "momentum_24h_long"]
        dataframe.loc[short_signal, ["enter_short", "enter_tag"]] = [1, "momentum_24h_short"]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(
        self, pair: str, trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs,
    ):
        if current_time >= trade.open_date_utc + timedelta(hours=4):
            return "fixed_hold_4h"
        return None

    def leverage(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_leverage: float, max_leverage: float,
        entry_tag: str | None, side: str, **kwargs,
    ) -> float:
        return 1.0
