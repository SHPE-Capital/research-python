"""Close-to-close features on daily bars (C1e). Rows are sessions, so shifts are in sessions."""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature, register_feature


@register_feature
class PastReturnDays(Feature):
    """close_d / close_{d-days} - 1 on daily bars.

    Known at session d's close (`ts_available`); uses only sessions <= d. NaN for the first `days` sessions.
    """

    feature_type = "past_return_days"

    def __init__(self, days: int):
        if days <= 0:
            raise ValueError("days must be positive")
        self.days = int(days)

    @property
    def name(self) -> str:
        return f"past_return_{self.days}d"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        return close / close.shift(self.days) - 1.0


@register_feature
class Momentum121(Feature):
    """close_{d-skip} / close_{d-lookback} - 1 on daily bars (12-1 momentum with the defaults).

    Skips the most recent `skip_days` sessions. Known at session d's close; uses only sessions <= d - skip_days.
    """

    feature_type = "momentum_12_1"

    def __init__(self, skip_days: int = 21, lookback_days: int = 252):
        if skip_days < 0 or lookback_days <= skip_days:
            raise ValueError("need 0 <= skip_days < lookback_days")
        self.skip_days = int(skip_days)
        self.lookback_days = int(lookback_days)

    @property
    def name(self) -> str:
        return "momentum_12_1" if (self.skip_days, self.lookback_days) == (21, 252) else (
            f"momentum_{self.lookback_days}_{self.skip_days}d"
        )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        return close.shift(self.skip_days) / close.shift(self.lookback_days) - 1.0
