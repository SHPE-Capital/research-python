"""Example features (C0). Copy this file's pattern for new features.

Each feature is time-based, not row-based: bars can be missing for minutes
without trades, so "5 rows ago" is not "5 minutes ago".
"""

from __future__ import annotations

import pandas as pd

from shpe_research.data.calendar import get_calendar, session_date
from shpe_research.features.base import Feature, register_feature


@register_feature
class MinuteOfSession(Feature):
    """Minutes from the session open to the bar's ts_available (1 for the first regular bar)."""

    feature_type = "minute_of_day"
    requires = ("ts_available",)

    @property
    def name(self) -> str:
        return "minute_of_day"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        return minutes_since_open(bars["ts_available"])


def minutes_since_open(ts: pd.Series) -> pd.Series:
    """Minutes between each timestamp and its session's open (NaN on non-session days)."""
    cal = get_calendar()
    day = session_date(ts - pd.Timedelta(minutes=1))
    opens = cal.schedule(day.min(), day.max())["open"].reindex(day.to_numpy())
    return pd.Series((ts.to_numpy() - opens.to_numpy()) / pd.Timedelta(minutes=1), index=ts.index)
