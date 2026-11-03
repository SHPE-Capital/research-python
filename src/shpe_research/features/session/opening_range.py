"""Opening-range features (F4).  Level: beginner-intermediate.

Definitions (Christopher's ORB plan, sections 2 and 5):
    Opening range = regular bars starting 09:30 through 09:44 New York time
    (`range_minutes`=15). H = their max high, L = their min low.
    The range is known only once the 09:44 bar completes (ts_available 09:45).

    or_width            (H - L) / session opening price (first regular bar's open)
    or_breakout_dist    (close - H) / H

Contract:
    - Two Feature subclasses (or one parametrized by `kind`), names
      f"or_width_{range_minutes}m" and f"or_breakout_dist_{range_minutes}m".
    - NaN on every bar whose ts_available is before the range completes, and on
      sessions with no bars inside the range window.
    - Early-close sessions still use 09:30-09:44.

Timing and details:
    - Inputs: one symbol's contract bars (`ts` bar start, `ts_available` = ts + 1 minute, UTC).
      Prices are used as given (no split adjustment is applied here).
    - Range bars are those with `ts` from 09:30 (inclusive) to 09:30 + range_minutes (exclusive),
      New York time, grouped by the New York date of `ts`. A missing minute is simply absent from
      the range; nothing is filled. The range is complete at ts_available 09:30 + range_minutes.
    - A row gets a value only when its ts_available >= that completion time. H and L are running
      max/min over range bars up to the row, so they are final from that row on.
    - Output is float64, NaN during warm-up (rows before the range completes, pre-market rows,
      sessions with no range bar). Registry keys: or_width, or_breakout_dist.
"""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature, register_feature

_OPEN_MINUTE = 9 * 60 + 30
_NY = "America/New_York"


def _opening_range(bars: pd.DataFrame, range_minutes: int) -> pd.DataFrame:
    """Per row: session `open`, range `high`/`low`, NaN until the range is complete at the row's ts_available."""
    ts = bars["ts"].dt.tz_convert(_NY)
    avail = bars["ts_available"].dt.tz_convert(_NY)
    day = ts.dt.tz_localize(None).dt.normalize()
    avail_day = avail.dt.tz_localize(None).dt.normalize()
    minute = ts.dt.hour * 60 + ts.dt.minute
    avail_minute = avail.dt.hour * 60 + avail.dt.minute
    in_range = (minute >= _OPEN_MINUTE) & (minute < _OPEN_MINUTE + range_minutes)
    complete = (avail_day > day) | ((avail_day == day) & (avail_minute >= _OPEN_MINUTE + range_minutes))

    high = bars["high"].where(in_range) if "high" in bars else None
    low = bars["low"].where(in_range) if "low" in bars else None
    out = pd.DataFrame(index=bars.index, dtype="float64")
    if "open" in bars:
        out["open"] = bars["open"].where(in_range).groupby(day).transform("first").where(complete)
    out["high"] = high.groupby(day).cummax().groupby(day).ffill().where(complete)
    out["low"] = low.groupby(day).cummin().groupby(day).ffill().where(complete)
    return out.astype("float64")


@register_feature
class OpeningRangeWidth(Feature):
    """(H - L) / session opening price, for the opening range of `range_minutes` (5, 15, 30, ...).

    Opening price is the open of the first range bar (normally the 09:30 bar). Known from the range
    completion (ts_available 09:30 + range_minutes); NaN before that. Warm-up: range_minutes rows.
    """

    feature_type = "or_width"
    requires = ("open", "high", "low", "ts", "ts_available")

    def __init__(self, range_minutes: int = 15):
        if int(range_minutes) <= 0:
            raise ValueError("range_minutes must be positive")
        self.range_minutes = int(range_minutes)

    @property
    def name(self) -> str:
        return f"or_width_{self.range_minutes}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        rng = _opening_range(bars, self.range_minutes)
        return ((rng["high"] - rng["low"]) / rng["open"]).astype("float64")


@register_feature
class OpeningRangeBreakoutDistance(OpeningRangeWidth):
    """(close - H) / H against the opening-range high; negative inside or below the range.

    Uses the bar's close, known at its ts_available, and H once the range is complete. NaN before.
    """

    feature_type = "or_breakout_dist"
    requires = ("close", "high", "low", "ts", "ts_available")

    @property
    def name(self) -> str:
        return f"or_breakout_dist_{self.range_minutes}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        rng = _opening_range(bars, self.range_minutes)
        return ((bars["close"] - rng["high"]) / rng["high"]).astype("float64")
