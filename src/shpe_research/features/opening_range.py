"""TODO(F4): opening-range features.  Level: beginner-intermediate.

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

Done when:
    - tests/todo/test_features_todo.py::test_opening_range passes, including
      the worked example (H=100, L=98, close 100.10 -> breakout 0.001).
    - assert_no_lookahead passes; registered and imported in features/__init__.py.
"""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature


class OpeningRangeWidth(Feature):
    feature_type = "or_width"
    requires = ("open", "high", "low", "ts", "ts_available")

    def __init__(self, range_minutes: int = 15):
        self.range_minutes = int(range_minutes)

    @property
    def name(self) -> str:
        return f"or_width_{self.range_minutes}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("TODO(F4): opening-range width")


class OpeningRangeBreakoutDistance(OpeningRangeWidth):
    feature_type = "or_breakout_dist"
    requires = ("close", "high", "low", "ts", "ts_available")

    @property
    def name(self) -> str:
        return f"or_breakout_dist_{self.range_minutes}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("TODO(F4): breakout distance above the opening-range high")
