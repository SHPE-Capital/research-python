"""TODO(F3): realized volatility.  Level: beginner.

Definition:
    Sample standard deviation of the last `window` completed one-minute log
    returns (close to close), within the session. ORB plan default: window=15.

Contract:
    - Feature subclass, feature_type "realized_vol", name f"realized_vol_{window}m".
    - A "one-minute return" needs two bars exactly one minute apart; a gap
      (minute without trades) must not be treated as a one-minute return.
      Document how you handle gaps (skip them, or use time-scaled returns).
    - Fewer than `window` returns available in the session -> NaN.

Done when:
    - tests/todo/test_features_todo.py::test_realized_vol passes.
    - assert_no_lookahead passes; registered and imported in features/__init__.py.

Stretch (also unlocks volatility-scaled stops in L2): a daily-horizon
volatility forecast (e.g. HAR model on daily realized variance) as a separate feature.
"""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature


class RealizedVol(Feature):
    feature_type = "realized_vol"
    requires = ("close", "ts", "ts_available")

    def __init__(self, window: int = 15):
        self.window = int(window)

    @property
    def name(self) -> str:
        return f"realized_vol_{self.window}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("TODO(F3): realized volatility")
