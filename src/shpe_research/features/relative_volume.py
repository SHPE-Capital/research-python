"""TODO(F1): relative volume and the intraday volume curve.  Level: beginner-intermediate.

Definition (Christopher's ORB plan, section 5):
    RVOL at a bar = cumulative volume so far today (through this bar)
                    / mean cumulative volume at the same minute of the session
                      over the previous `lookback_sessions` valid sessions.
    Example: 150,000 shares by 09:50 against a prior same-time average of
    100,000 gives RVOL = 1.5.

Contract:
    - Feature subclass, feature_type "relative_volume", name f"rvol_{lookback_sessions}d".
    - Only PREVIOUS sessions form the reference; today's later bars must not.
    - Fewer than `lookback_sessions` valid prior sessions -> NaN (never a guess).
    - Same-minute alignment uses minutes since the open, so early-close days
      compare like with like (see features/returns.py:minutes_since_open).
    - Missing minutes have zero volume for the cumulative sum, but they are
      not invented bars.

Done when:
    - tests/todo/test_features_todo.py::test_relative_volume passes (remove its skip).
    - assert_no_lookahead passes on fixture bars spanning > lookback_sessions sessions.
    - Registered with @register_feature and imported in features/__init__.py.

Stretch: expose the volume curve itself (expected cumulative volume by minute)
as a helper; R1 and a "best trading window" study both need it.
"""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature


class RelativeVolume(Feature):
    feature_type = "relative_volume"
    requires = ("volume", "ts", "ts_available")

    def __init__(self, lookback_sessions: int = 20):
        self.lookback_sessions = int(lookback_sessions)

    @property
    def name(self) -> str:
        return f"rvol_{self.lookback_sessions}d"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("TODO(F1): relative volume")
