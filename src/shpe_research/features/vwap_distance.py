"""TODO(F2): distance from session VWAP.  Level: beginner.

Definition:
    session VWAP at a bar = sum(bar vwap * bar volume) / sum(bar volume) over
    the session's bars up to and including this bar. Bars without a vwap use
    their (high + low + close) / 3 as a documented fallback.
    feature = close / session VWAP - 1

Contract:
    - Feature subclass, feature_type "vwap_distance", name "vwap_distance".
    - Resets every session (group by session_date as in features/returns.py).
    - Zero cumulative volume -> NaN.

Done when:
    - tests/todo/test_features_todo.py::test_vwap_distance passes.
    - assert_no_lookahead passes; registered and imported in features/__init__.py.
"""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature


class VwapDistance(Feature):
    feature_type = "vwap_distance"
    requires = ("close", "vwap", "volume", "high", "low", "ts_available")

    @property
    def name(self) -> str:
        return "vwap_distance"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("TODO(F2): VWAP distance")
