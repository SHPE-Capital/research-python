"""TODO(F5): regime features (trending vs. mean-reverting conditions).  Level: intermediate.

Regimes enter models as continuous inputs, not as a separate on/off classifier
(see docs/tasks.md, R-track). Reversal and momentum teams share these.

Definitions (each over the last `window` one-minute returns, within the session
unless noted):
    variance_ratio   Var(sum of q consecutive returns) / (q * Var(1-minute return)),
                     q = 5. > 1 suggests trending, < 1 mean reversion.
    autocorr         lag-1 autocorrelation of one-minute returns.
    efficiency       |sum of returns| / sum of |returns| (Kaufman efficiency ratio).

Contract:
    - One Feature subclass per measure, names f"{measure}_{window}m".
    - NaN when fewer than `window` returns are available.
    - Gaps handled the same way as F3 (coordinate with the F3 owner).

Done when:
    - tests/todo/test_features_todo.py::test_regime_features passes: on fixture
      bars with signal_strength=0.5 (built-in mean reversion) the mean
      variance_ratio is below 1 and the mean autocorr is negative.
    - assert_no_lookahead passes; registered and imported in features/__init__.py.

Stretch: a multi-day regime feature from daily bars (e.g. 20-day realized vol
percentile), computed only from sessions before the decision.
"""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature


class RegimeFeature(Feature):
    feature_type = "regime"
    requires = ("close", "ts", "ts_available")
    MEASURES = ("variance_ratio", "autocorr", "efficiency")

    def __init__(self, measure: str, window: int = 60):
        if measure not in self.MEASURES:
            raise ValueError(f"measure must be one of {self.MEASURES}")
        self.measure = measure
        self.window = int(window)

    @property
    def name(self) -> str:
        return f"{self.measure}_{self.window}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError(f"TODO(F5): {self.measure}")
