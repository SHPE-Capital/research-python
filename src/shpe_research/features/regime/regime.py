"""Regime features (trending vs. mean-reverting conditions).  Level: intermediate.

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

Implementation notes:
    - Inputs: close, ts, ts_available. Output columns f"{measure}_{window}m", float64.
      Registry key "regime"; the column names are read by T3's regime filter.
    - Timing: the value on a row uses that row's close and earlier rows only, so it is known
      at that bar's ts_available.
    - Returns are log close-to-close. A one-minute return exists only when the previous bar is in the
      same session (New York date) and exactly one minute earlier. A gap (missing minute) or a session
      start gives no return, and any window containing one is NaN (nothing is filled or skipped).
      Warm-up is therefore `window` returns = `window` + 1 consecutive bars; rows before are NaN.
    - variance_ratio uses overlapping q=5 sums inside the window (window - q + 1 of them) and sample
      variances (ddof=1); window must be >= 2q. Zero variance gives NaN.
    - autocorr is the lag-1 correlation over the window - 1 return pairs inside the window.
    - efficiency is |sum r| / sum |r| (0 to 1); NaN when every return is zero.

Stretch (not implemented): a multi-day regime feature from daily bars (e.g. 20-day realized vol
percentile), computed only from sessions before the decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date
from shpe_research.features.base import Feature, register_feature

VARIANCE_RATIO_Q = 5


@register_feature
class RegimeFeature(Feature):
    feature_type = "regime"
    requires = ("close", "ts", "ts_available")
    MEASURES = ("variance_ratio", "autocorr", "efficiency")

    def __init__(self, measure: str, window: int = 60):
        if measure not in self.MEASURES:
            raise ValueError(f"measure must be one of {self.MEASURES}")
        self.measure = measure
        self.window = int(window)
        if self.window < 3:
            raise ValueError("window must be at least 3")
        if measure == "variance_ratio" and self.window < 2 * VARIANCE_RATIO_Q:
            raise ValueError(f"variance_ratio needs window >= {2 * VARIANCE_RATIO_Q}")

    @property
    def name(self) -> str:
        return f"{self.measure}_{self.window}m"

    def _returns(self, bars: pd.DataFrame) -> pd.Series:
        """One-minute log returns; NaN at session starts and after gaps."""
        ts = bars["ts"].reset_index(drop=True)
        day = session_date(ts)
        valid = (ts.diff() == pd.Timedelta(minutes=1)) & (day == day.shift(1))
        r = np.log(bars["close"].reset_index(drop=True).astype(float)).diff()
        return r.where(valid)

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        """One value per bar, known at that bar's ts_available; NaN until `window` valid returns."""
        w = self.window
        r = self._returns(bars)
        if self.measure == "efficiency":
            net = r.rolling(w, min_periods=w).sum().abs()
            total = r.abs().rolling(w, min_periods=w).sum()
            out = net / total.where(total > 0)
        elif self.measure == "autocorr":
            out = r.rolling(w - 1, min_periods=w - 1).corr(r.shift(1))
            # a window of w valid returns needs the oldest one as the lag of the first pair
            out = out.where(r.rolling(w, min_periods=w).count() == w)
        else:
            q = VARIANCE_RATIO_Q
            sums = r.rolling(q, min_periods=q).sum()
            var_sum = sums.rolling(w - q + 1, min_periods=w - q + 1).var()
            var_r = r.rolling(w, min_periods=w).var()
            out = var_sum / (q * var_r.where(var_r > 0))
        out = out.replace([np.inf, -np.inf], np.nan).astype("float64")
        return pd.Series(out.to_numpy(), index=bars.index, name=self.name)
