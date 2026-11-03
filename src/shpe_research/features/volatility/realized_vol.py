"""Realized volatility (F3).

Inputs: one symbol's bars sorted by ts (columns close, ts, ts_available).
Output: column ``realized_vol_{window}m``, float64, the sample standard deviation (ddof=1) of the last
`window` completed one-minute log returns (close to close) within the New York session. Not annualised:
the unit is "per one-minute return".
Warm-up: NaN until the session has `window` valid returns (so the first value is on session row `window`).
Timing: the value on a row uses that row's close and earlier rows of the same session only, so it is known
at the row's ts_available.
Gaps: a return needs two bars of the same session exactly one minute apart. A minute without trades makes
the next bar's return invalid; such returns are SKIPPED (not time-scaled, not filled), so the window is the
last `window` valid returns, which may span a gap. The bar right after a gap has no valid return of its
own and gets NaN.
Prices: whatever the bars hold (the contract's adjusted close); no D2 adjustment is applied here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date
from shpe_research.features.base import Feature, register_feature


@register_feature
class RealizedVol(Feature):
    """Std of the last `window` one-minute log returns in the session; see the module docstring."""

    feature_type = "realized_vol"
    requires = ("close", "ts", "ts_available")

    def __init__(self, window: int = 15):
        if window < 2:
            raise ValueError("window must be at least 2 (a sample std needs two returns)")
        self.window = int(window)

    @property
    def name(self) -> str:
        return f"realized_vol_{self.window}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        day = session_date(bars["ts"]).to_numpy()
        ts = bars["ts"]
        log_close = np.log(bars["close"].to_numpy(dtype=float))
        prev_ok = (day[1:] == day[:-1]) & ((ts.iloc[1:].to_numpy() - ts.iloc[:-1].to_numpy()) == np.timedelta64(1, "m"))
        ret = np.full(len(bars), np.nan)
        ret[1:] = np.where(prev_ok, np.diff(log_close), np.nan)
        returns = pd.Series(ret, index=bars.index)
        valid = returns.dropna()
        # Rolling over valid returns only (gaps skipped), restarting each session; trailing window.
        vol = valid.groupby(day[bars.index.get_indexer(valid.index)]).transform(lambda r: r.rolling(self.window).std())
        return vol.reindex(bars.index).astype("float64")
