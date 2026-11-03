"""Distance from session VWAP (F2).

Inputs: contract bars of one symbol sorted by ts (``close``, ``vwap``, ``volume``, ``high``, ``low``,
``ts_available``). Prices are as delivered in the bars (no further adjustment).
Output column: ``vwap_distance`` (float64) = close / session VWAP - 1.
Warm-up: none beyond the first bar with volume in a session; before that, and whenever cumulative volume
is zero, the value is NaN.
Timing: the value on a row uses that bar and earlier bars of the same session only, so it is known at the
row's ``ts_available``. Sessions are New York dates of ``ts_available - 1 minute`` (as in PastReturn), so
the final bar of a session belongs to that session. Missing minutes stay missing: they add nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date
from shpe_research.features.base import Feature, register_feature


@register_feature
class VwapDistance(Feature):
    """close / session VWAP - 1, with the VWAP accumulated from the session open to this bar inclusive.

    Session VWAP = sum(bar vwap * volume) / sum(volume). A bar without a vwap uses (high + low + close) / 3
    as its price (documented fallback). Bars with unusable price or volume contribute nothing. Zero
    cumulative volume -> NaN.
    """

    feature_type = "vwap_distance"
    requires = ("close", "vwap", "volume", "high", "low", "ts_available")

    @property
    def name(self) -> str:
        return "vwap_distance"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        day = session_date(bars["ts_available"] - pd.Timedelta(minutes=1)).to_numpy()
        close = bars["close"].to_numpy(dtype=float)
        typical = (bars["high"].to_numpy(dtype=float) + bars["low"].to_numpy(dtype=float) + close) / 3.0
        price = np.where(np.isnan(bars["vwap"].to_numpy(dtype=float)), typical, bars["vwap"].to_numpy(dtype=float))
        volume = bars["volume"].to_numpy(dtype=float)
        usable = np.isfinite(price) & np.isfinite(volume)
        weight = pd.Series(np.where(usable, volume, 0.0), index=bars.index)
        notional = pd.Series(np.where(usable, price * volume, 0.0), index=bars.index)
        cum_w = weight.groupby(day).cumsum().to_numpy()
        cum_pv = notional.groupby(day).cumsum().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(cum_w > 0, cum_pv / cum_w, np.nan)
            out = close / vwap - 1.0
        return pd.Series(out, index=bars.index, dtype="float64")
