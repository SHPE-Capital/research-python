"""Order-flow features (F6): signed volume ratio from quote/trade minute summaries.  Level: intermediate.

Inputs are the per-minute tick summaries produced by D3
(data/ticks.py:minute_summaries), joined onto bars by (symbol, ts). Bars alone
cannot produce these.

Definitions:
    displayed_imbalance   mean (bid_size - ask_size) / (bid_size + ask_size) over the minute
    spread_bps            mean quoted spread in basis points over the minute
    signed_volume_ratio   (buy_volume - sell_volume) / volume over the last `window` minutes
    ofi                   order-flow imbalance (Cont, Kukanov & Stoikov 2014) summed over
                          `window` minutes. Needs quote-level changes, so compute it from
                          raw quotes in data/ticks.py and add it to the minute summary
                          first (coordinate with the D3 owner).

Contract:
    - Feature subclasses that require the summary columns (see TICK_SUMMARY_COLUMNS
      in contracts/ticks.py) and return NaN where a minute has no summary.
    - A minute's summary is available at the minute's end, like a bar.

Implemented: signed_volume_ratio (SignedVolumeRatio). displayed_imbalance, spread_bps and ofi are not
implemented; ofi needs a D3 summary column that does not exist yet. The class is not yet registered with
@register_feature: C1k's check-kit test runs every registered feature on fixture bars that lack the
tick-summary columns.

signed_volume_ratio:
    Inputs: one symbol's bars with the D3 summary columns buy_volume, sell_volume, trade_volume joined
    on (symbol, ts), plus ts_available (minute end, UTC). Output: float64 column
    f"signed_volume_ratio_{window}m" = sum(buy - sell) / sum(trade_volume) over summaries whose
    ts_available lies in (t - window minutes, t], t being the row's ts_available. The window is
    time-based and trailing, so a missing minute is simply absent; nothing is filled.
    Timing: known at the row's ts_available (the minute's end), like a bar.
    NaN when: the row's own minute has no summary (NaN buy/sell/volume); the session has fewer
    than `window` minutes of regular-session history (window start before 09:30 New York);
    the window's volume is zero. Never carries across sessions. Warm-up: first `window` - 1 rows
    of a full session (the row with ts_available 09:30 + window is the first value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import NY, session_date
from shpe_research.features.base import Feature

_OPEN_MINUTE = 9 * 60 + 30


class SignedVolumeRatio(Feature):
    """Net signed trade volume over the last `window` minutes divided by volume; see module docstring."""

    feature_type = "signed_volume_ratio"
    requires = ("buy_volume", "sell_volume", "trade_volume", "ts_available")

    def __init__(self, window: int = 5):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)

    @property
    def name(self) -> str:
        return f"signed_volume_ratio_{self.window}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        avail = bars["ts_available"]
        signed = pd.Series((bars["buy_volume"] - bars["sell_volume"]).to_numpy(dtype=float), index=avail)
        volume = pd.Series(bars["trade_volume"].to_numpy(dtype=float), index=avail)
        rolling = f"{self.window}min"
        num = signed.rolling(rolling, closed="right", min_periods=1).sum().to_numpy()
        den = volume.rolling(rolling, closed="right", min_periods=1).sum().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(den > 0, num / den, np.nan)

        # Warm-up: the window must start inside the regular session (09:30 New York or later).
        start = (avail - pd.Timedelta(minutes=self.window)).dt.tz_convert(NY)
        start_minute = (start.dt.hour * 60 + start.dt.minute).to_numpy()
        warm = (start_minute >= _OPEN_MINUTE) & (
            session_date(start.dt.tz_convert("UTC")).to_numpy()
            == session_date(avail - pd.Timedelta(minutes=1)).to_numpy()
        )
        own_missing = bars[["buy_volume", "sell_volume", "trade_volume"]].isna().any(axis=1).to_numpy()
        values = np.where(warm & ~own_missing, values, np.nan)
        return pd.Series(values, index=bars.index, dtype="float64")
