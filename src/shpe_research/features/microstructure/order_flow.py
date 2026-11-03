"""TODO(F6): order-flow features from quote/trade minute summaries.  Level: intermediate.

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

Done when:
    - tests/todo/test_features_todo.py::test_order_flow passes on a synthetic
      quote/trade sample, and assert_no_lookahead passes.
    - Team 3's R4 study can build its baseline from these features.
"""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature


class SignedVolumeRatio(Feature):
    feature_type = "signed_volume_ratio"
    requires = ("buy_volume", "sell_volume", "trade_volume", "ts_available")

    def __init__(self, window: int = 5):
        self.window = int(window)

    @property
    def name(self) -> str:
        return f"signed_volume_ratio_{self.window}m"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("TODO(F6): signed volume ratio")
