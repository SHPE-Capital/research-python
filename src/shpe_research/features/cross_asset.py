"""TODO(F7, F8): features that need another symbol's bars.  Level: intermediate.

The Feature contract computes from ONE symbol's bars. These two tasks extend
it: a feature declares `context_symbols`, and compute_features passes those
symbols' bars as a second argument. Agree the extension with the other owner
and update docs/contracts.md (a minor contract version bump).

F8  market_return (Christopher's ORB plan, section 5)
    latest available completed benchmark close / benchmark session open - 1.
    The benchmark (SPY) bar used must have ts_available <= the decision bar's
    ts_available; sharing a timestamp label is not enough.

F7  pair spread features (Team 1, statistical arbitrage)
    For a pair (this symbol, other): hedge ratio from a rolling OLS on log
    prices over the previous `lookback_sessions` sessions (fitted only on data
    before the decision), spread = log(p) - beta * log(p_other) - alpha,
    z-score of the spread, and its half-life of mean reversion.
    The trading platform implements the same pieces in
    trading-platform/backend/src/services/indicators/{ols,cointegration,zscore}.ts;
    match its definitions so research and live agree.

Done when:
    - compute_features supports context symbols without changing single-symbol features.
    - tests/todo/test_features_todo.py::test_market_return and ::test_pair_spread pass,
      including an assert_no_lookahead-style check that truncating EITHER symbol's
      future bars leaves earlier values unchanged.
"""

from __future__ import annotations

import pandas as pd

from shpe_research.features.base import Feature


class MarketReturn(Feature):
    feature_type = "market_return"
    requires = ("ts_available",)
    context_symbols: tuple[str, ...]

    def __init__(self, benchmark: str = "SPY"):
        self.benchmark = benchmark
        self.context_symbols = (benchmark,)

    @property
    def name(self) -> str:
        return f"market_return_{self.benchmark.lower()}"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("TODO(F8): needs the context-symbol extension to compute_features")


class PairSpreadZScore(Feature):
    feature_type = "pair_spread_z"
    requires = ("close", "ts_available")
    context_symbols: tuple[str, ...]

    def __init__(self, other: str, lookback_sessions: int = 20):
        self.other = other
        self.lookback_sessions = int(lookback_sessions)
        self.context_symbols = (other,)

    @property
    def name(self) -> str:
        return f"pair_z_{self.other.lower()}_{self.lookback_sessions}d"

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("TODO(F7): needs the context-symbol extension to compute_features")
