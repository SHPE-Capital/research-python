"""F7: pair spread features for (this symbol, `other`), using the shared estimators in `pair_math`.

Inputs: this symbol's bars (leg 1, `p`) and the other leg's bars as a context symbol (`p_other`). Prices are
as supplied in the bars. Each value is indexed by the row's `ts_available`, the time the bar is known.

Definitions (match the platform's `services/indicators/{ols,cointegration,zscore}.ts`):

* hedge ratio `beta`: `pair_math.rolling_ols_state` (`hedge="rolling_ols"`, OLS of p on p_other over a
  trailing time window, re-fit every `recalc_bars` observations, only changing while the pair passes the
  Engle-Granger gate; observations strictly before the row) or a constant (`hedge="fixed"`, `beta` param,
  platform default 1). Kalman and TLS are research-only in `strategies/statarb/hedge.py` and are not
  available here (features never import strategies).
* spread: `p - beta * p_other` (`price_space="raw"`, the platform's definition and the default) or
  `log p - beta * log p_other` (`price_space="log"`; research only, `exportable` is False until P9 adds
  `priceSpace`). The stub's log-price-with-intercept description is replaced by this.
* z-score: `pair_math.spread_zscore` over the last `zscore_bars` rows (default 390 * lookback_sessions),
  window ending at the row INCLUSIVE (the platform pushes the current point first), ddof=1, min 30 values.
* half-life: `pair_math.fit_ou` on the trailing `zscore_bars` spread values, built with the beta known at
  the refresh row and re-fit every `hl_every` rows (default 30), carried between refreshes (causal). In bars;
  NaN when the spread does not mean-revert or fewer than 50 pairs are usable.

The rolling OLS window is `ceil(lookback_sessions * 7 / 5)` calendar days (time-based, as on the platform), so
a session boundary shrinks it; it approximates `lookback_sessions` sessions.

Missing data: a row where either leg has no fresh bar gives a NaN spread and z-score; beta and half-life carry
their last values (`pair_math` rules). Warm-up rows (before the first Engle-Granger test) are NaN, not the
fixed default 1.0. Columns: `pair_z_<other>_<n>d` (output="z"), `pair_beta_<other>_<n>d` ("beta"),
`pair_hl_<other>_<n>d` ("hl"); `output` selects which, as one Feature gives one column.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from shpe_research.features.base import Feature, register_feature
from shpe_research.features.cross_asset import pair_math
from shpe_research.features.cross_asset.market_return import align_context_exact

_PREFIX = {"z": "pair_z", "beta": "pair_beta", "hl": "pair_hl"}
_HEDGES = ("rolling_ols", "fixed")
_MIN_OU_PAIRS = 50


@register_feature
class PairSpreadZScore(Feature):
    """Pair spread z-score, hedge ratio or half-life (`output`), known at the row's `ts_available`."""

    feature_type = "pair_spread_z"
    requires = ("close", "ts_available")

    def __init__(
        self,
        other: str,
        lookback_sessions: int = 20,
        hedge: str = "rolling_ols",
        price_space: str = "raw",
        output: str = "z",
        zscore_bars: int | None = None,
        recalc_bars: int = pair_math.DEFAULT_OLS_RECALC_BARS,
        hl_every: int = 30,
        beta: float = 1.0,
    ):
        if hedge not in _HEDGES:
            raise ValueError(f"hedge must be one of {_HEDGES} in features, got {hedge!r}")
        if price_space not in ("raw", "log"):
            raise ValueError(f"price_space must be 'raw' or 'log', got {price_space!r}")
        if output not in _PREFIX:
            raise ValueError(f"output must be one of {sorted(_PREFIX)}, got {output!r}")
        self.other = other
        self.lookback_sessions = int(lookback_sessions)
        if self.lookback_sessions < 1:
            raise ValueError("lookback_sessions must be at least 1")
        self.hedge = hedge
        self.price_space = price_space
        self.output = output
        self.zscore_bars = int(zscore_bars) if zscore_bars is not None else 390 * self.lookback_sessions
        self.recalc_bars = int(recalc_bars)
        self.hl_every = int(hl_every)
        self.fixed_beta = float(beta)
        self.window_ms = float(math.ceil(self.lookback_sessions * 7 / 5) * 86_400_000)
        self.context_symbols = (other,)

    @property
    def exportable(self) -> bool:
        """True if the platform can run this definition today (raw prices, `fixed` or `rolling_ols`)."""
        return self.price_space == "raw" and self.hedge in _HEDGES

    @property
    def name(self) -> str:
        return f"{_PREFIX[self.output]}_{self.other.lower()}_{self.lookback_sessions}d"

    def _beta(self, p1: pd.Series, p2: pd.Series) -> pd.Series:
        if self.hedge == "fixed":
            return pair_math.fixed_hedge(p1.index, self.fixed_beta)
        state = pair_math.rolling_ols_state(
            p1, p2, window_ms=self.window_ms, recalc_bars=self.recalc_bars, price_space=self.price_space
        )
        return state["beta"].where(state["tau"].notna())  # NaN until the first Engle-Granger test has run

    def _half_life(self, a: np.ndarray, b: np.ndarray, beta: np.ndarray) -> np.ndarray:
        out = np.full(len(a), np.nan)
        w = self.zscore_bars
        last = np.nan
        for r in range(len(a)):
            if (r + 1) % self.hl_every == 0 and np.isfinite(beta[r]):
                lo = max(0, r - w + 1)
                fit = pair_math.fit_ou(pd.Series(a[lo : r + 1] - beta[r] * b[lo : r + 1]))
                last = fit.half_life_bars if fit.n >= _MIN_OU_PAIRS and np.isfinite(fit.half_life_bars) else np.nan
            out[r] = last
        return out

    def compute(self, bars: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.Series:
        if not context or self.other not in context:
            raise ValueError(f"{self.name} needs context bars for {self.other}")
        idx = pd.DatetimeIndex(bars["ts_available"])
        p1 = pd.Series(bars["close"].to_numpy(dtype=float), index=idx)
        p2 = pd.Series(align_context_exact(bars, context[self.other], "close").to_numpy(), index=idx)
        beta = self._beta(p1, p2)  # fits in the chosen price space itself
        if self.price_space == "log":
            p1, p2 = np.log(p1.where(p1 > 0)), np.log(p2.where(p2 > 0))
        if self.output == "beta":
            return pd.Series(beta.to_numpy(), index=bars.index, dtype="float64")
        if self.output == "z":
            spread = p1 - beta * p2
            z = pair_math.spread_zscore(spread, self.zscore_bars, min_obs=min(30, self.zscore_bars))
            return pd.Series(z.to_numpy(), index=bars.index, dtype="float64")
        hl = self._half_life(p1.to_numpy(), p2.to_numpy(), beta.to_numpy())
        return pd.Series(hl, index=bars.index, dtype="float64")
