"""Hedge-ratio registry for the stat-arb family (SS2).

`hedge_ratio(p1, p2, method, **params)` returns beta per row of `p1`, where the spread is
`p1 - beta * p2` in raw prices (the platform's definition).

Keys and what is exportable to the platform (`PairsStrategyConfig.hedgeRatioMethod`):

    fixed        platform default; exportable
    rolling_ols  platform parity (`features/cross_asset/pair_math.py`); exportable
    kalman       research only until P9
    tls          research only until P9

The estimators that a feature (F7) also needs live in `pair_math.py` and are re-exported here, never
copied (`features/` must not import `strategies/`). `kalman` and `tls` are strategy-only and live here.

Timing, as in `pair_math`: every Series is indexed by the time its value is known; the ratio at index
t uses observations strictly before t unless `include_current=True` (the platform's reading).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from shpe_research.features.cross_asset.pair_math import (
    DEFAULT_OLS_RECALC_BARS,
    DEFAULT_OLS_WINDOW_MS,
    MIN_OBSERVATIONS,
    apply_coint_hysteresis,
    engle_granger,
    engle_granger_detail,
    fixed_hedge,
    paired_observations,
    rolling_ols_hedge,
    rolling_ols_state,
    window_slice,
)

__all__ = [
    "HEDGE_REGISTRY",
    "apply_coint_hysteresis",
    "engle_granger",
    "engle_granger_detail",
    "fixed_hedge",
    "hedge_ratio",
    "kalman_hedge",
    "register_hedge",
    "rolling_ols_hedge",
    "rolling_ols_state",
    "tls_hedge",
]

HEDGE_REGISTRY: dict[str, type] = {}


def register_hedge(cls: type) -> type:
    """Register a hedge estimator class under its `hedge_type`; duplicates raise ValueError."""
    if cls.hedge_type in HEDGE_REGISTRY:
        raise ValueError(f"hedge_type {cls.hedge_type!r} is already registered")
    HEDGE_REGISTRY[cls.hedge_type] = cls
    return cls


def hedge_ratio(p1: pd.Series, p2: pd.Series, method: str = "fixed", **params: Any) -> pd.Series:
    """Hedge ratio beta for each row of `p1` using `method` (a `HEDGE_REGISTRY` key).

    Inputs are price Series on the same DatetimeIndex (the time each value is known). The value at
    index t uses data strictly before t unless the method is called with `include_current=True`.
    Rows without a paired observation carry the previous value; before any estimate it is 1.0.
    """
    try:
        cls = HEDGE_REGISTRY[method]
    except KeyError:
        raise ValueError(f"unknown hedge method {method!r}; registered: {sorted(HEDGE_REGISTRY)}") from None
    return cls.estimate(p1, p2, **params)


def _carry(values: np.ndarray, idx: pd.DatetimeIndex, full: pd.Index, initial: float) -> pd.Series:
    """Put per-observation values back on the full index, carrying the last value over missing rows."""
    out = pd.Series(values, index=idx).reindex(full).ffill().fillna(initial)
    out.name = "beta"
    return out


def kalman_hedge(
    p1: pd.Series,
    p2: pd.Series,
    *,
    delta: float = 1e-5,
    obs_var: float = 1e-2,
    fit_intercept: bool = True,
    include_current: bool = False,
    initial_beta: float = 1.0,
    initial_var: float = 1.0,
) -> pd.Series:
    """Time-varying beta from a Kalman filter on `p1 = beta * p2 + alpha + noise` (raw prices).

    State `(beta, alpha)` follows a random walk with covariance `delta / (1 - delta) * I`; the
    observation noise variance is `obs_var` (price units squared). The prior is `initial_beta` with
    variance `initial_var` (alpha: 0 with variance `1e4 * initial_var` when fitted). `delta` and
    `obs_var` have no platform counterpart; tuning them is a trial. With `include_current=False` the
    value at t is the posterior after observing the bar before t. Rows with a missing leg are skipped
    (no update, no time step) and carry the previous value. Not exportable until P9.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if obs_var <= 0:
        raise ValueError("obs_var must be positive")
    idx, y, x = paired_observations(p1, p2)
    k = 2 if fit_intercept else 1
    theta = np.zeros(k)
    theta[0] = initial_beta
    cov = np.eye(k) * initial_var
    if fit_intercept:
        cov[1, 1] = 1e4 * initial_var
    q = np.eye(k) * delta / (1 - delta)
    out = np.empty(len(idx))
    for i in range(len(idx)):
        if not include_current:
            out[i] = theta[0]
        h = np.array([x[i], 1.0][:k])
        cov = cov + q
        s = float(h @ cov @ h) + obs_var
        gain = cov @ h / s
        theta = theta + gain * (y[i] - float(h @ theta))
        cov = cov - np.outer(gain, h @ cov)
        if include_current:
            out[i] = theta[0]
    return _carry(out, idx, p1.index, initial_beta)


def tls_hedge(
    p1: pd.Series,
    p2: pd.Series,
    *,
    window_ms: float = DEFAULT_OLS_WINDOW_MS,
    recalc_bars: int = DEFAULT_OLS_RECALC_BARS,
    include_current: bool = False,
    initial_beta: float = 1.0,
) -> pd.Series:
    """Rolling total-least-squares (orthogonal regression) slope of `p1` on `p2`, raw prices.

    Same schedule as `rolling_ols`: every `recalc_bars` observations the last `window_ms` ms are
    re-fit, with fewer than 5 observations leaving beta unchanged. The slope is the first principal
    direction of the centred window, so swapping the legs inverts beta exactly. There is no
    cointegration gate (the platform gate is OLS-based). Not exportable until P9.
    """
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if recalc_bars < 1:
        raise ValueError("recalc_bars must be at least 1")
    idx, y, x = paired_observations(p1, p2)
    t_ns = idx.asi8
    window_ns = int(round(window_ms * 1_000_000))
    out = np.empty(len(idx))
    beta, counter = float(initial_beta), 0
    for i in range(len(idx)):
        counter += 1
        if counter >= recalc_bars:
            counter = 0
            w = window_slice(t_ns, i, window_ns, include_current)
            if w.stop - w.start >= MIN_OBSERVATIONS:
                data = np.column_stack([y[w], x[w]])
                data = data - data.mean(axis=0)
                _, vecs = np.linalg.eigh(data.T @ data)
                v = vecs[:, -1]  # direction of largest variance
                if v[1] != 0:
                    beta = float(v[0] / v[1])
        out[i] = beta
    return _carry(out, idx, p1.index, initial_beta)


@register_hedge
class FixedHedge:
    """`fixed`: constant `beta` (default 1.0, the platform default)."""

    hedge_type: ClassVar[str] = "fixed"

    @staticmethod
    def estimate(p1: pd.Series, p2: pd.Series, beta: float = 1.0) -> pd.Series:
        return fixed_hedge(p1.index, beta)


@register_hedge
class RollingOlsHedge:
    """`rolling_ols`: `window_ms` 14.4e6, `recalc_bars` 5, cointegration gate on (platform v6)."""

    hedge_type: ClassVar[str] = "rolling_ols"

    @staticmethod
    def estimate(p1: pd.Series, p2: pd.Series, **params: Any) -> pd.Series:
        return rolling_ols_hedge(p1, p2, **params)


@register_hedge
class KalmanHedge:
    """`kalman`: `delta`, `obs_var`; research only until P9."""

    hedge_type: ClassVar[str] = "kalman"

    @staticmethod
    def estimate(p1: pd.Series, p2: pd.Series, **params: Any) -> pd.Series:
        return kalman_hedge(p1, p2, **params)


@register_hedge
class TlsHedge:
    """`tls`: `window_ms`, `recalc_bars`; research only until P9."""

    hedge_type: ClassVar[str] = "tls"

    @staticmethod
    def estimate(p1: pd.Series, p2: pd.Series, **params: Any) -> pd.Series:
        return tls_hedge(p1, p2, **params)
