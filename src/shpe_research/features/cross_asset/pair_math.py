"""Shared pair estimators: hedge ratios (SS2) and Ornstein-Uhlenbeck fits (SS3).

They live at the feature level so that both `features/cross_asset/pair_spread.py`
(F7) and `strategies/statarb/` can import them without an upward import.
SS2 and SS3 each append only under their own section header below and never
edit the other's; whoever merges second keeps both blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp

# --- Hedge ratios (SS2) -------------------------------------------------------
#
# Timing. Every price Series is indexed by the time its value is KNOWN (the bar's `ts_available`, which
# is also what the platform stamps on a completed bar). A hedge ratio returned at index t is what a
# decision made at t may use. With `include_current=False` (the default) it uses observations strictly
# before t; `include_current=True` also uses the observation stamped t, which is what the platform's
# PairsStrategy does (it pushes the bar into its window before it re-estimates). Both readings are
# implemented because the SS2 packet and the platform differ here.
#
# Missing data. An observation exists only where BOTH legs have a finite price, as on the platform
# (one observation per minute in which both legs printed). Windows are measured in time, not bars, so a
# session boundary shrinks them. Where fewer than 5 observations are in the window the previous value
# is carried; rows without an observation also carry the previous value.

#: MacKinnon (1991) critical values, two variables, intercept: the platform's CRITICAL_VALUES table.
ENGLE_GRANGER_CRITICAL_VALUES: dict[float, float] = {0.01: -3.9001, 0.05: -3.3377, 0.10: -3.0462}
#: Platform default `cointHysteresis`, in tau units.
DEFAULT_COINT_HYSTERESIS = 0.15
#: Platform defaults `olsWindowMs` (4 h) and `olsRecalcIntervalBars`.
DEFAULT_OLS_WINDOW_MS = 14_400_000
DEFAULT_OLS_RECALC_BARS = 5
MIN_OBSERVATIONS = 5  # computeEngleGranger returns null below this


@dataclass(frozen=True)
class EngleGrangerResult:
    """Outcome of `engle_granger_detail`; mirrors the platform's `EngleGrangerResult`."""

    beta: float  # OLS slope of p1 on p2
    alpha: float  # OLS intercept
    stat: float  # Dickey-Fuller tau of the residuals; more negative = stronger evidence
    critical_value: float  # MacKinnon value for the requested level
    cointegrated: bool  # stat < critical_value


def paired_observations(p1: pd.Series, p2: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Rows where both legs have a finite price, as (index, leg-1 prices, leg-2 prices).

    Both indexes must be DatetimeIndex and strictly increasing, else ValueError. If the indexes differ,
    `p2` is reindexed to `p1`'s index (missing labels become NaN, so those rows are dropped).
    """
    for name, s in (("p1", p1), ("p2", p2)):
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError(f"{name} needs a DatetimeIndex")
        if not s.index.is_monotonic_increasing or not s.index.is_unique:
            raise ValueError(f"{name} index must be strictly increasing")
    if not p1.index.equals(p2.index):
        p2 = p2.reindex(p1.index)
    a = p1.to_numpy(dtype=float)
    b = p2.to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return p1.index[ok], a[ok], b[ok]


def window_slice(t_ns: np.ndarray, k: int, window_ns: int, include_current: bool) -> slice:
    """Positions of the observations in the time window ending at observation `k`.

    Keeps times `>= t[k] - window` (the platform's RollingTimeWindow keeps entries at the cutoff), up to
    `k` inclusive when `include_current`, else up to `k - 1`.
    """
    lo = int(np.searchsorted(t_ns, t_ns[k] - window_ns, side="left"))
    return slice(lo, k + 1 if include_current else k)


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float] | None:
    """OLS of y on x with intercept: (beta, alpha); None if fewer than 2 points or x is constant."""
    n = len(y)
    if n < 2 or n != len(x):
        return None
    dx = x - x.sum() / n
    dy = y - y.sum() / n
    var_x = float(dx @ dx)
    if var_x == 0:
        return None
    beta = float(dx @ dy) / var_x
    return beta, float(y.sum() / n - beta * x.sum() / n)


def _engle_granger(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float] | None:
    """(beta, alpha, tau) exactly as services/indicators/cointegration.ts computes them, or None."""
    n = len(y)
    if n < MIN_OBSERVATIONS or n != len(x):
        return None
    fit = _ols(y, x)
    if fit is None:
        return None
    beta, alpha = fit
    resid = y - alpha - beta * x
    lag = resid[:-1]
    delta = np.diff(resid)
    denom = float(lag @ lag)
    if denom == 0:
        return None
    gamma = float(lag @ delta) / denom  # Delta e[t] = gamma * e[t-1]: no intercept, no lags
    err = delta - gamma * lag
    sigma2 = float(err @ err) / (n - 2)
    se = float(np.sqrt(sigma2 / denom))
    if se == 0:
        return None
    return beta, alpha, gamma / se


def engle_granger_detail(p1: pd.Series, p2: pd.Series, alpha: float = 0.05) -> EngleGrangerResult | None:
    """Engle-Granger two-step test as the platform runs it (`computeEngleGranger`).

    Step 1 regresses `p1` on `p2` with an intercept (raw prices). Step 2 is a no-lag, no-intercept
    Dickey-Fuller regression of the residual change on the lagged residual; tau = gamma / se. The
    verdict compares tau with the MacKinnon (1991) critical value for `alpha` (0.01, 0.05 or 0.10;
    anything else raises ValueError, where the platform silently uses 0.05).

    Inputs are two equal-length price Series (oldest first); rows where either leg is NaN are
    dropped. Returns None when fewer than 5 observations remain, leg 2 is constant, or the residual
    fit is degenerate. It uses only the values passed in, so the caller decides what is known at t.
    """
    try:
        crit = ENGLE_GRANGER_CRITICAL_VALUES[alpha]
    except KeyError:
        raise ValueError(f"alpha must be one of {sorted(ENGLE_GRANGER_CRITICAL_VALUES)}, got {alpha!r}") from None
    a = p1.to_numpy(dtype=float)
    b = p2.to_numpy(dtype=float)
    if len(a) != len(b):
        raise ValueError("p1 and p2 must have the same length")
    ok = np.isfinite(a) & np.isfinite(b)
    res = _engle_granger(a[ok], b[ok])
    if res is None:
        return None
    beta, intercept, tau = res
    return EngleGrangerResult(beta, intercept, tau, crit, bool(tau < crit))


def engle_granger(p1: pd.Series, p2: pd.Series) -> tuple[float, float]:
    """(tau, p-value) of the Engle-Granger test; (nan, nan) when the test is undefined.

    tau is the platform's statistic (see `engle_granger_detail`). The platform itself compares tau
    with critical values and computes no p-value; the p-value here is MacKinnon's response surface for
    the case the platform's critical values come from (two variables, constant):
    `statsmodels.tsa.adfvalues.mackinnonp(tau, regression="c", N=2)`.
    """
    res = engle_granger_detail(p1, p2)
    if res is None:
        return float("nan"), float("nan")
    return res.stat, float(mackinnonp(res.stat, regression="c", N=2))


def apply_coint_hysteresis(was_cointegrated: bool, stat: float, critical_value: float, band: float) -> bool:
    """The platform's `applyCointHysteresis`: a dead band of `band` tau units around the critical value."""
    if was_cointegrated:
        return bool(stat < critical_value + band)
    return bool(stat < critical_value - band)


def fixed_hedge(index: pd.Index, beta: float = 1.0) -> pd.Series:
    """Constant hedge ratio (platform `fixedHedgeRatio`, default 1) on `index`. Uses no data."""
    return pd.Series(float(beta), index=index, name="beta")


def rolling_ols_state(
    p1: pd.Series,
    p2: pd.Series,
    *,
    window_ms: float = DEFAULT_OLS_WINDOW_MS,
    recalc_bars: int = DEFAULT_OLS_RECALC_BARS,
    coint_alpha: float = 0.05,
    coint_hysteresis: float = DEFAULT_COINT_HYSTERESIS,
    gate: bool = True,
    include_current: bool = False,
    price_space: str = "raw",
    initial_beta: float = 1.0,
) -> pd.DataFrame:
    """Rolling OLS hedge ratio with the platform's cointegration gate, one row per row of `p1`.

    Columns: `beta` (the ratio usable at that row), `tau` (last Engle-Granger statistic, NaN before
    the first test) and `cointegrated` (gate verdict after hysteresis, False before the first test).

    Every `recalc_bars` observations the last `window_ms` milliseconds are re-fit by OLS of `p1` on
    `p2`. With `gate=True` (platform behaviour) beta only changes while the pair is cointegrated at
    `coint_alpha`, with `coint_hysteresis` as dead band; otherwise the previous beta is kept. Before
    any update beta is `initial_beta`. `price_space="log"` fits log prices (a dollar ratio; research
    only, not exportable). See the module notes for timing and missing data.
    """
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if recalc_bars < 1:
        raise ValueError("recalc_bars must be at least 1")
    if price_space not in ("raw", "log"):
        raise ValueError(f"price_space must be 'raw' or 'log', got {price_space!r}")
    crit = ENGLE_GRANGER_CRITICAL_VALUES.get(coint_alpha)
    if crit is None:
        raise ValueError(f"coint_alpha must be one of {sorted(ENGLE_GRANGER_CRITICAL_VALUES)}")
    idx, a, b = paired_observations(p1, p2)
    if price_space == "log":
        if (a <= 0).any() or (b <= 0).any():
            raise ValueError("log price space needs positive prices")
        a, b = np.log(a), np.log(b)
    t_ns = idx.asi8
    window_ns = int(round(window_ms * 1_000_000))
    m = len(idx)
    beta_out = np.empty(m)
    tau_out = np.full(m, np.nan)
    coint_out = np.zeros(m, dtype=bool)
    beta, tau, coint, counter = float(initial_beta), float("nan"), False, 0
    for k in range(m):
        counter += 1
        if counter >= recalc_bars:
            counter = 0
            w = window_slice(t_ns, k, window_ns, include_current)
            res = _engle_granger(a[w], b[w])
            if res is not None:
                new_beta, _, tau = res
                coint = apply_coint_hysteresis(coint, tau, crit, coint_hysteresis)
                if coint or not gate:
                    beta = new_beta
        beta_out[k], tau_out[k], coint_out[k] = beta, tau, coint
    state = pd.DataFrame({"beta": beta_out, "tau": tau_out, "cointegrated": coint_out}, index=idx)
    state = state.reindex(p1.index)
    state["beta"] = state["beta"].ffill().fillna(float(initial_beta))
    state["tau"] = state["tau"].ffill()
    state["cointegrated"] = state["cointegrated"].astype("boolean").ffill().fillna(False).astype(bool)
    return state


def rolling_ols_hedge(p1: pd.Series, p2: pd.Series, **params) -> pd.Series:
    """Platform `rolling_ols` hedge ratio: the `beta` column of `rolling_ols_state` (same parameters)."""
    return rolling_ols_state(p1, p2, **params)["beta"]


# --- Ornstein-Uhlenbeck fit and half-life (SS3) -------------------------------



@dataclass(frozen=True)
class OUFit:
    """Result of an AR(1) / Ornstein-Uhlenbeck fit to a spread.

    Units are per bar. ``half_life_bars`` is ``inf`` when the spread does not
    mean-revert (``kappa <= 0``) or the fit is degenerate. ``n`` is the number
    of (s_{t-1}, delta s_t) pairs used.
    """

    kappa: float
    mu: float
    sigma: float
    half_life_bars: float
    r2: float
    n: int


def fit_ou(spread: pd.Series) -> OUFit:
    """Fit ``delta s_t = a + b s_{t-1} + eps`` by OLS and return the OU parameters.

    Input: spread values in bar order (any index). Output: ``kappa = -b``,
    ``mu = -a / b``, ``sigma`` = residual std (ddof=2), ``half_life_bars =
    ln 2 / kappa`` (``inf`` if ``b >= 0``), ``r2`` and ``n``.
    Timing: the fit uses the whole series passed in, so call it on data known
    at the decision time only (e.g. a trailing window). Pairs containing a NaN
    are dropped; fewer than 3 usable pairs, or a constant series, give
    ``half_life_bars = inf`` with NaN parameters.
    """
    s = pd.Series(spread, dtype=float).reset_index(drop=True)
    x = s.shift(1)
    dy = s.diff()
    ok = x.notna() & dy.notna()
    xv, yv = x[ok].to_numpy(), dy[ok].to_numpy()
    n = len(xv)
    nan = float("nan")
    if n < 3 or np.ptp(xv) == 0:
        return OUFit(nan, nan, nan, math.inf, nan, n)
    xm, ym = xv.mean(), yv.mean()
    b = float(((xv - xm) * (yv - ym)).sum() / ((xv - xm) ** 2).sum())
    a = float(ym - b * xm)
    resid = yv - (a + b * xv)
    ss_res = float((resid**2).sum())
    ss_tot = float(((yv - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
    sigma = math.sqrt(ss_res / (n - 2))
    if b >= 0:
        return OUFit(-b, nan, sigma, math.inf, r2, n)
    kappa = -b
    return OUFit(kappa, -a / b, sigma, math.log(2) / kappa, r2, n)


def spread_zscore(spread: pd.Series, window_bars: int, min_obs: int = 30) -> pd.Series:
    """Rolling z-score of a spread, matching the platform ``PairsStrategy``.

    At position t the reference window is the last ``window_bars`` values
    ending at t **inclusive** (the platform pushes the current point before
    computing mean and std). Mean and sample std (ddof=1) use only values at or
    before t, so truncating the series after t leaves z_t unchanged. Output is
    NaN until the window holds ``min_obs`` non-NaN values, and NaN at a bar whose
    own spread is NaN (missing bar); NaNs inside the window are skipped. z = 0
    when the std is 0.
    """
    if window_bars < 2:
        raise ValueError("window_bars must be at least 2")
    if not 2 <= min_obs <= window_bars:
        raise ValueError("min_obs must be between 2 and window_bars")
    s = pd.Series(spread, dtype=float)
    roll = s.rolling(window_bars, min_periods=min_obs)
    mean, std = roll.mean(), roll.std(ddof=1)
    z = (s - mean) / std
    z = z.where(std != 0, 0.0)
    return z.where(mean.notna() & s.notna())
