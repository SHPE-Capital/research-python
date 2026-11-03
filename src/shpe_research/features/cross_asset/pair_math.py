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

# --- Hedge ratios (SS2) -------------------------------------------------------


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
