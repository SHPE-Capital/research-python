"""B2: rolling-high breakout. C > max(H over preceding n bars) + b*ATR14.

The current bar is excluded from the preceding high. The window is the n
preceding bars in the continuous series (carries across sessions). Invalid
(False) if any of the n highs, the close or the ATR is non-finite, or fewer
than n preceding bars exist.
"""

from __future__ import annotations

import numpy as np

from shpe_research.features.momentum.wilder import FLOAT_TOL, _as_float

BREAKOUT_N = (10, 20, 40, 60)
BREAKOUT_B = (0.0, 0.10, 0.25)


def preceding_high(high: np.ndarray, n: int) -> np.ndarray:
    """max(H[t-n .. t-1]); NaN if the window is incomplete or has NaN."""
    h = _as_float(high)
    out = np.full(h.shape, np.nan, dtype=h.dtype)
    n_t = h.shape[1]
    if n_t <= n:
        return out
    m = h[:, 0 : n_t - n].copy()
    for k in range(1, n):
        m = np.maximum(m, h[:, k : n_t - n + k])  # NaN propagates
    out[:, n:] = m
    return out


def breakout_signal_from_high(
    close: np.ndarray, prec_high: np.ndarray, atr14: np.ndarray, b: float
) -> np.ndarray:
    """Breakout mask from a precomputed preceding-high array and ATR14."""
    c = _as_float(close)
    with np.errstate(invalid="ignore"):
        thr = prec_high + b * atr14
        ok = np.isfinite(c) & np.isfinite(thr) & np.isfinite(atr14)
        return ok & (c - thr > FLOAT_TOL * np.maximum(np.abs(c), np.abs(thr)))


def breakout_signal(
    close: np.ndarray, high: np.ndarray, atr14: np.ndarray, n: int, b: float
) -> np.ndarray:
    """Breakout mask given an already-computed ATR14 array."""
    return breakout_signal_from_high(close, preceding_high(high, n), atr14, b)
