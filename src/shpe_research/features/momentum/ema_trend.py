"""B2: EMA trend. (EMAfast - EMAslow)/ATR14 > d and EMAslow > EMAslow 5 bars ago.

EMA alpha = 2/(length+1), seeded with the mean of the first `length` closes
(first valid at t=length-1), carried across sessions. Non-finite close resets.
Invalid on ATR <= 0 / NaN or if either EMA (including the slow EMA 5 bars
earlier) is NaN.
"""

from __future__ import annotations

import numpy as np

from shpe_research.features.momentum.wilder import FLOAT_TOL, _as_float, seeded_recursive

EMA_FAST = (5, 8, 12)
EMA_SLOW = (20, 35, 60)
EMA_D = (0.0, 0.10, 0.25)
SLOW_LOOKBACK = 5


def ema(close: np.ndarray, length: int) -> np.ndarray:
    return seeded_recursive(close, length, 2.0 / (length + 1))


def ema_trend_signal(
    ema_fast: np.ndarray, ema_slow: np.ndarray, atr14: np.ndarray, d: float
) -> np.ndarray:
    """Trend mask from precomputed EMAs and ATR14 arrays."""
    slow_prev = np.full(ema_slow.shape, np.nan, dtype=ema_slow.dtype)
    slow_prev[:, SLOW_LOOKBACK:] = ema_slow[:, :-SLOW_LOOKBACK]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (ema_fast - ema_slow) / atr14
        ok = np.isfinite(norm) & (atr14 > 0) & np.isfinite(slow_prev)
        rising = ema_slow - slow_prev > FLOAT_TOL * np.abs(ema_slow)
        return ok & (norm - d > FLOAT_TOL * max(1.0, abs(d))) & rising


def ema_trend_from_close(
    close: np.ndarray, atr14: np.ndarray, fast: int, slow: int, d: float
) -> np.ndarray:
    c = _as_float(close)
    return ema_trend_signal(ema(c, fast), ema(c, slow), atr14, d)
