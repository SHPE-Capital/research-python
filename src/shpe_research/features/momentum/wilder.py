# ruff: noqa: E741
"""B1: Wilder ATR, +DI/-DI and ADX (batch form).

Conventions (documented reference; the platform port must reproduce them):

* TR_t = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|); NaN at t=0 (no previous close).
  The series is continuous across sessions (previous close may be the prior
  session's last bar).
* Wilder smoothing of length n: the seed is the simple mean of the first n
  valid inputs (available at the n-th valid input), then
  ``s_t = s_{t-1} + (x_t - s_{t-1}) / n``.
* ATR_n = Wilder smooth of TR (first value at t=n).
* +DM = up if up > down and up > 0 else 0; -DM = down if down > up and
  down > 0 else 0; up = H_t-H_{t-1}, down = L_{t-1}-L_t.
* +DI = 100 * smooth(+DM)/smooth(TR); same for -DI. DX = 100*|+DI - -DI| /
  (+DI + -DI). ADX = Wilder smooth of DX (seed = mean of first a DX values, so
  first valid ADX at t = 2a-1).
* A non-finite input resets the recursion (warm-up restarts), so outputs are
  NaN until n valid inputs accumulate again. Zero denominators give NaN.
* Signals compare with relative tolerance FLOAT_TOL, never by rounding prices.

The loops run over time only and are vectorized across symbols, so they can be
Numba-jitted later.
"""

from __future__ import annotations

import os

import numpy as np

NUMBA_ENV = "SHPE_MOMENTUM_NUMBA"
FLOAT_TOL = 1e-6
ATR_LENGTH = 14
ADX_PERIODS = (7, 14, 21)
ADX_THRESHOLDS = (15, 20, 25, 30)


def use_numba() -> bool:
    """True when the Numba kernels should run (default). ``SHPE_MOMENTUM_NUMBA=0`` selects the NumPy reference."""
    if os.environ.get(NUMBA_ENV, "1").strip().lower() in ("0", "false", "off", "no", ""):
        return False
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def _as_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x if x.dtype in (np.float32, np.float64) else x.astype(np.float64)


def seeded_recursive(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Recursive smoother seeded with the mean of the first n valid inputs.

    ``s_t = s_{t-1} + alpha * (x_t - s_{t-1})``. Wilder uses alpha=1/n, EMA
    uses 2/(n+1). Non-finite input resets the state. State is float64;
    output has the input dtype.
    """
    x = _as_float(x)
    n_sym, n_t = x.shape
    out = np.full((n_sym, n_t), np.nan, dtype=x.dtype)
    if use_numba():
        from shpe_research.features.momentum._kernels import seeded_recursive_kernel

        seeded_recursive_kernel(np.ascontiguousarray(x), int(n), float(alpha), out)
        return out
    state = np.full(n_sym, np.nan)
    acc = np.zeros(n_sym)
    cnt = np.zeros(n_sym, dtype=np.int64)
    for t in range(n_t):
        xt = x[:, t].astype(np.float64)
        bad = ~np.isfinite(xt)
        ready = cnt >= n
        upd = ready & ~bad
        state = np.where(upd, state + alpha * (xt - state), state)
        warm = ~ready & ~bad
        acc = np.where(warm, acc + xt, acc)
        cnt = np.where(warm, cnt + 1, cnt)
        seed_now = warm & (cnt == n)
        state = np.where(seed_now, acc / n, state)
        state = np.where(bad, np.nan, state)
        acc = np.where(bad, 0.0, acc)
        cnt = np.where(bad, 0, cnt)
        out[:, t] = np.where(cnt >= n, state, np.nan)
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range; NaN at t=0 and wherever an input is non-finite."""
    h, l, c = _as_float(high), _as_float(low), _as_float(close)
    tr = np.full(h.shape, np.nan, dtype=h.dtype)
    pc = c[:, :-1]
    tr[:, 1:] = np.maximum(h[:, 1:] - l[:, 1:], np.maximum(np.abs(h[:, 1:] - pc), np.abs(l[:, 1:] - pc)))
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = ATR_LENGTH) -> np.ndarray:
    """Wilder ATR of length n (default 14); first valid value at t=n."""
    return seeded_recursive(true_range(high, low, close), n, 1.0 / n)


def adx_di(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, a: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (adx, plus_di, minus_di) for period a, in 0..100 points."""
    h, l = _as_float(high), _as_float(low)
    up = np.full(h.shape, np.nan, dtype=h.dtype)
    dn = np.full(h.shape, np.nan, dtype=h.dtype)
    up[:, 1:] = h[:, 1:] - h[:, :-1]
    dn[:, 1:] = l[:, :-1] - l[:, 1:]
    with np.errstate(invalid="ignore"):
        plus_dm = np.where(up > dn, np.maximum(up, 0.0), 0.0)
        minus_dm = np.where(dn > up, np.maximum(dn, 0.0), 0.0)
    plus_dm = np.where(np.isfinite(up) & np.isfinite(dn), plus_dm, np.nan)
    minus_dm = np.where(np.isfinite(up) & np.isfinite(dn), minus_dm, np.nan)
    k = 1.0 / a
    s_tr = seeded_recursive(true_range(high, low, close), a, k)
    s_p = seeded_recursive(plus_dm, a, k)
    s_m = seeded_recursive(minus_dm, a, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(s_tr > 0, 100.0 * s_p / s_tr, np.nan)
        mdi = np.where(s_tr > 0, 100.0 * s_m / s_tr, np.nan)
        den = pdi + mdi
        dx = np.where(den > 0, 100.0 * np.abs(pdi - mdi) / den, np.nan)
    adx = seeded_recursive(dx, a, k)
    dt = s_tr.dtype
    return adx.astype(dt), pdi.astype(dt), mdi.astype(dt)


def adx_signal(adx: np.ndarray, plus_di: np.ndarray, minus_di: np.ndarray, q: float) -> np.ndarray:
    """ADX > q and +DI > -DI; False on any NaN."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(adx) & np.isfinite(plus_di) & np.isfinite(minus_di)
        return ok & (adx - q > FLOAT_TOL * max(1.0, abs(q))) & (plus_di - minus_di > FLOAT_TOL)
