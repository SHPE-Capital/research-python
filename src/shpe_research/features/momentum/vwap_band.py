# ruff: noqa: E741
"""B3: session VWAP band. 0 <= (C - VWAP)/ATR14 <= e.

VWAP = cumsum(((H+L+C)/3)*V)/cumsum(V), reset at each session change
(``session_idx`` is a (minutes,) non-decreasing array). A bar with a
non-finite H/L/C/V is invalid and excluded from the sums (it does not poison
the rest of the session). Zero cumulative volume or ATR <= 0 is invalid.
"""

from __future__ import annotations

import numpy as np

from shpe_research.features.momentum.wilder import FLOAT_TOL, _as_float, use_numba

VWAP_E = (1.0, 2.0, 3.0)


def _segments(session_idx: np.ndarray) -> list[tuple[int, int]]:
    s = np.asarray(session_idx)
    if s.size == 0:
        return []
    cuts = np.flatnonzero(s[1:] != s[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [s.size]))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def session_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    session_idx: np.ndarray,
) -> np.ndarray:
    """Session VWAP (NaN where the bar is invalid or cumulative volume is 0)."""
    h, l, c, v = (_as_float(a) for a in (high, low, close, volume))
    good = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v)
    pv = np.where(good, (h + l + c) / 3.0 * v, 0.0).astype(np.float64)
    vv = np.where(good, v, 0.0).astype(np.float64)
    out = np.full(h.shape, np.nan, dtype=h.dtype)
    if use_numba():
        from shpe_research.features.momentum._kernels import segment_cumsum_ratio_kernel

        segs = _segments(session_idx)
        starts = np.array([a for a, _ in segs], dtype=np.int64)
        ends = np.array([b for _, b in segs], dtype=np.int64)
        r = np.full(h.shape, np.nan)
        segment_cumsum_ratio_kernel(np.ascontiguousarray(pv), np.ascontiguousarray(vv), good, starts, ends, r)
        return r.astype(h.dtype)
    for a, b in _segments(session_idx):
        cpv = np.cumsum(pv[:, a:b], axis=1)
        cv = np.cumsum(vv[:, a:b], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(cv > 0, cpv / cv, np.nan)
        out[:, a:b] = np.where(good[:, a:b], r, np.nan)
    return out


def vwap_band_signal(
    close: np.ndarray, vwap: np.ndarray, atr14: np.ndarray, e: float
) -> np.ndarray:
    """Band mask from precomputed VWAP and ATR14 arrays."""
    c = _as_float(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (c - vwap) / atr14
        ok = np.isfinite(z) & (atr14 > 0)
        return ok & (z >= -FLOAT_TOL) & (z <= e + FLOAT_TOL * max(1.0, e))
