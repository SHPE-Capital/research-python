"""B3: clock-window relative volume.

sum(volume over the latest w bars of the CURRENT session) divided by the
median of the same clock-time window (same minute-of-session) over the
preceding 20 sessions; signal if ratio >= v. A historical window is usable if
it exists (early-close days lack late minutes) and all its volumes are finite.
At least MIN_WINDOWS (15) usable windows are required. The opening window never
carries over: a window needs w bars within the same session (minute >= w-1,
and no gap in minute-of-session). Zero/NaN median is invalid.

``session_idx`` is a (minutes,) non-decreasing array; ``minute`` (optional,
same shape) is the minute-of-session, derived as position within the session
when omitted.
"""

from __future__ import annotations

import warnings

import numpy as np

from shpe_research.features.momentum.vwap_band import _segments
from shpe_research.features.momentum.wilder import FLOAT_TOL, _as_float, use_numba

RVOL_W = (3, 5, 10)
RVOL_V = (1.0, 1.5, 2.0)
HISTORY_SESSIONS = 20
MIN_WINDOWS = 15


def minute_of_session(session_idx: np.ndarray) -> np.ndarray:
    m = np.zeros(len(session_idx), dtype=np.int64)
    for a, b in _segments(session_idx):
        m[a:b] = np.arange(b - a)
    return m


def window_sums(volume: np.ndarray, minute: np.ndarray, w: int) -> np.ndarray:
    """Sum of the latest w bars within the session; NaN if incomplete/NaN."""
    v = _as_float(volume).astype(np.float64)
    n_t = v.shape[1]
    fin = np.isfinite(v)
    zero = np.zeros((v.shape[0], 1))
    cs = np.concatenate([zero, np.cumsum(np.where(fin, v, 0.0), axis=1)], axis=1)
    cn = np.concatenate([zero, np.cumsum(~fin, axis=1)], axis=1)
    out = np.full(v.shape, np.nan)
    if n_t < w:
        return out
    s = cs[:, w:] - cs[:, :-w]
    bad = (cn[:, w:] - cn[:, :-w]) > 0
    t = np.arange(w - 1, n_t)
    contiguous = (minute[t] >= w - 1) & (minute[t] - minute[t - (w - 1)] == w - 1)
    out[:, w - 1 :] = np.where(bad | ~contiguous[None, :], np.nan, s)
    return out


def rvol_ratio(
    volume: np.ndarray,
    session_idx: np.ndarray,
    w: int,
    minute: np.ndarray | None = None,
) -> np.ndarray:
    """Current window sum / median of same window over preceding sessions."""
    sess = np.asarray(session_idx)
    mn = minute_of_session(sess) if minute is None else np.asarray(minute)
    segs = _segments(sess)
    k = len(segs)
    if k == 0:
        return window_sums(volume, mn, w).astype(_as_float(volume).dtype)
    rank = np.zeros(len(sess), dtype=np.int64)
    for i, (a, b) in enumerate(segs):
        rank[a:b] = i
    n_min = int(mn.max()) + 1
    if use_numba():
        from shpe_research.features.momentum._kernels import rvol_ratio_kernel

        vol = _as_float(volume)
        n_t = vol.shape[1]
        contiguous = np.zeros(n_t, dtype=np.bool_)
        if n_t >= w:
            t = np.arange(w - 1, n_t)
            contiguous[w - 1 :] = (mn[t] >= w - 1) & (mn[t] - mn[t - (w - 1)] == w - 1)
        out = np.empty(vol.shape, dtype=np.float64)
        rvol_ratio_kernel(
            np.ascontiguousarray(vol), int(w), contiguous, rank, mn.astype(np.int64), k, n_min,
            HISTORY_SESSIONS, MIN_WINDOWS, out,
        )
        return out.astype(vol.dtype)
    ws = window_sums(volume, mn, w)
    grid = np.full((ws.shape[0], k, n_min), np.nan)
    grid[:, rank, mn] = ws
    med = np.full_like(grid, np.nan)
    for i in range(1, k):
        prior = grid[:, max(0, i - HISTORY_SESSIONS) : i, :]
        usable = np.sum(np.isfinite(prior), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(prior, axis=1)
        med[:, i, :] = np.where(usable >= MIN_WINDOWS, m, np.nan)
    denom = med[:, rank, mn]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, ws / denom, np.nan).astype(_as_float(volume).dtype)


def rvol_signal_from_ratio(ratio: np.ndarray, v: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(ratio) & (ratio >= v * (1.0 - FLOAT_TOL))


def rvol_signal(
    volume: np.ndarray,
    session_idx: np.ndarray,
    w: int,
    v: float,
    minute: np.ndarray | None = None,
) -> np.ndarray:
    return rvol_signal_from_ratio(rvol_ratio(volume, session_idx, w, minute), v)
