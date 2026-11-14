"""Numba kernels for the momentum indicator time loops (single-threaded, GIL-free).

Each kernel reproduces the arithmetic of the NumPy reference in the sibling module operation for
operation (float64 state, input-dtype outputs), so results agree with the reference to rounding. The
callers choose the path through ``wilder.use_numba``; this module imports numba, so it is only imported
when the Numba path is on. ``parallel=False`` on purpose: the benchmark parallelises over candidates.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def seeded_recursive_kernel(x, n, alpha, out):
    """Fill ``out`` with the seeded recursive smoother of ``x`` (symbols x time)."""
    n_sym, n_t = x.shape
    for i in range(n_sym):
        state = np.nan
        acc = 0.0
        cnt = 0
        for t in range(n_t):
            xt = np.float64(x[i, t])
            if not np.isfinite(xt):
                state = np.nan
                acc = 0.0
                cnt = 0
            elif cnt >= n:
                state = state + alpha * (xt - state)
            else:
                acc += xt
                cnt += 1
                if cnt == n:
                    state = acc / n
            if cnt >= n:
                out[i, t] = state
            else:
                out[i, t] = np.nan


@njit(cache=True, nogil=True)
def preceding_high_kernel(h, n, out):
    """out[i, t] = max(h[i, t-n .. t-1]) for t >= n, NaN if the window holds a NaN (O(1) per bar)."""
    n_sym, n_t = h.shape
    ring = np.empty(n + 1, dtype=np.int64)  # monotonic deque of indices, decreasing values
    size = n + 1
    for i in range(n_sym):
        head = 0
        count = 0
        last_nan = -1
        for t in range(1, n_t):
            j = t - 1  # newest bar of the window ending at t-1
            v = h[i, j]
            if np.isnan(v):
                last_nan = j
            else:
                while count > 0 and h[i, ring[(head + count - 1) % size]] <= v:
                    count -= 1
                ring[(head + count) % size] = j
                count += 1
            while count > 0 and ring[head] < t - n:
                head = (head + 1) % size
                count -= 1
            if t >= n:
                if last_nan >= t - n or count == 0:
                    out[i, t] = np.nan
                else:
                    out[i, t] = h[i, ring[head]]


@njit(cache=True, nogil=True)
def segment_cumsum_ratio_kernel(pv, vv, good, starts, ends, out):
    """Per-session cumsum(pv)/cumsum(vv); NaN where the bar is bad or cumulative volume is not positive."""
    n_sym = pv.shape[0]
    for i in range(n_sym):
        for s in range(starts.shape[0]):
            cpv = 0.0
            cv = 0.0
            for t in range(starts[s], ends[s]):
                cpv += pv[i, t]
                cv += vv[i, t]
                if good[i, t] and cv > 0:
                    out[i, t] = cpv / cv
                else:
                    out[i, t] = np.nan


@njit(cache=True, nogil=True)
def rvol_ratio_kernel(vol, w, contiguous, rank, mn, n_sess, n_min, history, min_windows, out):
    """Window sum / median of the same clock window over the prior ``history`` sessions.

    ``contiguous[t]`` says bar t closes a gap-free in-session window of w bars. Window sums use the same
    prefix-sum differences as the NumPy reference. A zero/NaN median or fewer than ``min_windows``
    finite prior windows gives NaN.
    """
    n_sym, n_t = vol.shape
    cs = np.zeros(n_t + 1)
    cn = np.zeros(n_t + 1, dtype=np.int64)
    ws = np.full(n_t, np.nan)
    grid = np.empty((n_sess, n_min))
    med = np.empty((n_sess, n_min))
    buf = np.empty(history)
    for i in range(n_sym):
        for t in range(n_t):
            v = np.float64(vol[i, t])
            fin = np.isfinite(v)
            cs[t + 1] = cs[t] + (v if fin else 0.0)
            cn[t + 1] = cn[t] + (0 if fin else 1)
        ws[:] = np.nan
        for t in range(w - 1, n_t):
            if contiguous[t] and (cn[t + 1] - cn[t + 1 - w]) == 0:
                ws[t] = cs[t + 1] - cs[t + 1 - w]
        grid[:, :] = np.nan
        for t in range(n_t):
            grid[rank[t], mn[t]] = ws[t]
        med[:, :] = np.nan
        # per clock minute, slide a sorted window over the prior sessions (insert newest, drop oldest)
        for m in range(n_min):
            c = 0
            for r in range(1, n_sess):
                old = r - 1 - history
                if old >= 0 and np.isfinite(grid[old, m]):
                    g = grid[old, m]
                    k = 0
                    while buf[k] != g:
                        k += 1
                    while k < c - 1:
                        buf[k] = buf[k + 1]
                        k += 1
                    c -= 1
                g = grid[r - 1, m]
                if np.isfinite(g):
                    k = c
                    while k > 0 and buf[k - 1] > g:
                        buf[k] = buf[k - 1]
                        k -= 1
                    buf[k] = g
                    c += 1
                if c >= min_windows and c > 0:
                    if c % 2 == 1:
                        med[r, m] = buf[c // 2]
                    else:
                        med[r, m] = (buf[c // 2 - 1] + buf[c // 2]) / 2.0
        for t in range(n_t):
            d = med[rank[t], mn[t]]
            if d > 0:
                out[i, t] = ws[t] / d
            else:
                out[i, t] = np.nan
