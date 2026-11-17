"""Numba kernels vs the NumPy reference path (SHPE_MOMENTUM_NUMBA=0) on random panels."""

from __future__ import annotations

import numpy as np
import pytest

from shpe_research.features.momentum import readiness
from shpe_research.features.momentum.breakout import preceding_high
from shpe_research.features.momentum.clock_rvol import rvol_ratio, rvol_signal
from shpe_research.features.momentum.ema_trend import ema
from shpe_research.features.momentum.vwap_band import session_vwap
from shpe_research.features.momentum.wilder import (
    NUMBA_ENV,
    adx_di,
    atr,
    seeded_recursive,
    use_numba,
)

pytest.importorskip("numba")

RTOL = 1e-6


def make_panel(seed: int, dtype, n_sym: int = 6, n_sessions: int = 24):
    """Random walk panel with NaN gaps, a NaN burst, and early-close sessions (fewer minutes)."""
    rng = np.random.default_rng(seed)
    lens = rng.choice([60, 60, 60, 60, 35], size=n_sessions)  # 35 = early close
    sess = np.repeat(np.arange(n_sessions), lens)
    n_t = sess.size
    close = 50 + np.cumsum(rng.normal(0, 0.2, (n_sym, n_t)), axis=1)
    high = close + rng.uniform(0, 0.3, close.shape)
    low = close - rng.uniform(0, 0.3, close.shape)
    vol = rng.uniform(100, 1000, close.shape).round()
    for a in (close, high, low, vol):
        a[rng.random(a.shape) < 0.01] = np.nan
    close[0, 200:230] = np.nan
    vol[1, 400:410] = np.nan
    high[2, 300] = np.inf  # non-finite reset
    return tuple(a.astype(dtype) for a in (high, low, close, vol)) + (sess,)


def both(monkeypatch, fn):
    monkeypatch.setenv(NUMBA_ENV, "0")
    ref = fn()
    monkeypatch.setenv(NUMBA_ENV, "1")
    new = fn()
    return ref, new


def assert_same(ref, new):
    if isinstance(ref, tuple):
        for r, n in zip(ref, new, strict=True):
            assert_same(r, n)
        return
    assert ref.dtype == new.dtype and ref.shape == new.shape
    if ref.dtype == np.bool_:
        np.testing.assert_array_equal(ref, new)
    else:
        np.testing.assert_array_equal(np.isnan(ref), np.isnan(new))
        np.testing.assert_allclose(new, ref, rtol=RTOL, equal_nan=True)


def test_env_switch(monkeypatch):
    monkeypatch.setenv(NUMBA_ENV, "0")
    assert not use_numba()
    monkeypatch.setenv(NUMBA_ENV, "1")
    assert use_numba()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_indicators_match_reference(monkeypatch, dtype, seed):
    h, lo, c, v, sess = make_panel(seed, dtype)
    for n, alpha in ((1, 1.0), (14, 1 / 14), (60, 2 / 61)):
        assert_same(*both(monkeypatch, lambda n=n, alpha=alpha: seeded_recursive(c, n, alpha)))
    assert_same(*both(monkeypatch, lambda: atr(h, lo, c)))
    for a in (7, 14, 21):
        assert_same(*both(monkeypatch, lambda a=a: adx_di(h, lo, c, a)))
    for length in (5, 20, 60):
        assert_same(*both(monkeypatch, lambda length=length: ema(c, length)))
    for n in (1, 10, 60):
        assert_same(*both(monkeypatch, lambda n=n: preceding_high(h, n)))
    assert_same(*both(monkeypatch, lambda: session_vwap(h, lo, c, v, sess)))
    for w in (1, 3, 10):
        assert_same(*both(monkeypatch, lambda w=w: rvol_ratio(v, sess, w)))
        assert_same(*both(monkeypatch, lambda w=w: rvol_signal(v, sess, w, 1.5)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_readiness_masks_identical(monkeypatch, dtype):
    h, lo, c, v, sess = make_panel(7, dtype)
    ref, new = both(monkeypatch, lambda: readiness.readiness_masks(h, lo, c, v, sess))
    assert ref.keys() == new.keys()
    for k in ref:
        np.testing.assert_array_equal(ref[k], new[k], err_msg=k)
    assert any(m.any() for m in new.values())


def test_rvol_explicit_minute_with_gap(monkeypatch):
    h, lo, c, v, sess = make_panel(3, np.float64)
    minute = np.zeros(sess.size, dtype=np.int64)
    for s in np.unique(sess):
        idx = np.flatnonzero(sess == s)
        minute[idx] = np.arange(idx.size)
    minute[minute >= 20] += 1  # a one-minute hole inside every session
    assert_same(*both(monkeypatch, lambda: rvol_ratio(v, sess, 5, minute)))


def test_edges(monkeypatch):
    x = np.arange(3.0).reshape(1, 3)
    assert_same(*both(monkeypatch, lambda: seeded_recursive(x, 5, 0.2)))  # shorter than warm-up
    assert_same(*both(monkeypatch, lambda: preceding_high(x, 3)))  # n_t <= n
    assert_same(*both(monkeypatch, lambda: preceding_high(x, 2)))
    assert_same(*both(monkeypatch, lambda: rvol_ratio(x, np.zeros(3, dtype=int), 2)))  # single session
    empty = np.zeros((2, 0))
    assert_same(*both(monkeypatch, lambda: seeded_recursive(empty, 3, 0.5)))


def test_readonly_and_noncontiguous_inputs(monkeypatch):
    h, lo, c, v, sess = make_panel(4, np.float32)
    c2 = np.asfortranarray(c)
    c2.setflags(write=False)
    assert_same(*both(monkeypatch, lambda: ema(c2, 20)))
    assert_same(*both(monkeypatch, lambda: preceding_high(h[:, ::2], 10)))
