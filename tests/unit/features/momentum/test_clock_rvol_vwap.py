# ruff: noqa: E741
import numpy as np

from shpe_research.features.momentum.clock_rvol import (
    minute_of_session,
    rvol_ratio,
    rvol_signal,
)
from shpe_research.features.momentum.vwap_band import session_vwap, vwap_band_signal


def _sessions(lengths):
    return np.concatenate([np.full(n, i) for i, n in enumerate(lengths)])


def test_vwap_hand_calc_and_reset():
    sess = np.array([0, 0, 0, 1, 1])
    h = np.array([[11.0, 12, 13, 20, 22]])
    l = np.array([[9.0, 10, 11, 18, 20]])
    c = np.array([[10.0, 11, 12, 19, 21]])
    v = np.array([[100.0, 100, 200, 50, 50]])
    vw = session_vwap(h, l, c, v, sess)[0]
    assert vw[0] == 10.0
    assert vw[1] == 10.5
    assert abs(vw[2] - (1000 + 1100 + 2400) / 400) < 1e-12
    assert vw[3] == 19.0  # reset: only session 1 bars
    assert vw[4] == 20.0


def test_vwap_zero_volume_and_nan_invalid():
    sess = np.zeros(4, dtype=int)
    one = np.full((1, 4), 10.0)
    v = np.array([[0.0, 5, np.nan, 5]])
    vw = session_vwap(one, one, one, v, sess)[0]
    assert np.isnan(vw[0]) and vw[1] == 10.0 and np.isnan(vw[2]) and vw[3] == 10.0


def test_vwap_band_bounds():
    c = np.array([[100.0, 101, 99.99, 103, 103.01, 100]])
    vw = np.full_like(c, 100.0)
    atr = np.ones_like(c)
    s = vwap_band_signal(c, vw, atr, 3.0)[0]
    assert s.tolist() == [True, True, False, True, False, True]
    assert vwap_band_signal(c, vw, np.zeros_like(c), 3.0).sum() == 0
    assert vwap_band_signal(c, vw, np.full_like(c, np.nan), 3.0).sum() == 0


def _vol_panel(lengths, base=100.0):
    sess = _sessions(lengths)
    return np.full((1, sess.size), base), sess


def test_rvol_fewer_than_15_windows_invalid():
    v, sess = _vol_panel([10] * 16)  # session 14 has 14 priors, session 15 has 15
    r = rvol_ratio(v, sess, 3)[0]
    m = minute_of_session(sess)
    in14 = (sess == 14) & (m >= 2)
    in15 = (sess == 15) & (m >= 2)
    assert np.isnan(r[in14]).all()
    assert np.allclose(r[in15], 1.0)
    assert rvol_signal(v, sess, 3, 1.0)[0][in15].all()
    assert not rvol_signal(v, sess, 3, 1.0)[0][in14].any()


def test_rvol_opening_window_boundary_no_carryover():
    v, sess = _vol_panel([10] * 17)
    s = rvol_signal(v, sess, 5, 1.0)[0]
    m = minute_of_session(sess)
    last = sess == 16
    assert not s[last & (m < 4)].any()  # window would need previous session bars
    assert s[last & (m == 4)].all()
    assert s[last & (m == 9)].all()


def test_rvol_median_and_threshold():
    lengths = [10] * 21
    v, sess = _vol_panel(lengths)
    v = v.copy()
    # make sessions 0..19 volumes at minute 4 vary: median of window sums known
    base_idx = np.flatnonzero(sess == 20)[4]
    for k in range(20):
        v[0, np.flatnonzero(sess == k)[2:5]] = 100 + k  # window (2..4) sum 3*(100+k)
    v[0, np.flatnonzero(sess == 20)[2:5]] = 300.0
    r = rvol_ratio(v, sess, 3)[0, base_idx]
    # window sums at last 20 sessions (1..20) -> sessions 1..19 -> 3*(101..119), session 0 dropped?
    # preceding 20 sessions of session 20 = 0..19 -> median of 3*(100..119) = 3*109.5
    assert abs(r - 900 / (3 * 109.5)) < 1e-9
    assert rvol_signal(v, sess, 3, 2.0)[0, base_idx]
    assert not rvol_signal(v, sess, 3, 3.0)[0, base_idx]


def test_rvol_early_close_days():
    # 20 history sessions: 5 are early closes (6 bars) so minutes 6..9 have only 15 windows
    lengths = [10] * 15 + [6] * 5 + [10]
    v, sess = _vol_panel(lengths)
    r = rvol_ratio(v, sess, 3)[0]
    m = minute_of_session(sess)
    cur = sess == 20
    assert np.allclose(r[cur & (m >= 2) & (m <= 9)], 1.0)  # >=15 windows still
    # one more early close in history -> 14 windows for late minutes => invalid
    lengths = [10] * 14 + [6] * 6 + [10]
    v, sess = _vol_panel(lengths)
    r = rvol_ratio(v, sess, 3)[0]
    m = minute_of_session(sess)
    cur = sess == 20
    assert np.allclose(r[cur & (m >= 2) & (m <= 5)], 1.0)
    assert np.isnan(r[cur & (m >= 6)]).all()


def test_rvol_zero_median_and_nan_invalid():
    v, sess = _vol_panel([10] * 17, base=0.0)
    assert not rvol_signal(v, sess, 3, 1.0).any()
    v, sess = _vol_panel([10] * 17)
    v[0, np.flatnonzero(sess == 16)[3]] = np.nan
    s = rvol_signal(v, sess, 3, 1.0)[0]
    last = np.flatnonzero(sess == 16)
    assert not s[last[3]] and not s[last[4]] and not s[last[5]] and s[last[6]]
