import numpy as np

from shpe_research.features.momentum.breakout import breakout_signal, preceding_high
from shpe_research.features.momentum.ema_trend import ema, ema_trend_signal


def test_current_bar_excluded_from_threshold():
    # bar 5 has a huge high; threshold at bar 5 must not include it
    h = np.array([[10.0, 10, 10, 10, 10, 1000.0, 10]])
    ph = preceding_high(h, 5)
    assert np.isnan(ph[0, :5]).all()
    assert ph[0, 5] == 10.0
    assert ph[0, 6] == 1000.0  # now it is in the preceding window
    c = np.array([[10.0, 10, 10, 10, 10, 11.0, 11.0]])
    atr = np.ones_like(c)
    sig = breakout_signal(c, h, atr, 5, 0.0)
    assert sig[0, 5] and not sig[0, 6]


def test_breakout_b_times_atr_and_strictness():
    h = np.full((1, 12), 10.0)
    atr = np.full((1, 12), 2.0)
    c = np.full((1, 12), 10.0)
    c[0, 10] = 10.5  # > 10 + 0.25*2=10.5? equal -> False
    c[0, 11] = 10.6
    assert not breakout_signal(c, h, atr, 10, 0.25)[0, 10]
    assert breakout_signal(c, h, atr, 10, 0.25)[0, 11]
    assert breakout_signal(c, h, atr, 10, 0.10)[0, 10]


def test_breakout_invalid_on_nan_and_warmup():
    h = np.full((1, 15), 10.0)
    h[0, 7] = np.nan
    c = np.full((1, 15), 20.0)
    atr = np.full((1, 15), 1.0)
    s = breakout_signal(c, h, atr, 5, 0.0)
    assert not s[0, :5].any()  # warm-up
    assert s[0, 5] and s[0, 6] and s[0, 7]  # windows end before the NaN high
    assert not s[0, 8:13].any()  # windows containing the NaN high
    assert s[0, 13]
    atr[0, 13] = np.nan
    assert not breakout_signal(c, h, atr, 5, 0.0)[0, 13]


def test_ema_hand_calc():
    c = np.array([[1.0, 2, 3, 4, 5, 6, 7]])
    e = ema(c, 3)[0]
    # alpha=0.5, seed=mean(1,2,3)=2 at t=2
    assert np.isnan(e[:2]).all()
    assert e[2] == 2.0
    assert e[3] == 3.0  # 2 + .5*(4-2)
    assert e[4] == 4.0
    assert e[5] == 5.0


def test_ema_trend_requires_slow_rising_5_bars_ago():
    n = 30
    fast = np.full((1, n), 12.0)
    slow = np.full((1, n), 10.0)
    atr = np.full((1, n), 4.0)  # (12-10)/4 = 0.5
    s = ema_trend_signal(fast, slow, atr, 0.25)
    assert not s.any()  # slow flat: not greater than 5 bars ago
    slow = np.linspace(10, 10.29, n)[None]
    s = ema_trend_signal(fast, slow, atr, 0.25)
    assert not s[0, :5].any() and s[0, 5:].all()
    assert not ema_trend_signal(fast, slow, atr, 0.5)[0].any()  # strict >
    assert ema_trend_signal(fast, slow, atr, 0.4)[0, 5]
    slow2 = slow.copy()
    slow2[0, 10] = np.nan
    s2 = ema_trend_signal(fast, slow2, atr, 0.25)
    assert not s2[0, 10] and not s2[0, 15] and s2[0, 16]
    assert not ema_trend_signal(fast, slow, np.zeros_like(atr), 0.0).any()
