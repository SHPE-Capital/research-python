# ruff: noqa: E741
import numpy as np

from shpe_research.features.momentum.readiness import (
    convergence_reports,
    first_valid_bars,
    first_valid_index,
    readiness_masks,
)


def _panel(n_sess=22, bars=390, symbols=2, seed=3):
    rng = np.random.default_rng(seed)
    n = n_sess * bars
    c = 100 + np.cumsum(rng.normal(0, 0.05, (symbols, n)), axis=1)
    h = c + rng.uniform(0.01, 0.1, c.shape)
    l = c - rng.uniform(0.01, 0.1, c.shape)
    v = rng.uniform(500, 1500, c.shape)
    sess = np.repeat(np.arange(n_sess), bars)
    return h, l, c, v, sess


def test_first_valid_index():
    m = np.array([[False, False, True], [False, False, False]])
    assert first_valid_index(m).tolist() == [2, -1]


def test_first_valid_bars_per_setting():
    h, l, c, v, sess = _panel(n_sess=17, bars=40)
    fv = first_valid_bars(readiness_masks(h, l, c, v, sess))
    assert (fv["atr14"] == 70).all()  # 5 x 14
    assert (fv["adx_a7"] == 35).all()  # 5 x 7, >= 2a-1
    assert (fv["adx_a21"] == 105).all()
    assert (fv["breakout_n60"] == 70).all()  # max(n, 5*ATR)
    assert (fv["ema_f5_s60"] == 300).all()  # 5 x 60
    assert (fv["ema_f12_s20"] == 100).all()
    assert (fv["vwap"] == 70).all()
    # rvol: needs 15 prior sessions (session 15 start = 600) + opening window w-1
    assert (fv["rvol_w3"] == 15 * 40 + 2).all()
    assert (fv["rvol_w10"] == 15 * 40 + 9).all()


def test_readiness_zero_mult_shows_structural_first_valid():
    h, l, c, v, sess = _panel(n_sess=3, bars=40)
    fv = first_valid_bars(readiness_masks(h, l, c, v, sess, warmup_mult=0))
    assert (fv["atr14"] == 14).all()
    assert (fv["adx_a21"] == 41).all()  # 2a-1
    assert (fv["ema_f5_s20"] == max(20 - 1 + 5, 14)).all()
    assert (fv["rvol_w3"] == -1).all()  # <15 sessions


def test_convergence_reports_longest_settings():
    rng = np.random.default_rng(11)
    n = 4000
    c = 100 + np.cumsum(rng.normal(0, 0.1, (3, n)), axis=1)
    h = c + rng.uniform(0, 0.2, c.shape)
    l = c - rng.uniform(0, 0.2, c.shape)
    reports = {r.name: r for r in convergence_reports(h, l, c)}
    assert set(reports) == {"ema60", "adx21"}
    assert reports["ema60"].passed and reports["ema60"].warmup_bars >= 300
    assert reports["adx21"].passed and reports["adx21"].warmup_bars >= 105
    assert reports["adx21"].max_abs_diff <= reports["adx21"].tolerance


def test_convergence_extends_when_tolerance_tight():
    rng = np.random.default_rng(11)
    c = 100 + np.cumsum(rng.normal(0, 0.1, (2, 4000)), axis=1)
    h, l = c + 0.1, c - 0.1
    loose = convergence_reports(h, l, c, adx_tol=5.0)[1]
    tight = convergence_reports(h, l, c, adx_tol=1e-3)[1]
    assert tight.warmup_bars >= loose.warmup_bars
