"""SS2: hedge-ratio estimators in features/cross_asset/pair_math.py.

The Engle-Granger vector below was produced by running the platform's own
`computeEngleGranger` (services/indicators/cointegration.ts, significance 0.05) on
these 60 prices; see the test for the p-value reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.features.cross_asset import pair_math as pm

P1 = [
    71.06073549240396, 72.42919463920151, 72.86821036906665, 72.33655696901643, 71.67864702869103,
    71.06958029658459, 70.72093378246409, 71.11595003641983, 71.39587109677477, 71.17829631281667,
    70.62745896375506, 69.44037254798006, 69.79254191170573, 69.44477314333598, 70.04326186158151,
    70.54448979563335, 71.0523669859803, 71.13354838882442, 70.4102646533742, 70.12922030560077,
    70.5058343522715, 70.97936003453077, 71.65338456145334, 72.28195211214873, 72.01103861335199,
    71.14662205539537, 70.94620749057158, 71.40944401258487, 71.47691250865782, 71.08208982368642,
    71.02498239604272, 71.23586217030783, 70.64554406764114, 69.91719335162145, 70.94121885873189,
    71.40676167998025, 71.77907521155815, 71.73908230640609, 71.9989777723036, 70.83516420362476,
    70.37113232404808, 70.54959205014647, 70.62301762798984, 70.5465804156484, 70.58223350451783,
    70.67743905249763, 70.22190918840653, 69.42994639522526, 69.43537648920302, 70.26375270906726,
    70.80199397036199, 70.91802349777794, 71.28362663267218, 71.48382160204838, 70.80509548791638,
    70.36360131886218, 71.3704994884243, 71.86126663718241, 71.00599328688871, 70.7014214587797,
]
P2 = [
    50.2, 50.68396491225695, 50.75499629339482, 50.56618419204484, 50.73255252227928,
    50.91698060670142, 50.58610774621177, 50.23032841624335, 50.2913654980935, 50.21392691476503,
    49.75588766085908, 49.59136596783328, 49.837497217252604, 49.82439745634308, 49.62300893379814,
    49.87933572819807, 50.33007164190236, 50.35437579592422, 50.32183584917808, 50.700605492471425,
    50.94703976382231, 50.6909163521848, 50.54608795330281, 50.71977473486275, 50.5609342362969,
    50.06515525066756, 49.93375050620521, 50.04698525583156, 49.79075966764859, 49.49156991249833,
    49.71348587062448, 49.999735261538234, 49.897965047640454, 49.9552692245276, 50.44007180700587,
    50.68226865485583, 50.52866969513975, 50.635580508011344, 50.94517543137377, 50.79120787162002,
    50.39875907082229, 50.39265847451736, 50.422165081930125, 50.0021668349781, 49.66505414773098,
    49.81502924786072, 49.8638471020035, 49.59267842245992, 49.66085850275345, 50.11093561296834,
    50.24601790573342, 50.159930520172196, 50.471412032223554, 50.86975733825044, 50.76316696218693,
    50.57322207449984, 50.758596858947776, 50.77547646551508, 50.322280067472065, 50.05619626207872,
]
R = {
    'beta': 1.0601742344563383,
    'alpha': 17.64267412972449,
    'testStatistic': -3.154129283792321,
    'criticalValue': -3.3377,
    'isCointegrated': False,
}


def minute_index(n: int, start: str = "2026-10-01T13:30:00Z") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1min")


def planted_pair(n: int = 700, beta: float = 1.2, seed: int = 3, phi: float = 0.9, sd: float = 0.15):
    """Raw prices p1 = 5 + beta * p2 + AR(1) noise, so the pair is cointegrated."""
    rng = np.random.default_rng(seed)
    idx = minute_index(n)
    p2 = 100 + np.cumsum(rng.normal(0, 0.6, n))
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0, sd)
    return pd.Series(5 + beta * p2 + e, idx), pd.Series(p2, idx)


def test_fixed():
    p1, _ = planted_pair(50)
    out = pm.fixed_hedge(p1.index, beta=1.7)
    assert out.index.equals(p1.index)
    assert (out == 1.7).all()
    assert (pm.fixed_hedge(p1.index) == 1.0).all()  # platform default


def test_rolling_ols_recovers_beta():
    p1, p2 = planted_pair()
    beta = pm.rolling_ols_hedge(p1, p2)
    tail = beta.iloc[300:]
    assert abs(tail.mean() - 1.2) < 0.05
    assert (tail - 1.2).abs().max() < 0.05


def test_rolling_ols_past_only():
    p1, p2 = planted_pair()
    full = pm.rolling_ols_hedge(p1, p2)
    for cut in (320, 411, 500):
        trunc = pm.rolling_ols_hedge(p1.iloc[: cut + 1], p2.iloc[: cut + 1])
        assert trunc.iloc[cut] == full.iloc[cut]
        pd.testing.assert_series_equal(trunc, full.iloc[: cut + 1])
    # the strict default ignores bar t itself: perturbing it leaves beta at t unchanged
    p1b = p1.copy()
    p1b.iloc[500] += 50.0
    assert pm.rolling_ols_hedge(p1b, p2).iloc[500] == full.iloc[500]
    # the platform reading (bar t included) does react to it, as long as t is a recalculation bar
    pf = pm.rolling_ols_hedge(p1, p2, include_current=True)
    pfb = pm.rolling_ols_hedge(p1b, p2, include_current=True)
    assert (pf != pfb).any()


def test_rolling_ols_updates_every_recalc_bars():
    p1, p2 = planted_pair(400)
    beta = pm.rolling_ols_hedge(p1, p2)
    changes = np.flatnonzero(beta.diff().fillna(0).to_numpy() != 0)
    assert len(changes) > 10
    assert set(np.diff(changes)) == {5}


def test_engle_granger_matches_platform():
    stat, pvalue = pm.engle_granger(pd.Series(P1), pd.Series(P2))
    assert stat == pytest.approx(R["testStatistic"], abs=1e-6)
    res = pm.engle_granger_detail(pd.Series(P1), pd.Series(P2), alpha=0.05)
    assert res.beta == pytest.approx(R["beta"], abs=1e-9)
    assert res.alpha == pytest.approx(R["alpha"], abs=1e-9)
    assert res.critical_value == R["criticalValue"]
    assert res.cointegrated is R["isCointegrated"]
    # The platform computes no p-value (it compares tau with the MacKinnon critical value). We read
    # "MacKinnon p-value" as statsmodels' mackinnonp(tau, regression="c", N=2), the table the
    # platform's critical values (-3.9001, -3.3377, -3.0462) come from. Reference value below.
    assert pvalue == pytest.approx(0.07793972439712285, abs=1e-6)
    # p < 0.05 exactly when tau is below the 5% critical value
    assert (pvalue < 0.05) == (stat < -3.3377)


def test_engle_granger_critical_values_and_degenerate():
    p1, p2 = planted_pair(100)
    assert pm.engle_granger_detail(p1, p2, alpha=0.01).critical_value == -3.9001
    assert pm.engle_granger_detail(p1, p2, alpha=0.10).critical_value == -3.0462
    assert pm.engle_granger_detail(p1.iloc[:4], p2.iloc[:4]) is None  # fewer than 5 observations
    assert pm.engle_granger_detail(p1, p2 * 0 + 3.0) is None  # constant leg 2: OLS undefined
    stat, p = pm.engle_granger(p1.iloc[:4], p2.iloc[:4])
    assert np.isnan(stat) and np.isnan(p)


def test_coint_hysteresis_matches_platform():
    f = pm.apply_coint_hysteresis
    assert f(False, -3.40, -3.3377, 0.15) is False  # not past the band yet
    assert f(False, -3.50, -3.3377, 0.15) is True
    assert f(True, -3.20, -3.3377, 0.15) is True  # inside the band, keep the verdict
    assert f(True, -3.10, -3.3377, 0.15) is False


def broken_pair():
    """Cointegrated for 350 bars, then leg 1 becomes an unrelated random walk."""
    n = 900
    rng = np.random.default_rng(11)
    idx = minute_index(n)
    p2 = 100 + np.cumsum(rng.normal(0, 0.6, n))
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 0.15)
    p1 = 5 + 1.2 * p2 + e
    p1[350:] = 125 + np.cumsum(rng.normal(0, 0.4, n - 350))
    return pd.Series(p1, idx), pd.Series(p2, idx)


def test_beta_frozen_when_not_cointegrated():
    p1, p2 = broken_pair()
    state = pm.rolling_ols_state(p1, p2)
    assert {"beta", "tau", "cointegrated"} <= set(state.columns)
    gate = state["cointegrated"].to_numpy()
    assert gate[:340].any() and not gate[-100:].any()
    off_start = 350 + int(np.argmax(~gate[350:]))
    off = state["beta"].iloc[off_start:]
    assert off.nunique() == 1  # frozen at the last cointegrated value
    assert off.iloc[0] == state["beta"].iloc[off_start - 1]
    # with the gate disabled beta keeps moving
    free = pm.rolling_ols_hedge(p1, p2, gate=False)
    assert free.iloc[off_start:].nunique() > 5
    # before the first cointegrated verdict the pair keeps the fixed default of 1.0
    first_true = int(np.argmax(gate))
    assert (state["beta"].iloc[:first_true] == 1.0).all()


def test_session_boundary_keeps_beta_until_window_refills():
    n = 400
    idx = minute_index(n).append(minute_index(n, "2026-10-02T13:30:00Z"))
    p1a, p2a = planted_pair(n, seed=5)
    p1b, p2b = planted_pair(n, seed=6)
    p1 = pd.Series(np.r_[p1a.to_numpy(), p1b.to_numpy()], idx)
    p2 = pd.Series(np.r_[p2a.to_numpy(), p2b.to_numpy()], idx)
    beta = pm.rolling_ols_hedge(p1, p2)
    # day 2 opens with an empty (time-based) window: fewer than 5 observations, so beta carries over
    assert beta.iloc[n] == beta.iloc[n - 1]
    assert beta.iloc[n : n + 4].nunique() == 1


def test_missing_bar_is_not_an_observation():
    p1, p2 = planted_pair(300)
    p1m = p1.copy()
    p1m.iloc[150] = np.nan
    out = pm.rolling_ols_hedge(p1m, p2)
    assert out.index.equals(p1.index)
    assert out.iloc[150] == out.iloc[149]
    assert not out.isna().any()
    # same as dropping the minute from both legs, then carrying beta over it
    ref = pm.rolling_ols_hedge(p1m.drop(p1m.index[150]), p2.drop(p2.index[150]))
    pd.testing.assert_series_equal(out.drop(out.index[150]), ref)


def test_log_price_space_is_a_dollar_ratio():
    p1, p2 = planted_pair(500)
    raw = pm.rolling_ols_hedge(p1, p2, price_space="raw").iloc[-1]
    log = pm.rolling_ols_hedge(p1, p2, price_space="log").iloc[-1]
    assert abs(raw - 1.2) < 0.05
    assert log != raw and 0.5 < log < 2.0
    with pytest.raises(ValueError):
        pm.rolling_ols_hedge(p1, p2, price_space="sqrt")


def test_rolling_ols_validates_inputs():
    p1, p2 = planted_pair(50)
    with pytest.raises(ValueError):
        pm.rolling_ols_hedge(p1, p2, recalc_bars=0)
    with pytest.raises(ValueError):
        pm.rolling_ols_hedge(p1, p2, window_ms=0)
    with pytest.raises(ValueError):
        pm.rolling_ols_hedge(p1, p2.iloc[::-1])  # unsorted index
