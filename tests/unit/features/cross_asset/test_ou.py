"""Acceptance tests for SS3: OU fit, half-life and spread z-score."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from shpe_research.features.cross_asset import pair_math
from shpe_research.features.cross_asset.pair_math import OUFit, fit_ou, spread_zscore
from shpe_research.strategies.statarb import ou


def _ar1(half_life: float, n: int, seed: int, mu: float = 0.0, sigma: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    phi = math.exp(-math.log(2) / half_life)
    x = np.empty(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = mu + phi * (x[i - 1] - mu) + sigma * rng.standard_normal()
    return pd.Series(x)


@pytest.mark.parametrize("half_life", [12, 30])
def test_half_life_recovered(half_life):
    fit = fit_ou(_ar1(half_life, 20_000, seed=half_life, mu=5.0))
    assert isinstance(fit, OUFit)
    assert fit.half_life_bars == pytest.approx(half_life, rel=0.3)
    assert fit.mu == pytest.approx(5.0, abs=1.0)
    assert fit.n == 19_999
    assert 0 < fit.r2 < 1


def test_random_walk_inf():
    rw = pd.Series(np.cumsum(np.random.default_rng(3).standard_normal(5_000)))
    fit = fit_ou(rw)
    assert fit.half_life_bars > 100 or math.isinf(fit.half_life_bars)


def test_explosive_is_inf():
    fit = fit_ou(pd.Series(1.01 ** np.arange(200)))
    assert math.isinf(fit.half_life_bars)


def test_fit_ou_short_or_constant_is_nan_safe():
    assert math.isinf(fit_ou(pd.Series([1.0, 1.0, 1.0, 1.0])).half_life_bars)
    assert fit_ou(pd.Series([1.0])).n == 0


def test_fit_ou_drops_nan_pairs():
    s = _ar1(12, 500, seed=1)
    s.iloc[100] = np.nan
    assert fit_ou(s).n == 497


def test_zscore_zero_std():
    z = spread_zscore(pd.Series([2.0] * 50), window_bars=40, min_obs=30)
    assert z.iloc[:29].isna().all()
    assert (z.iloc[29:] == 0.0).all()


def test_zscore_past_only():
    s = _ar1(12, 200, seed=7)
    full = spread_zscore(s, window_bars=40)
    cut = spread_zscore(s.iloc[:121], window_bars=40)
    assert full.iloc[120] == cut.iloc[120]
    pd.testing.assert_series_equal(full.iloc[:121], cut)


def test_zscore_matches_hand_computation():
    s = pd.Series(np.arange(1.0, 41.0))
    z = spread_zscore(s, window_bars=30, min_obs=30)
    w = s.iloc[10:40]  # window includes the current observation (platform behaviour)
    assert z.iloc[39] == pytest.approx((w.iloc[-1] - w.mean()) / w.std(ddof=1))
    assert z.iloc[28] != z.iloc[28]  # NaN before min_obs


def test_zscore_missing_bar_is_nan_and_ignored():
    s = pd.Series(np.arange(1.0, 61.0))
    s.iloc[45] = np.nan
    z = spread_zscore(s, window_bars=30, min_obs=25)
    assert math.isnan(z.iloc[45])
    assert not math.isnan(z.iloc[50])


def test_zscore_validates_args():
    with pytest.raises(ValueError):
        spread_zscore(pd.Series([1.0, 2.0]), window_bars=1)
    with pytest.raises(ValueError):
        spread_zscore(pd.Series([1.0, 2.0]), window_bars=10, min_obs=11)


def test_reexports_are_the_same_objects():
    assert ou.fit_ou is pair_math.fit_ou
    assert ou.spread_zscore is pair_math.spread_zscore


def test_half_life_screen():
    rng = np.random.default_rng(0)
    phi = 0.95
    x = np.zeros(5_000)
    for i in range(1, len(x)):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    assert ou.passes_half_life_screen(pd.Series(x), max_half_life_bars=30)
    assert not ou.passes_half_life_screen(pd.Series(x), max_half_life_bars=5)
    assert not ou.passes_half_life_screen(pd.Series(np.cumsum(rng.standard_normal(5_000))), max_half_life_bars=30)
    assert not ou.passes_half_life_screen(pd.Series(x), max_half_life_bars=30, min_half_life_bars=20)
