"""SS2: strategy-side hedge registry plus the Kalman and TLS estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.features.cross_asset import pair_math as pm
from shpe_research.strategies.statarb import hedge as hg


def drifting_pair(n: int = 1500, b0: float = 1.0, b1: float = 1.4, seed: int = 7):
    """p1 = beta_t * p2 + noise with beta drifting linearly from b0 to b1."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-01T13:30:00Z", periods=n, freq="1min")
    p2 = 100 + 8 * np.sin(np.arange(n) / 150.0) + np.cumsum(rng.normal(0, 0.2, n))
    beta = np.linspace(b0, b1, n)
    p1 = beta * p2 + rng.normal(0, 0.2, n)
    return pd.Series(p1, idx), pd.Series(p2, idx), pd.Series(beta, idx)


def test_kalman_tracks_drift():
    p1, p2, true = drifting_pair()
    kal = hg.kalman_hedge(p1, p2)
    ols = pm.rolling_ols_hedge(p1, p2, gate=False)
    sl = slice(400, None)  # both estimators warmed up
    err_kal = (kal - true).abs().iloc[sl].mean()
    err_ols = (ols - true).abs().iloc[sl].mean()
    assert err_kal < err_ols
    assert err_kal < 0.1


def test_kalman_past_only():
    p1, p2, _ = drifting_pair(600)
    full = hg.kalman_hedge(p1, p2)
    cut = 400
    trunc = hg.kalman_hedge(p1.iloc[: cut + 1], p2.iloc[: cut + 1])
    assert trunc.iloc[cut] == full.iloc[cut]
    p1b = p1.copy()
    p1b.iloc[cut] += 30.0
    assert hg.kalman_hedge(p1b, p2).iloc[cut] == full.iloc[cut]  # bar t itself is not used
    assert hg.kalman_hedge(p1b, p2, include_current=True).iloc[cut] != hg.kalman_hedge(
        p1, p2, include_current=True
    ).iloc[cut]


def test_kalman_missing_bar_carries_state():
    p1, p2, _ = drifting_pair(300)
    p1m = p1.copy()
    p1m.iloc[100] = np.nan
    out = hg.kalman_hedge(p1m, p2)
    assert out.index.equals(p1.index) and not out.isna().any()
    assert out.iloc[100] == out.iloc[99]


def test_tls_recovers_beta_and_is_leg_symmetric():
    rng = np.random.default_rng(2)
    n = 700
    idx = pd.date_range("2026-10-01T13:30:00Z", periods=n, freq="1min")
    latent = 100 + np.cumsum(rng.normal(0, 0.3, n))
    p2 = pd.Series(latent + rng.normal(0, 0.3, n), idx)
    p1 = pd.Series(1.2 * latent + rng.normal(0, 0.36, n), idx)
    b = hg.tls_hedge(p1, p2).iloc[-1]
    assert abs(b - 1.2) < 0.06
    # swapping the legs inverts the slope exactly (OLS does not)
    b_rev = hg.tls_hedge(p2, p1).iloc[-1]
    assert b * b_rev == pytest.approx(1.0, rel=1e-9)


def test_tls_past_only_and_session_gap():
    p1, p2, _ = drifting_pair(500)
    full = hg.tls_hedge(p1, p2)
    trunc = hg.tls_hedge(p1.iloc[:401], p2.iloc[:401])
    assert trunc.iloc[400] == full.iloc[400]
    assert (full.iloc[:4] == 1.0).all()  # not enough observations yet: initial beta


def test_registry_keys():
    for key in ("fixed", "rolling_ols", "kalman", "tls"):
        assert key in hg.HEDGE_REGISTRY
        assert hg.HEDGE_REGISTRY[key].hedge_type == key
    p1, p2, _ = drifting_pair(400)
    for key in ("fixed", "rolling_ols", "kalman", "tls"):
        out = hg.hedge_ratio(p1, p2, key)
        assert isinstance(out, pd.Series) and out.index.equals(p1.index)
        assert not out.isna().any()
    assert (hg.hedge_ratio(p1, p2, "fixed", beta=2.0) == 2.0).all()
    with pytest.raises(ValueError, match="unknown hedge"):
        hg.hedge_ratio(p1, p2, "nope")


def test_registry_rejects_duplicates_and_matches_pair_math():
    with pytest.raises(ValueError, match="already registered"):

        @hg.register_hedge
        class Dup:
            hedge_type = "fixed"

    # re-exports are the same objects, not copies
    assert hg.rolling_ols_hedge is pm.rolling_ols_hedge
    assert hg.engle_granger is pm.engle_granger
    assert hg.fixed_hedge is pm.fixed_hedge


def test_defaults_match_platform():
    p1, p2, _ = drifting_pair(300)
    a = hg.hedge_ratio(p1, p2, "rolling_ols")
    b = pm.rolling_ols_hedge(p1, p2, window_ms=14_400_000, recalc_bars=5)
    pd.testing.assert_series_equal(a, b)
