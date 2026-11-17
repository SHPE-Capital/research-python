"""F7: pair spread z-score, hedge ratio and half-life."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.features import compute_features
from shpe_research.features.base import assert_no_lookahead, build_feature
from shpe_research.features.cross_asset import pair_math as pm
from shpe_research.features.cross_asset.pair_spread import PairSpreadZScore


@pytest.fixture(scope="module")
def pair():
    bars = synthetic_bars(
        symbols=["AAA"], start="2026-03-02", end="2026-03-27", cointegrated_pairs=[{"beta": 1.2, "half_life_bars": 60}]
    )
    a, b = bars.attrs["planted"]["cointegrated_pairs"]["pairs"][0]["symbols"]
    return bars[bars["symbol"] == a].reset_index(drop=True), bars[bars["symbol"] == b].reset_index(drop=True), b


def test_registered_and_names():
    f = build_feature({"type": "pair_spread_z", "other": "BBB"})
    assert f.name == "pair_z_bbb_20d" and f.context_symbols == ("BBB",)
    assert PairSpreadZScore("BBB", output="beta").name == "pair_beta_bbb_20d"
    assert PairSpreadZScore("BBB", output="hl").name == "pair_hl_bbb_20d"
    with pytest.raises(ValueError):
        PairSpreadZScore("BBB", hedge="kalman")
    with pytest.raises(ValueError):
        PairSpreadZScore("BBB", output="alpha")


def test_truncate_either_symbol(pair):
    a, b, name_b = pair
    feats = [PairSpreadZScore(name_b, 3, output=o) for o in ("z", "beta", "hl")]
    full = compute_features(a.assign(symbol="A"), feats, context={name_b: b})
    for cut in (1200, 1500):
        t = a["ts_available"].iloc[cut]
        n = int((a["ts_available"] <= t).sum())
        for la, lb in ((a[a["ts_available"] <= t], b), (a, b[b["ts_available"] <= t])):
            part = compute_features(la.assign(symbol="A"), feats, context={name_b: lb})
            pd.testing.assert_frame_equal(part.iloc[:n].reset_index(drop=True), full.iloc[:n].reset_index(drop=True))


def test_assert_no_lookahead_own_leg(pair):
    a, b, name_b = pair
    feature = PairSpreadZScore(name_b, 3)
    full = feature.compute(a, {name_b: b})
    pre = feature.compute(a.iloc[:1500], {name_b: b})
    np.testing.assert_allclose(pre.to_numpy(), full.iloc[:1500].to_numpy(), equal_nan=True)
    assert callable(assert_no_lookahead)


def test_matches_platform_vector():
    """beta and z-score against independent from-definition numbers on raw prices.

    No platform-generated vector exists in the repo, so the reference is recomputed here with numpy from
    the platform's definitions (OLS of p1 on p2 over the strict past window, spread p1 - beta * p2,
    z over a window ending at t inclusive with ddof=1).
    """
    rng = np.random.default_rng(4)
    n = 600
    idx = pd.date_range("2026-10-01T13:31Z", periods=n, freq="1min")
    p2 = 100 + np.cumsum(rng.normal(0, 0.6, n))
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 0.15)
    p1 = 5 + 1.2 * p2 + e
    a = pd.DataFrame({"symbol": "A", "ts": idx - pd.Timedelta(minutes=1), "ts_available": idx, "close": p1})
    b = a.assign(symbol="B", close=p2)
    f = PairSpreadZScore("B", 1, price_space="raw", zscore_bars=100)
    beta = PairSpreadZScore("B", 1, output="beta").compute(a, {"B": b})
    z = f.compute(a, {"B": b})
    ref = pm.rolling_ols_hedge(pd.Series(p1, idx), pd.Series(p2, idx), window_ms=f.window_ms)
    ok = beta.notna().to_numpy()
    assert ok.sum() > 300
    np.testing.assert_allclose(beta.to_numpy()[ok], ref.to_numpy()[ok], atol=1e-12)
    k = 400
    spread = p1 - beta.to_numpy() * p2
    win = spread[k - 99 : k + 1]
    assert z.iloc[k] == pytest.approx((spread[k] - win.mean()) / win.std(ddof=1), abs=1e-6)
    j = int(np.flatnonzero(np.diff(beta.to_numpy()[ok]) != 0)[-1]) + 1 + int(np.argmax(ok))
    lo = int(np.searchsorted(idx.asi8, idx.asi8[j] - int(f.window_ms * 1e6), side="left"))
    assert beta.iloc[j] == pytest.approx(np.polyfit(p2[lo:j], p1[lo:j], 1)[0], abs=1e-6)


def test_log_space_not_exportable():
    assert PairSpreadZScore("B", price_space="raw").exportable is True
    assert PairSpreadZScore("B", price_space="log").exportable is False
    assert PairSpreadZScore("B", hedge="fixed").exportable is True
    with pytest.raises(ValueError):
        PairSpreadZScore("B", price_space="sqrt")


def test_missing_other_leg_gives_nan(pair):
    a, b, name_b = pair
    z = PairSpreadZScore(name_b, 3).compute(a, {name_b: b.drop(b.index[1500])})
    assert np.isnan(z.iloc[1500]) and np.isfinite(z.iloc[1499])
