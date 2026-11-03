"""SS1: pair selection on a formation window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.strategies.statarb.selection import (
    PAIRSET_COLUMNS,
    prices_from_bars,
    select_pairs,
)

PAIRS = [
    {"beta": 1.3, "half_life_bars": 60, "spread_sd": 0.01},
    {"beta": 0.8, "half_life_bars": 120, "spread_sd": 0.01},
    {"trap": True},
]
START, END = "2026-03-02", "2026-04-10"


@pytest.fixture(scope="module")
def pair_prices():
    bars = synthetic_bars(start=START, end=END, cointegrated_pairs=PAIRS)
    return prices_from_bars(bars, "10min"), bars.attrs["planted"]["cointegrated_pairs"]


def _key(a, b):
    return frozenset((a, b))


def test_planted_pairs_found(pair_prices):
    prices, planted = pair_prices
    ps = select_pairs(prices, (prices.index[0], prices.index[-1]), "all_pairs", pvalue_max=1.0)
    assert list(ps.columns) == PAIRSET_COLUMNS
    top3 = {_key(a, b) for a, b in zip(ps["leg1"][:3], ps["leg2"][:3], strict=True)}
    for p in planted["pairs"]:
        assert _key(*p["symbols"]) in top3
    # beta and half-life are filled and in the right neighbourhood
    row = ps[(ps.leg1 == "P1A") & (ps.leg2 == "P1B")].iloc[0]
    assert row.beta == pytest.approx(1.3, rel=0.1)
    assert row.half_life_bars == pytest.approx(6.0, rel=0.6)  # planted 60 one-minute bars = 6 ten-minute bars
    assert np.isfinite(row.coint_pvalue) and row.coint_pvalue < 0.01


def test_trap_pair_rejected(pair_prices):
    prices, planted = pair_prices
    window = (prices.index[0], prices.index[-1])
    loose = select_pairs(prices, window, "all_pairs", pvalue_max=1.0)
    assert any(_key(a, b) == _key(*planted["trap"]["symbols"]) for a, b in zip(loose.leg1, loose.leg2, strict=True))
    strict = select_pairs(prices, window, "all_pairs", pvalue_max=0.05)
    got = {_key(a, b) for a, b in zip(strict.leg1, strict.leg2, strict=True)}
    assert _key(*planted["trap"]["symbols"]) not in got
    for p in planted["pairs"]:
        assert _key(*p["symbols"]) in got


def test_formation_only(pair_prices):
    prices, _ = pair_prices
    end = prices.index[len(prices) // 2]
    start = prices.index[0]
    cut = prices.loc[:end]
    for method in ("all_pairs", "distance", "correlation", "pca_cluster"):
        kw = {"pvalue_max": 1.0}
        full = select_pairs(prices, (start, end), method, **kw)
        trunc = select_pairs(cut, (start, end), method, **kw)
        pd.testing.assert_frame_equal(full, trunc)
        assert full.attrs["n_tested"] == trunc.attrs["n_tested"]
    # changing data after the end changes nothing either
    mutated = prices.copy()
    mutated.loc[mutated.index > end] *= 3.0
    pd.testing.assert_frame_equal(
        select_pairs(prices, (start, end), "all_pairs"), select_pairs(mutated, (start, end), "all_pairs")
    )


def test_n_tested(pair_prices):
    prices, _ = pair_prices
    n = prices.shape[1]
    ps = select_pairs(prices, (prices.index[0], prices.index[-1]), "all_pairs")
    assert ps.attrs["n_tested"] == n * (n - 1) // 2
    assert ps.attrs["method"] == "all_pairs"
    assert ps.attrs["formation"] == (prices.index[0], prices.index[-1])
    assert len(ps) <= ps.attrs["n_tested"]
    # within-sector restriction reduces the count
    sectors = {s: ("a" if i % 2 else "b") for i, s in enumerate(prices.columns)}
    sub = select_pairs(prices, (prices.index[0], prices.index[-1]), "all_pairs", sectors=sectors)
    assert sub.attrs["n_tested"] < ps.attrs["n_tested"]


def test_methods_fill_columns_and_filters(pair_prices):
    prices, planted = pair_prices
    window = (prices.index[0], prices.index[-1])
    for method in ("distance", "correlation", "pca_cluster", "all_pairs"):
        ps = select_pairs(prices, window, method, pvalue_max=1.0, max_pairs=4)
        assert len(ps) <= 4
        assert ps.attrs["method"] == method
        if len(ps):
            assert ps[["beta", "half_life_bars", "coint_pvalue", "score"]].notna().all().all()
    ps = select_pairs(prices, window, "all_pairs", pvalue_max=1.0, min_half_life_bars=1e9)
    assert ps.empty and ps.attrs["n_tested"] > 0
    ps = select_pairs(prices, window, "all_pairs", pvalue_max=1.0, hedge="fixed", beta=1.0)
    assert (ps.beta == 1.0).all()
    with pytest.raises(ValueError):
        select_pairs(prices, window, "nope")


def test_missing_bars_and_empty_window(pair_prices):
    prices, _ = pair_prices
    holey = prices.copy()
    holey.iloc[100:200, 0] = np.nan
    ps = select_pairs(holey, (holey.index[0], holey.index[-1]), "all_pairs", pvalue_max=1.0)
    assert ps.attrs["n_tested"] == prices.shape[1] * (prices.shape[1] - 1) // 2
    with pytest.raises(ValueError):
        after = prices.index[-1] + pd.Timedelta(days=1)
        select_pairs(prices, (after, after + pd.Timedelta(days=1)), "distance")


def test_pca_cluster_ari():
    syms = [f"S{i:02d}" for i in range(18)]
    bars = synthetic_bars(symbols=syms, start=START, end=END, factors={"sectors": 3})
    sector = bars.attrs["planted"]["sector"]
    prices = prices_from_bars(bars, "10min")
    ps = select_pairs(prices, (prices.index[0], prices.index[-1]), "pca_cluster", pvalue_max=1.0)
    labels = ps.attrs["clusters"].reindex(syms)
    assert adjusted_rand_score(sector.reindex(syms), labels) >= 0.8
    k = labels[labels >= 0].value_counts()
    assert ps.attrs["n_tested"] == int((k * (k - 1) // 2).sum())
    assert ps["cluster"].notna().all()


def test_deterministic(pair_prices):
    prices, _ = pair_prices
    w = (prices.index[0], prices.index[-1])
    a = select_pairs(prices, w, "pca_cluster", pvalue_max=1.0)
    b = select_pairs(prices, w, "pca_cluster", pvalue_max=1.0)
    pd.testing.assert_frame_equal(a, b)
