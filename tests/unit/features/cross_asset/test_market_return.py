"""F8: market_return and market_beta against a benchmark context symbol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.features import compute_features
from shpe_research.features.base import build_feature
from shpe_research.features.cross_asset.market_return import MarketBeta, MarketReturn


def benchmark_from(bars: pd.DataFrame, factor: pd.Series | None = None) -> pd.DataFrame:
    """SPY bars on the same clock as `bars`' first symbol."""
    g = bars[bars["symbol"] == bars["symbol"].iloc[0]].reset_index(drop=True)
    rng = np.random.default_rng(1)
    step = factor.reindex(g["ts"]).to_numpy() if factor is not None else rng.normal(0, 0.0007, len(g))
    close = 400 * np.exp(np.cumsum(step))
    open_ = np.r_[400.0, close[:-1]]
    return pd.DataFrame(
        {"symbol": "SPY", "ts": g["ts"], "ts_available": g["ts_available"], "open": open_, "close": close}
    )


@pytest.fixture(scope="module")
def two_sessions():
    return synthetic_bars(symbols=["AAA"], start="2026-04-01", end="2026-04-02")


def test_registered():
    assert isinstance(build_feature({"type": "market_return"}), MarketReturn)
    assert isinstance(build_feature({"type": "market_beta", "window_bars": 100}), MarketBeta)
    assert MarketBeta(window_bars=100).name == "beta_spy_100"
    assert MarketBeta(window_bars=100, output="resid_return").name == "resid_return_spy_100"
    with pytest.raises(ValueError):
        MarketBeta(output="alpha")


def test_truncate_either_symbol(two_sessions):
    bars = two_sessions
    spy = benchmark_from(bars)
    feats = [MarketReturn(), MarketBeta(window_bars=60), MarketBeta(window_bars=60, output="resid_return")]
    full = compute_features(bars, feats, context={"SPY": spy})
    for cut in (100, 390, 500):
        t = bars["ts_available"].iloc[cut]
        n = int((bars["ts_available"] <= t).sum())
        for b, s in ((bars[bars["ts_available"] <= t], spy), (bars, spy[spy["ts_available"] <= t])):
            part = compute_features(b, feats, context={"SPY": s})
            pd.testing.assert_frame_equal(part.iloc[:n].reset_index(drop=True), full.iloc[:n].reset_index(drop=True))


def test_beta_planted():
    syms = ["AAA", "BBB", "CCC"]
    bars = synthetic_bars(symbols=syms, start="2026-03-02", end="2026-04-03", factors={"market": 0.0005, "sectors": 2})
    planted = bars.attrs["planted"]
    bars.attrs = {}  # pandas cannot concat frames whose attrs hold Series
    spy = benchmark_from(bars, planted["factors"]["market_return"])
    out = compute_features(bars, [MarketBeta(window_bars=390 * 5)], context={"SPY": spy})
    for sym in syms:
        got = out.loc[out["symbol"] == sym, "beta_spy_1950"].iloc[-1]
        assert abs(got - planted["beta"][sym]) < 0.1


def test_resid_return_is_return_minus_beta_times_benchmark(two_sessions):
    spy = benchmark_from(two_sessions)
    w = 60
    beta = MarketBeta(window_bars=w).compute(two_sessions, {"SPY": spy})
    resid = MarketBeta(window_bars=w, output="resid_return").compute(two_sessions, {"SPY": spy})
    day = two_sessions["ts"].dt.date
    rs = two_sessions["close"].groupby(day).pct_change()
    rb = spy["close"].groupby(spy["ts"].dt.date).pct_change()
    k = 500
    assert resid.iloc[k] == pytest.approx(rs.iloc[k] - beta.iloc[k] * rb.iloc[k])
    # the window excludes the current bar: beta at k is the OLS slope on rows k-w..k-1
    x, y = rb.iloc[k - w : k], rs.iloc[k - w : k]
    assert beta.iloc[k] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-6)
    assert beta.iloc[:10].isna().all()  # warm-up is NaN, not zero


def test_missing_benchmark_nan(two_sessions):
    spy = benchmark_from(two_sessions)
    spy_gap = spy.drop(spy.index[200])
    ret = MarketReturn().compute(two_sessions, {"SPY": spy_gap})
    assert np.isnan(ret.iloc[200])  # no forward fill across the gap
    assert ret.iloc[199] == pytest.approx(spy["close"].iloc[199] / spy["open"].iloc[0] - 1)
    assert ret.iloc[201] == pytest.approx(spy["close"].iloc[201] / spy["open"].iloc[0] - 1)
    resid = MarketBeta(window_bars=60, output="resid_return").compute(two_sessions, {"SPY": spy_gap})
    assert np.isnan(resid.iloc[200])


def test_session_open_resets(two_sessions):
    spy = benchmark_from(two_sessions)
    ret = MarketReturn().compute(two_sessions, {"SPY": spy})
    day = two_sessions["ts"].dt.date
    day2 = two_sessions.index[day != day.iloc[0]][0]
    assert ret.iloc[day2] == pytest.approx(spy["close"].iloc[day2] / spy["open"].iloc[day2] - 1)
