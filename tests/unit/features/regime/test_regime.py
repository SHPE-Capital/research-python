from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.features.base import FEATURE_REGISTRY, assert_no_lookahead, build_feature
from shpe_research.features.regime.regime import RegimeFeature


def _bars(closes, start="2026-10-01T13:30Z", skip=()):
    ts = pd.date_range(start, periods=len(closes) + len(skip), freq="1min").astype("datetime64[ns, UTC]")
    ts = pd.DatetimeIndex([t for i, t in enumerate(ts) if i not in skip])
    return pd.DataFrame({"ts": ts, "ts_available": ts + pd.Timedelta(minutes=1), "close": np.asarray(closes, float)})


def test_registered_and_named():
    assert FEATURE_REGISTRY["regime"] is RegimeFeature
    f = build_feature({"type": "regime", "measure": "efficiency", "window": 4})
    assert f.name == "efficiency_4m"
    with pytest.raises(ValueError):
        RegimeFeature("nope")
    with pytest.raises(ValueError):
        RegimeFeature("variance_ratio", window=9)  # needs window >= 2q


def test_efficiency_hand_values_and_warmup():
    closes = 100 * np.exp(np.cumsum([0, 0.01, 0.01, -0.01, 0.02, 0.0]))
    out = RegimeFeature("efficiency", 4).compute(_bars(closes))
    assert out.iloc[:4].isna().all()  # 4 returns need 5 bars
    assert out.iloc[4] == pytest.approx(0.03 / 0.05)  # returns .01 .01 -.01 .02
    assert out.iloc[5] == pytest.approx(0.02 / 0.04)  # returns .01 -.01 .02 0
    assert out.dtype == np.float64


def test_autocorr_alternating_is_minus_one_and_trend_efficiency_one():
    zig = 100 * np.exp(np.cumsum([0, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]))
    ac = RegimeFeature("autocorr", 6).compute(_bars(zig))
    assert ac.iloc[-1] == pytest.approx(-1.0)
    trend = 100 * np.exp(np.arange(12) * 0.01)
    assert RegimeFeature("efficiency", 5).compute(_bars(trend)).iloc[-1] == pytest.approx(1.0)


def test_variance_ratio_hand_value():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.001, 30)
    closes = 100 * np.exp(np.concatenate([[0], np.cumsum(r)]))
    w, q = 20, 5
    win = r[-w:]
    sums = np.convolve(win, np.ones(q), "valid")
    expected = sums.var(ddof=1) / (q * win.var(ddof=1))
    assert RegimeFeature("variance_ratio", w).compute(_bars(closes)).iloc[-1] == pytest.approx(expected)


def test_gap_and_session_boundary_give_nan():
    closes = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.001, 20)))
    gapped = _bars(closes, skip=(8,))  # one missing minute
    out = RegimeFeature("efficiency", 5).compute(gapped)
    # rows whose last 5 returns all use consecutive minutes are valid; the 2 rows around the gap are not
    assert out.iloc[:5].isna().all()
    assert out.iloc[8:13].isna().all()
    assert out.iloc[13:].notna().all()
    # next session: window never spans the overnight boundary
    two = pd.concat([_bars(closes[:10]), _bars(closes[10:], start="2026-10-02T13:30Z")], ignore_index=True)
    out2 = RegimeFeature("efficiency", 5).compute(two)
    assert out2.iloc[10:15].isna().all() and out2.iloc[15:].notna().all()


def test_zero_variance_is_nan_not_zero():
    flat = _bars(np.full(12, 100.0))
    for m in RegimeFeature.MEASURES:
        assert RegimeFeature(m, 10).compute(flat).isna().all()


def test_no_lookahead_and_centered_detects_cheat():
    closes = 100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.001, 200)))
    bars = _bars(closes)
    for m in RegimeFeature.MEASURES:
        assert_no_lookahead(RegimeFeature(m, 20), bars)

    class Centered(RegimeFeature):
        def compute(self, b):
            r = np.log(b["close"]).diff()
            return r.rolling(20, center=True).std()

    with pytest.raises(AssertionError):
        assert_no_lookahead(Centered("autocorr", 20), bars)
