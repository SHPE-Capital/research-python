"""Acceptance tests for the label tasks. Remove a test's skip marker in the PR that implements it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.labels.forward_return import ForwardReturn


def test_multi_horizon():
    from shpe_research.labels.multi_horizon import forward_returns

    bars = synthetic_bars(symbols=["AAA"], start="2026-10-01", end="2026-10-02")
    out = forward_returns(bars, horizons=(5, 30))
    for h in (5, 30):
        single = ForwardReturn(h).compute(bars)[f"fwd_return_{h}m"]
        np.testing.assert_allclose(out[f"fwd_return_{h}m"], single, equal_nan=True)
    costed = forward_returns(bars, horizons=(30,), cost_bps=10)
    np.testing.assert_allclose(costed["fwd_return_30m"], out["fwd_return_30m"] - 0.001, equal_nan=True)


def test_multi_horizon_end_ts_and_vol_normalized():
    from shpe_research.features.volatility.realized_vol import RealizedVol
    from shpe_research.labels.multi_horizon import forward_returns

    bars = synthetic_bars(symbols=["AAA", "BBB"], start="2026-10-01", end="2026-10-02")
    raw = forward_returns(bars, horizons=(5, 30))
    assert {"symbol", "decision_ts", "label_end_ts_5m", "label_end_ts_30m"} <= set(raw.columns)
    assert (raw["label_end_ts_5m"] - raw["decision_ts"] == pd.Timedelta(minutes=5)).all()
    assert raw.index.equals(bars.index)
    norm = forward_returns(bars, horizons=(30,), vol_normalized=True)
    sym = bars["symbol"] == "AAA"
    vol = RealizedVol(15).compute(bars[sym].sort_values("ts"))
    expected = raw.loc[sym, "fwd_return_30m"] / (vol * np.sqrt(30))
    got = norm.loc[sym, "fwd_return_30m"]
    ok = expected.notna() & np.isfinite(expected)
    assert ok.any()
    np.testing.assert_allclose(got[ok], expected[ok])
    assert got[vol.isna()].isna().all()  # warm-up stays NaN, not zero


def _bars(rows):
    ts = to_utc_ns(pd.date_range("2026-10-01T13:30Z", periods=len(rows), freq="1min"))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    return df.assign(symbol="AAA", ts=ts, ts_available=ts + pd.Timedelta(minutes=1), volume=1000,
                     vwap=df["close"], trade_count=pd.array([10] * len(df), dtype="Int64"), session="regular")


@pytest.mark.skip(reason="TODO(L2)")
def test_exit_policy():
    from shpe_research.labels.exit_policy import ExitPolicy, simulate_exits

    bars = _bars([
        [100.0, 100.2, 99.9, 100.1],  # decision after this bar
        [100.2, 100.5, 100.1, 100.4],  # entry at this open (+costs)
        [100.4, 100.6, 100.3, 100.5],
        [98.5, 98.8, 98.0, 98.4],  # gaps below the 99.0 stop: fills at 98.5, not 99.0
    ])
    candidates = pd.DataFrame({"symbol": ["AAA"], "decision_ts": [bars["ts_available"].iloc[0]], "stop": [99.0]})
    policy = ExitPolicy(max_minutes=30, half_spread_bps=0, slippage_bps=0)
    out = simulate_exits(bars, candidates, policy).iloc[0]
    assert out["entry_price"] == pytest.approx(100.2)
    assert out["exit_reason"] == "stop"
    assert out["exit_price"] == pytest.approx(98.5)
    assert out["net_return"] == pytest.approx(98.5 / 100.2 - 1)
