"""T1 acceptance: the L2 exit-policy test, ported onto trade/ (stop gap fills at the open)."""

from __future__ import annotations

import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns
from shpe_research.trade.simulate import simulate_trades
from shpe_research.trade.types import ExitPolicy, FillModel

ZERO = FillModel(half_spread_bps=0, slippage_bps=0, commission_per_share=0)


def _bars(rows):
    ts = to_utc_ns(pd.date_range("2026-10-01T13:30Z", periods=len(rows), freq="1min"))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    return df.assign(symbol="AAA", ts=ts, ts_available=ts + pd.Timedelta(minutes=1), volume=1000,
                     vwap=df["close"], trade_count=pd.array([10] * len(df), dtype="Int64"), session="regular")


def test_exit_policy():
    bars = _bars([
        [100.0, 100.2, 99.9, 100.1],  # decision after this bar
        [100.2, 100.5, 100.1, 100.4],  # entry at this open (+costs)
        [100.4, 100.6, 100.3, 100.5],
        [98.5, 98.8, 98.0, 98.4],  # gaps below the 99.0 stop: fills at 98.5, not 99.0
    ])
    candidates = pd.DataFrame({
        "symbol": ["AAA"], "decision_ts": [bars["ts_available"].iloc[0]], "side": ["long"],
        "strategy_id": ["t:1"], "stop_price": [99.0],
    })
    out = simulate_trades(bars, candidates, ExitPolicy(stop_rule="price"), fill=ZERO).iloc[0]
    assert out["entry_price"] == pytest.approx(100.2)
    assert out["exit_reason"] == "stop_loss"
    assert out["exit_price"] == pytest.approx(98.5)
    assert out["net_return"] == pytest.approx(98.5 / 100.2 - 1)
