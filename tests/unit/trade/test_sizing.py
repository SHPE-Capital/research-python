"""T2 acceptance tests: sizing rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.trade.sizing import EXPORTABLE_RULES, size_candidates
from shpe_research.trade.types import SizingRule


def _bars(close=123.45, symbol="AAA"):
    ts = pd.date_range("2026-01-05 14:30", periods=3, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"symbol": symbol, "ts": ts, "ts_available": ts + pd.Timedelta(minutes=1), "open": close,
         "high": close, "low": close, "close": close, "volume": 1000}
    )


def _cands(bars, n=1, **cols):
    df = pd.DataFrame({"symbol": "AAA", "decision_ts": [bars["ts_available"].iloc[0]] * n,
                       "side": "long", "strategy_id": "t:1"})
    for k, v in cols.items():
        df[k] = v
    return df


def test_fixed_notional_matches_platform():
    b = _bars(123.45)
    q = SizingRule("fixed_notional", notional_usd=5000).size(_cands(b), b, 100_000)
    assert q.iloc[0] == 40 == int(np.floor(5000 / 123.45))
    assert q.dtype.kind == "i"


def test_risk_per_trade():
    b = _bars(100.0)
    c = _cands(b, stop_price=98.0)
    assert SizingRule("risk_per_trade", risk_usd=100).size(c, b, 1e5).iloc[0] == 50
    assert SizingRule("risk_per_trade", risk_pct=0.001).size(c, b, 1e5).iloc[0] == 50


def test_floor_and_zero():
    b = _bars(600.0)
    q = SizingRule("fixed_notional", notional_usd=500).size(_cands(b), b, 1e5)
    assert q.iloc[0] == 0
    # zero stop distance and missing bar also give 0
    c = _cands(b, stop_price=600.0)
    assert SizingRule("risk_per_trade", risk_usd=100).size(c, b, 1e5).iloc[0] == 0
    c2 = _cands(b)
    c2["decision_ts"] = pd.Timestamp("2025-01-01", tz="UTC")
    assert SizingRule("fixed_qty", qty=5).size(c2, b, 1e5).iloc[0] == 5
    assert SizingRule("fixed_notional", notional_usd=5000).size(c2, b, 1e5).iloc[0] == 0


def test_prob_to_size_monotone():
    b = _bars(100.0)
    probs = [0.3, 0.5, 0.55, 0.7, 0.9, 1.0]
    c = _cands(b, n=len(probs), p=probs)
    q = SizingRule("prob_to_size", prob_column="p", min_prob=0.5, max_notional_usd=10_000).size(c, b, 1e5)
    assert list(q[:2]) == [0, 0]
    assert (q.diff().dropna() >= 0).all()
    assert q.iloc[-1] == 100


def test_kelly_capped():
    b = _bars(100.0)
    probs = [0.4, 0.6, 0.9, 0.99]
    c = _cands(b, n=4, p=probs)
    rule = SizingRule("kelly_fractional", fraction=0.5, prob_column="p", payoff_ratio=1.5, cap_notional_usd=20_000)
    q = rule.size(c, b, 1e6)
    assert (q * 100.0 <= 20_000 + 1e-9).all()
    assert q.iloc[0] == 0 and q.iloc[-1] == 200


def test_inverse_vol():
    b = _bars(100.0)
    c = _cands(b, vol=0.01)
    assert SizingRule("inverse_vol", target_risk_usd=50, vol_column="vol").size(c, b, 1e5).iloc[0] == 50


def test_native_and_positional_params():
    b = _bars()
    assert SizingRule("native").size(_cands(b, qty=7), b, 1e5).iloc[0] == 7
    assert SizingRule("native").size(_cands(b), b, 1e5).iloc[0] == 0
    assert SizingRule("fixed_qty", qty=3).params == {"qty": 3}
    assert size_candidates(SizingRule("fixed_qty", qty=3), _cands(b), b, 1e5).iloc[0] == 3


def test_unknown_rule_and_missing_param():
    with pytest.raises(ValueError):
        SizingRule("nope")
    with pytest.raises(ValueError):
        SizingRule("fixed_notional")


def test_exportable_flags():
    assert EXPORTABLE_RULES == {"fixed_qty", "fixed_notional", "native"}
    for r, kw in [("fixed_qty", {"qty": 1}), ("fixed_notional", {"notional_usd": 1}), ("native", {})]:
        assert SizingRule(r, **kw).exportable
    assert not SizingRule("risk_per_trade", risk_usd=1).exportable
    assert not SizingRule("inverse_vol", target_risk_usd=1, vol_column="v").exportable
    assert not SizingRule("prob_to_size", prob_column="p", min_prob=0.5, max_notional_usd=1).exportable
