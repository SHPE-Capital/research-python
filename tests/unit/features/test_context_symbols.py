from __future__ import annotations

import pandas as pd
import pytest

from shpe_research.features.base import (
    Feature,
    align_context,
    assert_no_lookahead,
    compute_features,
)
from tests.conftest import make_bars


class _CtxClose(Feature):
    """Context close aligned backward on ts_available (test double for F8)."""

    feature_type = "_ctx_test"
    context_symbols = ("SPY",)

    @property
    def name(self) -> str:
        return "ctx_close"

    def compute(self, bars: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.Series:
        return align_context(bars, context["SPY"], "close")


def test_align_backward_only():
    bars = make_bars("AAA", n=3)
    ctx = make_bars("SPY", n=3, seed=1)
    ctx["close"] = [10.0, 20.0, 30.0]
    # Context bars 1 and 2 become available 1 ns after the matching row: not usable yet.
    ctx.loc[1, "ts_available"] = bars.loc[1, "ts_available"] + pd.Timedelta(1, "ns")
    ctx.loc[2, "ts_available"] = bars.loc[2, "ts_available"] + pd.Timedelta(1, "ns")
    out = align_context(bars, ctx, "close")
    assert out.index.equals(bars.index)
    assert out.tolist() == [10.0, 10.0, 20.0]
    # exactly equal availability is usable
    ctx.loc[1, "ts_available"] = bars.loc[1, "ts_available"]
    assert align_context(bars, ctx, "close").tolist() == [10.0, 20.0, 20.0]


def test_align_does_not_cross_sessions():
    day1 = make_bars("AAA", n=2, start="2026-10-01T13:30:00Z")
    day2 = make_bars("AAA", n=2, start="2026-10-02T13:30:00Z")
    bars = pd.concat([day1, day2], ignore_index=True)
    ctx = make_bars("SPY", n=2, start="2026-10-01T13:30:00Z")  # no bars on day 2
    ctx["close"] = [1.0, 2.0]
    out = align_context(bars, ctx, "close")
    assert out.iloc[:2].tolist() == [1.0, 2.0]
    assert out.iloc[2:].isna().all()


def test_align_missing_context_is_nan():
    bars = make_bars("AAA", n=3)
    out = align_context(bars, make_bars("SPY").iloc[:0], "close")
    assert out.isna().all() and len(out) == 3


def test_context_not_traded():
    bars = pd.concat([make_bars("AAA", seed=0), make_bars("BBB", seed=2)], ignore_index=True)
    spy = make_bars("SPY", seed=1)
    out = compute_features(bars, [_CtxClose()], context={"SPY": spy})
    assert set(out["symbol"]) == {"AAA", "BBB"}
    assert out["ctx_close"].notna().sum() > 0


def test_missing_context_symbol_raises():
    with pytest.raises(ValueError, match="SPY"):
        compute_features(make_bars("AAA"), [_CtxClose()], context={"QQQ": make_bars("QQQ")})
    with pytest.raises(ValueError, match="SPY"):
        compute_features(make_bars("AAA"), [_CtxClose()])


def test_no_context_features_unchanged():
    from shpe_research.features import PastReturn

    bars = make_bars("AAA")
    a = compute_features(bars, [PastReturn(window=5)])
    b = compute_features(bars, [PastReturn(window=5)], context={"SPY": make_bars("SPY")})
    pd.testing.assert_frame_equal(a, b)


def test_context_feature_passes_no_lookahead():
    bars, spy = make_bars("AAA", n=120), make_bars("SPY", n=120, seed=1)

    class Bound(_CtxClose):
        feature_type = "_ctx_bound"

        def compute(self, bars, context=None):
            return super().compute(bars, {"SPY": spy})

    assert_no_lookahead(Bound(), bars)
