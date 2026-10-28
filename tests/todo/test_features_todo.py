"""Acceptance tests for the feature tasks. Remove a test's skip marker in the PR that implements it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.data.calendar import session_date
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.features.base import assert_no_lookahead


@pytest.fixture(scope="module")
def month():
    return synthetic_bars(symbols=["AAA"], start="2026-04-01", end="2026-05-15")


@pytest.mark.skip(reason="TODO(F1)")
def test_relative_volume(month):
    from shpe_research.features.relative_volume import RelativeVolume

    feature = RelativeVolume(lookback_sessions=20)
    values = feature.compute(month)
    days = session_date(month["ts"])
    ordered = np.sort(days.unique())
    assert values[days.isin(ordered[:20])].isna().all()
    assert values[days == ordered[20]].notna().all()

    # Hand check: minute 20 of session 21 against the same minute of sessions 1-20.
    minute = month.groupby(days).cumcount()
    cumvol = month.groupby(days)["volume"].cumsum()
    reference = cumvol[(minute == 19) & days.isin(ordered[:20])].mean()
    row = month.index[(minute == 19) & (days == ordered[20])][0]
    assert values[row] == pytest.approx(cumvol[row] / reference)
    assert_no_lookahead(feature, month)


@pytest.mark.skip(reason="TODO(F2)")
def test_vwap_distance(month):
    from shpe_research.features.vwap_distance import VwapDistance

    values = VwapDistance().compute(month)
    days = session_date(month["ts"])
    pv = (month["vwap"] * month["volume"]).groupby(days).cumsum()
    expected = month["close"] / (pv / month["volume"].groupby(days).cumsum()) - 1
    np.testing.assert_allclose(values, expected, rtol=1e-12)
    assert_no_lookahead(VwapDistance(), month)


@pytest.mark.skip(reason="TODO(F3)")
def test_realized_vol(month):
    from shpe_research.features.realized_vol import RealizedVol

    values = RealizedVol(window=15).compute(month)
    days = session_date(month["ts"])
    returns = np.log(month["close"]).groupby(days).diff()
    expected = returns.groupby(days).transform(lambda r: r.rolling(15).std())
    np.testing.assert_allclose(values, expected, rtol=1e-9, equal_nan=True)
    assert_no_lookahead(RealizedVol(window=15), month)


@pytest.mark.skip(reason="TODO(F4)")
def test_opening_range():
    from shpe_research.features.opening_range import OpeningRangeBreakoutDistance, OpeningRangeWidth

    bars = synthetic_bars(symbols=["AAA"], start="2026-10-01", end="2026-10-01")
    first15 = bars.index[:15]
    bars.loc[first15, ["open", "close"]] = 99.0
    bars.loc[first15, "high"] = 99.5
    bars.loc[first15, "low"] = 98.5
    bars.loc[bars.index[3], "high"] = 100.0  # H
    bars.loc[bars.index[7], "low"] = 98.0  # L
    bars.loc[bars.index[15], ["open", "high", "low", "close"]] = [100.0, 100.2, 99.9, 100.10]
    breakout = OpeningRangeBreakoutDistance().compute(bars)
    width = OpeningRangeWidth().compute(bars)
    assert breakout.iloc[:14].isna().all()  # range unknown until 09:45
    assert breakout.iloc[15] == pytest.approx(0.001)
    assert width.iloc[15] == pytest.approx(2.0 / 99.0)
    assert_no_lookahead(OpeningRangeBreakoutDistance(), bars)


@pytest.mark.skip(reason="TODO(F5)")
def test_regime_features():
    from shpe_research.features.regime import RegimeFeature

    reverting = synthetic_bars(symbols=["AAA"], start="2026-04-01", end="2026-04-30", signal_strength=0.5)
    assert RegimeFeature("variance_ratio", 60).compute(reverting).mean() < 1
    assert RegimeFeature("autocorr", 60).compute(reverting).mean() < 0
    for measure in RegimeFeature.MEASURES:
        assert_no_lookahead(RegimeFeature(measure, 60), reverting)


@pytest.mark.skip(reason="TODO(F6): needs D3 minute summaries joined onto bars")
def test_order_flow():
    from shpe_research.features.order_flow import SignedVolumeRatio

    bars = pd.DataFrame(
        {
            "ts_available": pd.date_range("2026-10-01T13:31Z", periods=6, freq="1min").astype("datetime64[ns, UTC]"),
            "buy_volume": [10, 0, 5, 5, 0, 10],
            "sell_volume": [0, 10, 5, 0, 5, 10],
            "trade_volume": [10, 10, 10, 5, 5, 20],
        }
    )
    values = SignedVolumeRatio(window=2).compute(bars)
    assert values.iloc[1] == pytest.approx(0.0)
    assert values.iloc[3] == pytest.approx(5 / 15)


@pytest.mark.skip(reason="TODO(F8): needs context symbols in compute_features")
def test_market_return():
    raise NotImplementedError("Write with the F7 owner once the context-symbol extension is agreed")


@pytest.mark.skip(reason="TODO(F7): needs context symbols in compute_features")
def test_pair_spread():
    raise NotImplementedError("Write with the F8 owner once the context-symbol extension is agreed")
