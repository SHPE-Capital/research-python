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


def test_relative_volume(month):
    from shpe_research.features.volume.relative_volume import RelativeVolume

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


def test_vwap_distance(month):
    from shpe_research.features.volume.vwap_distance import VwapDistance

    values = VwapDistance().compute(month)
    days = session_date(month["ts"])
    pv = (month["vwap"] * month["volume"]).groupby(days).cumsum()
    expected = month["close"] / (pv / month["volume"].groupby(days).cumsum()) - 1
    np.testing.assert_allclose(values, expected, rtol=1e-12)
    assert_no_lookahead(VwapDistance(), month)


def test_realized_vol(month):
    from shpe_research.features.volatility.realized_vol import RealizedVol

    values = RealizedVol(window=15).compute(month)
    days = session_date(month["ts"])
    returns = np.log(month["close"]).groupby(days).diff()
    expected = returns.groupby(days).transform(lambda r: r.rolling(15).std())
    np.testing.assert_allclose(values, expected, rtol=1e-9, equal_nan=True)
    assert_no_lookahead(RealizedVol(window=15), month)


def test_opening_range():
    from shpe_research.features.session.opening_range import OpeningRangeBreakoutDistance, OpeningRangeWidth

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


def test_regime_features():
    from shpe_research.features.regime.regime import RegimeFeature

    reverting = synthetic_bars(symbols=["AAA"], start="2026-04-01", end="2026-04-30", signal_strength=0.5)
    assert RegimeFeature("variance_ratio", 60).compute(reverting).mean() < 1
    assert RegimeFeature("autocorr", 60).compute(reverting).mean() < 0
    for measure in RegimeFeature.MEASURES:
        assert_no_lookahead(RegimeFeature(measure, 60), reverting)


def test_order_flow():
    from shpe_research.features.microstructure.order_flow import SignedVolumeRatio

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


def _spy_frame(ts_available, opens, closes):
    ts_available = pd.DatetimeIndex(ts_available)
    return pd.DataFrame(
        {
            "symbol": "SPY",
            "ts": ts_available - pd.Timedelta(minutes=1),
            "ts_available": ts_available,
            "open": opens,
            "close": closes,
        }
    )


def test_market_return():
    from shpe_research.features.cross_asset.market_return import MarketReturn

    avail = pd.date_range("2026-10-01T13:31Z", periods=3, freq="1min").astype("datetime64[ns, UTC]")
    bars = pd.DataFrame(
        {"symbol": "AAA", "ts": avail - pd.Timedelta(minutes=1), "ts_available": avail, "close": [10.0, 10.1, 10.2]}
    )
    spy = _spy_frame(avail, [400.0, 401.0, 402.0], [401.0, 402.0, 404.0])
    values = MarketReturn().compute(bars, {"SPY": spy})
    np.testing.assert_allclose(values.to_numpy(), [401 / 400 - 1, 402 / 400 - 1, 404 / 400 - 1], rtol=1e-12)
    assert MarketReturn().name == "market_return_spy"

    # A benchmark bar that becomes available 1 ns after the row's ts_available is not used.
    late = spy.copy()
    late.loc[2, "ts_available"] = avail[2] + pd.Timedelta(1, "ns")
    late.loc[2, "close"] = 999.0
    assert MarketReturn().compute(bars, {"SPY": late}).iloc[2] != pytest.approx(999 / 400 - 1)
    head = MarketReturn().compute(bars.iloc[:2], {"SPY": late}).to_numpy()
    np.testing.assert_allclose(head, values.to_numpy()[:2], rtol=1e-12)


def test_pair_spread():
    from shpe_research.features.cross_asset.pair_spread import PairSpreadZScore

    bars = synthetic_bars(
        symbols=["AAA"],
        start="2026-03-02",
        end="2026-04-03",
        cointegrated_pairs=[{"beta": 1.2, "half_life_bars": 60}],
    )
    spec = bars.attrs["planted"]["cointegrated_pairs"]["pairs"][0]
    a, b = spec["symbols"]
    leg_a = bars[bars["symbol"] == a].reset_index(drop=True)
    leg_b = bars[bars["symbol"] == b].reset_index(drop=True)
    # The fixture plants the spread in log prices, so the planted beta is read in log price space.
    kw = {"other": b, "price_space": "log"}  # default lookback_sessions=20
    beta = PairSpreadZScore(output="beta", **kw).compute(leg_a, {b: leg_b})
    hl = PairSpreadZScore(output="hl", **kw).compute(leg_a, {b: leg_b})
    assert abs(beta.iloc[-500:].mean() - spec["beta"]) < 0.05
    assert abs(hl.iloc[-1] - spec["half_life_bars"]) <= 0.3 * spec["half_life_bars"]
    z = PairSpreadZScore(**kw).compute(leg_a, {b: leg_b})
    assert z.iloc[:3].isna().all() and z.iloc[-100:].notna().all()
