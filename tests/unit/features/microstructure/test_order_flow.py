"""Unit tests for SignedVolumeRatio (F6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.features.base import assert_no_lookahead, compute_features
from shpe_research.features.microstructure.order_flow import SignedVolumeRatio
from shpe_research.testing.checks import check_feature


def _frame(starts, buy, sell, vol):
    ts = pd.DatetimeIndex(pd.to_datetime(starts, utc=True)).as_unit("ns")
    return pd.DataFrame(
        {"symbol": "AAA", "ts": ts, "ts_available": ts + pd.Timedelta(minutes=1),
         "buy_volume": buy, "sell_volume": sell, "trade_volume": vol}
    )


def test_hand_values_missing_minute_and_session_boundary():
    # Day 1 (EDT): 13:30, 13:31, [13:32 missing], 13:33; Day 2: 13:30, 13:31
    starts = ["2026-10-01T13:30Z", "2026-10-01T13:31Z", "2026-10-01T13:33Z",
              "2026-10-02T13:30Z", "2026-10-02T13:31Z"]
    bars = _frame(starts, [10.0, 0, 4, 10, 2], [0.0, 10, 0, 0, 6], [10.0, 10, 4, 10, 8])
    v = SignedVolumeRatio(window=2).compute(bars)
    assert np.isnan(v.iloc[0])  # warm-up
    assert v.iloc[1] == pytest.approx(0.0)  # (10-10)/20
    assert v.iloc[2] == pytest.approx(1.0)  # window (13:32, 13:34] holds only the 13:33 minute; 13:32 stays missing
    assert np.isnan(v.iloc[3])  # new session: warm-up, no carry-over
    assert v.iloc[4] == pytest.approx(6 / 18)  # (12 buy - 6 sell) / 18


def test_no_carry_over_across_sessions():
    bars = _frame(["2026-10-01T19:59Z", "2026-10-02T13:30Z", "2026-10-02T13:31Z"], [9.0, 1, 1], [0.0, 0, 3],
                  [9.0, 1, 4])
    v = SignedVolumeRatio(window=2).compute(bars)
    assert np.isnan(v.iloc[1])
    assert v.iloc[2] == pytest.approx((2 - 3) / 5)


def test_missing_summary_and_zero_volume_are_nan():
    bars = _frame(["2026-10-01T13:30Z", "2026-10-01T13:31Z", "2026-10-01T13:32Z"],
                  [1.0, np.nan, 0], [0.0, np.nan, 0], [1.0, np.nan, 0])
    v = SignedVolumeRatio(window=2).compute(bars)
    assert np.isnan(v.iloc[1])  # no summary this minute
    assert np.isnan(v.iloc[2])  # window (13:31, 13:33] has zero volume


def _two_sessions():
    rng = np.random.default_rng(1)
    starts = pd.date_range("2026-03-02T14:30Z", periods=390, freq="1min").append(
        pd.date_range("2026-03-03T14:30Z", periods=390, freq="1min")
    )
    buy = rng.integers(1, 50, len(starts)).astype(float)
    sell = rng.integers(1, 50, len(starts)).astype(float)
    return _frame(starts, buy, sell, buy + sell)


def test_name_dtype_and_no_lookahead():
    f = SignedVolumeRatio(window=3)
    assert f.name == "signed_volume_ratio_3m"
    rng = np.random.default_rng(0)
    starts = pd.date_range("2026-10-01T13:30Z", periods=40, freq="1min").delete([7, 8, 20])
    buy = rng.integers(0, 50, len(starts)).astype(float)
    sell = rng.integers(0, 50, len(starts)).astype(float)
    bars = _frame(starts, buy, sell, buy + sell)
    out = compute_features(bars, [f])
    assert out[f.name].dtype == np.float64
    assert_no_lookahead(f, bars)
    check_feature(f, _two_sessions())


def test_noncausal_variant_is_caught():
    class Leaky(SignedVolumeRatio):
        def compute(self, bars):
            return super().compute(bars).shift(-1)

    starts = pd.date_range("2026-10-01T13:30Z", periods=20, freq="1min")
    bars = _frame(starts, np.arange(20.0), np.ones(20), np.arange(20.0) + 1)
    with pytest.raises(AssertionError):
        assert_no_lookahead(Leaky(window=2), bars)
