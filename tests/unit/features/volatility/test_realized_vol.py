"""Unit tests for RealizedVol (F3): hand-computed values, session reset, gaps, warm-up, causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.features import FEATURE_REGISTRY, build_feature
from shpe_research.features.base import assert_no_lookahead
from shpe_research.features.volatility.realized_vol import RealizedVol


def _bars(ts_list, closes):
    ts = pd.DatetimeIndex(pd.to_datetime(ts_list, utc=True))
    return pd.DataFrame(
        {"symbol": "AAA", "ts": ts, "ts_available": ts + pd.Timedelta(minutes=1), "close": np.asarray(closes, float)}
    )


DAY1 = ["2026-04-01 13:31", "2026-04-01 13:32", "2026-04-01 13:33", "2026-04-01 13:34", "2026-04-01 13:35"]


def test_registered_and_named():
    assert FEATURE_REGISTRY["realized_vol"] is RealizedVol
    f = build_feature({"type": "realized_vol", "window": 3})
    assert f.name == "realized_vol_3m"


def test_hand_computed_and_warmup():
    closes = [100.0, 101.0, 100.0, 102.0, 101.0]
    vals = RealizedVol(window=3).compute(_bars(DAY1, closes))
    r = np.diff(np.log(closes))
    assert vals.iloc[:3].isna().all()  # needs 3 returns -> first available on row 3
    assert vals.iloc[3] == pytest.approx(np.std(r[:3], ddof=1))
    assert vals.iloc[4] == pytest.approx(np.std(r[1:4], ddof=1))
    assert vals.dtype == np.float64


def test_session_reset():
    ts = DAY1 + ["2026-04-02 13:31", "2026-04-02 13:32", "2026-04-02 13:33", "2026-04-02 13:34"]
    closes = [100, 101, 100, 102, 101, 50, 51, 50, 52]
    vals = RealizedVol(window=3).compute(_bars(ts, closes))
    assert vals.iloc[5:8].isna().all()  # overnight gap is not a return; warm-up restarts
    assert vals.iloc[8] == pytest.approx(np.std(np.diff(np.log(closes[5:])), ddof=1))


def test_gap_is_not_a_one_minute_return():
    ts = ["2026-04-01 13:31", "2026-04-01 13:32", "2026-04-01 13:33", "2026-04-01 13:35", "2026-04-01 13:36"]
    closes = [100.0, 101.0, 100.0, 105.0, 104.0]
    vals = RealizedVol(window=2).compute(_bars(ts, closes))
    r = [np.log(101 / 100), np.log(100 / 101), np.log(104 / 105)]  # the 13:33 -> 13:35 jump is skipped
    assert vals.iloc[2] == pytest.approx(np.std(r[:2], ddof=1))
    assert np.isnan(vals.iloc[3])  # the bar after a gap has no valid return of its own
    assert vals.iloc[4] == pytest.approx(np.std([r[1], r[2]], ddof=1))


def test_invalid_window():
    with pytest.raises(ValueError):
        RealizedVol(window=1)


def test_no_lookahead_with_gaps():
    bars = synthetic_bars(symbols=["AAA"], start="2026-04-01", end="2026-04-03")
    bars = bars.drop(bars.index[[10, 11, 50, 200]]).reset_index(drop=True)
    assert_no_lookahead(RealizedVol(window=15), bars)


def test_lookahead_check_catches_centered_window():
    class Centered(RealizedVol):
        def compute(self, bars):
            r = np.log(bars["close"]).diff()
            return r.rolling(self.window, center=True).std()

    bars = synthetic_bars(symbols=["AAA"], start="2026-04-01", end="2026-04-02")
    with pytest.raises(AssertionError):
        assert_no_lookahead(Centered(window=15), bars)
