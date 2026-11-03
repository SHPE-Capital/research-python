"""Opening-range features (F4): width and breakout distance for 5, 15 and 30 minute ranges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.features import FEATURE_REGISTRY, build_feature
from shpe_research.features.base import assert_no_lookahead
from shpe_research.features.session.opening_range import OpeningRangeBreakoutDistance, OpeningRangeWidth


def _two_days() -> pd.DataFrame:
    bars = synthetic_bars(symbols=["AAA"], start="2026-10-01", end="2026-10-02").reset_index(drop=True)
    return bars


def test_registered():
    assert FEATURE_REGISTRY["or_width"] is OpeningRangeWidth
    assert FEATURE_REGISTRY["or_breakout_dist"] is OpeningRangeBreakoutDistance
    assert build_feature({"type": "or_width", "range_minutes": 5}).name == "or_width_5m"
    assert OpeningRangeBreakoutDistance(30).name == "or_breakout_dist_30m"


@pytest.mark.parametrize("m", [5, 15, 30])
def test_hand_values_and_warmup(m):
    bars = _two_days()
    day2 = bars.index[bars["ts"].dt.tz_convert("America/New_York").dt.date == pd.Timestamp("2026-10-02").date()]
    first = day2[0]
    rng = day2[:m]
    bars.loc[rng, ["open", "close"]] = 50.0
    bars.loc[rng, "high"] = 50.5
    bars.loc[rng, "low"] = 49.5
    bars.loc[day2[1], "high"] = 52.0  # H
    bars.loc[day2[m - 1], "low"] = 48.0  # L (last range bar)
    bars.loc[first, "open"] = 50.0
    bars.loc[day2[m], "close"] = 52.26
    width = OpeningRangeWidth(m).compute(bars)
    brk = OpeningRangeBreakoutDistance(m).compute(bars)
    assert width.dtype == np.float64
    assert width.loc[day2[: m - 1]].isna().all()
    assert brk.loc[day2[: m - 1]].isna().all()
    assert width.loc[day2[m - 1]] == pytest.approx(4.0 / 50.0)  # the last range bar completes at open+m
    assert width.loc[day2[m]] == pytest.approx(4.0 / 50.0)
    assert brk.loc[day2[m]] == pytest.approx(0.005)
    assert_no_lookahead(OpeningRangeWidth(m), bars)
    assert_no_lookahead(OpeningRangeBreakoutDistance(m), bars)


def test_session_boundary_no_carry_over():
    bars = _two_days()
    out = OpeningRangeWidth(15).compute(bars)
    ny = bars["ts"].dt.tz_convert("America/New_York")
    day2 = ny.dt.date == pd.Timestamp("2026-10-02").date()
    assert out[day2].iloc[:14].isna().all()  # day 1's range is not used on day 2
    assert out[day2].iloc[14:].notna().all()


def test_missing_minute_stays_missing_and_range_still_uses_present_bars():
    bars = _two_days()
    dropped = bars.drop(index=bars.index[5]).reset_index(drop=True)  # a minute inside day 1's range
    out = OpeningRangeWidth(15).compute(dropped)
    assert out.iloc[:13].isna().all()  # one range row is gone, so the 09:44 bar is row 13
    assert out.iloc[13:389].notna().all()  # day 1 now has 389 bars
    assert_no_lookahead(OpeningRangeBreakoutDistance(15), dropped)


def test_session_without_range_bars_is_nan():
    bars = _two_days()
    late = bars[bars["ts"].dt.tz_convert("America/New_York").dt.hour >= 10].reset_index(drop=True)
    assert OpeningRangeWidth(15).compute(late).isna().all()
    assert OpeningRangeBreakoutDistance(15).compute(late).isna().all()


def test_bad_range_minutes():
    with pytest.raises(ValueError):
        OpeningRangeWidth(0)
