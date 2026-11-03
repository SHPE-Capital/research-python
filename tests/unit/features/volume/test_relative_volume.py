"""Hand-checked cases for RelativeVolume (F1): missing minutes, session boundaries, early closes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.features.base import FEATURE_REGISTRY, assert_no_lookahead, build_feature
from shpe_research.features.volume.relative_volume import RelativeVolume


def _bars(spec: dict[str, dict[int, float]], open_utc: dict[str, str]) -> pd.DataFrame:
    """spec: {session: {minute_of_session (1 = first bar): volume}}; ts is the bar's start."""
    rows = []
    for day, minutes in spec.items():
        open_ts = pd.Timestamp(f"{day} {open_utc[day]}", tz="UTC")
        for m, vol in sorted(minutes.items()):
            ts = open_ts + pd.Timedelta(minutes=m - 1)
            rows.append({"symbol": "AAA", "ts": ts, "ts_available": ts + pd.Timedelta(minutes=1), "volume": vol})
    return pd.DataFrame(rows)


APRIL = {"2026-04-06": "13:30", "2026-04-07": "13:30", "2026-04-08": "13:30"}


def test_registered_and_named():
    assert FEATURE_REGISTRY["relative_volume"] is RelativeVolume
    feature = build_feature({"type": "relative_volume", "lookback_sessions": 2})
    assert feature.name == "rvol_2d"


def test_hand_values_with_missing_minute():
    # Minute 2 is missing on 04-06: its cumulative volume at minute 2 equals minute 1's (zero-volume minute).
    bars = _bars(
        {
            "2026-04-06": {1: 10, 3: 30},  # cum: m1=10, m2=10, m3=40
            "2026-04-07": {1: 20, 2: 20, 3: 20},  # cum: 20, 40, 60
            "2026-04-08": {1: 30, 2: 10, 3: 10},  # cum: 30, 40, 50
        },
        APRIL,
    )
    values = RelativeVolume(lookback_sessions=2).compute(bars)
    assert values.iloc[:5].isna().all()  # first two sessions have no full reference
    # 04-08: reference mean of 04-06 and 04-07 cumulative volume at the same minute.
    assert values.iloc[5] == pytest.approx(30 / ((10 + 20) / 2))
    assert values.iloc[6] == pytest.approx(40 / ((10 + 40) / 2))  # minute 2 uses carried cum of the gap
    assert values.iloc[7] == pytest.approx(50 / ((40 + 60) / 2))
    assert values.dtype == np.float64


def test_fewer_valid_sessions_than_lookback_is_nan():
    bars = _bars({"2026-04-06": {1: 10}, "2026-04-07": {1: 10}}, APRIL)
    assert RelativeVolume(lookback_sessions=2).compute(bars).isna().all()


def test_today_does_not_enter_reference():
    base = {
        "2026-04-06": {1: 10, 2: 10},
        "2026-04-07": {1: 10, 2: 10},
        "2026-04-08": {1: 10, 2: 10},
    }
    changed = {**base, "2026-04-08": {1: 10, 2: 1000}}
    a = RelativeVolume(1).compute(_bars(base, APRIL))
    b = RelativeVolume(1).compute(_bars(changed, APRIL))
    assert a.iloc[4] == b.iloc[4]  # minute 1 of 04-08 unaffected by minute 2 volume


def test_early_close_reference_does_not_cover_late_minutes():
    open_utc = {"2026-11-27": "14:30", "2026-11-30": "14:30"}  # 11-27 closes at 13:00 NY (210 minutes)
    bars = _bars({"2026-11-27": {1: 5, 210: 5}, "2026-11-30": {1: 10, 210: 10, 211: 10, 300: 10}}, open_utc)
    values = RelativeVolume(1).compute(bars)
    assert values.iloc[2] == pytest.approx(10 / 5)
    assert values.iloc[3] == pytest.approx(20 / 10)
    assert values.iloc[4:].isna().all()  # reference session was closed at those minutes: never a guess


def test_no_lookahead_on_hand_frame():
    spec = {
        d: {m: float(10 + (i * 7 + m) % 5) for m in range(1, 40) if m % 11}
        for i, d in enumerate(["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-13"])
    }
    bars = _bars(spec, {d: "13:30" for d in spec})
    assert_no_lookahead(RelativeVolume(2), bars)
