from __future__ import annotations

import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.calendar import get_calendar, session_date


@pytest.fixture(scope="module")
def cal():
    return get_calendar()


def test_holidays_and_early_closes(cal):
    sessions = cal.sessions("2026-11-23", "2026-11-30")
    assert pd.Timestamp("2026-11-26") not in sessions  # Thanksgiving
    schedule = cal.schedule("2026-11-25", "2026-11-27")
    assert bool(schedule.loc["2026-11-27", "early_close"])
    assert not bool(schedule.loc["2026-11-25", "early_close"])
    open_, close = cal.session_bounds("2026-11-27")
    assert open_ == pd.Timestamp("2026-11-27T14:30Z") and close == pd.Timestamp("2026-11-27T18:00Z")


def test_session_bounds_follow_daylight_saving(cal):
    assert cal.session_bounds("2026-03-06")[0] == pd.Timestamp("2026-03-06T14:30Z")  # EST
    assert cal.session_bounds("2026-03-09")[0] == pd.Timestamp("2026-03-09T13:30Z")  # EDT


@pytest.mark.parametrize(
    ("ts", "tag"),
    [
        ("2026-10-01T07:59Z", "off"),  # 03:59 New York
        ("2026-10-01T08:00Z", "pre"),
        ("2026-10-01T13:29Z", "pre"),
        ("2026-10-01T13:30Z", "regular"),
        ("2026-10-01T19:59Z", "regular"),
        ("2026-10-01T20:00Z", "post"),
        ("2026-10-01T23:59Z", "post"),  # 19:59 New York
        ("2026-10-02T00:00Z", "off"),  # 20:00 New York
        ("2026-11-26T15:00Z", "off"),  # Thanksgiving
        ("2026-11-27T17:59Z", "regular"),
        ("2026-11-27T18:00Z", "post"),  # early close
        ("2026-10-03T15:00Z", "off"),  # Saturday
    ],
)
def test_tag_sessions(cal, ts, tag):
    assert cal.tag_sessions(to_utc_ns(pd.Series([ts]))).iloc[0] == tag


def test_regular_minutes_and_window(cal):
    assert len(cal.regular_minutes("2026-10-01")) == 390
    assert len(cal.regular_minutes("2026-11-27")) == 210
    morning = cal.regular_minutes("2026-10-01", "09:30", "11:00")
    assert len(morning) == 90 and morning[0] == pd.Timestamp("2026-10-01T13:30Z")


def test_daily_bar_availability_is_session_close(cal):
    # Alpaca stamps daily bars at New York midnight.
    ts = to_utc_ns(pd.Series(["2026-11-27T05:00Z", "2026-11-25T05:00Z", "2026-11-26T05:00Z"]))
    closes = cal.session_close_for(ts)
    assert closes.iloc[0] == pd.Timestamp("2026-11-27T18:00Z")
    assert closes.iloc[1] == pd.Timestamp("2026-11-25T21:00Z")
    assert pd.isna(closes.iloc[2])
    assert session_date(ts).iloc[0] == pd.Timestamp("2026-11-27")


def test_previous_sessions(cal):
    prev = cal.previous_sessions("2026-11-30", 3)
    assert list(prev.strftime("%m-%d")) == ["11-24", "11-25", "11-27"]
