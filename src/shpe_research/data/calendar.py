"""NYSE trading calendar: sessions, early closes, and session tags for timestamps.

Wraps `exchange_calendars` (XNYS), which knows holidays and early closes.
Session tags used by the bars contract:

    regular  open <= ts < close of that day's session (13:00 close on early-close days)
    pre      a session day, 04:00 New York time <= ts < open
    post     a session day, close <= ts < 20:00 New York time
    off      anything else (weekends, holidays, overnight)

Dates are New York calendar dates, returned as naive midnight Timestamps.
"""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from shpe_research.contracts.base import UTC_NS

NY = "America/New_York"
_PRE_OPEN_MINUTE = 4 * 60
_POST_CLOSE_MINUTE = 20 * 60
_REGULAR_CLOSE = pd.Timedelta(hours=16)


def session_date(ts: pd.Series) -> pd.Series:
    """New York calendar date of each UTC timestamp, as naive midnight Timestamps."""
    return ts.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()


def _ny_time(day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day.date()} {hhmm}").tz_localize(NY).tz_convert("UTC")


class TradingCalendar:
    def __init__(self, name: str = "XNYS", start: str = "2000-01-01", end: str | None = None):
        cal = xcals.get_calendar(name, start=start, end=end)
        schedule = cal.schedule[["open", "close"]].copy()
        schedule.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
        schedule["open"] = schedule["open"].astype(UTC_NS)
        schedule["close"] = schedule["close"].astype(UTC_NS)
        close_ny = schedule["close"].dt.tz_convert(NY)
        schedule["early_close"] = (close_ny - close_ny.dt.normalize()) < _REGULAR_CLOSE
        self.name = name
        self._schedule = schedule

    @property
    def first_session(self) -> pd.Timestamp:
        return self._schedule.index[0]

    @property
    def last_session(self) -> pd.Timestamp:
        return self._schedule.index[-1]

    def schedule(self, start, end) -> pd.DataFrame:
        """Sessions with start <= date <= end: columns open, close (UTC) and early_close."""
        return self._schedule.loc[pd.Timestamp(start).normalize() : pd.Timestamp(end).normalize()]

    def sessions(self, start, end) -> pd.DatetimeIndex:
        return self.schedule(start, end).index

    def is_session(self, day) -> bool:
        return pd.Timestamp(day).normalize() in self._schedule.index

    def session_bounds(self, day) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(open, close) in UTC of the session on `day`."""
        row = self._schedule.loc[pd.Timestamp(day).normalize()]
        return row["open"], row["close"]

    def previous_sessions(self, before, count: int) -> pd.DatetimeIndex:
        """The `count` sessions strictly before `before`, oldest first."""
        idx = self._schedule.index
        end = idx.searchsorted(pd.Timestamp(before).normalize(), side="left")
        if end < count:
            raise ValueError(f"calendar starts at {idx[0].date()}; cannot go back {count} sessions from {before}")
        return idx[end - count : end]

    def tag_sessions(self, ts: pd.Series) -> pd.Series:
        """Session tag ("pre", "regular", "post", "off") for each UTC timestamp."""
        ny = ts.dt.tz_convert(NY)
        day = ny.dt.tz_localize(None).dt.normalize()
        minute = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
        bounds = self._schedule.reindex(day.to_numpy())
        open_ = bounds["open"].to_numpy()
        close = bounds["close"].to_numpy()
        t = ts.to_numpy()
        in_session = ~pd.isna(open_)
        conditions = [
            ~in_session,
            (t >= open_) & (t < close),
            (minute >= _PRE_OPEN_MINUTE) & (t < open_),
            (t >= close) & (minute < _POST_CLOSE_MINUTE),
        ]
        tags = np.select(conditions, ["off", "regular", "pre", "post"], default="off")
        return pd.Series(tags, index=ts.index, dtype=object).astype(str)

    def session_close_for(self, ts: pd.Series) -> pd.Series:
        """Close of the session on each timestamp's New York date (NaT on non-session days).

        This is when a daily bar stamped `ts` becomes available.
        """
        closes = self._schedule["close"].reindex(session_date(ts).to_numpy())
        return pd.Series(closes.to_numpy(), index=ts.index).astype(UTC_NS)

    def regular_minutes(self, day, start_time: str | None = None, end_time: str | None = None) -> pd.DatetimeIndex:
        """Expected 1-minute bar starts (UTC) of `day`'s regular session.

        Optionally clipped to a New York wall-clock window, e.g. ("09:30", "11:00").
        """
        day = pd.Timestamp(day).normalize()
        open_, close = self.session_bounds(day)
        if start_time is not None:
            open_ = max(open_, _ny_time(day, start_time))
        if end_time is not None:
            close = min(close, _ny_time(day, end_time))
        return pd.date_range(open_, close, freq="1min", inclusive="left").astype(UTC_NS)


@lru_cache(maxsize=1)
def get_calendar() -> TradingCalendar:
    """The shared NYSE calendar (built once per process)."""
    return TradingCalendar()
