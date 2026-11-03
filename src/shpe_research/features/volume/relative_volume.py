"""Relative volume and the intraday volume curve.  Level: beginner-intermediate.

Definition (Christopher's ORB plan, section 5):
    RVOL at a bar = cumulative volume so far today (through this bar)
                    / mean cumulative volume at the same minute of the session
                      over the previous `lookback_sessions` valid sessions.
    Example: 150,000 shares by 09:50 against a prior same-time average of
    100,000 gives RVOL = 1.5.

Contract:
    - Feature subclass, feature_type "relative_volume", name f"rvol_{lookback_sessions}d".
    - Only PREVIOUS sessions form the reference; today's later bars must not.
    - Fewer than `lookback_sessions` valid prior sessions -> NaN (never a guess).
    - Same-minute alignment uses minutes since the open, so early-close days
      compare like with like (see features/session/minute_of_session.py).
    - Missing minutes have zero volume for the cumulative sum, but they are
      not invented bars.

The curve itself (expected cumulative volume by minute) is exposed as `RelativeVolume.volume_curve`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import get_calendar, session_date
from shpe_research.features.base import Feature, register_feature
from shpe_research.features.session.minute_of_session import minutes_since_open


def _cumulative_volume_by_minute(bars: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Per-session cumulative volume on a minute grid.

    Returns (sessions, row_session, row_minute, curve). `curve[s, m - 1]` is session s's cumulative volume
    through minute m (a missing minute adds zero), and NaN for minutes after that session's close. Rows whose
    minute is outside 1..session length get row_session = -1.
    """
    day = session_date(bars["ts_available"] - pd.Timedelta(minutes=1))
    minute = minutes_since_open(bars["ts_available"]).to_numpy(dtype=float)
    schedule = get_calendar().schedule(day.min(), day.max())
    length = ((schedule["close"] - schedule["open"]) / pd.Timedelta(minutes=1)).round().astype(int)
    in_session = day.isin(schedule.index).to_numpy()
    length_per_row = length.reindex(day.to_numpy()).to_numpy(dtype=float)
    ok = in_session & np.isfinite(minute) & (minute >= 1) & (minute <= length_per_row) & (minute == np.round(minute))
    sessions = pd.DatetimeIndex(np.sort(day[ok].unique()))
    grid = int(length.max()) if len(length) else 0
    curve = np.zeros((len(sessions), grid))
    row_session = np.full(len(bars), -1, dtype=int)
    if len(sessions):
        row_session[ok] = sessions.get_indexer(day[ok])
        rows = np.flatnonzero(ok)
        volume = np.nan_to_num(bars["volume"].to_numpy(dtype=float)[rows])
        np.add.at(curve, (row_session[rows], minute[rows].astype(int) - 1), volume)
        curve = np.cumsum(curve, axis=1)
        closes = length.reindex(sessions).to_numpy()
        curve[np.arange(grid)[None, :] >= closes[:, None]] = np.nan
    return sessions, row_session, np.where(ok, minute, np.nan), curve


@register_feature
class RelativeVolume(Feature):
    """Cumulative volume today through this bar / mean cumulative volume at the same minute of prior sessions.

    Inputs: `volume`, `ts`, `ts_available` of one symbol's regular-session bars, sorted by ts.
    Output: float64 column `rvol_<lookback_sessions>d`.
    Timing: the value on a row uses that row's bar, earlier bars of the same session and complete PRIOR
    sessions only, so it is known at the row's `ts_available`.
    Reference: the previous `lookback_sessions` sessions that have data in `bars` (calendar sessions
    without data are skipped). A prior session's cumulative volume at minute m is its volume through
    minute m, with missing minutes counted as zero volume. Minutes are minutes since that session's open,
    so early-close days line up with full days.
    Warm-up: NaN until `lookback_sessions` prior valid sessions exist, NaN if any reference session was
    already closed at that minute (early close), and NaN if the reference mean is zero.
    Prices are not used, so split adjustment does not matter; volume is as reported on the contract bars.
    """

    feature_type = "relative_volume"
    requires = ("volume", "ts", "ts_available")

    def __init__(self, lookback_sessions: int = 20):
        if lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive")
        self.lookback_sessions = int(lookback_sessions)

    @property
    def name(self) -> str:
        return f"rvol_{self.lookback_sessions}d"

    def volume_curve(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Expected cumulative volume by minute: one row per session, columns 1..N, known at that session's open.

        Row s is the mean cumulative-volume curve of the `lookback_sessions` valid sessions before s (NaN
        during warm-up or past an early close). It uses no data from session s itself.
        """
        sessions, _, _, curve = _cumulative_volume_by_minute(bars)
        return pd.DataFrame(
            self._expected(curve), index=sessions, columns=pd.RangeIndex(1, curve.shape[1] + 1, name="minute")
        )

    def _expected(self, curve: np.ndarray) -> np.ndarray:
        n = self.lookback_sessions
        expected = np.full(curve.shape, np.nan)
        for s in range(n, len(curve)):
            expected[s] = curve[s - n : s].mean(axis=0)  # NaN where any reference session was closed
        return expected

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        sessions, row_session, minute, curve = _cumulative_volume_by_minute(bars)
        out = np.full(len(bars), np.nan)
        valid = row_session >= 0
        if valid.any():
            expected = self._expected(curve)
            s = row_session[valid]
            m = minute[valid].astype(int) - 1
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = curve[s, m] / expected[s, m]
            ratio[~np.isfinite(ratio)] = np.nan
            out[valid] = ratio
        return pd.Series(out, index=bars.index, dtype="float64")
