"""HAR-RV components (Corsi 2009) on daily bars, built from intraday returns (C1e)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date
from shpe_research.features.base import Feature, register_feature

_NY = "America/New_York"
_COLUMNS = ("rv_d", "rv_w", "rv_m")


def _daily_rv(minute: pd.DataFrame, symbol: str, bar_minutes: int) -> pd.Series:
    """Realized variance per session date: sum of squared log returns of `bar_minutes` closes, regular session."""
    m = minute[(minute["symbol"] == symbol) & (minute["session"] == "regular")].sort_values("ts")
    if m.empty:
        return pd.Series(dtype="float64")
    day = session_date(m["ts"]).reset_index(drop=True)
    opens = (day + pd.Timedelta(hours=9, minutes=30)).dt.tz_localize(_NY).dt.tz_convert("UTC")
    avail = m["ts_available"].reset_index(drop=True)
    step = int(pd.Timedelta(minutes=bar_minutes).value)
    delta = (avail - opens).astype("int64").to_numpy()
    bucket = -(-delta // step)  # ceil: a bar ending inside (k-1, k] steps belongs to bucket k
    frame = pd.DataFrame(
        {"day": day, "bucket": bucket, "open": m["open"].to_numpy(), "close": m["close"].to_numpy()}
    )
    first_open = frame.groupby("day")["open"].first()
    closes = frame.groupby(["day", "bucket"], sort=True)["close"].last().reset_index()
    ref = closes.groupby("day")["close"].shift(1)
    ref = ref.fillna(closes["day"].map(first_open))
    closes["r2"] = np.log(closes["close"] / ref) ** 2
    return closes.groupby("day")["r2"].sum()


@register_feature
class HarRv(Feature):
    """Daily realized variance of session d and its 5- and 22-session means (columns rv_d, rv_w, rv_m).

    Needs the minute bars, passed as `context["minute_bars"]`. RV of session d sums squared log returns of
    `bar_minutes` closes inside the regular session (the first return starts at the session's first open), so
    it is fully known at the daily bar's `ts_available`. `rv_w` and `rv_m` are trailing means over sessions
    <= d (NaN until the window is full). A Feature returns one Series, so `column` picks which of the three
    this instance yields; `compute_frame` returns all three.
    """

    feature_type = "har_rv"
    requires = ("symbol", "ts", "ts_available")
    context_symbols = ("minute_bars",)

    def __init__(self, bar_minutes: int = 5, components: tuple[int, int, int] = (1, 5, 22), column: str = "rv_d"):
        if bar_minutes <= 0:
            raise ValueError("bar_minutes must be positive")
        components = tuple(int(c) for c in components)
        if len(components) != 3 or min(components) < 1:
            raise ValueError("components must be three positive window lengths")
        if column not in _COLUMNS:
            raise ValueError(f"column must be one of {_COLUMNS}")
        self.bar_minutes = int(bar_minutes)
        self.components = components
        self.column = column

    @property
    def name(self) -> str:
        return self.column if self.bar_minutes == 5 else f"{self.column}_{self.bar_minutes}m"

    def compute_frame(self, bars: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """rv_d, rv_w, rv_m for one symbol's daily `bars` (aligned to bars.index).

        Timing: row d uses minute bars of session d only (all available by the daily bar's `ts_available`)
        and daily RV of earlier sessions.
        """
        if "minute_bars" not in context:
            raise ValueError("har_rv needs context['minute_bars']")
        if bars.empty:
            return pd.DataFrame({c: pd.Series(dtype="float64") for c in _COLUMNS}, index=bars.index)
        rv = _daily_rv(context["minute_bars"], str(bars["symbol"].iloc[0]), self.bar_minutes)
        days = session_date(bars["ts"])
        rv_d = pd.Series(days.map(rv).to_numpy(dtype=float), index=bars.index)
        d, w, m = self.components
        return pd.DataFrame(
            {
                "rv_d": rv_d.rolling(d, min_periods=d).mean(),
                "rv_w": rv_d.rolling(w, min_periods=w).mean(),
                "rv_m": rv_d.rolling(m, min_periods=m).mean(),
            },
            index=bars.index,
        )

    def compute(self, bars: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.Series:
        return self.compute_frame(bars, context or {})[self.column]
