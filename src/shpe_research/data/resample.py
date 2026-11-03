"""Minute bars to daily bars (C1e)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.calendar import session_date

_NY = "America/New_York"
_DAILY_COLUMNS = [
    "symbol", "ts", "ts_available", "open", "high", "low", "close", "volume", "vwap", "trade_count", "session",
]


def resample_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """One daily bar per symbol and regular session from intraday contract bars.

    Inputs: contract bars sorted by (symbol, ts) at a sub-hourly timeframe. Only `session == "regular"` rows
    are used; sessions without regular bars produce no row.
    Output: contract bars with `ts` = the session date at 09:30 New York (in UTC), `ts_available` = the
    `ts_available` of the session's last regular bar (16:00, or the early close), open = first bar's open,
    high/low = extremes, close = last bar's close, volume summed, vwap volume-weighted over the bars that have
    one (NaN if none), trade_count summed (NA if all NA) and `session = "regular"`.
    Timing: a daily row is known only at its `ts_available`; nothing after the last regular bar is read.
    """
    regular = bars[bars["session"] == "regular"]
    if not regular.empty:
        width = (regular["ts_available"] - regular["ts"]).median()
        if width > pd.Timedelta(hours=1):
            raise ValueError("resample_daily expects minute (intraday) bars, got bars wider than one hour")
    if regular.empty:
        out = pd.DataFrame({c: pd.Series(dtype=bars[c].dtype) for c in _DAILY_COLUMNS})
        return out.reset_index(drop=True)
    regular = regular.sort_values(["symbol", "ts"], kind="stable").copy()
    regular["_day"] = session_date(regular["ts"])
    px_vol = regular["vwap"] * regular["volume"]
    regular["_pv"] = px_vol
    regular["_vv"] = regular["volume"].where(regular["vwap"].notna(), 0)
    grouped = regular.groupby(["symbol", "_day"], sort=True)
    out = grouped.agg(
        ts_available=("ts_available", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        _pv=("_pv", "sum"),
        _vv=("_vv", "sum"),
        trade_count=("trade_count", lambda s: s.sum(min_count=1)),
    ).reset_index()
    out["vwap"] = np.where(out["_vv"] > 0, out["_pv"] / out["_vv"].where(out["_vv"] > 0), np.nan)
    opens = out["_day"] + pd.Timedelta(hours=9, minutes=30)
    out["ts"] = to_utc_ns(pd.Series(opens.dt.tz_localize(_NY).dt.tz_convert("UTC")))
    out["session"] = "regular"
    out["volume"] = out["volume"].astype("int64")
    out["trade_count"] = out["trade_count"].astype("Int64")
    out["ts_available"] = to_utc_ns(out["ts_available"])
    return out[_DAILY_COLUMNS].sort_values(["symbol", "ts"]).reset_index(drop=True)
