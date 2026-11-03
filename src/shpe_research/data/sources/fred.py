"""`fred` source: macro series with point-in-time availability.

`ts` is the observation date; `ts_available` is the release date plus `release_lag`, never the observation
date. With the real API the release date is the vintage's `realtime_start` (ALFRED), and the first vintage of
each observation is kept, so later revisions never leak backwards. Needs FRED_API_KEY in the environment.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, ClassVar

import httpx
import pandas as pd

from shpe_research.data.sources.base import clip_available, register_source, series_frame

URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_alfred(series: str, start: Any, end: Any, transport: httpx.BaseTransport | None = None) -> list[dict]:
    """First-release observations of `series` as `{date, release_date, value}` dicts (FRED's missing "." dropped)."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set; add it to .env or work from a snapshot")
    params = {"series_id": series, "api_key": key, "file_type": "json",
              "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}
    with httpx.Client(timeout=30.0, transport=transport) as http:
        res = http.get(URL, params=params)
    res.raise_for_status()
    first: dict[str, dict] = {}
    for obs in res.json()["observations"]:
        if obs["value"] == ".":
            continue
        seen = first.get(obs["date"])
        if seen is None or obs["realtime_start"] < seen["release_date"]:
            first[obs["date"]] = {"date": obs["date"], "release_date": obs["realtime_start"], "value": obs["value"]}
    return list(first.values())


@register_source
class FredSource:
    source_type: ClassVar[str] = "fred"

    def __init__(self, series: str | list[str], release_lag: str = "1D", fetch: Callable | None = None):
        self.series = [series] if isinstance(series, str) else list(series)
        self.release_lag = pd.Timedelta(release_lag)
        self._fetch = fetch or fetch_alfred

    def load(self, symbols: list[str] | None, start: Any, end: Any, timeframe: str = "1Day") -> pd.DataFrame:
        """Rows `ts, ts_available, series, value` whose `ts_available` is inside the date window.

        `symbols` and `timeframe` are ignored. Missing observations are dropped, not filled.
        Raises ValueError if any release date plus lag precedes its observation date.
        """
        rows = [
            {"ts": r["date"], "ts_available": pd.Timestamp(r["release_date"], tz="UTC") + self.release_lag,
             "series": s, "value": r["value"]}
            for s in self.series
            for r in self._fetch(s, start, end)
        ]
        return clip_available(series_frame(rows), start, end)
