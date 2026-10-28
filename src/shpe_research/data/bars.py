"""D1: download Alpaca bars into the local store, concurrently and resumably.

    uv run shpe data bars --symbols SPY,KO --from 2026-10-01 --to 2026-10-27

Work is split into units (one symbol x one period, see data/store.py) and run
on a thread pool (default 12 workers) that shares one rate limit. Complete
units already in the store are skipped, so re-running a download is a cheap
no-op and an interrupted download resumes where it stopped. A period that is
still in progress (this month) is stored as incomplete and refetched next time.

Whole periods are fetched even when you ask for a few days: one month of
1-minute bars for one symbol is 2-3 requests, and whole-period units keep
resuming and checksumming simple.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from shpe_research.contracts.bars import DAILY, empty_bars, timeframe_delta
from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.alpaca import AlpacaDataClient, sip_safe_end
from shpe_research.data.calendar import TradingCalendar, get_calendar
from shpe_research.data.jobs import run_units
from shpe_research.data.store import BarUnit, DataStore, period_window, periods_for

log = logging.getLogger(__name__)


def normalize_bars(raw: list[dict[str, Any]], symbol: str, timeframe: str, cal: TradingCalendar | None = None):
    """Raw Alpaca bars (t, o, h, l, c, v, vw, n) as contract bars."""
    if not raw:
        return empty_bars()
    cal = cal or get_calendar()
    df = pd.DataFrame.from_records(raw)
    ts = to_utc_ns(df["t"])
    if timeframe == DAILY:
        available = cal.session_close_for(ts)
        on_session = available.notna()
        available = available.fillna(ts + pd.Timedelta(days=1))
        session = np.where(on_session, "regular", "off")
    else:
        available = ts + timeframe_delta(timeframe)
        session = cal.tag_sessions(ts).to_numpy()
    bars = pd.DataFrame(
        {
            "symbol": symbol,
            "ts": ts,
            "ts_available": available.astype(ts.dtype),
            "open": df["o"].astype(float),
            "high": df["h"].astype(float),
            "low": df["l"].astype(float),
            "close": df["c"].astype(float),
            "volume": df["v"].round().astype("int64"),
            "vwap": df["vw"].astype(float) if "vw" in df else np.nan,
            "trade_count": pd.array(df["n"], dtype="Int64") if "n" in df else pd.array([pd.NA] * len(df), "Int64"),
            "session": pd.Series(session, dtype=object).astype(str).to_numpy(),
        }
    )
    bars = bars.sort_values("ts", kind="stable")
    duplicated = bars.duplicated("ts", keep="last")
    if duplicated.any():
        log.warning("%s: dropped %d duplicate bars (kept the last revision)", symbol, int(duplicated.sum()))
    return bars[~duplicated].reset_index(drop=True)


@dataclass
class DownloadSummary:
    units: int = 0
    skipped: int = 0
    fetched: int = 0
    rows: int = 0
    requests: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        text = (
            f"{self.units} units: {self.fetched} fetched ({self.rows:,} bars, {self.requests} requests), "
            f"{self.skipped} already stored, {len(self.failed)} failed"
        )
        for key, err in self.failed.items():
            text += f"\n  FAILED {key}: {err}"
        return text


def download_bars(
    symbols: list[str],
    start,
    end,
    timeframe: str = "1Min",
    adjustment: str = "raw",
    feed: str = "sip",
    workers: int = 12,
    store: DataStore | None = None,
    client: AlpacaDataClient | None = None,
    now: pd.Timestamp | None = None,
    force: bool = False,
) -> DownloadSummary:
    """Fetch every unit covering New York dates start..end for `symbols` that the store lacks."""
    store = store or DataStore()
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    units = [
        BarUnit(feed, timeframe, adjustment, s.upper(), p) for s in symbols for p in periods_for(timeframe, start, end)
    ]
    summary = DownloadSummary(units=len(units))
    todo = []
    for unit in units:
        if not force and store.bar_unit_complete(unit):
            summary.skipped += 1
        else:
            todo.append(unit)
    if not todo:
        return summary

    owns_client = client is None
    client = client or AlpacaDataClient()
    cal = get_calendar()
    sent_before = client.requests_sent

    def fetch(unit: BarUnit) -> int:
        window_start, window_end = period_window(unit.period)
        fetch_end = sip_safe_end(window_end, now) if feed == "sip" else min(window_end, now)
        if fetch_end <= window_start:
            return -1  # period entirely in the future
        raw = client.bars(unit.symbol, window_start, fetch_end, timeframe, adjustment, feed)
        bars = normalize_bars(raw, unit.symbol, timeframe, cal)
        store.write_bar_unit(unit, bars, window_start, fetch_end, complete=fetch_end >= window_end)
        return len(bars)

    try:
        results, summary.failed = run_units(todo, fetch, workers, "bars", key=lambda u: u.key)
    finally:
        store.bars_manifest.flush()
        summary.requests = client.requests_sent - sent_before
        if owns_client:
            client.close()
    fetched = [rows for rows in results.values() if rows >= 0]
    summary.fetched = len(fetched)
    summary.rows = sum(fetched)
    summary.skipped += len(results) - len(fetched)
    return summary
