"""D3: download quotes and trades, and summarize them per minute.

    uv run shpe data ticks --symbols KO --from 2026-10-27 --to 2026-10-27
    uv run shpe data ticks --symbols KO --from 2026-10-27 --to 2026-10-27 --kinds quotes \
        --start-time 09:30 --end-time 10:00
    uv run shpe data tick-summary --symbols KO --from 2026-10-27 --to 2026-10-27

Tick data is large (SPY: about 3M quotes plus trades per SIP session), so a
unit is one symbol, kind and New York date, and the default window is the
regular session. `--session extended` fetches 04:00-20:00; `--start-time` and
`--end-time` narrow the window further. The manifest records the window each
unit covers; a later request for a wider window refetches that unit.

Normalization drops one-sided (bid or ask <= 0) and crossed (ask < bid) quotes
and records how many were dropped. Quote sizes stay in round lots.

Minute summaries (contracts/ticks.py:TICK_SUMMARY_COLUMNS) join onto 1-minute
bars by (symbol, ts) and feed the order-flow features (F6). Trades are signed
with Lee-Ready: above the mid of the last quote strictly before the trade is a
buy, below is a sell, and at the mid the tick test decides (uptick buy,
downtick sell, using the last non-zero price change).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from shpe_research.contracts.base import UTC_NS, to_utc_ns
from shpe_research.contracts.ticks import (
    QUOTE_SCHEMA,
    SCHEMAS,
    TICK_SUMMARY_COLUMNS,
    TRADE_SCHEMA,
    validate_quotes,
    validate_trades,
)
from shpe_research.data.alpaca import AlpacaDataClient, sip_safe_end
from shpe_research.data.calendar import NY, get_calendar
from shpe_research.data.jobs import run_units
from shpe_research.data.store import DataStore, TickUnit

log = logging.getLogger(__name__)

KINDS = ("quotes", "trades")


# ---------------------------------------------------------------- normalization


def _strings(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].fillna("").astype(str) if column in df else pd.Series("", index=df.index)


def _lists(df: pd.DataFrame, column: str) -> list[list[str]]:
    if column not in df:
        return [[] for _ in range(len(df))]
    return [list(v) if isinstance(v, list | tuple | np.ndarray) else [] for v in df[column]]


def normalize_quotes(raw: list[dict[str, Any]], symbol: str) -> tuple[pd.DataFrame, int]:
    """Contract quotes from raw Alpaca quotes, and how many one-sided or crossed quotes were dropped."""
    if not raw:
        return QUOTE_SCHEMA.empty_table().to_pandas(), 0
    df = pd.DataFrame.from_records(raw)
    quotes = pd.DataFrame(
        {
            "symbol": symbol,
            "ts": to_utc_ns(df["t"]),
            "bid_price": df["bp"].astype(float),
            "ask_price": df["ap"].astype(float),
            "bid_size": df["bs"].astype("int64"),
            "ask_size": df["as"].astype("int64"),
            "bid_exchange": _strings(df, "bx"),
            "ask_exchange": _strings(df, "ax"),
            "tape": _strings(df, "z"),
            "conditions": _lists(df, "c"),
        }
    )
    valid = (quotes["bid_price"] > 0) & (quotes["ask_price"] > 0) & (quotes["ask_price"] >= quotes["bid_price"])
    quotes = quotes[valid].sort_values("ts", kind="stable").reset_index(drop=True)
    validate_quotes(quotes)
    return quotes, int((~valid).sum())


def normalize_trades(raw: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    """Contract trades from raw Alpaca trades (prints with non-positive price or size are dropped)."""
    if not raw:
        return TRADE_SCHEMA.empty_table().to_pandas()
    df = pd.DataFrame.from_records(raw)
    trades = pd.DataFrame(
        {
            "symbol": symbol,
            "ts": to_utc_ns(df["t"]),
            "price": df["p"].astype(float),
            "size": df["s"].round().astype("int64"),
            "exchange": _strings(df, "x"),
            "tape": _strings(df, "z"),
            "conditions": _lists(df, "c"),
            "trade_id": _strings(df, "i"),
        }
    )
    trades = trades[(trades["price"] > 0) & (trades["size"] > 0)]
    trades = trades.sort_values("ts", kind="stable").reset_index(drop=True)
    validate_trades(trades)
    return trades


# ---------------------------------------------------------------- download


def session_window(day, session: str = "regular", start_time: str | None = None, end_time: str | None = None):
    """[start, end) in UTC for `day`: the regular session, or 04:00-20:00 for "extended", optionally narrowed."""
    cal = get_calendar()
    day = pd.Timestamp(day).normalize()
    if session == "extended":
        start = pd.Timestamp(f"{day.date()} 04:00").tz_localize(NY).tz_convert("UTC")
        end = pd.Timestamp(f"{day.date()} 20:00").tz_localize(NY).tz_convert("UTC")
    else:
        start, end = cal.session_bounds(day)
    if start_time:
        start = max(start, pd.Timestamp(f"{day.date()} {start_time}").tz_localize(NY).tz_convert("UTC"))
    if end_time:
        end = min(end, pd.Timestamp(f"{day.date()} {end_time}").tz_localize(NY).tz_convert("UTC"))
    return start, end


@dataclass
class TickDownloadSummary:
    units: int = 0
    skipped: int = 0
    fetched: int = 0
    rows: int = 0
    dropped: int = 0
    requests: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        text = (
            f"{self.units} units: {self.fetched} fetched ({self.rows:,} records, {self.dropped:,} invalid quotes "
            f"dropped, {self.requests} requests), {self.skipped} already stored, {len(self.failed)} failed"
        )
        for key, err in self.failed.items():
            text += f"\n  FAILED {key}: {err}"
        return text


TickJob = tuple[TickUnit, pd.Timestamp, pd.Timestamp, bool]  # unit, window start, window end, whole session


def _plan_jobs(symbols, start, end, kinds, feed, session, start_time, end_time, now) -> list[TickJob]:
    jobs = []
    for day in get_calendar().sessions(start, end):
        full_start, full_end = session_window(day, session)
        w_start, w_end = session_window(day, session, start_time, end_time)
        w_end = sip_safe_end(w_end, now) if feed == "sip" else min(w_end, now)
        if w_end <= w_start:
            continue
        complete = w_start <= full_start and w_end >= full_end
        date = day.strftime("%Y-%m-%d")
        jobs += [(TickUnit(feed, kind, s.upper(), date), w_start, w_end, complete) for s in symbols for kind in kinds]
    return jobs


def download_ticks(
    symbols: list[str],
    start,
    end,
    kinds: tuple[str, ...] = KINDS,
    feed: str = "sip",
    session: str = "regular",
    start_time: str | None = None,
    end_time: str | None = None,
    workers: int = 12,
    store: DataStore | None = None,
    client: AlpacaDataClient | None = None,
    now: pd.Timestamp | None = None,
    force: bool = False,
) -> TickDownloadSummary:
    """Fetch quotes and/or trades for each session in New York dates start..end that the store does not cover."""
    store = store or DataStore()
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    planned = _plan_jobs(symbols, start, end, kinds, feed, session, start_time, end_time, now)
    summary = TickDownloadSummary(units=len(planned))
    jobs = []
    for job in planned:
        if not force and store.tick_unit_covers(job[0], job[1], job[2]):
            summary.skipped += 1
        else:
            jobs.append(job)
    if not jobs:
        return summary

    owns_client = client is None
    client = client or AlpacaDataClient()
    sent_before = client.requests_sent

    def fetch(job) -> tuple[int, int]:
        unit, w_start, w_end, complete = job
        raw = client.ticks(unit.kind, unit.symbol, w_start, w_end, feed)
        if unit.kind == "quotes":
            frame, dropped = normalize_quotes(raw, unit.symbol)
        else:
            frame, dropped = normalize_trades(raw, unit.symbol), 0
        table = pa.Table.from_pandas(frame, schema=SCHEMAS[unit.kind], preserve_index=False)
        store.write_tick_unit(unit, table, w_start, w_end, complete, dropped=dropped)
        return len(frame), dropped

    try:
        results, summary.failed = run_units(jobs, fetch, workers, "ticks", key=lambda job: job[0].key)
    finally:
        store.ticks_manifest.flush()
        summary.requests = client.requests_sent - sent_before
        if owns_client:
            client.close()
    summary.fetched = len(results)
    summary.rows = sum(rows for rows, _ in results.values())
    summary.dropped = sum(dropped for _, dropped in results.values())
    return summary


# ---------------------------------------------------------------- minute summaries


def sign_trades(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.Series:
    """+1 buy, -1 sell, 0 unknown for each trade (Lee-Ready with tick-test fallback), aligned to trades.index."""
    if trades.empty:
        return pd.Series(dtype="int64")
    q = quotes.assign(mid=(quotes["bid_price"] + quotes["ask_price"]) / 2)[["symbol", "ts", "mid"]]
    t = trades[["symbol", "ts", "price"]].assign(_row=np.arange(len(trades)))
    t = t.sort_values("ts", kind="stable")
    merged = pd.merge_asof(t, q.sort_values("ts", kind="stable"), on="ts", by="symbol", allow_exact_matches=False)
    merged = merged.sort_values("_row")
    quote_sign = np.sign(merged["price"].to_numpy() - merged["mid"].to_numpy())
    # Tick test: sign of the last non-zero price change for the same symbol.
    ordered = trades.assign(_row=np.arange(len(trades))).sort_values(["symbol", "ts"], kind="stable")
    change = np.sign(ordered.groupby("symbol")["price"].diff()).replace(0, np.nan)
    tick = change.groupby(ordered["symbol"]).ffill().fillna(0)
    tick_sign = pd.Series(tick.to_numpy(), index=ordered["_row"].to_numpy()).sort_index().to_numpy()
    sign = np.where(np.isnan(quote_sign) | (quote_sign == 0), tick_sign, quote_sign)
    return pd.Series(sign.astype("int64"), index=trades.index)


def minute_summaries(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol and minute with quotes or trades (contracts/ticks.py:TICK_SUMMARY_COLUMNS)."""
    q = quotes.assign(minute=quotes["ts"].dt.floor("1min"))
    mid = (q["bid_price"] + q["ask_price"]) / 2
    q = q.assign(
        spread_bps=(q["ask_price"] - q["bid_price"]) / mid * 1e4,
        imbalance=(q["bid_size"] - q["ask_size"]) / (q["bid_size"] + q["ask_size"]).replace(0, np.nan),
        mid=mid,
    )
    qg = q.groupby(["symbol", "minute"])
    quote_part = pd.DataFrame(
        {
            "quote_count": qg.size(),
            "spread_bps_mean": qg["spread_bps"].mean(),
            "spread_bps_median": qg["spread_bps"].median(),
            "imbalance_mean": qg["imbalance"].mean(),
            "mid_close": qg["mid"].last(),
        }
    )
    sign = sign_trades(trades, quotes)
    t = trades.assign(
        minute=trades["ts"].dt.floor("1min"),
        buy=np.where(sign > 0, trades["size"], 0),
        sell=np.where(sign < 0, trades["size"], 0),
    )
    tg = t.groupby(["symbol", "minute"])
    trade_part = pd.DataFrame(
        {"trade_count": tg.size(), "trade_volume": tg["size"].sum(), "buy_volume": tg["buy"].sum(),
         "sell_volume": tg["sell"].sum()}
    )
    out = quote_part.join(trade_part, how="outer")
    counts = ["quote_count", "trade_count", "trade_volume", "buy_volume", "sell_volume"]
    out[counts] = out[counts].fillna(0).astype("int64")
    out = out.reset_index().rename(columns={"minute": "ts"})
    out["ts"] = out["ts"].astype(UTC_NS)
    out["ts_available"] = out["ts"] + pd.Timedelta(minutes=1)
    float_cols = ["spread_bps_mean", "spread_bps_median", "imbalance_mean", "mid_close"]
    out[float_cols] = out[float_cols].astype(float)
    return out[list(TICK_SUMMARY_COLUMNS)].sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)
