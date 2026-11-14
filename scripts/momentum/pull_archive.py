# ruff: noqa: E501
"""A1: staged, resumable minute-bar archive pull with a SQLite manifest and measured Alpaca limits.

    uv run python scripts/momentum/pull_archive.py --stage pilot --root "<store root outside git>"
    uv run python scripts/momentum/pull_archive.py --stage pilot --root ... --symbols SPY,AAPL \
        --start 2024-03-04 --end 2024-03-05 --dry-run        # tiny dry run, writes under <root>/dryrun

Stages (pool size x years): pilot (fixed ~77 names x 2 years), 300x2, 500x5, 1000x10. Only `pilot` has a built-in
pool; larger stages need --symbols-file (one ticker per line). Only `pilot` is meant to be run for now.

Why not data/store.py's JSON manifest: it rewrites the whole file every 20 units. Here every finished unit is one
SQLite row (WAL, committed immediately), so cost per unit stays constant at ~240,000 units. Unit files use the
same layout and Parquet schema as DataStore (bars/feed=.../symbol=.../period=YYYY-MM.parquet), so DataStore can read
the files later; only the index differs. store.py itself is untouched.

Resume: a unit (feed, symbol, month) is skipped when its row is complete and covers the wanted window and its file
exists. Failed units write nothing and are retried on the next run.

Measured and logged: HTTP status counts (429s, Retry-After), requests per minute achieved (mean and peak in any
60 s window), bars per request, bytes on disk, wall time, SIP vs IEX entitlement, and how far back minute bars
exist for delisted or renamed symbols. Credentials come only from the environment (ALPACA_DATA_KEY and
ALPACA_DATA_SECRET); nothing here logs them.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from shpe_research.contracts.bars import BarMetadata, bars_to_table, validate_bars
from shpe_research.data.alpaca import AlpacaDataClient, AlpacaError, sip_safe_end
from shpe_research.data.bars import normalize_bars
from shpe_research.data.calendar import get_calendar
from shpe_research.data.jobs import run_units
from shpe_research.data.store import BarUnit, ny_midnight, period_window, periods_for, write_parquet_atomic

log = logging.getLogger("pull_archive")

MANIFEST_DB = "archive_manifest.sqlite"
TIMEFRAME = "1Min"
ADJUSTMENT = "raw"
MAX_PLANNED_GB = 10.0
MIN_FREE_GB_AFTER = 20.0
BYTES_PER_ROW_ESTIMATE = 30  # conservative zstd Parquet bytes per minute bar
BARS_PER_SESSION_ESTIMATE = 700  # pre/regular/post mix for a liquid name; upper bound is 960
WARMUP_MONTHS = 3
PROBE_FLOOR = "2016-01-01T00:00:00Z"


@dataclass(frozen=True)
class Stage:
    name: str
    symbols: int | None  # None: built-in pilot pool
    years: int


STAGES = {
    "pilot": Stage("pilot", None, 2),
    "300x2": Stage("300x2", 300, 2),
    "500x5": Stage("500x5", 500, 5),
    "1000x10": Stage("1000x10", 1000, 10),
}

# Pilot pool, chosen from general knowledge (not looked up): ~58 liquid US common stocks, SPY, and delisted or renamed
# names. DELISTED lists the delisting or ticker-change date as remembered; the probe measures what Alpaca returns.
LIQUID = (
    "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM V MA XOM JNJ PG HD COST WMT KO PEP MRK ABBV CVX BAC ORCL CRM "
    "ADBE NFLX AMD CSCO MCD TMO ABT LIN DIS WFC INTC QCOM TXN CAT GE HON IBM AMGN LOW UPS BA GS MS BLK NKE SBUX "
    "T VZ PFE CMCSA DE PYPL UBER MU PLTR"
).split()
DELISTED = {
    "SPLK": "acquired by Cisco, Mar 2024",
    "JNPR": "acquired by HPE, Jul 2025",
    "ANSS": "acquired by Synopsys, Jul 2025",
    "HES": "acquired by Chevron, Jul 2025",
    "WBA": "taken private, Aug 2025",
    "DFS": "merged into Capital One, May 2025",
    "PXD": "acquired by Exxon, May 2024",
    "MRO": "acquired by ConocoPhillips, Nov 2024",
    "CTLT": "taken private, Dec 2024",
    "SRCL": "taken private, Nov 2024",
    "SAVE": "Spirit Airlines, delisted Nov 2024 (bankruptcy)",
    "SQ": "renamed XYZ (Block), Jan 2025",
    "ALTR": "acquired by Siemens, Mar 2025",
    "ATVI": "acquired by Microsoft, Oct 2023 (before window)",
    "SIVB": "bank failure, Mar 2023 (before window)",
    "TWTR": "taken private, Oct 2022 (before window)",
}
BENCHMARK = "SPY"


def pilot_pool() -> list[str]:
    return sorted({*LIQUID, *DELISTED, BENCHMARK})


# ---------------------------------------------------------------------------------------------- measurement


class CountingTransport(httpx.BaseTransport):
    """Wraps a transport and records status, bytes and send time of every HTTP request (including retries)."""

    def __init__(self, inner: httpx.BaseTransport, clock: Callable[[], float] = time.monotonic):
        self._inner = inner
        self._clock = clock
        self._lock = threading.Lock()
        self.times: deque[float] = deque()
        self.status_counts: dict[int, int] = {}
        self.retry_after: list[str] = []
        self.bytes_received = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        sent = self._clock()
        response = self._inner.handle_request(request)
        response.read()
        with self._lock:
            self.times.append(sent)
            self.status_counts[response.status_code] = self.status_counts.get(response.status_code, 0) + 1
            self.bytes_received += len(response.content)
            if response.status_code == 429:
                self.retry_after.append(response.headers.get("retry-after", "none"))
        return response

    def close(self) -> None:
        self._inner.close()

    def stats(self) -> dict:
        with self._lock:
            times = sorted(self.times)
            statuses = dict(self.status_counts)
            retry_after = list(self.retry_after)
        total = len(times)
        span = (times[-1] - times[0]) if total > 1 else 0.0
        peak, lo = 0, 0
        for hi, t in enumerate(times):
            while t - times[lo] > 60.0:
                lo += 1
            peak = max(peak, hi - lo + 1)
        return {
            "http_requests": total,
            "status_counts": {str(k): v for k, v in sorted(statuses.items())},
            "n_429": statuses.get(429, 0),
            "retry_after_seen": sorted(set(retry_after)),
            "requests_per_min_mean": round(total / (span / 60.0), 1) if span > 0 else None,
            "requests_per_min_peak_60s": peak,
            "bytes_received": self.bytes_received,
            "ok_requests": statuses.get(200, 0),
        }


# ---------------------------------------------------------------------------------------------- manifest

_SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    feed TEXT NOT NULL, timeframe TEXT NOT NULL, adjustment TEXT NOT NULL, symbol TEXT NOT NULL, period TEXT NOT NULL,
    year INTEGER NOT NULL, file TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, rows INTEGER NOT NULL,
    regular_rows INTEGER NOT NULL, expected_regular INTEGER NOT NULL, coverage REAL,
    first_ts TEXT, last_ts TEXT, win_start TEXT NOT NULL, win_end TEXT NOT NULL, complete INTEGER NOT NULL,
    requests INTEGER NOT NULL, stage TEXT NOT NULL, fetched_at TEXT NOT NULL,
    PRIMARY KEY (feed, timeframe, adjustment, symbol, period)
);
CREATE INDEX IF NOT EXISTS units_symbol_year ON units (symbol, year);
CREATE TABLE IF NOT EXISTS probes (
    kind TEXT NOT NULL, symbol TEXT NOT NULL, feed TEXT NOT NULL, status INTEGER, first_ts TEXT, last_ts TEXT,
    note TEXT, probed_at TEXT NOT NULL, PRIMARY KEY (kind, symbol, feed)
);
CREATE TABLE IF NOT EXISTS runs (
    stage TEXT NOT NULL, started_at TEXT NOT NULL, summary TEXT NOT NULL
);
"""


class ArchiveManifest:
    """One SQLite row per unit, committed on write (constant cost per unit, unlike a rewritten JSON file)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _key(unit: BarUnit) -> tuple:
        return (unit.feed, unit.timeframe, unit.adjustment, unit.symbol, unit.period)

    def get(self, unit: BarUnit) -> dict | None:
        with self._lock:
            cur = self._db.execute(
                "SELECT * FROM units WHERE feed=? AND timeframe=? AND adjustment=? AND symbol=? AND period=?",
                self._key(unit),
            )
            row = cur.fetchone()
            return dict(zip([c[0] for c in cur.description], row, strict=True)) if row else None

    def put(self, unit: BarUnit, entry: dict) -> None:
        cols = [
            "year", "file", "sha256", "bytes", "rows", "regular_rows", "expected_regular", "coverage", "first_ts",
            "last_ts", "win_start", "win_end", "complete", "requests", "stage", "fetched_at",
        ]
        with self._lock:
            self._db.execute(
                f"INSERT OR REPLACE INTO units (feed,timeframe,adjustment,symbol,period,{','.join(cols)}) "
                f"VALUES (?,?,?,?,?,{','.join('?' * len(cols))})",
                (*self._key(unit), *[entry[c] for c in cols]),
            )
            self._db.commit()

    def put_probe(self, kind: str, symbol: str, feed: str, status: int | None, first_ts, last_ts, note: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO probes VALUES (?,?,?,?,?,?,?,?)",
                (kind, symbol, feed, status, first_ts, last_ts, note, _now_iso()),
            )
            self._db.commit()

    def put_run(self, stage: str, started_at: str, summary: dict) -> None:
        with self._lock:
            self._db.execute("INSERT INTO runs VALUES (?,?,?)", (stage, started_at, json.dumps(summary)))
            self._db.commit()

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            cur = self._db.execute(sql, args)
            names = [c[0] for c in cur.description]
            return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    def year_coverage(self) -> list[dict]:
        """Per symbol-year row counts and regular-session coverage (the A1 "done" view)."""
        return self.query(
            "SELECT symbol, year, SUM(rows) AS rows, SUM(regular_rows) AS regular_rows, "
            "SUM(expected_regular) AS expected_regular, "
            "ROUND(1.0*SUM(regular_rows)/NULLIF(SUM(expected_regular),0), 4) AS coverage, "
            "MIN(first_ts) AS first_ts, MAX(last_ts) AS last_ts, SUM(bytes) AS bytes, "
            "MIN(complete) AS all_complete FROM units GROUP BY symbol, year ORDER BY symbol, year"
        )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------- planning


def expected_regular_minutes(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Regular-session minutes (early closes counted) for sessions whose open falls in [start, end)."""
    cal = get_calendar()
    ny = "America/New_York"
    sched = cal.schedule(start.tz_convert(ny).tz_localize(None), end.tz_convert(ny).tz_localize(None))
    sched = sched[(sched["open"] >= start) & (sched["open"] < end)]
    return int(((sched["close"] - sched["open"]) / pd.Timedelta(minutes=1)).sum())


def wanted_window(period: str, range_start: pd.Timestamp, range_end: pd.Timestamp, now: pd.Timestamp, feed: str):
    """(start, end) UTC to fetch for a month unit: the month clipped to the pull range and, for SIP, to now - 16 min."""
    p_start, p_end = period_window(period)
    start = max(p_start, ny_midnight(range_start))
    end = min(p_end, ny_midnight(pd.Timestamp(range_end) + pd.Timedelta(days=1)))
    end = sip_safe_end(end, now) if feed == "sip" else min(end, now)
    return start, end


def plan_size(n_symbols: int, start, end) -> dict:
    sessions = len(get_calendar().sessions(start, end))
    rows = n_symbols * sessions * BARS_PER_SESSION_ESTIMATE
    return {"sessions": sessions, "rows_est": rows, "gb_est": round(rows * BYTES_PER_ROW_ESTIMATE / 1e9, 2)}


def disk_guard(root: Path, planned_gb: float, max_gb: float = MAX_PLANNED_GB) -> None:
    if planned_gb > max_gb:
        raise SystemExit(f"STOP: planned pull ~{planned_gb} GB exceeds the {max_gb} GB limit (raise --max-gb deliberately).")
    probe = root if root.exists() else next(p for p in root.parents if p.exists())
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb - planned_gb < MIN_FREE_GB_AFTER:
        raise SystemExit(f"STOP: {free_gb:.1f} GB free, planned {planned_gb} GB would leave under {MIN_FREE_GB_AFTER} GB.")


# ---------------------------------------------------------------------------------------------- probes


def probe_entitlement(client: AlpacaDataClient, manifest: ArchiveManifest, day: str) -> dict:
    """One 1-minute SPY request per feed on a wholly historical day: records HTTP status and bars returned."""
    out = {}
    start = ny_midnight(day)
    end = start + pd.Timedelta(days=1)
    for feed in ("sip", "iex"):
        try:
            n = len(client.bars(BENCHMARK, start, end, TIMEFRAME, ADJUSTMENT, feed))
            out[feed] = {"status": 200, "bars": n}
            manifest.put_probe("entitlement", BENCHMARK, feed, 200, None, None, f"{n} bars on {day}")
        except AlpacaError as err:
            out[feed] = {"status": err.status, "bars": 0}
            manifest.put_probe("entitlement", BENCHMARK, feed, err.status, None, None, str(err)[:200])
    return out


def _edge_bar(client: AlpacaDataClient, symbol: str, start: str, end: str, feed: str, sort: str) -> str | None:
    page = client._get(  # noqa: SLF001 - single-bar probe; the public bars() would page through everything
        f"/stocks/{symbol}/bars",
        {"timeframe": TIMEFRAME, "adjustment": ADJUSTMENT, "feed": feed, "start": start, "end": end, "limit": 1, "sort": sort},
    )
    bars = page.get("bars") or []
    return bars[0]["t"] if bars else None


def probe_history(client: AlpacaDataClient, manifest: ArchiveManifest, symbol: str, end: pd.Timestamp, feed: str) -> dict:
    """Earliest and latest 1-minute bar Alpaca has for `symbol` between the probe floor and `end`."""
    end_s = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        first = _edge_bar(client, symbol, PROBE_FLOOR, end_s, feed, "asc")
        last = _edge_bar(client, symbol, PROBE_FLOOR, end_s, feed, "desc") if first else None
        result = {"status": 200, "first_ts": first, "last_ts": last}
        note = "minute bars returned" if first else "no minute bars at all"
    except AlpacaError as err:
        result = {"status": err.status, "first_ts": None, "last_ts": None}
        note = str(err)[:200]
    manifest.put_probe("history", symbol, feed, result["status"], result["first_ts"], result["last_ts"], note)
    return result


# ---------------------------------------------------------------------------------------------- the pull


def run_stage(
    stage: str,
    symbols: list[str],
    start,
    end,
    feed: str,
    root: Path,
    client: AlpacaDataClient,
    transport: CountingTransport | None = None,
    now: pd.Timestamp | None = None,
    workers: int = 8,
    probe_symbols: list[str] | None = None,
    max_gb: float = MAX_PLANNED_GB,
    check_disk: bool = True,
) -> dict:
    """Pull every missing (symbol, month) unit for `symbols` over New York dates start..end; returns the summary."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    started_iso, t0 = _now_iso(), time.monotonic()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if ny_midnight(end + pd.Timedelta(days=1)) > now - pd.Timedelta(minutes=16) and feed == "sip":
        log.warning("range end reaches the last 16 minutes; the final unit is clipped and stored incomplete")
    root.mkdir(parents=True, exist_ok=True)
    plan = plan_size(len(symbols), start, end)
    log.info("stage %s: %d symbols, %s", stage, len(symbols), plan)
    if check_disk:
        disk_guard(root, plan["gb_est"], max_gb)

    manifest = ArchiveManifest(root / MANIFEST_DB)
    try:
        entitlement = probe_entitlement(client, manifest, _probe_day(start))
        log.info("entitlement: %s", entitlement)
        if feed == "sip" and entitlement["sip"]["status"] != 200:
            raise SystemExit(f"STOP: SIP not available on these keys ({entitlement['sip']}); not pulling history.")

        cal = get_calendar()
        units = [
            BarUnit(feed, TIMEFRAME, ADJUSTMENT, s, p) for s in symbols for p in periods_for(TIMEFRAME, start, end)
        ]
        todo, skipped = [], 0
        for unit in units:
            w_start, w_end = wanted_window(unit.period, start, end, now, feed)
            if w_end <= w_start:
                skipped += 1
                continue
            entry = manifest.get(unit)
            if (
                entry and entry["complete"] and (root / entry["file"]).exists()
                and pd.Timestamp(entry["win_start"]) <= w_start and pd.Timestamp(entry["win_end"]) >= w_end
            ):
                skipped += 1
            else:
                todo.append(unit)
        log.info("%d units total, %d already complete, %d to fetch", len(units), skipped, len(todo))

        counter = {"raw_bars": 0, "done": 0}
        counter_lock = threading.Lock()

        def fetch(unit: BarUnit) -> int:
            w_start, w_end = wanted_window(unit.period, start, end, now, feed)
            before = client.requests_sent
            raw = client.bars(unit.symbol, w_start, w_end, TIMEFRAME, ADJUSTMENT, feed)
            bars = normalize_bars(raw, unit.symbol, TIMEFRAME, cal)
            if len(bars):
                validate_bars(bars)
            meta = BarMetadata(feed=feed, timeframe=TIMEFRAME, adjustment=ADJUSTMENT)
            sha, rows = write_parquet_atomic(root / unit.relpath, bars_to_table(bars, meta))
            regular = int((bars["session"] == "regular").sum()) if len(bars) else 0
            expected = expected_regular_minutes(w_start, w_end)
            nominal_end = min(period_window(unit.period)[1], ny_midnight(end + pd.Timedelta(days=1)))
            manifest.put(unit, {
                "year": int(unit.period[:4]), "file": unit.relpath, "sha256": sha,
                "bytes": (root / unit.relpath).stat().st_size, "rows": rows, "regular_rows": regular,
                "expected_regular": expected, "coverage": round(regular / expected, 4) if expected else None,
                "first_ts": bars["ts"].iloc[0].isoformat() if len(bars) else None,
                "last_ts": bars["ts"].iloc[-1].isoformat() if len(bars) else None,
                "win_start": w_start.isoformat(), "win_end": w_end.isoformat(),
                "complete": int(w_end >= nominal_end), "requests": client.requests_sent - before,
                "stage": stage, "fetched_at": _now_iso(),
            })
            with counter_lock:
                counter["raw_bars"] += len(raw)
                counter["done"] += 1
                if counter["done"] % 100 == 0:
                    log.info("progress: %d/%d units, %s raw bars", counter["done"], len(todo), f"{counter['raw_bars']:,}")
            return len(raw)

        results, failed = run_units(todo, fetch, workers, f"pull[{stage}]", key=lambda u: u.key)

        history = {}
        for symbol in probe_symbols or []:
            history[symbol] = probe_history(client, manifest, symbol, min(ny_midnight(end + pd.Timedelta(days=1)), sip_safe_end(now, now)), feed)

        elapsed = time.monotonic() - t0
        net = transport.stats() if transport else {}
        disk_bytes = manifest.query("SELECT COALESCE(SUM(bytes),0) AS b, COALESCE(SUM(rows),0) AS r, COUNT(*) AS n FROM units")[0]
        ok = max(1, net.get("ok_requests", 0))
        summary = {
            "stage": stage, "feed": feed, "symbols": len(symbols), "start": str(start.date()), "end": str(end.date()),
            "units_total": len(units), "units_skipped": skipped, "units_fetched": len(results), "units_failed": len(failed),
            "failed": dict(list(failed.items())[:20]),
            "raw_bars_fetched": counter["raw_bars"], "elapsed_s": round(elapsed, 1),
            "bars_per_ok_request": round(counter["raw_bars"] / ok, 1),
            "rows_in_manifest": disk_bytes["r"], "bytes_on_disk": disk_bytes["b"], "units_in_manifest": disk_bytes["n"],
            "bytes_per_row": round(disk_bytes["b"] / disk_bytes["r"], 2) if disk_bytes["r"] else None,
            "entitlement": entitlement, "network": net, "delisted_history_probe": history,
            "planned": plan,
        }
        manifest.put_run(stage, started_iso, summary)
        (root / "reports").mkdir(exist_ok=True)
        (root / "reports" / f"{stage}_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        return summary
    finally:
        manifest.close()


def _probe_day(start: pd.Timestamp) -> str:
    """First weekday session on or after `start` (the entitlement probe needs a real trading day)."""
    return str(get_calendar().sessions(start, start + pd.Timedelta(days=10))[0].date())


# ---------------------------------------------------------------------------------------------- CLI


def _setup_logging(root: Path, stage: str) -> None:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(root / "logs" / f"{stage}.log")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=sorted(STAGES), default="pilot")
    ap.add_argument("--root", required=True, type=Path, help="store root (outside git)")
    ap.add_argument("--symbols", help="comma-separated override pool")
    ap.add_argument("--symbols-file", type=Path)
    ap.add_argument("--start", help="first New York date to analyse (default: end minus stage years)")
    ap.add_argument("--end", default="2025-10-31", help="last New York date, inclusive")
    ap.add_argument("--warmup-months", type=int, default=WARMUP_MONTHS)
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=None, help="request budget per minute (default config, 190)")
    ap.add_argument("--max-gb", type=float, default=MAX_PLANNED_GB)
    ap.add_argument("--dry-run", action="store_true", help="write under <root>/dryrun instead of the real store")
    args = ap.parse_args(argv)

    spec = STAGES[args.stage]
    if args.symbols:
        symbols = sorted({s.strip().upper() for s in args.symbols.split(",") if s.strip()})
    elif args.symbols_file:
        symbols = sorted({ln.strip().upper() for ln in args.symbols_file.read_text().splitlines() if ln.strip()})
    elif spec.symbols is None:
        symbols = pilot_pool()
    else:
        raise SystemExit(f"stage {spec.name} needs --symbols-file (about {spec.symbols} tickers)")

    end = pd.Timestamp(args.end)
    analysis_start = pd.Timestamp(args.start) if args.start else end - pd.DateOffset(years=spec.years) + pd.Timedelta(days=1)
    start = analysis_start if args.start else analysis_start - pd.DateOffset(months=args.warmup_months)
    root = args.root / "dryrun" if args.dry_run else args.root
    root.mkdir(parents=True, exist_ok=True)
    _setup_logging(root, args.stage)

    from shpe_research.data.alpaca import Credentials, shared_limiter  # env-only credentials
    from shpe_research.data.ratelimit import RateLimiter

    creds = Credentials.from_env()
    limiter = RateLimiter(args.rpm) if args.rpm else shared_limiter()
    transport = CountingTransport(httpx.HTTPTransport())
    with AlpacaDataClient(creds, limiter=limiter, transport=transport) as client:
        probes = [] if args.dry_run else sorted(set(DELISTED) & set(symbols)) + ([BENCHMARK] if BENCHMARK in symbols else [])
        summary = run_stage(
            args.stage, symbols, start, end, args.feed, root, client, transport, workers=args.workers,
            probe_symbols=probes, max_gb=args.max_gb,
        )
    log.info("summary: %s", json.dumps({k: v for k, v in summary.items() if k != "failed"}, sort_keys=True))
    return 1 if summary["units_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
