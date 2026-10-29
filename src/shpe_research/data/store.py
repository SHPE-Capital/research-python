"""Local Parquet store for market data, with a checksummed manifest.

Layout under the data directory (RESEARCH_DATA_DIR, default ./data):

    bars/feed=sip/timeframe=1Min/adjustment=raw/symbol=AAPL/period=2026-10.parquet
    bars/feed=sip/timeframe=1Day/adjustment=split/symbol=AAPL/period=2026.parquet
    bars/manifest.json
    ticks/feed=sip/kind=quotes/symbol=KO/date=2026-10-27.parquet   (data/ticks.py)
    ticks/manifest.json

A "unit" is one file: one symbol over one period. Intraday bars use New York
calendar months; daily bars use years. Periods follow New York dates, so a
session and its extended hours never straddle two files.

Every unit write is atomic (temporary file, then rename), so a crash never
leaves a half-written file that looks complete. The manifest records each
unit's sha256, row count, fetched window and whether the period is complete;
a period that is still in progress is refetched on the next download.
Run one downloader process per data directory at a time; threads within a
process share the manifest safely.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from shpe_research.config import data_dir
from shpe_research.contracts.bars import DAILY, BarMetadata, bars_to_table, table_to_bars, validate_bars
from shpe_research.contracts.version import CONTRACT_VERSION
from shpe_research.data.calendar import NY

MANIFEST = "manifest.json"


class StoreError(RuntimeError):
    pass


def ny_midnight(day) -> pd.Timestamp:
    """Start of a New York calendar day, in UTC."""
    return pd.Timestamp(pd.Timestamp(day).date()).tz_localize(NY).tz_convert("UTC")


def periods_for(timeframe: str, start, end) -> list[str]:
    """Period labels covering New York dates start..end inclusive ("2026-10" months, or "2026" years for 1Day)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if timeframe == DAILY:
        return [str(y) for y in range(start.year, end.year + 1)]
    return [p.strftime("%Y-%m") for p in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")]


def period_window(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of a period in UTC, on New York midnights."""
    if len(period) == 4:
        start = pd.Timestamp(f"{period}-01-01")
        end = pd.Timestamp(f"{int(period) + 1}-01-01")
    else:
        start = pd.Timestamp(f"{period}-01")
        end = start + pd.offsets.MonthBegin(1)
    return ny_midnight(start), ny_midnight(end)


@dataclass(frozen=True)
class BarUnit:
    feed: str
    timeframe: str
    adjustment: str
    symbol: str
    period: str

    @property
    def key(self) -> str:
        return (
            f"feed={self.feed}/timeframe={self.timeframe}/adjustment={self.adjustment}"
            f"/symbol={self.symbol}/period={self.period}"
        )

    @property
    def relpath(self) -> str:
        return f"bars/{self.key}.parquet"


@dataclass(frozen=True)
class TickUnit:
    """Quotes or trades for one symbol on one New York date."""

    feed: str
    kind: str
    symbol: str
    date: str  # YYYY-MM-DD

    @property
    def key(self) -> str:
        return f"feed={self.feed}/kind={self.kind}/symbol={self.symbol}/date={self.date}"

    @property
    def relpath(self) -> str:
        return f"ticks/{self.key}.parquet"


class Manifest:
    """JSON index of stored units. Thread-safe; call flush() to persist pending updates."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._dirty = 0
        if path.exists():
            data = json.loads(path.read_text())
            self._units: dict[str, dict[str, Any]] = data.get("units", {})
        else:
            self._units = {}

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._units.get(key)
            return dict(entry) if entry else None

    def set(self, key: str, entry: dict[str, Any], flush_every: int = 20) -> None:
        with self._lock:
            self._units[key] = entry
            self._dirty += 1
            should_flush = self._dirty >= flush_every
        if should_flush:
            self.flush()

    def items(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            return sorted((k, dict(v)) for k, v in self._units.items())

    def flush(self) -> None:
        with self._lock:
            payload = json.dumps({"contract_version": CONTRACT_VERSION, "units": self._units}, indent=1, sort_keys=True)
            self._dirty = 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(self.path, payload.encode())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet_atomic(path: Path, table: pa.Table) -> tuple[str, int]:
    """Write `table` to `path` atomically; returns (sha256, rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return sha256_file(path), table.num_rows


def _entry(relpath: str, sha: str, rows: int, start: pd.Timestamp, end: pd.Timestamp, complete: bool, **extra):
    return {
        "file": relpath,
        "sha256": sha,
        "rows": rows,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "complete": complete,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "contract_version": CONTRACT_VERSION,
        **extra,
    }


class DataStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else data_dir()
        self.bars_manifest = Manifest(self.root / "bars" / MANIFEST)
        self.ticks_manifest = Manifest(self.root / "ticks" / MANIFEST)

    def manifests(self) -> dict[str, Manifest]:
        return {"bars": self.bars_manifest, "ticks": self.ticks_manifest}

    # ------------------------------------------------------------ bars

    def write_bar_unit(
        self,
        unit: BarUnit,
        bars: pd.DataFrame,
        fetched_start: pd.Timestamp,
        fetched_end: pd.Timestamp,
        complete: bool,
    ) -> dict[str, Any]:
        validate_bars(bars)
        meta = BarMetadata(feed=unit.feed, timeframe=unit.timeframe, adjustment=unit.adjustment)
        sha, rows = write_parquet_atomic(self.root / unit.relpath, bars_to_table(bars, meta))
        entry = _entry(unit.relpath, sha, rows, fetched_start, fetched_end, complete)
        self.bars_manifest.set(unit.key, entry)
        return entry

    def bar_unit_complete(self, unit: BarUnit) -> bool:
        entry = self.bars_manifest.get(unit.key)
        return bool(entry and entry["complete"] and (self.root / entry["file"]).exists())

    def read_bars(
        self,
        symbols: list[str],
        start,
        end,
        timeframe: str = "1Min",
        feed: str = "sip",
        adjustment: str = "raw",
        sessions: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Contract bars for New York dates start..end inclusive. Raises StoreError if any part was never fetched."""
        lo, hi = ny_midnight(start), ny_midnight(pd.Timestamp(end) + pd.Timedelta(days=1))
        tables, missing = [], []
        for symbol in symbols:
            for period in periods_for(timeframe, start, end):
                unit = BarUnit(feed, timeframe, adjustment, symbol, period)
                entry = self.bars_manifest.get(unit.key)
                p_start, p_end = period_window(period)
                need_end = min(hi, p_end)
                path = self.root / unit.relpath
                if entry is None or not path.exists() or pd.Timestamp(entry["end"]) < need_end:
                    covered = entry["end"] if entry else "nothing"
                    missing.append(f"{unit.key} (stored through {covered})")
                    continue
                tables.append(pq.read_table(path))
        if missing:
            raise StoreError(
                "bars not in the local store:\n  " + "\n  ".join(missing) +
                f"\nDownload them with: uv run shpe data bars --symbols {','.join(symbols)} "
                f"--from {pd.Timestamp(start).date()} --to {pd.Timestamp(end).date()} --timeframe {timeframe} "
                f"--adjustment {adjustment} --feed {feed}"
            )
        if not tables:
            raise StoreError("no symbols requested")
        bars = table_to_bars(pa.concat_tables([t.replace_schema_metadata(None) for t in tables]))
        bars = bars[(bars["ts"] >= lo) & (bars["ts"] < hi)]
        if sessions is not None:
            bars = bars[bars["session"].isin(sessions)]
        return bars.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------ ticks

    def write_tick_unit(
        self, unit: TickUnit, table: pa.Table, fetched_start: pd.Timestamp, fetched_end: pd.Timestamp,
        complete: bool, dropped: int = 0,
    ) -> dict[str, Any]:
        sha, rows = write_parquet_atomic(self.root / unit.relpath, table)
        entry = _entry(unit.relpath, sha, rows, fetched_start, fetched_end, complete, dropped=dropped)
        self.ticks_manifest.set(unit.key, entry)
        return entry

    def tick_unit_covers(self, unit: TickUnit, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """Whether the stored unit was fetched over a window containing [start, end)."""
        entry = self.ticks_manifest.get(unit.key)
        return bool(
            entry
            and (self.root / entry["file"]).exists()
            and pd.Timestamp(entry["start"]) <= start
            and pd.Timestamp(entry["end"]) >= end
        )

    def read_ticks(self, kind: str, symbols: list[str], start, end, feed: str = "sip") -> pd.DataFrame:
        """Stored quotes or trades for New York dates start..end inclusive, sorted by (symbol, ts)."""
        from shpe_research.contracts.ticks import SCHEMAS

        tables, missing = [], []
        for symbol in symbols:
            for day in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"):
                unit = TickUnit(feed, kind, symbol, day.strftime("%Y-%m-%d"))
                entry = self.ticks_manifest.get(unit.key)
                if entry is None:
                    continue  # weekends and holidays have no unit; downloads record every session
                path = self.root / unit.relpath
                if not path.exists():
                    missing.append(unit.key)
                    continue
                tables.append(pq.read_table(path))
        if missing:
            raise StoreError("tick files listed in the manifest are missing:\n  " + "\n  ".join(missing))
        if not tables:
            return SCHEMAS[kind].empty_table().to_pandas()
        df = pa.concat_tables(tables).to_pandas()
        return df.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------ integrity

    def verify(self) -> list[str]:
        """Problems found re-checking every manifest entry's file and checksum (empty list means clean)."""
        problems = []
        for name, manifest in self.manifests().items():
            for key, entry in manifest.items():
                path = self.root / entry["file"]
                if not path.exists():
                    problems.append(f"{name}/{key}: file missing")
                elif sha256_file(path) != entry["sha256"]:
                    problems.append(f"{name}/{key}: checksum mismatch")
        return problems
