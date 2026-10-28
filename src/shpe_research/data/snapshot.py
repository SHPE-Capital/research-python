"""D5: frozen, versioned, checksummed dataset snapshots.

    uv run shpe data snapshot build --name workshop-v0 --universe data/universes/workshop_v0.yaml \
        --from 2026-07-01 --to 2026-09-30 --extra-symbols SPY
    uv run shpe data snapshot verify snapshots/workshop-v0.zip

A snapshot copies bars for a fixed symbol list and date range out of the local
store into one self-describing folder (and a zip of it):

    snapshots/<name>/
        snapshot.json     contract version, research-repo commit, symbols and roles,
                          every file's sha256 and row count, and the store entries it came from
        universe.yaml     the D4 manifest, when built from one
        bars/<timeframe>_<adjustment>/<SYMBOL>.parquet

Everyone who loads the same snapshot gets identical data, so workshop results
are comparable and reruns are reproducible. `load_snapshot` verifies checksums
before returning anything.

Snapshots contain Alpaca market data. Keep them members-only (a private
drive or private release), never in this repo or anywhere public; Alpaca's
data terms restrict redistribution.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from shpe_research.config import git_commit, snapshots_dir
from shpe_research.contracts.bars import BarMetadata, bars_to_table, table_to_bars, validate_bars
from shpe_research.contracts.version import CONTRACT_VERSION
from shpe_research.data.store import BarUnit, DataStore, periods_for, sha256_file, write_parquet_atomic

MANIFEST = "snapshot.json"
NOTICE = "Contains Alpaca market data: members-only, do not redistribute or publish."
DEFAULT_SERIES = (("1Min", "raw"), ("1Day", "split"))


def _symbols_from_universe(path: Path) -> tuple[list[str], dict[str, str]]:
    manifest = yaml.safe_load(Path(path).read_text())
    roles = {s: "dev" for s in manifest["dev"]} | {s: "held_out" for s in manifest["held_out"]}
    return sorted(roles), roles


def build_snapshot(
    name: str,
    start,
    end,
    symbols: list[str] | None = None,
    universe: str | Path | None = None,
    extra_symbols: list[str] | None = None,
    series: tuple[tuple[str, str], ...] = DEFAULT_SERIES,
    feed: str = "sip",
    store: DataStore | None = None,
    out_root: str | Path | None = None,
    make_zip: bool = True,
) -> Path:
    """Copy bars for New York dates start..end into snapshots/<name>/ and return that folder."""
    store = store or DataStore()
    roles: dict[str, str] = {}
    if universe is not None:
        symbols, roles = _symbols_from_universe(Path(universe))
    if not symbols:
        raise ValueError("pass symbols or a universe manifest")
    extras = [s.upper() for s in (extra_symbols or [])]
    roles |= {s: "context" for s in extras if s not in roles}
    all_symbols = sorted({*(s.upper() for s in symbols), *extras})

    out = Path(out_root or snapshots_dir()) / name
    if out.exists():
        raise FileExistsError(f"{out} exists; snapshots are immutable, so pick a new name (e.g. {name}-v2)")
    files, sources = [], []
    for timeframe, adjustment in series:
        for symbol in all_symbols:
            bars = store.read_bars([symbol], start, end, timeframe=timeframe, feed=feed, adjustment=adjustment)
            rel = f"bars/{timeframe}_{adjustment}/{symbol}.parquet"
            meta = BarMetadata(feed=feed, timeframe=timeframe, adjustment=adjustment)
            sha, rows = write_parquet_atomic(out / rel, bars_to_table(bars, meta))
            files.append({"path": rel, "sha256": sha, "rows": rows, "symbol": symbol, "timeframe": timeframe,
                          "adjustment": adjustment})
            for period in periods_for(timeframe, start, end):
                entry = store.bars_manifest.get(BarUnit(feed, timeframe, adjustment, symbol, period).key) or {}
                sources.append({"unit": BarUnit(feed, timeframe, adjustment, symbol, period).key,
                                "sha256": entry.get("sha256")})
    if universe is not None:
        shutil.copyfile(universe, out / "universe.yaml")
    manifest = {
        "name": name,
        "notice": NOTICE,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "contract_version": CONTRACT_VERSION,
        "git_commit": git_commit(),
        "feed": feed,
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "symbols": all_symbols,
        "roles": {s: roles.get(s, "") for s in all_symbols},
        "series": [{"timeframe": t, "adjustment": a} for t, a in series],
        "universe": "universe.yaml" if universe is not None else None,
        "files": files,
        "sources": sources,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1))
    if make_zip:
        _zip(out)
    return out


def _zip(folder: Path) -> Path:
    path = folder.with_suffix(".zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:  # Parquet is already compressed
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                zf.write(file, f"{folder.name}/{file.relative_to(folder).as_posix()}")
    return path


def _extract(zip_path: Path) -> Path:
    """Extract next to the zip (once) and return the snapshot folder."""
    with zipfile.ZipFile(zip_path) as zf:
        roots = {Path(n).parts[0] for n in zf.namelist()}
        if len(roots) != 1:
            raise ValueError(f"{zip_path} is not a snapshot archive (expected one top-level folder)")
        folder = zip_path.parent / roots.pop()
        if not folder.exists():
            zf.extractall(zip_path.parent)
    return folder


def _folder(path: str | Path) -> Path:
    path = Path(path)
    return _extract(path) if path.suffix == ".zip" else path


def verify_snapshot(path: str | Path) -> list[str]:
    """Problems with a snapshot folder or zip: missing files, checksum mismatches, contract violations."""
    folder = _folder(path)
    manifest_path = folder / MANIFEST
    if not manifest_path.exists():
        return [f"{folder}: no {MANIFEST}"]
    manifest = json.loads(manifest_path.read_text())
    problems = []
    if manifest["contract_version"].split(".")[0] != CONTRACT_VERSION.split(".")[0]:
        problems.append(f"contract version {manifest['contract_version']} is incompatible with {CONTRACT_VERSION}")
    for f in manifest["files"]:
        file = folder / f["path"]
        if not file.exists():
            problems.append(f"{f['path']}: missing")
            continue
        if sha256_file(file) != f["sha256"]:
            problems.append(f"{f['path']}: checksum mismatch")
            continue
        try:
            validate_bars(table_to_bars(pq.read_table(file)))
        except ValueError as err:
            problems.append(f"{f['path']}: {err}")
    return problems


@dataclass
class Snapshot:
    folder: Path
    manifest: dict[str, Any]

    @property
    def name(self) -> str:
        return self.manifest["name"]

    def symbols(self, role: str = "all") -> list[str]:
        """All symbols, or those with a role: "dev", "held_out" or "context" (e.g. a benchmark)."""
        if role == "all":
            return list(self.manifest["symbols"])
        return [s for s, r in self.manifest["roles"].items() if r == role]

    def bars(
        self, timeframe: str = "1Min", adjustment: str = "raw", symbols: list[str] | None = None,
        start=None, end=None,
    ) -> pd.DataFrame:
        wanted = set(symbols or self.symbols())
        files = [f for f in self.manifest["files"]
                 if f["timeframe"] == timeframe and f["adjustment"] == adjustment and f["symbol"] in wanted]
        if not files:
            raise ValueError(f"snapshot {self.name} has no {timeframe}/{adjustment} bars for {sorted(wanted)}")
        bars = pd.concat([table_to_bars(pq.read_table(self.folder / f["path"])) for f in files], ignore_index=True)
        if start is not None:
            bars = bars[bars["ts"] >= pd.Timestamp(start).tz_localize("America/New_York").tz_convert("UTC")]
        if end is not None:
            stop = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
            bars = bars[bars["ts"] < stop]
        return bars.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)

    def fingerprint(self) -> str:
        """One hash over every file checksum: two snapshots with the same fingerprint hold the same data."""
        digest = hashlib.sha256()
        for f in sorted(self.manifest["files"], key=lambda f: f["path"]):
            digest.update(f"{f['path']}:{f['sha256']}".encode())
        return digest.hexdigest()


def load_snapshot(path: str | Path) -> Snapshot:
    """Open a snapshot folder or zip after verifying every checksum."""
    folder = _folder(path)
    problems = verify_snapshot(folder)
    if problems:
        raise ValueError(f"snapshot {folder} failed verification:\n  " + "\n  ".join(problems))
    return Snapshot(folder=folder, manifest=json.loads((folder / MANIFEST).read_text()))
