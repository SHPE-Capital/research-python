"""Where an experiment's bars come from, as named in its config's `data` section.

    data: {source: fixture, symbols: [...], start: ..., end: ..., seed: 7}
    data: {source: parquet, path: some/file_or_directory}
    data: {source: store, symbols: [...], start: ..., end: ..., timeframe: 1Min, feed: sip, adjustment: raw}
    data: {source: snapshot, path: snapshots/workshop-v0.zip, role: dev, start: ..., end: ...}
          (role: all | dev | held_out | context; or list `symbols` explicitly)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from shpe_research.contracts.bars import table_to_bars, validate_bars
from shpe_research.data.fixtures import synthetic_bars


def _fixture(cfg: dict[str, Any]) -> pd.DataFrame:
    params = {k: v for k, v in cfg.items() if k in ("symbols", "start", "end", "seed", "signal_strength")}
    return synthetic_bars(**params)


def _parquet(cfg: dict[str, Any]) -> pd.DataFrame:
    bars = table_to_bars(pq.read_table(Path(cfg["path"])))
    return bars.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)


def _store(cfg: dict[str, Any]) -> pd.DataFrame:
    from shpe_research.data.store import DataStore

    return DataStore(cfg.get("root")).read_bars(
        [s.upper() for s in cfg["symbols"]],
        cfg["start"],
        cfg["end"],
        timeframe=cfg.get("timeframe", "1Min"),
        feed=cfg.get("feed", "sip"),
        adjustment=cfg.get("adjustment", "raw"),
    )


def _snapshot(cfg: dict[str, Any]) -> pd.DataFrame:
    from shpe_research.data.snapshot import load_snapshot

    snap = load_snapshot(cfg["path"])
    symbols = cfg.get("symbols") or snap.symbols(cfg.get("role", "all"))
    return snap.bars(
        timeframe=cfg.get("timeframe", "1Min"),
        adjustment=cfg.get("adjustment", "raw"),
        symbols=[s.upper() for s in symbols],
        start=cfg.get("start"),
        end=cfg.get("end"),
    )


SOURCES: dict[str, Callable[[dict[str, Any]], pd.DataFrame]] = {
    "fixture": _fixture,
    "parquet": _parquet,
    "store": _store,
    "snapshot": _snapshot,
}


def load_bars(cfg: dict[str, Any]) -> pd.DataFrame:
    """Contract bars for an experiment's `data` config section."""
    source = cfg.get("source", "fixture")
    try:
        loader = SOURCES[source]
    except KeyError:
        raise ValueError(f"unknown data source {source!r}; known: {sorted(SOURCES)}") from None
    bars = loader(cfg)
    validate_bars(bars)
    return bars
