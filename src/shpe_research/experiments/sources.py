"""Where an experiment's bars come from, as named in its config's `data` section.

    data: {source: fixture, symbols: [...], start: ..., end: ..., seed: 7}
    data: {source: parquet, path: some/file_or_directory}
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


SOURCES: dict[str, Callable[[dict[str, Any]], pd.DataFrame]] = {
    "fixture": _fixture,
    "parquet": _parquet,
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
