"""Where an experiment's bars come from, as named in its config's `data` section.

    data: {source: fixture, symbols: [...], start: ..., end: ..., seed: 7}
    data: {source: parquet, path: some/file_or_directory}
    data: {source: store, symbols: [...], start: ..., end: ..., timeframe: 1Min, feed: sip, adjustment: raw}
    data: {source: snapshot, path: snapshots/workshop-v0.zip, role: dev, start: ..., end: ...}
          (role: all | dev | held_out | context; or list `symbols` explicitly)
    data: {source: file, path: bars.csv, format: csv}
    data: {source: alpaca, symbols: [...], start: ..., end: ..., feed: iex}

The sources themselves live in `shpe_research.data.sources` (C1g); this module keeps the config-facing
`load_bars` and the `SOURCES` key list so existing configs and imports keep working.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from shpe_research.contracts.bars import validate_bars
from shpe_research.data.sources import SOURCE_REGISTRY, build_source, register_source
from shpe_research.data.sources.base import _load_builtins

__all__ = ["SOURCES", "SOURCE_REGISTRY", "build_source", "load_bars", "register_source"]

_ALIASES = {"parquet": "file"}  # legacy key: `source: parquet` is the file source with format parquet
_LOAD_ARGS = ("symbols", "start", "end", "timeframe")
_load_builtins()
SOURCES = SOURCE_REGISTRY


def load_bars(cfg: dict[str, Any]) -> pd.DataFrame:
    """Contract bars for an experiment's `data` config section (`source` or `type` names the source).

    `symbols`, `start`, `end` and `timeframe` go to `Source.load`; every other key configures the source.
    Result rows are checked against the bars contract. Timing: `ts_available` comes from the source.
    """
    params = {k: v for k, v in cfg.items() if k not in ("source", "type")}
    key = cfg.get("source", cfg.get("type", "fixture"))
    key = _ALIASES.get(key, key)
    load_args = {k: params.pop(k) for k in _LOAD_ARGS if k in params}
    if key == "file" and cfg.get("source") == "parquet":
        params.setdefault("format", "parquet")
    source = build_source({"type": key, **params})
    bars = source.load(load_args.get("symbols"), load_args.get("start"), load_args.get("end"),
                       load_args.get("timeframe", "1Min"))
    validate_bars(bars)
    return bars
