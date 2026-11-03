"""Source protocol and registry (C1g, ADR 0011).

A source turns a config's `data` section into a frame. Bar sources return the
bars contract; event/series sources (`fred`, `edgar`) return `ts`,
`ts_available`, `series`, `value`. In every case `ts_available` is when the
value was knowable, never the period it describes.

    build_source({"type": "fixture", "seed": 7}).load(["AAA"], "2026-03-02", "2026-03-06", "1Min")
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar, Protocol, runtime_checkable

import pandas as pd

SOURCE_REGISTRY: dict[str, type] = {}
SERIES_COLUMNS = ["ts", "ts_available", "series", "value"]
_MODULES = ("fixture", "file", "alpaca", "fred", "edgar")


@runtime_checkable
class Source(Protocol):
    """`load(symbols, start, end, timeframe)`: `start`/`end` are inclusive New York dates (or None)."""

    source_type: ClassVar[str]

    def load(self, symbols: list[str] | None, start: Any, end: Any, timeframe: str) -> pd.DataFrame: ...


def register_source(cls: type) -> type:
    """Class decorator: register `cls` under its `source_type`."""
    key = cls.source_type
    if key in SOURCE_REGISTRY:
        raise ValueError(f"source_type {key!r} is already registered")
    SOURCE_REGISTRY[key] = cls
    return cls


def _load_builtins() -> None:
    for name in _MODULES:
        importlib.import_module(f"shpe_research.data.sources.{name}")


def build_source(cfg: dict[str, Any]) -> Source:
    """Instantiate the source named by `cfg["type"]`; every other key is a constructor parameter.

    The caller's dict is not mutated. An unknown type raises ValueError listing the registered keys.
    """
    _load_builtins()
    params = dict(cfg)
    key = params.pop("type", None)
    if key not in SOURCE_REGISTRY:
        raise ValueError(f"unknown data source {key!r}; known: {sorted(SOURCE_REGISTRY)}")
    return SOURCE_REGISTRY[key](**params)


def ny_window(start: Any, end: Any) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """UTC bounds [lo, hi) for inclusive New York dates `start`..`end`; None stays open."""
    lo = hi = None
    if start is not None:
        lo = pd.Timestamp(start).tz_localize("America/New_York").tz_convert("UTC")
    if end is not None:
        hi = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
    return lo, hi


def filter_bars(bars: pd.DataFrame, symbols: list[str] | None, start: Any, end: Any) -> pd.DataFrame:
    """Rows of `bars` for `symbols` with `ts` inside the New York date window, sorted by (symbol, ts)."""
    if symbols:
        bars = bars[bars["symbol"].isin([s.upper() for s in symbols])]
    lo, hi = ny_window(start, end)
    if lo is not None:
        bars = bars[bars["ts"] >= lo]
    if hi is not None:
        bars = bars[bars["ts"] < hi]
    return bars.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)


def series_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the `ts, ts_available, series, value` frame (extra row keys are kept after those four).

    Refuses `ts_available < ts`.
    """
    df = pd.DataFrame(rows)
    df = df[[*SERIES_COLUMNS, *[c for c in df.columns if c not in SERIES_COLUMNS]]]
    df["ts"] = pd.to_datetime(df["ts"], utc=True).astype("datetime64[ns, UTC]")
    df["ts_available"] = pd.to_datetime(df["ts_available"], utc=True).astype("datetime64[ns, UTC]")
    df["value"] = df["value"].astype(float)
    if (df["ts_available"] < df["ts"]).any():
        raise ValueError("ts_available is earlier than ts for some rows; a value cannot be known before its period")
    return df.sort_values(["series", "ts_available"], kind="stable").reset_index(drop=True)


def clip_available(df: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    """Keep series rows whose `ts_available` falls in the date window."""
    lo, hi = ny_window(start, end)
    if lo is not None:
        df = df[df["ts_available"] >= lo]
    if hi is not None:
        df = df[df["ts_available"] < hi]
    return df.reset_index(drop=True)
