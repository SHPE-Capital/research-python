"""`file` source (local Parquet or CSV, validated against the contract) and `snapshot` source (D5 snapshots)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pyarrow.parquet as pq

from shpe_research.contracts.bars import BAR_COLUMNS, table_to_bars, validate_bars
from shpe_research.data.sources.base import filter_bars, register_source


@register_source
class FileSource:
    source_type: ClassVar[str] = "file"

    def __init__(self, path: str, format: str = "parquet"):  # noqa: A002 - config key name
        if format not in ("parquet", "csv"):
            raise ValueError(f"file source format must be 'parquet' or 'csv', got {format!r}")
        self.path = Path(path)
        self.format = format

    def load(self, symbols: list[str] | None, start: Any, end: Any, timeframe: str = "1Min") -> pd.DataFrame:
        """Bars from the file, filtered to `symbols` and the New York date window when given.

        Raises ValueError (ContractError) when the file does not satisfy the bars contract. `ts_available`
        is whatever the file holds; the contract check enforces it is not before the bar's own `ts`.
        """
        if self.format == "parquet":
            bars = table_to_bars(pq.read_table(self.path))
        else:
            raw = pd.read_csv(self.path)
            if any(c not in raw.columns for c in BAR_COLUMNS):
                validate_bars(raw)  # raises a ContractError that lists the problems
            for col in ("ts", "ts_available"):
                raw[col] = pd.to_datetime(raw[col], utc=True).astype("datetime64[ns, UTC]")
            bars = raw[list(BAR_COLUMNS)].astype({"volume": "int64"})
            bars["trade_count"] = bars["trade_count"].astype("Int64")
        bars = filter_bars(bars, symbols, start, end)
        validate_bars(bars)
        return bars


@register_source
class SnapshotSource:
    source_type: ClassVar[str] = "snapshot"

    def __init__(self, path: str, role: str = "all", adjustment: str = "raw"):
        self.path = path
        self.role = role
        self.adjustment = adjustment

    def load(self, symbols: list[str] | None, start: Any, end: Any, timeframe: str = "1Min") -> pd.DataFrame:
        """Snapshot bars for `symbols` (default: every symbol with `role`) in the date window.

        Checksums are verified (D5) before anything loads; raises ValueError ("failed verification") when a
        file is missing, altered or breaks the contract.
        """
        from shpe_research.data.snapshot import load_snapshot

        snap = load_snapshot(self.path)
        wanted = symbols or snap.symbols(self.role)
        return snap.bars(timeframe=timeframe, adjustment=self.adjustment, symbols=[s.upper() for s in wanted],
                         start=start, end=end)
