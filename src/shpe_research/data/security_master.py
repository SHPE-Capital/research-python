"""C1m: security master, the record of which symbols existed and belonged where on each date.

Each row says: `symbol` was listed on `listing_exchange` with `asset_status` and `sector` from `valid_from`
(inclusive) until `valid_to` (exclusive; NaT means still valid). A symbol whose sector changes has one row per
sector, with touching date ranges. Dates are New York calendar dates.

Timing: `members(d)` and `sector_of(symbols, d)` use only rows with `valid_from <= d < valid_to`, i.e. what was
known to hold on `d`. A delisted name stays a member up to its `valid_to`; bars after that are not filled.

The table is stored as Parquet with a `.sha256.json` sidecar; `load` verifies the checksum before returning.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from shpe_research.data.store import sha256_file

COLUMNS = ["symbol", "valid_from", "valid_to", "sector", "listing_exchange", "asset_status", "source"]
FILTERABLE = ("sector", "listing_exchange", "asset_status", "source")


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256.json")


def _day(value) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None).normalize()


class SecurityMaster:
    """Point-in-time table of symbols; see the module docstring for the row semantics."""

    def __init__(self, table: pd.DataFrame):
        missing = [c for c in COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"security master is missing columns {missing}")
        t = table[COLUMNS].copy()
        t["symbol"] = t["symbol"].astype(str).str.strip().str.upper()
        t["valid_from"] = pd.to_datetime(t["valid_from"]).astype("datetime64[ns]")
        t["valid_to"] = pd.to_datetime(t["valid_to"]).astype("datetime64[ns]")
        if t["valid_from"].isna().any():
            raise ValueError("security master rows need a valid_from")
        if (t["valid_to"] <= t["valid_from"]).any():
            raise ValueError("security master rows need valid_to > valid_from")
        self.table = t.sort_values(["symbol", "valid_from"], kind="stable").reset_index(drop=True)
        self._check_overlaps()

    def _check_overlaps(self) -> None:
        for symbol, g in self.table.groupby("symbol"):
            ends = g["valid_to"].fillna(pd.Timestamp.max)
            if (g["valid_from"].iloc[1:].to_numpy() < ends.iloc[:-1].to_numpy()).any():
                raise ValueError(f"security master rows for {symbol} overlap in time")

    def _active(self, date) -> pd.DataFrame:
        d = _day(date)
        t = self.table
        return t[(t["valid_from"] <= d) & (t["valid_to"].isna() | (d < t["valid_to"]))]

    def members(self, date, filters: Mapping[str, Any] | None = None) -> list[str]:
        """Sorted symbols valid on `date`, optionally restricted by column filters.

        `filters` maps a column in `FILTERABLE` to one value or a list of values. Only information valid on
        `date` is used, so a name listed later is absent and a name delisted later is still present.
        """
        active = self._active(date)
        for column, wanted in (filters or {}).items():
            if column not in FILTERABLE:
                raise ValueError(f"unknown filter {column!r}; use one of {FILTERABLE}")
            values = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
            active = active[active[column].isin(values)]
        return sorted(active["symbol"].unique())

    def sector_of(self, symbols, date) -> pd.Series:
        """Sector of each symbol as valid on `date`; NaN for symbols not in the master on that date."""
        wanted = [str(s).strip().upper() for s in symbols]
        sectors = self._active(date).set_index("symbol")["sector"]
        return sectors.reindex(wanted).rename("sector")

    def save(self, path: str | Path) -> Path:
        """Write Parquet plus a checksum sidecar next to it; returns the Parquet path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(path, index=False)
        _sidecar(path).write_text(json.dumps({"file": path.name, "sha256": sha256_file(path),
                                              "rows": len(self.table)}, indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> SecurityMaster:
        """Read a saved master after verifying its checksum; raises ValueError on a missing or wrong one."""
        path = Path(path)
        sidecar = _sidecar(path)
        if not sidecar.exists():
            raise ValueError(f"{path}: no checksum file {sidecar.name}")
        expected = json.loads(sidecar.read_text())["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"{path}: checksum mismatch, the security master was modified")
        return cls(pd.read_parquet(path))
