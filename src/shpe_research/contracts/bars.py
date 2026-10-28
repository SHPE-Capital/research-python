"""Bars contract: OHLCV bars in one canonical shape, whatever their source.

Columns (one row per symbol and bar):

    symbol        str
    ts            UTC datetime, the START of the bar's interval (Alpaca's `t`)
    ts_available  UTC datetime, the earliest time the bar is fully known: the
                  end of its interval (for daily bars, that session's close,
                  early closes included). Decisions may use a bar only at or
                  after ts_available.
    open, high, low, close   float64, as delivered (adjustment recorded in metadata)
    volume        int64
    vwap          float64, NaN when the source has none
    trade_count   nullable Int64
    session       "pre" | "regular" | "post" | "off" (see data/calendar.py)

Rows are sorted by (symbol, ts) and unique on that pair. Minutes without
trades have no bar; they are never filled in. Feed, timeframe, adjustment,
source and contract version travel as metadata (BarMetadata), not columns.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import pandas as pd
import pyarrow as pa

from shpe_research.contracts.base import ContractError, check_columns
from shpe_research.contracts.version import CONTRACT_VERSION

SESSIONS = ("pre", "regular", "post", "off")

BAR_COLUMNS: dict[str, str] = {
    "symbol": "str",
    "ts": "utc",
    "ts_available": "utc",
    "open": "float",
    "high": "float",
    "low": "float",
    "close": "float",
    "volume": "int",
    "vwap": "float",
    "trade_count": "nullable_int",
    "session": "str",
}

BAR_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("ts_available", pa.timestamp("ns", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("vwap", pa.float64()),
        ("trade_count", pa.int64()),
        ("session", pa.string()),
    ]
)

_TIMEFRAME_DELTAS = {
    "1Min": pd.Timedelta(minutes=1),
    "5Min": pd.Timedelta(minutes=5),
    "15Min": pd.Timedelta(minutes=15),
    "30Min": pd.Timedelta(minutes=30),
    "1Hour": pd.Timedelta(hours=1),
}
DAILY = "1Day"
TIMEFRAMES = (*_TIMEFRAME_DELTAS, DAILY)


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    """Interval length of an intraday timeframe. Daily bars end at the session close instead."""
    try:
        return _TIMEFRAME_DELTAS[timeframe]
    except KeyError:
        raise ValueError(f"no fixed interval for timeframe {timeframe!r}; known: {TIMEFRAMES}") from None


@dataclass(frozen=True)
class BarMetadata:
    """Provenance stored alongside every bars file."""

    feed: str
    timeframe: str
    adjustment: str
    source: str = "alpaca-v2"
    contract_version: str = CONTRACT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_schema_metadata(self) -> dict[bytes, bytes]:
        return {b"shpe_research": json.dumps(asdict(self)).encode()}

    @classmethod
    def from_schema_metadata(cls, metadata: dict[bytes, bytes] | None) -> BarMetadata | None:
        if not metadata or b"shpe_research" not in metadata:
            return None
        return cls(**json.loads(metadata[b"shpe_research"]))


def empty_bars() -> pd.DataFrame:
    """A zero-row frame with the contract's columns and dtypes."""
    return BAR_SCHEMA.empty_table().to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get).astype(
        {"volume": "int64"}
    )


def validate_bars(df: pd.DataFrame) -> None:
    """Raise ContractError listing every way `df` breaks the bars contract."""
    problems = check_columns(df, BAR_COLUMNS)
    if problems:
        raise ContractError("bars", problems)
    if df.empty:
        return
    if df.duplicated(["symbol", "ts"]).any():
        problems.append(f"{int(df.duplicated(['symbol', 'ts']).sum())} duplicate (symbol, ts) rows")
    order = df[["symbol", "ts"]]
    if not order.equals(order.sort_values(["symbol", "ts"], kind="stable")):
        problems.append("rows must be sorted by (symbol, ts)")
    if (df["ts_available"] <= df["ts"]).any():
        problems.append("ts_available must be after ts on every row")
    bad_sessions = set(df["session"].unique()) - set(SESSIONS)
    if bad_sessions:
        problems.append(f"unknown session values {sorted(bad_sessions)}")
    if df[["open", "high", "low", "close"]].isna().any().any():
        problems.append("open/high/low/close may not be NaN")
    if problems:
        raise ContractError("bars", problems)


def bars_to_table(df: pd.DataFrame, metadata: BarMetadata) -> pa.Table:
    """Contract bars as an Arrow table carrying their provenance, ready to write as Parquet."""
    table = pa.Table.from_pandas(df[list(BAR_COLUMNS)], schema=BAR_SCHEMA, preserve_index=False)
    return table.replace_schema_metadata(metadata.to_schema_metadata())


def table_to_bars(table: pa.Table) -> pd.DataFrame:
    """Arrow table (e.g. read from Parquet) back to contract dtypes."""
    df = table.select(list(BAR_COLUMNS)).to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)
    return df.astype({"volume": "int64"})
