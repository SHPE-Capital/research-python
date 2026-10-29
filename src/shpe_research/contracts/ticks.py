"""Quote and trade contracts: tick records in one canonical shape.

Quotes (one row per displayed top-of-book update):

    symbol, ts (UTC ns, exchange/SIP timestamp), bid_price, ask_price (float64),
    bid_size, ask_size (int64, ROUND LOTS as Alpaca sends them, usually 100
    shares each; convert explicitly before using as a share count),
    bid_exchange, ask_exchange, tape (str), conditions (list of str)

Trades (one row per print):

    symbol, ts (UTC ns), price (float64), size (int64, shares), exchange, tape
    (str), conditions (list of str), trade_id (str)

Rows are sorted by (symbol, ts). Ticks share a timestamp legitimately, so ts
is not unique. A tick is available at its own ts; any receipt delay is an
explicit modelling assumption, never baked into the data.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa

from shpe_research.contracts.base import ContractError, check_columns

QUOTE_SIZE_UNIT = "round_lots"

QUOTE_COLUMNS: dict[str, str] = {
    "symbol": "str",
    "ts": "utc",
    "bid_price": "float",
    "ask_price": "float",
    "bid_size": "int",
    "ask_size": "int",
    "bid_exchange": "str",
    "ask_exchange": "str",
    "tape": "str",
    "conditions": "list",
}

TRADE_COLUMNS: dict[str, str] = {
    "symbol": "str",
    "ts": "utc",
    "price": "float",
    "size": "int",
    "exchange": "str",
    "tape": "str",
    "conditions": "list",
    "trade_id": "str",
}

QUOTE_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("bid_price", pa.float64()),
        ("ask_price", pa.float64()),
        ("bid_size", pa.int64()),
        ("ask_size", pa.int64()),
        ("bid_exchange", pa.string()),
        ("ask_exchange", pa.string()),
        ("tape", pa.string()),
        ("conditions", pa.list_(pa.string())),
    ]
)

TRADE_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("price", pa.float64()),
        ("size", pa.int64()),
        ("exchange", pa.string()),
        ("tape", pa.string()),
        ("conditions", pa.list_(pa.string())),
        ("trade_id", pa.string()),
    ]
)

SCHEMAS = {"quotes": QUOTE_SCHEMA, "trades": TRADE_SCHEMA}
COLUMNS = {"quotes": QUOTE_COLUMNS, "trades": TRADE_COLUMNS}

# Per-minute summaries of quotes and trades (data/ticks.py:minute_summaries), joinable onto
# 1-minute bars by (symbol, ts). Like a bar, a minute's summary is available at its end.
TICK_SUMMARY_COLUMNS: dict[str, str] = {
    "symbol": "str",
    "ts": "utc",  # minute start
    "ts_available": "utc",  # minute end
    "quote_count": "int",
    "spread_bps_mean": "float",  # mean of (ask - bid) / mid over the minute's quotes, in basis points
    "spread_bps_median": "float",
    "imbalance_mean": "float",  # mean of (bid_size - ask_size) / (bid_size + ask_size)
    "mid_close": "float",  # last mid of the minute
    "trade_count": "int",
    "trade_volume": "int",
    "buy_volume": "int",  # Lee-Ready: above the prevailing mid, or an uptick at the mid
    "sell_volume": "int",  # below the prevailing mid, or a downtick at the mid
}


def _validate(kind: str, df: pd.DataFrame) -> None:
    problems = check_columns(df, COLUMNS[kind])
    if not problems and not df.empty:
        order = df[["symbol", "ts"]]
        if not order.equals(order.sort_values(["symbol", "ts"], kind="stable")):
            problems.append("rows must be sorted by (symbol, ts)")
    if problems:
        raise ContractError(kind, problems)


def validate_quotes(df: pd.DataFrame) -> None:
    """Raise ContractError listing every way `df` breaks the quotes contract."""
    _validate("quotes", df)
    if not df.empty and ((df["bid_price"] <= 0) | (df["ask_price"] < df["bid_price"])).any():
        raise ContractError("quotes", ["one-sided or crossed quotes must be dropped during normalization"])


def validate_trades(df: pd.DataFrame) -> None:
    """Raise ContractError listing every way `df` breaks the trades contract."""
    _validate("trades", df)
    if not df.empty and ((df["price"] <= 0) | (df["size"] <= 0)).any():
        raise ContractError("trades", ["trades need positive price and size"])
