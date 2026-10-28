from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.contracts.base import to_utc_ns


def make_bars(symbol: str = "AAA", n: int = 60, start: str = "2026-10-01T13:30:00Z", seed: int = 0) -> pd.DataFrame:
    """Small contract-valid regular-session minute bars for one symbol."""
    rng = np.random.default_rng(seed)
    ts = to_utc_ns(pd.date_range(start, periods=n, freq="1min"))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": ts,
            "ts_available": ts + pd.Timedelta(minutes=1),
            "open": open_,
            "high": np.maximum(open_, close) * 1.0005,
            "low": np.minimum(open_, close) * 0.9995,
            "close": close,
            "volume": rng.integers(100, 10_000, n).astype("int64"),
            "vwap": (open_ + close) / 2,
            "trade_count": pd.array(rng.integers(1, 100, n), dtype="Int64"),
            "session": "regular",
        }
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()
