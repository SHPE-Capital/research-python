"""Deterministic synthetic bars for tests, the workshop and offline development.

Regular-session minute bars on real NYSE sessions (holidays and early closes
included) that satisfy the bars contract, so every layer can be built and
tested before anyone downloads market data.

The price process has a small, known amount of short-horizon mean reversion
(`signal_strength`): each minute's return leans against the previous five
minutes' returns. A correct pipeline should find it; a pipeline that finds a
much stronger signal is probably leaking future data. Real markets are not
this simple, so never draw conclusions about strategies from fixture results.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from shpe_research.contracts.bars import validate_bars
from shpe_research.contracts.base import UTC_NS
from shpe_research.data.calendar import get_calendar

DEFAULT_SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


def synthetic_bars(
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    start: str = "2026-03-02",
    end: str = "2026-08-28",
    seed: int = 7,
    signal_strength: float = 0.3,
) -> pd.DataFrame:
    """Contract-valid 1-minute regular-session bars for `symbols` over [start, end] sessions."""
    cal = get_calendar()
    schedule = cal.schedule(start, end)
    if schedule.empty:
        raise ValueError(f"no sessions between {start} and {end}")
    minute_starts = [
        pd.date_range(row.open, row.close, freq="1min", inclusive="left") for row in schedule.itertuples()
    ]
    ts = minute_starts[0].append(minute_starts[1:])
    session_lengths = np.array([len(m) for m in minute_starts])
    first_of_session = np.zeros(len(ts), dtype=bool)
    first_of_session[np.r_[0, np.cumsum(session_lengths)[:-1]]] = True
    minute_index = np.concatenate([np.arange(n) for n in session_lengths])
    u_shape = 1 + 2.5 * ((minute_index - 195) / 195) ** 2

    rng = np.random.default_rng(seed)
    frames = []
    n = len(ts)
    for symbol in symbols:
        sigma = rng.uniform(0.0006, 0.0015)
        # r_t = -(k/5) * (r_{t-1} + ... + r_{t-5}) + noise
        a = np.r_[1.0, np.full(5, signal_strength / 5)]
        returns = lfilter([1.0], a, rng.normal(0, sigma, n))
        gaps = np.where(first_of_session, rng.normal(0, 0.006, n), 0.0)
        close = rng.uniform(20, 300) * np.exp(np.cumsum(returns + gaps))
        open_ = np.r_[close[0] * np.exp(-returns[0]), close[:-1]] * np.exp(gaps)
        wick = np.abs(rng.normal(0, sigma / 2, (2, n)))
        high = np.maximum(open_, close) * np.exp(wick[0])
        low = np.minimum(open_, close) * np.exp(-wick[1])
        volume = (rng.uniform(2_000, 40_000) * u_shape * rng.lognormal(0, 0.5, n)).astype("int64") + 1
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "ts": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "vwap": (open_ + high + low + close) / 4,
                    "trade_count": pd.array(np.maximum(1, volume // 50), dtype="Int64"),
                }
            )
        )
    bars =pd.concat(frames, ignore_index=True)
    bars["ts"] = bars["ts"].astype(UTC_NS)
    bars.insert(2, "ts_available", bars["ts"] + pd.Timedelta(minutes=1))
    bars["session"] = "regular"
    bars = bars.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)
    validate_bars(bars)
    return bars
