"""C1e: minute bars resample to one daily bar per symbol and regular session."""

from __future__ import annotations

import pandas as pd
import pytest

from shpe_research.contracts.bars import validate_bars
from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.resample import resample_daily


def _bars(rows: list[tuple]) -> pd.DataFrame:
    """rows: (symbol, ts, open, high, low, close, volume, vwap, trade_count, session); ts_available = ts + 1 min."""
    df = pd.DataFrame(
        rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count", "session"]
    )
    df["ts"] = to_utc_ns(pd.to_datetime(df["ts"], utc=True))
    df.insert(2, "ts_available", df["ts"] + pd.Timedelta(minutes=1))
    df["volume"] = df["volume"].astype("int64")
    df["trade_count"] = pd.array(df["trade_count"], dtype="Int64")
    return df


def _two_sessions() -> pd.DataFrame:
    # 2026-10-01 is EDT (09:30 = 13:30Z); 2026-11-02 is EST (09:30 = 14:30Z).
    return _bars(
        [
            ("AAA", "2026-10-01T13:00:00Z", 9.0, 9.5, 8.5, 9.2, 50, 9.1, 5, "pre"),
            ("AAA", "2026-10-01T13:30:00Z", 10.0, 11.0, 9.0, 10.5, 100, 10.0, 1, "regular"),
            ("AAA", "2026-10-01T13:31:00Z", 11.0, 12.0, 10.0, 11.5, 200, 11.0, 2, "regular"),
            ("AAA", "2026-10-01T13:32:00Z", 12.0, 13.0, 11.0, 12.5, 300, 12.0, 3, "regular"),
            ("AAA", "2026-10-01T20:00:00Z", 13.0, 13.0, 13.0, 13.0, 10, 13.0, 1, "post"),
            ("AAA", "2026-11-02T14:30:00Z", 20.0, 22.0, 19.0, 21.0, 10, 20.5, 4, "regular"),
            ("AAA", "2026-11-02T14:31:00Z", 21.0, 21.5, 18.0, 18.5, 30, 19.5, 6, "regular"),
        ]
    )


def test_daily_ohlc_hand():
    daily = resample_daily(_two_sessions())
    validate_bars(daily)
    assert len(daily) == 2
    first, second = daily.iloc[0], daily.iloc[1]
    assert (first["open"], first["high"], first["low"], first["close"]) == (10.0, 13.0, 9.0, 12.5)
    assert first["volume"] == 600
    assert first["vwap"] == pytest.approx((10 * 100 + 11 * 200 + 12 * 300) / 600)
    assert first["trade_count"] == 6
    assert first["session"] == "regular"
    assert (second["open"], second["high"], second["low"], second["close"]) == (20.0, 22.0, 18.0, 18.5)
    assert second["volume"] == 40
    assert second["vwap"] == pytest.approx((20.5 * 10 + 19.5 * 30) / 40)


def test_ts_and_ts_available():
    daily = resample_daily(_two_sessions())
    assert list(daily["ts"]) == [pd.Timestamp("2026-10-01T13:30:00Z"), pd.Timestamp("2026-11-02T14:30:00Z")]
    # ts_available is the last regular bar's ts_available, not the post-market bar's or the calendar close
    assert list(daily["ts_available"]) == [pd.Timestamp("2026-10-01T13:33:00Z"), pd.Timestamp("2026-11-02T14:32:00Z")]


def test_full_and_early_close_sessions():
    bars = synthetic_bars(symbols=["AAA", "BBB"], start="2026-11-25", end="2026-11-27")
    daily = resample_daily(bars)
    validate_bars(daily)
    assert daily.groupby("symbol").size().tolist() == [2, 2]  # Thanksgiving has no session
    aaa = daily[daily["symbol"] == "AAA"]
    assert list(aaa["ts"]) == [pd.Timestamp("2026-11-25T14:30:00Z"), pd.Timestamp("2026-11-27T14:30:00Z")]
    assert list(aaa["ts_available"]) == [pd.Timestamp("2026-11-25T21:00:00Z"), pd.Timestamp("2026-11-27T18:00:00Z")]
    one = bars[(bars["symbol"] == "AAA") & (bars["ts"] < pd.Timestamp("2026-11-26T00:00:00Z"))]
    assert aaa.iloc[0]["close"] == one["close"].iloc[-1]
    assert aaa.iloc[0]["volume"] == one["volume"].sum()


def test_sessions_without_regular_bars_produce_no_row():
    only_pre = _bars([("AAA", "2026-10-01T13:00:00Z", 9.0, 9.5, 8.5, 9.2, 50, 9.1, 5, "pre")])
    out = resample_daily(only_pre)
    assert out.empty
    validate_bars(out)


def test_nan_vwap_when_no_bar_has_one_and_symbols_stay_separate():
    rows = [
        ("AAA", "2026-10-01T13:30:00Z", 1.0, 1.0, 1.0, 1.0, 10, float("nan"), 1, "regular"),
        ("BBB", "2026-10-01T13:30:00Z", 2.0, 2.0, 2.0, 2.0, 20, 2.0, 1, "regular"),
    ]
    daily = resample_daily(_bars(rows))
    assert daily["symbol"].tolist() == ["AAA", "BBB"]
    assert pd.isna(daily.iloc[0]["vwap"]) and daily.iloc[1]["vwap"] == 2.0


def test_rejects_daily_input():
    daily = resample_daily(_two_sessions())
    daily["ts_available"] = daily["ts"] + pd.Timedelta(hours=6, minutes=30)  # a true daily span
    with pytest.raises(ValueError, match="minute"):
        resample_daily(daily)
