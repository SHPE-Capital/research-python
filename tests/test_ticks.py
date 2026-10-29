from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shpe_research.cli import main
from shpe_research.contracts.base import to_utc_ns
from shpe_research.contracts.ticks import TICK_SUMMARY_COLUMNS
from shpe_research.data import ticks as ticks_module
from shpe_research.data.store import DataStore, TickUnit
from shpe_research.data.ticks import download_ticks, minute_summaries, normalize_quotes, sign_trades

from .fake_alpaca import FakeAlpaca, raw_ticks

DAY = "2026-10-27"
LATER = pd.Timestamp("2026-10-28T12:00Z")


@pytest.fixture
def fake():
    fake = FakeAlpaca()
    quotes, trades = raw_ticks("2026-10-27T13:30:00Z", minutes=30)
    fake.add("quotes", "KO", quotes)
    fake.add("trades", "KO", trades)
    return fake


def test_normalize_quotes_drops_invalid_and_keeps_nanoseconds():
    raw = [
        {"t": "2026-10-27T13:30:00.123456789Z", "bp": 10.0, "ap": 10.02, "bs": 3, "as": 4, "c": ["R"]},
        {"t": "2026-10-27T13:30:01Z", "bp": 0.0, "ap": 10.02, "bs": 0, "as": 4},
        {"t": "2026-10-27T13:30:02Z", "bp": 10.05, "ap": 10.02, "bs": 1, "as": 1},
    ]
    quotes, dropped = normalize_quotes(raw, "KO")
    assert dropped == 2
    assert len(quotes) == 1
    assert quotes["ts"].iloc[0].nanosecond == 789
    assert quotes["conditions"].iloc[0] == ["R"]


def test_download_ticks_windows_resume_and_widen(fake, tmp_path: Path):
    store = DataStore(tmp_path)
    kwargs = {"store": store, "now": LATER, "start_time": "09:30", "end_time": "09:40"}
    summary = download_ticks(["KO"], DAY, DAY, client=fake.client(page_limit=200), **kwargs)
    assert summary.fetched == 2
    assert summary.dropped > 0
    assert not summary.failed
    quotes = store.read_ticks("quotes", ["KO"], DAY, DAY)
    assert quotes["ts"].min() >= pd.Timestamp("2026-10-27T13:30Z")
    assert quotes["ts"].max() < pd.Timestamp("2026-10-27T13:40Z")
    entry = store.ticks_manifest.get(TickUnit("sip", "quotes", "KO", DAY).key)
    assert not entry["complete"]  # a narrowed window is not the whole session

    sent = len(fake.requests)
    again = download_ticks(["KO"], DAY, DAY, client=fake.client(), **kwargs)
    assert again.skipped == 2
    assert len(fake.requests) == sent

    wider = download_ticks(["KO"], DAY, DAY, client=fake.client(), **{**kwargs, "end_time": "09:50"})
    assert wider.fetched == 2
    assert store.read_ticks("trades", ["KO"], DAY, DAY)["ts"].max() >= pd.Timestamp("2026-10-27T13:40Z")
    assert store.verify() == []


def test_lee_ready_signs_against_the_prior_quote():
    ts = to_utc_ns(pd.Series(["2026-10-27T13:30:00Z", "2026-10-27T13:30:05Z"]))
    quotes = pd.DataFrame({"symbol": "KO", "ts": ts, "bid_price": [9.99, 10.09], "ask_price": [10.01, 10.11],
                           "bid_size": 1, "ask_size": 1})
    trade_ts = to_utc_ns(pd.Series(
        ["2026-10-27T13:30:01Z", "2026-10-27T13:30:02Z", "2026-10-27T13:30:03Z", "2026-10-27T13:30:04Z",
         "2026-10-27T13:30:05Z"]
    ))
    trades = pd.DataFrame({"symbol": "KO", "ts": trade_ts, "price": [10.01, 9.99, 10.00, 10.00, 10.05], "size": 100})
    signs = sign_trades(trades, quotes).tolist()
    # above mid, below mid, at mid after a rise (uptick), at mid with no change (last non-zero change was up),
    # and a trade stamped with a new quote is judged against the quote before it (mid 10.00) -> buy
    assert signs == [1, -1, 1, 1, 1]


def test_minute_summaries():
    q_ts = to_utc_ns(pd.Series(["2026-10-27T13:30:10Z", "2026-10-27T13:30:40Z", "2026-10-27T13:31:10Z"]))
    quotes = pd.DataFrame({"symbol": "KO", "ts": q_ts, "bid_price": [9.99, 9.99, 10.99],
                           "ask_price": [10.01, 10.01, 11.01], "bid_size": [3, 1, 2], "ask_size": [1, 3, 2]})
    t_ts = to_utc_ns(pd.Series(["2026-10-27T13:30:20Z", "2026-10-27T13:30:50Z"]))
    trades = pd.DataFrame({"symbol": "KO", "ts": t_ts, "price": [10.01, 9.99], "size": [100, 40]})
    out = minute_summaries(quotes, trades)
    assert list(out.columns) == list(TICK_SUMMARY_COLUMNS)
    first, second = out.iloc[0], out.iloc[1]
    assert first["quote_count"] == 2
    assert first["spread_bps_mean"] == pytest.approx(20.0)
    assert first["imbalance_mean"] == pytest.approx(0.0)
    assert first["buy_volume"] == 100
    assert first["sell_volume"] == 40
    assert first["ts_available"] == pd.Timestamp("2026-10-27T13:31Z")
    assert second["trade_count"] == 0
    assert second["mid_close"] == pytest.approx(11.0)


def test_cli_ticks_and_summary(fake, tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ticks_module, "AlpacaDataClient", lambda: fake.client())
    base = ["--symbols", "KO", "--from", DAY, "--to", DAY]
    assert main(["data", "ticks", *base, "--start-time", "09:30", "--end-time", "09:35"]) == 0
    out = tmp_path / "summary.parquet"
    assert main(["data", "tick-summary", *base, "--out", str(out)]) == 0
    assert len(pd.read_parquet(out)) == 5
    assert "minute rows" in capsys.readouterr().out
