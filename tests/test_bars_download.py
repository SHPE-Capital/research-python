from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shpe_research.cli import main
from shpe_research.contracts.bars import validate_bars
from shpe_research.data import bars as bars_module
from shpe_research.data.alpaca import AlpacaError, Credentials
from shpe_research.data.bars import download_bars
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.ratelimit import RateLimiter
from shpe_research.data.store import BarUnit, DataStore, StoreError
from shpe_research.experiments.sources import load_bars

from .fake_alpaca import FakeAlpaca, raw_bars, raw_daily

LATER = pd.Timestamp("2026-10-28T12:00Z")


@pytest.fixture(scope="module")
def october():
    return synthetic_bars(symbols=["SPY", "KO"], start="2025-10-01", end="2025-10-31")


@pytest.fixture
def fake(october):
    fake = FakeAlpaca()
    for symbol, group in october.groupby("symbol"):
        fake.add("bars", symbol, raw_bars(group), timeframe="1Min")
        fake.add("bars", symbol, raw_daily(group), timeframe="1Day")
    return fake


def test_download_paginates_stores_and_resumes(fake, october, tmp_path: Path):
    store = DataStore(tmp_path)
    summary = download_bars(["spy", "KO"], "2025-10-06", "2025-10-07", store=store, client=fake.client(), now=LATER)
    assert summary.fetched == 2
    assert not summary.failed
    assert summary.rows == len(october)
    assert summary.requests > 2  # pagination was exercised
    entry = store.bars_manifest.get(BarUnit("sip", "1Min", "raw", "SPY", "2025-10").key)
    assert entry["complete"]
    assert entry["rows"] == len(october) // 2

    got = store.read_bars(["KO", "SPY"], "2025-10-06", "2025-10-07")
    validate_bars(got)
    expected = october[october["ts"].dt.tz_convert("America/New_York").dt.day.isin([6, 7])]
    assert len(got) == len(expected) == 4 * 390
    order = ["symbol", "ts"]
    np.testing.assert_allclose(got.sort_values(order)["close"], expected.sort_values(order)["close"])
    assert set(got["session"]) == {"regular"}

    sent = len(fake.requests)
    again = download_bars(["SPY", "KO"], "2025-10-06", "2025-10-07", store=store, client=fake.client(), now=LATER)
    assert again.skipped == 2
    assert again.fetched == 0
    assert len(fake.requests) == sent

    via_config = load_bars({"source": "store", "root": str(tmp_path), "symbols": ["SPY"], "start": "2025-10-06",
                            "end": "2025-10-06"})
    assert len(via_config) == 390


def test_period_in_progress_is_refetched_and_reads_respect_coverage(fake, tmp_path: Path):
    store = DataStore(tmp_path)
    now = pd.Timestamp("2025-10-15T15:00Z")
    download_bars(["SPY"], "2025-10-01", "2025-10-31", store=store, client=fake.client(), now=now)
    entry = store.bars_manifest.get(BarUnit("sip", "1Min", "raw", "SPY", "2025-10").key)
    assert not entry["complete"]
    assert pd.Timestamp(entry["end"]) == now - pd.Timedelta(minutes=16)

    assert len(store.read_bars(["SPY"], "2025-10-01", "2025-10-14")) == 10 * 390
    with pytest.raises(StoreError, match="shpe data bars"):
        store.read_bars(["SPY"], "2025-10-01", "2025-10-20")

    sent = len(fake.requests)
    later = download_bars(["SPY"], "2025-10-01", "2025-10-31", store=store, client=fake.client(), now=LATER)
    assert later.fetched == 1
    assert len(fake.requests) > sent
    assert store.bars_manifest.get(BarUnit("sip", "1Min", "raw", "SPY", "2025-10").key)["complete"]


def test_client_trims_alpacas_inclusive_end(fake, october):
    spy = october[october["symbol"] == "SPY"]
    start, end = spy["ts"].iloc[0], spy["ts"].iloc[10]
    records = fake.client().bars("SPY", start, end)
    assert len(records) == 10
    assert pd.Timestamp(records[-1]["t"]) < end


def test_retries_then_succeeds_and_explains_403(fake, october):
    spy = october[october["symbol"] == "SPY"]
    fake.script.extend([429, 503])
    assert len(fake.client().bars("SPY", spy["ts"].iloc[0], spy["ts"].iloc[5])) == 5
    fake.script.extend([403])
    with pytest.raises(AlpacaError, match="15 minutes"):
        fake.client().bars("SPY", spy["ts"].iloc[0], spy["ts"].iloc[5])


def test_one_failing_symbol_does_not_stop_the_rest(fake, tmp_path: Path):
    fake.script.extend([422])
    summary = download_bars(["SPY", "KO"], "2025-10-06", "2025-10-06", store=DataStore(tmp_path), workers=1,
                            client=fake.client(), now=LATER)
    assert summary.fetched == 1
    assert len(summary.failed) == 1


def test_unknown_symbol_stores_an_empty_complete_unit(fake, tmp_path: Path):
    store = DataStore(tmp_path)
    summary = download_bars(["NOPE"], "2025-10-06", "2025-10-06", store=store, client=fake.client(), now=LATER)
    assert summary.fetched == 1
    assert summary.rows == 0
    assert store.bar_unit_complete(BarUnit("sip", "1Min", "raw", "NOPE", "2025-10"))
    assert store.read_bars(["NOPE"], "2025-10-06", "2025-10-06").empty


def test_daily_bars_are_available_at_the_session_close(fake, tmp_path: Path):
    store = DataStore(tmp_path)
    download_bars(["SPY"], "2025-10-01", "2025-10-31", timeframe="1Day", adjustment="split", store=store,
                  client=fake.client(), now=LATER)
    assert store.bars_manifest.get(BarUnit("sip", "1Day", "split", "SPY", "2025").key)["complete"]
    daily = store.read_bars(["SPY"], "2025-10-01", "2025-10-31", timeframe="1Day", adjustment="split")
    assert len(daily) == 23
    first = daily.iloc[0]
    assert first["ts"] == pd.Timestamp("2025-10-01T04:00Z")
    assert first["ts_available"] == pd.Timestamp("2025-10-01T20:00Z")


def test_verify_detects_tampering(fake, tmp_path: Path):
    store = DataStore(tmp_path)
    download_bars(["SPY"], "2025-10-06", "2025-10-06", store=store, client=fake.client(), now=LATER)
    assert DataStore(tmp_path).verify() == []
    path = tmp_path / BarUnit("sip", "1Min", "raw", "SPY", "2025-10").relpath
    path.write_bytes(path.read_bytes() + b"x")
    assert "checksum mismatch" in DataStore(tmp_path).verify()[0]


def test_missing_keys_explain_what_to_do(monkeypatch):
    monkeypatch.delenv("ALPACA_DATA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_DATA_SECRET", raising=False)
    with pytest.raises(AlpacaError, match=".env.example"):
        Credentials.from_env()


def test_rate_limiter_spaces_requests():
    clock = {"t": 0.0}
    waits = []
    limiter = RateLimiter(120, clock=lambda: clock["t"], sleep=waits.append)
    for _ in range(3):
        limiter.acquire()
    assert waits == [pytest.approx(0.5), pytest.approx(1.0)]


def test_cli_bars(fake, tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(bars_module, "AlpacaDataClient", lambda: fake.client())
    argv = ["data", "bars", "--symbols", "SPY", "--from", "2025-10-06", "--to", "2025-10-06", "--workers", "2"]
    assert main(argv) == 0
    assert "1 fetched" in capsys.readouterr().out
    assert main(["data", "verify"]) == 0
