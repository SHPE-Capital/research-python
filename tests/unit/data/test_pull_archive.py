# ruff: noqa: E501
"""A1: staged archive pull (mocked HTTP; no network)."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest

from shpe_research.data.alpaca import AlpacaDataClient, Credentials
from shpe_research.data.calendar import get_calendar
from shpe_research.data.ratelimit import RateLimiter

SCRIPT = Path(__file__).parents[3] / "scripts" / "momentum" / "pull_archive.py"
_spec = importlib.util.spec_from_file_location("pull_archive", SCRIPT)
pa = importlib.util.module_from_spec(_spec)
sys.modules["pull_archive"] = pa
_spec.loader.exec_module(pa)

NOW = pd.Timestamp("2026-11-13T12:00:00Z")
NY = "America/New_York"


class FakeAlpaca:
    """Serves regular-session minute bars; `dead_after` maps a symbol to its last live date."""

    def __init__(self, dead_after=None, fail_symbols=(), throttle_first=0, deny_feeds=()):
        self.dead_after = dead_after or {}
        self.fail_symbols = set(fail_symbols)
        self.throttle = throttle_first
        self.deny = set(deny_feeds)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        q = request.url.params
        symbol = request.url.path.split("/")[-2]
        if self.throttle > 0:
            self.throttle -= 1
            return httpx.Response(429, headers={"retry-after": "1"}, text="slow down")
        if q["feed"] in self.deny:
            return httpx.Response(403, text="forbidden")
        if symbol in self.fail_symbols:
            return httpx.Response(422, text="bad symbol")
        lo, hi = pd.Timestamp(q["start"]), pd.Timestamp(q["end"])
        cal = get_calendar()
        minutes = []
        for day in cal.sessions(lo.tz_convert(NY).tz_localize(None), hi.tz_convert(NY).tz_localize(None)):
            if symbol in self.dead_after and day > pd.Timestamp(self.dead_after[symbol]):
                continue
            minutes.extend(cal.regular_minutes(day))
        minutes = [m for m in minutes if lo <= m <= hi]
        if q.get("sort") == "desc":
            minutes.reverse()
        limit = int(q["limit"])
        offset = int(q.get("page_token", 0))
        bars = [
            {"t": m.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5, "v": 100, "vw": 10.2, "n": 5}
            for m in minutes[offset : offset + limit]
        ]
        nxt = str(offset + limit) if offset + limit < len(minutes) else None
        return httpx.Response(200, json={"symbol": symbol, "bars": bars, "next_page_token": nxt})


def run(root, fake, symbols, start="2024-03-04", end="2024-03-08", sleeps=None, **kw):
    transport = pa.CountingTransport(httpx.MockTransport(fake))
    client = AlpacaDataClient(
        Credentials("test-key", "test-secret"), limiter=RateLimiter(100_000), transport=transport,
        page_limit=5_000, sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )
    with client:
        return pa.run_stage(
            "pilot", symbols, start, end, "sip", root, client, transport, now=NOW, workers=2, check_disk=False, **kw
        )


def test_pull_records_rows_coverage_checksums(tmp_path):
    summary = run(tmp_path, FakeAlpaca(), ["AAA", "SPY"])
    assert summary["units_fetched"] == 2 and summary["units_failed"] == 0
    db = sqlite3.connect(tmp_path / pa.MANIFEST_DB)
    rows = db.execute(
        "SELECT rows, regular_rows, expected_regular, coverage, sha256, complete, bytes FROM units"
    ).fetchall()
    db.close()
    assert len(rows) == 2
    for n, regular, expected, coverage, sha, complete, size in rows:
        assert n == regular == expected == 5 * 390
        assert coverage == 1.0 and len(sha) == 64 and complete == 1 and size > 0
    assert summary["bars_per_ok_request"] > 0
    assert summary["entitlement"]["sip"]["status"] == 200
    assert (tmp_path / "reports" / "pilot_summary.json").exists()
    manifest = pa.ArchiveManifest(tmp_path / pa.MANIFEST_DB)
    assert manifest.year_coverage()[0]["year"] == 2024
    manifest.close()


def test_resume_skips_complete_units_and_refetches_missing_file(tmp_path):
    run(tmp_path, FakeAlpaca(), ["AAA"])
    second = run(tmp_path, FakeAlpaca(), ["AAA"])
    assert second["units_fetched"] == 0 and second["units_skipped"] == 1
    next(tmp_path.glob("bars/**/*.parquet")).unlink()
    third = run(tmp_path, FakeAlpaca(), ["AAA"])
    assert third["units_fetched"] == 1


def test_failed_unit_is_reported_and_retried_later(tmp_path):
    summary = run(tmp_path, FakeAlpaca(fail_symbols={"BAD"}), ["AAA", "BAD"])
    assert summary["units_failed"] == 1 and summary["units_fetched"] == 1
    again = run(tmp_path, FakeAlpaca(), ["AAA", "BAD"])
    assert again["units_failed"] == 0 and again["units_fetched"] == 1 and again["units_skipped"] == 1


def test_429_is_counted_with_retry_after(tmp_path):
    sleeps: list[float] = []
    summary = run(tmp_path, FakeAlpaca(throttle_first=2), ["AAA"], sleeps=sleeps)
    assert summary["network"]["n_429"] == 2
    assert summary["network"]["retry_after_seen"] == ["1"]
    assert summary["units_failed"] == 0 and 1.0 in sleeps


def test_sip_not_entitled_stops_before_pulling(tmp_path):
    with pytest.raises(SystemExit, match="SIP not available"):
        run(tmp_path, FakeAlpaca(deny_feeds={"sip"}), ["AAA"])
    assert not list(tmp_path.glob("bars/**/*.parquet"))


def test_delisted_symbol_history_probe_and_partial_coverage(tmp_path):
    summary = run(tmp_path, FakeAlpaca(dead_after={"DEAD": "2024-03-05"}), ["DEAD"], probe_symbols=["DEAD"])
    probe = summary["delisted_history_probe"]["DEAD"]
    assert probe["status"] == 200
    assert probe["first_ts"].startswith("2016-01-04")  # the fake serves history back to the probe floor
    assert probe["last_ts"].startswith("2024-03-05")
    manifest = pa.ArchiveManifest(tmp_path / pa.MANIFEST_DB)
    cov = manifest.year_coverage()[0]
    manifest.close()
    assert cov["rows"] == 2 * 390 and cov["coverage"] == pytest.approx(2 / 5, abs=1e-3)


def test_disk_guard_refuses_over_budget(tmp_path):
    with pytest.raises(SystemExit, match="exceeds"):
        pa.disk_guard(tmp_path, 10.5)


def test_pilot_pool_has_spy_liquid_and_delisted():
    pool = pa.pilot_pool()
    assert "SPY" in pool and set(pa.DELISTED) <= set(pool) and 70 <= len(pool) <= 90


def test_larger_stage_requires_symbols_file(tmp_path):
    with pytest.raises(SystemExit, match="symbols-file"):
        pa.main(["--stage", "300x2", "--root", str(tmp_path)])


def test_window_clipped_to_sip_delay():
    _, end = pa.wanted_window("2026-11", pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-30"), NOW, "sip")
    assert end == NOW - pd.Timedelta(minutes=16)
