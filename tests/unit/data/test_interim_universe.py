# ruff: noqa: E501
"""Interim A1 symbol list: asset filter, D4 selection and symbols file (mocked HTTP, synthetic store)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import yaml

from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.calendar import get_calendar
from shpe_research.data.store import BarUnit, DataStore, period_window


def _load(name: str):
    path = Path(__file__).parents[3] / "scripts" / "momentum" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


iu = _load("build_interim_universe")
pa = _load("pull_archive")

AS_OF = "2025-06-02"
WINDOW = 15


def asset(symbol, name, exchange="NASDAQ", status="active", tradable=True, cls="us_equity"):
    return {"symbol": symbol, "name": name, "exchange": exchange, "status": status, "tradable": tradable, "class": cls}


def test_filter_common_stock_keeps_listed_common_and_drops_products():
    assets = [
        asset("AAPL", "Apple Inc. Common Stock"),
        asset("JPM", "JPMorgan Chase & Co.", exchange="NYSE"),
        asset("NTRS", "Northern Trust Corporation Common Stock"),  # "Trust" outside ARCA/BATS is fine
        asset("SPLK", "Splunk Inc.", status="inactive", tradable=False),
        asset("SPY", "SPDR S&P 500 ETF Trust", exchange="ARCA"),
        asset("GLD", "SPDR Gold Trust", exchange="ARCA"),
        asset("QQQ", "Invesco QQQ Trust, Series 1"),
        asset("OTCX", "Some Pink Sheet Co", exchange="OTC"),
        asset("ABCDW", "ABCD Corp Warrants"),
        asset("BRK.B", "Berkshire Hathaway Inc. Class B", exchange="NYSE"),
        asset("BACPK", "Bank of America 6% Preferred", exchange="NYSE"),
        asset("SPAC", "Blank Check Acquisition Corp Class A"),
        asset("HALT", "Halted Co", tradable=False),
        asset("CRYP", "Coin", cls="crypto"),
        asset("DUP", "Old Dup Co", status="inactive", tradable=False),
        asset("DUP", "New Dup Co"),
    ]
    df = iu.filter_common_stock(assets)
    assert df["symbol"].tolist() == ["AAPL", "DUP", "JPM", "NTRS", "SPLK"]
    assert df.set_index("symbol").loc["DUP", "status"] == "active"
    assert set(df["sector"]) == {"unknown"}


def test_fetch_assets_falls_back_to_paper_on_401(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_KEY", "k")
    monkeypatch.setenv("ALPACA_DATA_SECRET", "s")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        assert request.url.path == "/v2/assets"
        assert request.url.params["status"] == "all"
        if request.url.host == "api.alpaca.markets":
            return httpx.Response(401, json={"message": "unauthorized"})
        return httpx.Response(200, json=[asset("AAPL", "Apple Inc.")])

    assert iu.fetch_assets_any(transport=httpx.MockTransport(handler)) == [asset("AAPL", "Apple Inc.")]
    assert seen == ["api.alpaca.markets", "paper-api.alpaca.markets"]


def _daily(symbol: str, sessions: pd.DatetimeIndex, price: float, dollar_volume: float, vol: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = to_utc_ns(pd.Series(sessions.tz_localize("America/New_York")))
    close = price * np.exp(np.cumsum(rng.normal(0, vol, len(sessions))))
    return pd.DataFrame({
        "symbol": symbol, "ts": ts, "ts_available": get_calendar().session_close_for(ts),
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": (dollar_volume / close).astype("int64"), "vwap": close,
        "trade_count": pd.array([1000] * len(close), dtype="Int64"), "session": "regular",
    })


@pytest.fixture
def setup(tmp_path: Path):
    store = DataStore(tmp_path / "data")
    sessions = get_calendar().previous_sessions(AS_OF, WINDOW)
    rng = np.random.default_rng(0)
    assets = []
    for i in range(40):
        symbol = f"S{chr(65 + i // 26)}{chr(65 + i % 26)}"
        assets.append(asset(symbol, f"{symbol} Inc."))
        bars = _daily(symbol, sessions, rng.uniform(20, 200), rng.uniform(15e6, 300e6), rng.uniform(0.005, 0.04), seed=i)
        store.write_bar_unit(BarUnit("sip", "1Day", "split", symbol, "2025"), bars, *period_window("2025"), complete=True)
    assets.append(asset("THIN", "Thin Co"))
    store.write_bar_unit(BarUnit("sip", "1Day", "split", "THIN", "2025"), _daily("THIN", sessions, 50, 1e6, 0.02, 99),
                         *period_window("2025"), complete=True)
    assets.append(asset("GONE", "Gone Co", status="inactive", tradable=False))  # no bars: excluded, not an error
    assets.append(asset("IVV", "iShares Core S&P 500 ETF", exchange="ARCA"))
    store.bars_manifest.flush()

    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({
        "name": "t_interim", "candidates": "placeholder.csv", "as_of": AS_OF, "window_sessions": WINDOW, "feed": "sip",
        "screens": {"min_median_close": 5, "min_median_dollar_volume": 10_000_000, "min_morning_coverage": None},
        "cells": {"liquidity_bins": 3, "volatility_bins": 3, "per_cell": 3, "held_out_per_cell": 0},
        "sector_cap": None, "seed": 7,
    }))
    assets_json = tmp_path / "assets.json"
    assets_json.write_text(json.dumps(assets))
    return store, config, assets, assets_json, tmp_path / "out"


def _symbols(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text().splitlines()
    return [ln for ln in lines if ln.startswith("#")], [ln for ln in lines if not ln.startswith("#")]


def test_build_writes_symbols_file_matching_manifest(setup):
    store, config, assets, assets_json, out = setup
    result, path = iu.build(config, out, assets, assets_json, store=store)

    header, symbols = _symbols(path)
    assert header and any("sha256" in h for h in header)
    assert symbols == sorted(symbols)
    assert "SPY" in symbols
    selected = set(result.selected["symbol"])
    assert len(selected) == 27  # 9 cells x 3
    assert set(symbols) == selected | {"SPY"}
    assert "THIN" not in symbols and "GONE" not in symbols and "IVV" not in symbols
    manifest = yaml.safe_load((out / "t_interim.yaml").read_text())
    assert set(manifest["dev"]) == selected and manifest["held_out"] == []
    assert (out / "candidates_interim.csv").exists()

    # pull_archive reads it, skipping the header
    assert pa.read_symbols_file(path) == symbols


def test_build_is_reproducible(setup):
    store, config, assets, assets_json, out = setup
    _, first = iu.build(config, out / "a", assets, assets_json, store=store)
    _, second = iu.build(config, out / "b", assets, assets_json, store=store)
    strip_git = lambda p: [ln for ln in p.read_text().splitlines() if not ln.startswith("# git")]  # noqa: E731
    assert strip_git(first) == strip_git(second)


def test_read_symbols_file_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("# header\n\naapl\n  MSFT  \n# another\nAAPL\n")
    assert pa.read_symbols_file(f) == ["AAPL", "MSFT"]
