from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from shpe_research.cli import main
from shpe_research.contracts.base import to_utc_ns
from shpe_research.data.calendar import get_calendar
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.store import BarUnit, DataStore, period_window
from shpe_research.data.universe import UniverseConfig, build_universe, required_downloads

AS_OF = "2026-06-01"
SECTORS = ("Tech", "Health", "Energy")


def make_daily(symbol: str, sessions: pd.DatetimeIndex, price: float, dollar_volume: float, vol: float, seed: int):
    rng = np.random.default_rng(seed)
    cal = get_calendar()
    ts = to_utc_ns(pd.Series(sessions.tz_localize("America/New_York")))
    close = price * np.exp(np.cumsum(rng.normal(0, vol, len(sessions))))
    return pd.DataFrame(
        {
            "symbol": symbol, "ts": ts, "ts_available": cal.session_close_for(ts),
            "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
            "volume": (dollar_volume / close).astype("int64"), "vwap": close,
            "trade_count": pd.array([1000] * len(close), dtype="Int64"), "session": "regular",
        }
    )


def write_daily(store: DataStore, bars: pd.DataFrame) -> None:
    unit = BarUnit("sip", "1Day", "split", bars["symbol"].iloc[0], "2026")
    store.write_bar_unit(unit, bars, *period_window("2026"), complete=True)


@pytest.fixture
def universe_store(tmp_path: Path):
    """30 eligible names across 3 sectors, plus one too cheap, one too thin, one with short history, one absent."""
    store = DataStore(tmp_path / "data")
    sessions = get_calendar().previous_sessions(AS_OF, 15)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        symbol = f"S{i:02d}"
        rows.append((symbol, SECTORS[i % 3]))
        write_daily(store, make_daily(symbol, sessions, rng.uniform(20, 200), rng.uniform(30e6, 300e6),
                                      rng.uniform(0.005, 0.04), seed=i))
    write_daily(store, make_daily("CHEAP", sessions, 5, 100e6, 0.02, seed=100))
    write_daily(store, make_daily("THIN", sessions, 50, 1e6, 0.02, seed=101))
    write_daily(store, make_daily("SHORT", sessions[-5:], 50, 100e6, 0.02, seed=102))
    rows += [("CHEAP", "Tech"), ("THIN", "Tech"), ("SHORT", "Tech"), ("GHOST", "Tech")]
    store.bars_manifest.flush()
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(rows, columns=["symbol", "sector"]).to_csv(candidates, index=False)
    return store, candidates


def _config(candidates: Path, **overrides) -> UniverseConfig:
    base = {
        "name": "test", "candidates": candidates, "as_of": AS_OF, "window_sessions": 10,
        "screens": {"min_median_close": 10, "min_median_dollar_volume": 25e6, "min_morning_coverage": None},
        "cells": {"liquidity_bins": 2, "volatility_bins": 3, "per_cell": 3, "held_out_per_cell": 1},
        "sector_cap": 7, "seed": 1,
    }
    return UniverseConfig(**{**base, **overrides})


def test_screens_record_every_exclusion(universe_store):
    store, candidates = universe_store
    table = build_universe(_config(candidates), store).table.set_index("symbol")
    assert "median close below" in table.loc["CHEAP", "exclusion"]
    assert "dollar volume below" in table.loc["THIN", "exclusion"]
    assert "incomplete daily history" in table.loc["SHORT", "exclusion"]
    assert "no split-adjusted daily bars" in table.loc["GHOST", "exclusion"]
    assert (table.loc[["CHEAP", "THIN", "SHORT", "GHOST"], "status"] == "excluded").all()


def test_cells_sampling_and_dev_held_out_split(universe_store):
    store, candidates = universe_store
    result = build_universe(_config(candidates), store)
    manifest = result.manifest
    assert len(manifest["cells"]) == 6
    for cell in manifest["cells"]:
        assert cell["eligible"] == 5
        assert cell["selected"] == 3
        assert len(cell["held_out"]) == 1
    assert manifest["counts"]["selected"] == 18
    assert manifest["counts"]["held_out"] == 6
    assert set(manifest["dev"]).isdisjoint(manifest["held_out"])


def test_sector_cap_limits_selection_and_reports_shortfall(universe_store):
    store, candidates = universe_store
    result = build_universe(_config(candidates, sector_cap=5), store)
    selected = result.selected
    assert selected.groupby("sector").size().max() <= 5
    assert (result.table["status"] == "sector_capped").any()
    shortfall = sum(c["shortfall"] for c in result.manifest["cells"])
    assert shortfall == 18 - len(selected)


def test_selection_is_reproducible_and_depends_only_on_the_seed(universe_store):
    store, candidates = universe_store
    first = build_universe(_config(candidates), store).manifest
    again = build_universe(_config(candidates), store).manifest
    other = build_universe(_config(candidates, seed=2), store).manifest
    assert (first["dev"], first["held_out"]) == (again["dev"], again["held_out"])
    assert (first["dev"], first["held_out"]) != (other["dev"], other["held_out"])


def test_morning_coverage_screen(tmp_path: Path):
    store = DataStore(tmp_path / "data")
    window = get_calendar().previous_sessions(AS_OF, 10)
    minute = synthetic_bars(symbols=["AAA", "BBB"], start=str(window[0].date()), end=str(window[-1].date()))
    minute = minute[~((minute["symbol"] == "BBB") & (minute.index % 5 == 0))]  # BBB misses 20% of minutes
    for symbol, group in minute.groupby("symbol"):
        unit = BarUnit("sip", "1Min", "raw", symbol, "2026-05")
        store.write_bar_unit(unit, group.reset_index(drop=True), *period_window("2026-05"), complete=True)
        write_daily(store, make_daily(symbol, window, 50, 100e6, 0.02, seed=1))
    store.bars_manifest.flush()
    candidates = tmp_path / "c.csv"
    candidates.write_text("symbol,sector\nAAA,Tech\nBBB,Tech\nCCC,Tech\n")
    cfg = _config(candidates, screens={"min_morning_coverage": 0.95}, sector_cap=None,
                  cells={"liquidity_bins": 1, "volatility_bins": 1, "per_cell": 1, "held_out_per_cell": 0})
    table = build_universe(cfg, store).table.set_index("symbol")
    assert table.loc["AAA", "status"] == "selected"
    assert table.loc["AAA", "morning_coverage"] == pytest.approx(1.0)
    assert "morning coverage below" in table.loc["BBB", "exclusion"]
    assert table.loc["CCC", "status"] == "excluded"
    assert [c["timeframe"] for c in required_downloads(cfg)] == ["1Day", "1Min"]


def test_cli_writes_manifest(universe_store, tmp_path: Path, monkeypatch, capsys):
    store, candidates = universe_store
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(store.root))
    cfg = {
        "name": "cli", "candidates": candidates.name, "as_of": AS_OF, "window_sessions": 10,
        "screens": {"min_median_close": 10, "min_median_dollar_volume": 25000000, "min_morning_coverage": None},
        "cells": {"liquidity_bins": 2, "volatility_bins": 3, "per_cell": 3, "held_out_per_cell": 1},
        "sector_cap": 7, "seed": 1,
    }
    path = tmp_path / "u.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["data", "universe", "--config", str(path)]) == 0
    manifest = yaml.safe_load((store.root / "universes" / "cli.yaml").read_text())
    assert manifest["counts"]["selected"] == 18
    assert len(manifest["candidates"]) == 34
    assert "18 selected (12 dev, 6 held out)" in capsys.readouterr().out
