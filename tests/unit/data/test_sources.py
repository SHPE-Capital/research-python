"""C1g: pluggable data sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from shpe_research.data import fixtures
from shpe_research.data.bars import normalize_bars
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.snapshot import build_snapshot
from shpe_research.data.sources import SOURCE_REGISTRY, build_source
from shpe_research.data.store import BarUnit, DataStore, period_window
from shpe_research.experiments.sources import load_bars

from ...fakes.fake_alpaca import raw_daily

ROOT = Path(__file__).parents[3]
KEYS = {"alpaca", "edgar", "file", "fixture", "fred", "snapshot"}


def test_registry_keys():
    assert KEYS <= set(SOURCE_REGISTRY)
    for key in KEYS:
        assert SOURCE_REGISTRY[key].source_type == key
    with pytest.raises(ValueError) as err:
        build_source({"type": "nope"})
    for key in KEYS:
        assert key in str(err.value)


def test_fixture_passthrough(monkeypatch):
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return synthetic_bars(symbols=["AAA"], start="2026-03-02", end="2026-03-02")

    monkeypatch.setattr(fixtures, "synthetic_bars", fake)
    monkeypatch.setattr("shpe_research.data.sources.fixture.synthetic_bars", fake)
    cfg = {"type": "fixture", "momentum": {"AAA": 0.1}, "seed": 3, "brand_new_option": 5}
    build_source(cfg).load(["AAA"], "2026-03-02", "2026-03-02", "1Min")
    assert seen == {"momentum": {"AAA": 0.1}, "seed": 3, "brand_new_option": 5, "symbols": ["AAA"],
                    "start": "2026-03-02", "end": "2026-03-02"}
    assert cfg["type"] == "fixture"  # the caller's dict is not mutated


def test_fred_release_lag():
    def fetch(series, start, end):
        return [
            {"date": "2026-01-01", "release_date": "2026-02-12", "value": "3.1"},
            {"date": "2026-02-01", "release_date": "2026-03-12", "value": "3.3"},
        ]

    src = build_source({"type": "fred", "series": "CPIAUCSL", "release_lag": "1D", "fetch": fetch})
    out = src.load(None, "2026-01-01", "2026-12-31", "1Day")
    assert list(out.columns) == ["ts", "ts_available", "series", "value"]
    assert (out["series"] == "CPIAUCSL").all()
    assert out["value"].tolist() == [3.1, 3.3]
    assert out["ts_available"].tolist() == [pd.Timestamp("2026-02-13", tz="UTC"), pd.Timestamp("2026-03-13", tz="UTC")]
    assert (out["ts_available"] > out["ts"]).all()


def test_fred_rejects_availability_before_observation():
    def fetch(series, start, end):
        return [{"date": "2026-03-01", "release_date": "2026-02-01", "value": "1"}]

    with pytest.raises(ValueError, match="ts_available"):
        build_source({"type": "fred", "series": "X", "fetch": fetch}).load(None, "2026-01-01", "2026-12-31", "1Day")


def test_edgar_uses_acceptance_time():
    def fetch(cik, form):
        return [{"accession": "a-1", "form": "10-K", "report_date": "2025-12-31",
                 "acceptance": "2026-02-20T21:15:30.000Z"}]

    out = build_source({"type": "edgar", "form": "10-K", "cik": 320193, "fetch": fetch}).load(
        None, "2026-01-01", "2026-12-31", "1Day")
    assert out["ts_available"].iloc[0] == pd.Timestamp("2026-02-20 21:15:30", tz="UTC")
    assert out["ts"].iloc[0] == pd.Timestamp("2025-12-31", tz="UTC")


def test_file_source_round_trip_and_contract(tmp_path: Path):
    bars = synthetic_bars(symbols=["AAA", "BBB"], start="2026-03-02", end="2026-03-03")
    pq_path = tmp_path / "b.parquet"
    import pyarrow.parquet as pq

    from shpe_research.contracts.bars import BarMetadata, bars_to_table

    pq.write_table(bars_to_table(bars, BarMetadata("fixture", "1Min", "raw", source="fixture")), pq_path)
    got = build_source({"type": "file", "path": str(pq_path), "format": "parquet"}).load(["AAA"], None, None, "1Min")
    assert set(got["symbol"]) == {"AAA"}
    bad = tmp_path / "bad.csv"
    bars.drop(columns=["close"]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        build_source({"type": "file", "path": str(bad), "format": "csv"}).load(None, None, None, "1Min")


@pytest.fixture
def snapshot_folder(tmp_path: Path) -> Path:
    store = DataStore(tmp_path / "store")
    minute = synthetic_bars(symbols=["AAA"], start="2026-09-01", end="2026-09-30")
    group = minute.reset_index(drop=True)
    store.write_bar_unit(BarUnit("sip", "1Min", "raw", "AAA", "2026-09"), group, *period_window("2026-09"),
                         complete=True)
    daily = normalize_bars(raw_daily(group).drop(columns="_ts").to_dict("records"), "AAA", "1Day")
    store.write_bar_unit(BarUnit("sip", "1Day", "split", "AAA", "2026"), daily, *period_window("2026"),
                         complete=True)
    store.bars_manifest.flush()
    uni = tmp_path / "u.yaml"
    uni.write_text(yaml.safe_dump({"name": "u", "dev": ["AAA"], "held_out": []}))
    return build_snapshot("ws", "2026-09-01", "2026-09-30", universe=uni, store=store, out_root=tmp_path / "snaps",
                          make_zip=False)


def test_snapshot_checksum_checked(snapshot_folder: Path):
    ok = build_source({"type": "snapshot", "path": str(snapshot_folder)}).load(None, None, None, "1Min")
    assert len(ok) > 0
    target = snapshot_folder / "bars" / "1Min_raw" / "AAA.parquet"
    target.write_bytes(target.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(ValueError, match="failed verification"):
        build_source({"type": "snapshot", "path": str(snapshot_folder)}).load(None, None, None, "1Min")


def test_legacy_load_bars_keys(snapshot_folder: Path):
    assert len(load_bars({"source": "snapshot", "path": str(snapshot_folder), "role": "all"})) > 0
    assert len(load_bars({"source": "fixture", "symbols": ["AAA"], "start": "2026-03-02", "end": "2026-03-02"})) > 0
    with pytest.raises(ValueError, match="fixture"):
        load_bars({"source": "bogus"})


def test_existing_configs_load():
    configs = sorted((ROOT / "configs").rglob("*.yaml"))
    assert configs
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        data = cfg.get("data") if isinstance(cfg, dict) else None
        if not data or data.get("source", "fixture") not in {"fixture", "parquet"}:
            continue  # snapshot/store configs need data that is not in the repo
        small = {**data, "start": data.get("start", "2026-03-02"), "end": data.get("start", "2026-03-02")}
        assert len(load_bars(small)) > 0, path
