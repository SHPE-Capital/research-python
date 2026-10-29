from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from shpe_research.cli import main
from shpe_research.data import snapshot as snapshot_module
from shpe_research.data.bars import normalize_bars
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.snapshot import build_snapshot, load_snapshot, verify_snapshot
from shpe_research.data.store import BarUnit, DataStore, period_window
from shpe_research.experiments.runner import ExperimentConfig, run_experiment

from .conftest import HELLO
from .fake_alpaca import raw_daily

START, END = "2026-09-01", "2026-09-30"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    store = DataStore(root)
    minute = synthetic_bars(symbols=["AAA", "BBB", "SPY"], start=START, end=END)
    for symbol, group in minute.groupby("symbol"):
        group = group.reset_index(drop=True)
        store.write_bar_unit(BarUnit("sip", "1Min", "raw", symbol, "2026-09"), group, *period_window("2026-09"),
                             complete=True)
        daily = normalize_bars(raw_daily(group).drop(columns="_ts").to_dict("records"), symbol, "1Day")
        store.write_bar_unit(BarUnit("sip", "1Day", "split", symbol, "2026"), daily, *period_window("2026"),
                             complete=True)
    store.bars_manifest.flush()
    return store


@pytest.fixture
def universe(tmp_path: Path) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(yaml.safe_dump({"name": "u", "dev": ["AAA"], "held_out": ["BBB"]}))
    return path


def test_build_verify_and_load(store, universe, tmp_path: Path):
    folder = build_snapshot("ws", START, END, universe=universe, extra_symbols=["spy"], store=store,
                            out_root=tmp_path / "snaps")
    assert verify_snapshot(folder) == []
    assert (folder / "universe.yaml").exists()
    assert folder.with_suffix(".zip").exists()
    snap = load_snapshot(folder)
    assert snap.symbols() == ["AAA", "BBB", "SPY"]
    assert snap.symbols("dev") == ["AAA"]
    assert snap.symbols("held_out") == ["BBB"]
    assert snap.symbols("context") == ["SPY"]
    assert len(snap.manifest["files"]) == 6
    assert len(snap.bars(symbols=["AAA"])) == 21 * 390
    assert len(snap.bars("1Day", "split")) == 3 * 21
    assert len(snap.bars(symbols=["AAA"], start="2026-09-29", end="2026-09-30")) == 2 * 390
    assert all(source["sha256"] for source in snap.manifest["sources"])


def test_zip_round_trip_gives_identical_data(store, universe, tmp_path: Path):
    folder = build_snapshot("ws", START, END, universe=universe, store=store, out_root=tmp_path / "a")
    elsewhere = tmp_path / "b"
    elsewhere.mkdir()
    zipped = Path(shutil.copy(folder.with_suffix(".zip"), elsewhere))
    from_zip = load_snapshot(zipped)
    assert from_zip.folder == elsewhere / "ws"
    assert from_zip.fingerprint() == load_snapshot(folder).fingerprint()
    pd.testing.assert_frame_equal(from_zip.bars(), load_snapshot(folder).bars())


def test_tampering_is_caught_and_snapshots_are_immutable(store, universe, tmp_path: Path):
    folder = build_snapshot("ws", START, END, universe=universe, store=store, out_root=tmp_path, make_zip=False)
    target = folder / "bars" / "1Min_raw" / "AAA.parquet"
    target.write_bytes(target.read_bytes()[:-10] + b"0123456789")
    assert verify_snapshot(folder) == ["bars/1Min_raw/AAA.parquet: checksum mismatch"]
    with pytest.raises(ValueError, match="failed verification"):
        load_snapshot(folder)
    with pytest.raises(FileExistsError):
        build_snapshot("ws", START, END, universe=universe, store=store, out_root=tmp_path)


def test_hello_model_runs_from_a_snapshot(store, tmp_path: Path):
    folder = build_snapshot("ws", START, END, symbols=["AAA", "BBB", "SPY"], store=store, out_root=tmp_path,
                            series=(("1Min", "raw"),), make_zip=False)
    cfg = ExperimentConfig.from_yaml(HELLO)
    cfg.data = {"source": "snapshot", "path": str(folder), "role": "all"}
    result = run_experiment(cfg, out_root=tmp_path / "runs")
    assert result.metrics["validation"]["n_rows"] > 0


def test_cli_build_and_verify(store, universe, tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(store.root))
    monkeypatch.setattr(snapshot_module, "snapshots_dir", lambda: tmp_path / "snaps")
    argv = ["data", "snapshot", "build", "--name", "cli", "--universe", str(universe), "--from", START,
            "--to", END, "--extra-symbols", "SPY"]
    assert main(argv) == 0
    assert "3 symbols" in capsys.readouterr().out
    assert main(["data", "snapshot", "verify", str(tmp_path / "snaps" / "cli.zip")]) == 0
    assert "verified" in capsys.readouterr().out
