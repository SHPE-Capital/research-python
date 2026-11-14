from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from shpe_research.contracts.weekly_config import from_json

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "momentum_golden"


@pytest.fixture(scope="module")
def mg():
    spec = importlib.util.spec_from_file_location("make_golden", ROOT / "scripts" / "momentum" / "make_golden.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_golden"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def generated(mg):
    return mg.build_all()


def test_regeneration_is_byte_identical_to_checked_in_files(generated):
    for rel, text in generated.items():
        raw = (FIX / rel).read_bytes().replace(bytes([13, 10]), bytes([10]))  # tolerate autocrlf checkouts
        assert raw == text.encode("utf-8"), f"{rel} is stale: run make_golden.py"
    on_disk = {p.relative_to(FIX).as_posix() for p in FIX.rglob("*") if p.is_file()}
    assert on_disk == set(generated)


def test_two_builds_are_identical(mg, generated):
    assert mg.build_all() == generated


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_weekly_configs_validate():
    for d in FIX.iterdir():
        cfg = from_json((d / "weekly_config.json").read_text(encoding="utf-8"))
        assert cfg.selected is not None and cfg.account.max_positions >= 1


def test_scenario_edge_cases_are_present():
    m1 = json.loads((FIX / "s1_trend_gap" / "manifest.json").read_text(encoding="utf-8"))
    assert any(s["early_close"] for s in m1["sessions"])
    (gap,) = m1["missing_bars"]
    bars = _rows(FIX / "s1_trend_gap" / "bars.jsonl")
    assert not any(b["i"] == gap["i"] and b["asset_id"] == gap["asset_id"] for b in bars)
    inds = _rows(FIX / "s1_trend_gap" / "indicators.jsonl")
    row = next(r for r in inds if r["i"] == gap["i"] and r["asset_id"] == gap["asset_id"])
    assert row["atr14"] is None and row["signal"] is False
    # warm-up boundary: ATR14 readiness starts at bar 70 for an undisturbed symbol
    first = min(r["i"] for r in inds if r["asset_id"] == "a-0001" and r["ready_atr14"])
    assert first == 70
    assert all(r["atr14"] is None for r in inds if r["asset_id"] == "a-0001" and r["i"] < 14)


def test_signals_match_indicator_rows():
    for d in FIX.iterdir():
        inds = {(r["i"], r["asset_id"]) for r in _rows(d / "indicators.jsonl") if r["signal"]}
        sigs = {(r["i"], r["asset_id"]) for r in _rows(d / "signals.jsonl")}
        assert inds == sigs and sigs


def test_fresh_order_matches_account_engine(mg):
    """The first scripted decision uses a fresh account, so its sized qty is what C4 actually fills."""
    for spec in mg.SCENARIOS:
        d = FIX / spec.name
        first = _rows(d / "orders.jsonl")[0]
        assert first["profile"] == "fresh"
        ledger = _rows(d / "ledger.jsonl")
        o = next(o for o in first["orders"])
        entry = [e for e in ledger if e["entry_i"] == first["i"] + 1 and e["asset_id"] == o["asset_id"]]
        if entry:  # filled (not cancelled above the limit)
            assert entry[0]["qty"] == o["qty"]
