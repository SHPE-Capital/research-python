"""Tests for the E1 benchmark harness (scripts/momentum/bench.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "momentum" / "bench.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("momentum_bench", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["momentum_bench"] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_machine_profile_has_core_fields(bench, tmp_path):
    p = bench.machine_profile(tmp_path)
    assert p["cores_logical"] >= 1 and p["ram_total_gb"] > 0 and p["scratch_free_gb"] > 0


def test_candidate_grid_size_and_determinism(bench):
    assert len(bench.candidate_grid(256)) == 256
    assert bench.candidate_grid(256) == bench.candidate_grid(256)


def test_replay_deterministic_and_cache_hits(bench, tmp_path):
    bars, _ = bench.timed_panel(2, "2026-03-02", "2026-03-04", tmp_path)
    units = bench.split_sessions(bars)
    m = bench.measure_replay(units, 8)  # asserts cold == warm results internally
    assert m["cache_hit_rate_overall"] > 0.5
    assert m["warm_s_per_candidate"] <= m["cold_s_per_candidate"] * 2


def test_projection_math(bench):
    r = bench.StepResult("2", "OK", {"cold_s_per_candidate": 1.0, "warm_s_per_candidate": 0.5, "symbol_sessions": 10})
    p = bench.project(r, history_sessions=10, candidates=2, weeks=2, policies=3, seeds=1, n_symbols=10)
    # per unit 0.1s; calibration = 0.1*10*10*2 = 20s; total = 20*2*3*1*2x reserve = 240s
    assert p["cold"]["weekend_job_hours"] == pytest.approx(20 / 3600, abs=1e-3)
    assert p["cold"]["full_run_hours_with_2x_reserve"] == pytest.approx(240 / 3600, abs=0.01)


def test_steps_3_4_blocked_without_c4(bench):
    assert all(s.status == "BLOCKED" for s in bench.steps_3_4())
