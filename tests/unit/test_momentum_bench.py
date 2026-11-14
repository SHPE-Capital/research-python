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


def _matrix(bench, **kw):
    replay = {"cold_s_per_candidate": 2.0, "warm_s_per_candidate": 1.0, "symbol_sessions": 10}
    build = {"build_s_per_symbol_session": 0.01, "panel_bytes_per_symbol_session": 1000.0}
    runs = {1: {"status": "OK", "wall_s": 8.0, "speedup": 1.0, "worker_peak_rss_mb": 100.0},
            8: {"status": "OK", "wall_s": 1.0, "speedup": 8.0, "worker_peak_rss_mb": 120.0},
            16: {"status": "SKIPPED_RAM"}}
    args = dict(ns=[10], years=[1], threads=[1, 8, 16], replay=replay, build=build, parquet_bytes_per_unit=500.0,
                thread_runs=runs, candidates=4, weeks=1, policies=1, seeds=1)
    return bench.scale_matrix(**(args | kw))


def test_scale_matrix_cache_beats_no_cache_and_threads_help(bench):
    r1, r8, r16 = _matrix(bench)
    assert r16["status"] == "SKIPPED_RAM"
    assert r1["weekly_h_d6_cache"] < r1["weekly_h_no_cache"]
    assert r8["weekly_h_no_cache"] < r1["weekly_h_no_cache"]
    units = 10 * bench.SESSIONS_PER_YEAR
    assert r1["panel_gb"] == pytest.approx(units * 1000 / 1e9)
    assert r1["parquet_gb"] == pytest.approx(units * 500 / 1e9)
    assert r8["ram_total_gb"] == pytest.approx(r8["ram_per_worker_mb"] * 8 / 1e3)
    # no-cache single worker: build units*0.01 s + units*4 candidates*0.2 s per unit
    assert r1["weekly_h_no_cache"] == pytest.approx((units * 0.01 + units * 4 * 0.2) / 3600)


def test_scale_matrix_pull_time_only_when_rate_given(bench):
    assert _matrix(bench)[0]["pull_hours"] is None
    r = _matrix(bench, pull_rows_per_s=1000.0)[0]
    assert r["pull_hours"] == pytest.approx(10 * bench.SESSIONS_PER_YEAR * bench.MINUTES_PER_SESSION / 1000 / 3600)


def test_load_protocol_absent_returns_none(bench, tmp_path):
    assert bench.load_protocol(tmp_path / "nope.yaml") is None
    p = tmp_path / "p.yaml"
    p.write_text("universe_size: 500\n", encoding="utf-8")
    assert bench.load_protocol(p) == {"universe_size": 500}
