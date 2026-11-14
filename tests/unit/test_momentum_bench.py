"""Tests for the E1 benchmark harness (scripts/momentum/bench.py) on the real account engine and panel loader."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "momentum" / "bench.py"
WARMUP = 5  # small history prefix keeps the pure-Python indicator cost low


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("momentum_bench", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["momentum_bench"] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def panel_paths(bench, tmp_path_factory):
    paths, build_s = bench.make_panel(tmp_path_factory.mktemp("panel"), 6, 3, seed=0)
    assert build_s >= 0
    return paths


def test_machine_profile_has_core_fields(bench, tmp_path):
    p = bench.machine_profile(tmp_path)
    assert p["cores_logical"] >= 1 and p["ram_total_gb"] > 0 and p["scratch_free_gb"] > 0


def test_sample_configs_seeded_and_distinct(bench):
    a, b = bench.sample_configs(256, 3), bench.sample_configs(256, 3)
    assert a == b and len(set(a)) == 256
    assert a != bench.sample_configs(256, 4)
    with pytest.raises(ValueError):
        bench.sample_configs(10**7, 0)


def test_week_panel_prefix_is_history_only(bench, panel_paths):
    p0, w0 = bench.build_week_panel(panel_paths, 0, WARMUP)
    p2, w2 = bench.build_week_panel(panel_paths, 2, WARMUP)
    assert w0 == 0  # first chunk has no history
    assert w2 == WARMUP
    sessions = np.unique(np.asarray(p2.session_index))
    assert len(sessions) in (WARMUP + 4, WARMUP + 5) and p2.n_symbols == 6  # MLK week has 4 sessions
    assert p2.n_minutes == p2.close.shape[1]


def test_replay_deterministic_across_pools_and_cache(bench, panel_paths):
    kernel = bench.resolve_kernel("python")
    cfgs = bench.sample_configs(3, 1)
    digests = []
    for workers, pool in ((1, "serial"), (2, "thread")):
        with bench.Runner(panel_paths, kernel, workers, pool, WARMUP) as r:
            cold = bench.run_pass(r, [2], cfgs)
            warm = bench.run_pass(r, [2], cfgs)
        assert cold["digest"] == warm["digest"]
        assert warm["cache_hit_rate"] > cold["cache_hit_rate"]
        digests.append(cold["digest"])
    assert digests[0] == digests[1]


def test_step1_deterministic_rerun(bench, tmp_path):
    r = bench.step1(tmp_path, bench.resolve_kernel("python"), 0, WARMUP, n_cands=2)
    assert r["deterministic"] is True and r["sessions_returned"] == 5


def test_fast_kernel_missing_is_clear_error(bench, monkeypatch):
    monkeypatch.setitem(sys.modules, "shpe_research.trade.account_fast", None)  # forces ImportError
    with pytest.raises(SystemExit, match="account_fast"):
        bench.resolve_kernel("fast")


def test_pool_choice(bench):
    py, fast = bench.Kernel("python", len, False), bench.Kernel("fast", len, True)
    assert bench.choose_pool(py, 1, "auto") == "serial"
    assert bench.choose_pool(py, 8, "auto") == "process"
    assert bench.choose_pool(fast, 8, "auto") == "thread"
    assert bench.choose_pool(fast, 8, "process") == "process"


def test_budget_gate_skips_big_cells_with_reason(bench):
    big = bench.estimate_cell(1000, 10, 16, "process")
    why = bench.skip_reason(big, 11.0, 30.0, 1e12)
    assert why is not None and "exceeds" in why
    small = bench.estimate_cell(12, 0.05, 1, "serial")
    assert bench.skip_reason(small, 11.0, 30.0, 1e12) is None
    assert "free space" in bench.skip_reason(small, 11.0, 30.0, 1.0)
    assert "disk" in bench.skip_reason(bench.estimate_cell(1000, 10, 1, "serial"), 1e6, 5.0, 1e12)


def test_projection_math_with_and_without_cache(bench):
    params = {"weeks": 3, "policies": 3, "seeds": 2, "budget": 10, "carry_over": 0.5,
              "policies_share_replays": True}
    p = bench.project(2.0, window_weeks=5, params=params)
    cands = 20
    assert p["without_cache"]["candidate_weeks"] == 3 * 3 * cands * 5
    first, step = cands * 5, cands * (1 + 0.5 * 4)
    assert p["with_cache"]["candidate_weeks"] == pytest.approx(first + 2 * step)
    assert p["with_cache"]["full_run_hours_2x_reserve"] == pytest.approx((first + 2 * step) * 2 * 2 / 3600, abs=0.01)
    assert p["with_cache"]["candidate_weeks"] < p["without_cache"]["candidate_weeks"]
    unshared = bench.project(2.0, window_weeks=5, params={**params, "policies_share_replays": False})
    assert unshared["with_cache"]["candidate_weeks"] == pytest.approx(3 * (first + 2 * step))


def test_render_markdown_reports_skips(bench):
    result = {
        "meta": {"kernel": "python", "pool_rule": "x", "smoke": True, "ram_budget_gb": 11, "disk_budget_gb": 30,
                 "panel_source": "synthetic"},
        "profile": {"cpu": "cpu", "cores_logical": 16, "ram_total_gb": 16},
        "params": {}, "steps": {},
        "matrix": [{"n": 1000, "years": 10, "skipped": "estimated RAM 99.0 GB exceeds budget 11.0 GB"}],
    }
    md = bench.render_markdown(result)
    assert "SKIPPED: estimated RAM 99.0 GB" in md


def _measured_cell(n, years, per_cw, ram_w, threads=("1", "8")):
    th = {t: {"pool": "thread", "s_per_candidate_week": per_cw / int(t), "rss": {"peak_per_worker_mb": ram_w,
                                                                                "peak_total_mb": ram_w * int(t)}}
          for t in threads}
    return {"n": n, "years": years, "window_weeks": 52, "disk_bytes": 52 * 1000 * n, "panel_bytes": 52 * 1000 * n,
            "threads": th, "basis": "measured"}


def _args(bench):
    ns = bench.build_parser().parse_args([])
    ns.params, ns.threads = dict(bench.DEFAULT_PARAMS), [1, 8]
    return ns


def test_fill_projections_labels_and_scales_linearly(bench):
    args = _args(bench)
    pw = bench.prefix_weeks(args.warmup)
    base = _measured_cell(100, 1, 0.01, 50.0)
    base["disk_bytes"] = base["panel_bytes"] = (52 + pw) * 1000 * 100
    big = {"n": 400, "years": 2, "skipped": "estimated RAM 99.0 GB exceeds budget 12.0 GB"}
    cells = [base, big]
    bench.fill_projections(cells, args)
    assert base["basis"] == "measured" and big["basis"] == "projected"
    assert big["threads"]["1"]["s_per_candidate_week"] == pytest.approx(0.04)
    assert big["threads"]["8"]["rss"]["peak_per_worker_mb"] == pytest.approx(200.0)
    assert big["panel_bytes"] == pytest.approx((104 + pw) * 1000 * 400)
    assert big["best_threads"] == 8
    assert big["projection"]["with_cache"]["full_run_hours_2x_reserve"] < \
        big["projection"]["without_cache"]["full_run_hours_2x_reserve"]
    assert big["pull_hours"] is None
    args.pull_rows_per_s = 1000.0
    assert bench.pull_hours(10, 1, 1000.0) == pytest.approx(10 * 252 * 390 / 1000 / 3600)


def test_e1_table_has_profile_and_both_bases(bench):
    args = _args(bench)
    base = _measured_cell(100, 1, 0.01, 50.0)
    big = {"n": 400, "years": 2, "skipped": "too big"}
    cells = [base, big]
    bench.fill_projections(cells, args)
    result = {"meta": {"kernel": "fast", "pool_rule": "threads", "ram_budget_gb": 12, "disk_budget_gb": 40},
              "profile": {"os": "os", "cpu": "cpu", "cores_physical": 8, "cores_logical": 16, "ram_total_gb": 17,
                          "ram_available_gb": 10, "scratch_free_gb": 70, "python": "3", "numpy": "2", "pandas": "2"},
              "params": args.params, "steps": {}, "matrix": cells}
    md = bench.render_e1_table(result)
    assert "Machine profile" in md and "| measured |" in md and "| projected |" in md


def test_fast_kernel_matches_oracle_and_counts_trades(bench, panel_paths):
    cfgs = bench.sample_configs(4, 2)
    out = {}
    for name in ("python", "fast"):
        with bench.Runner(panel_paths, bench.resolve_kernel(name), 1, "serial", WARMUP) as r:
            outs, _ = r.run([(2, tuple(cfgs))])
        out[name] = outs[0]["results"]
    for (dr_p, n_p), (dr_f, n_f) in zip(out["python"], out["fast"], strict=True):
        np.testing.assert_allclose(dr_p, dr_f, rtol=0, atol=1e-12)
        assert n_p == n_f  # fast kernel has no ledger by default; the count comes from summary["trades"]
