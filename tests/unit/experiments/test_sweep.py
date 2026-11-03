"""M2: grid expansion, MLflow-recorded sweeps, seed spread, validation-only selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from shpe_research.experiments.sweep import (
    SweepRun,
    expand_grid,
    run_sweep,
    select_best,
    summarize,
    trial_count,
)

HELLO = Path(__file__).parents[3] / "configs/experiments/sweeps/hello_sweep.yaml"


def test_expand_grid_applies_dotted_overrides_in_order():
    base = {"seed": 0, "model": {"type": "logistic", "params": {"C": 1.0}}}
    points = expand_grid(base, {"model.params.C": [0.1, 1.0], "seed": [1, 2]})
    assert [p[0] for p in points] == [
        {"model.params.C": 0.1, "seed": 1},
        {"model.params.C": 0.1, "seed": 2},
        {"model.params.C": 1.0, "seed": 1},
        {"model.params.C": 1.0, "seed": 2},
    ]
    assert points[2][1]["model"]["params"]["C"] == 1.0 and points[2][1]["seed"] == 1
    assert base["model"]["params"]["C"] == 1.0 and base["seed"] == 0  # base is not mutated


def test_expand_grid_rejects_unknown_path():
    with pytest.raises(KeyError, match="model.parms"):
        expand_grid({"model": {"params": {}}}, {"model.parms.C": [1]})


def _runs(values: dict[tuple[float, int], float]) -> list[SweepRun]:
    return [
        SweepRun(Path(f"r{i}"), {"depth": d, "seed": s}, {"validation": {"ic_mean": v}}, order=(d_idx(d),))
        for i, ((d, s), v) in enumerate(values.items())
    ]


def d_idx(depth: float) -> int:
    return [4, 8, 12].index(depth)


def test_select_best_uses_mean_over_seeds_and_reports_spread():
    runs = _runs({(4, 1): 0.01, (4, 2): 0.03, (8, 1): 0.10, (8, 2): 0.12, (12, 1): 0.02, (12, 2): 0.02})
    table = summarize(runs, "validation.ic_mean")
    by_depth = {r["config"]["depth"]: r for r in table}
    assert by_depth[8]["mean"] == pytest.approx(0.11)
    assert by_depth[8]["n_seeds"] == 2 and by_depth[8]["sd"] == pytest.approx(0.01414, abs=1e-4)
    assert select_best(runs, "validation.ic_mean")["config"] == {"depth": 8}


def test_select_best_prefers_simpler_within_one_standard_error():
    # depth 12 is nominally best but within one SE of depth 4: take the simpler depth 4
    runs = _runs({(4, 1): 0.08, (4, 2): 0.12, (12, 1): 0.09, (12, 2): 0.13})
    assert select_best(runs, "validation.ic_mean")["config"] == {"depth": 4}


def test_select_best_minimize_direction():
    runs = _runs({(4, 1): 0.5, (8, 1): 0.2, (12, 1): 0.3})
    assert select_best(runs, "validation.ic_mean", direction="minimize")["config"] == {"depth": 8}


def test_selection_refuses_test_metrics():
    runs = _runs({(4, 1): 0.5})
    with pytest.raises(ValueError, match="test"):
        select_best(runs, "test.ic_mean")


def test_run_sweep_records_every_run_in_mlflow(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = (tmp_path / "mlruns").as_uri()
    runs = run_sweep(HELLO, out_root=tmp_path / "runs", tracking_uri=uri)
    assert len(runs) == 4
    assert len({r.name for r in runs}) == 4  # distinct run dirs even within one second
    for r in runs:
        assert (r / "report.md").exists()

    client = mlflow.tracking.MlflowClient(uri)
    exp = client.get_experiment_by_name("hello_sweep")
    recorded = client.search_runs([exp.experiment_id])
    assert len(recorded) == 4
    assert {r.data.params["seed"] for r in recorded} == {"1", "2"}
    assert {r.data.params["model.params.C"] for r in recorded} == {"0.1", "1.0"}
    assert all("validation.n_rows" in r.data.metrics for r in recorded)
    assert all("test.n_rows" in r.data.metrics for r in recorded)  # test is recorded, never selected on
    assert trial_count(uri, "hello_sweep") == 4

    run_sweep(HELLO, out_root=tmp_path / "runs", tracking_uri=uri)  # attempts accumulate
    assert trial_count(uri, "hello_sweep") == 8


def test_run_sweep_rejects_typos_before_running(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(HELLO.read_text().replace("  session: regular", "  sesion: regular"))
    with pytest.raises(Exception, match="sesion"):
        run_sweep(bad, out_root=tmp_path / "runs", tracking_uri=(tmp_path / "ml").as_uri())
    assert not (tmp_path / "runs").exists()
