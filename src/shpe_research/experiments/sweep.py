"""TODO(M2): config sweeps, seed ensembles and an experiment registry.  Level: intermediate.

Goal: generating many models is a config change, and every attempt is recorded.

    sweep config     a base experiment config plus a `grid:` section, e.g.
                     grid: {model.params.max_depth: [4, 8, 12], seed: [1, 2, 3]}
    run_sweep()      expands the grid, runs each point with run_experiment, and
                     logs params, metrics and artifacts to MLflow
                     (`uv sync --extra ml`; local ./mlruns store by default)
    seed ensembles   ORB plan: three seeds per configuration, deployed model =
                     mean of the three. Report per-seed spread; never select a lucky seed.
    selection        choose by VALIDATION metrics only, under constraints frozen
                     before looking (ORB plan section 6); ties within one standard
                     error go to the simpler configuration.
    trial count      the registry's total number of attempts feeds E2's deflated Sharpe.

Done when:
    - `shpe sweep configs/experiments/<sweep>.yaml` runs a 2x2 grid on fixture data
      and every run appears in MLflow with its config and metrics.
    - tests/todo/test_experiments_todo.py::test_sweep passes.
"""

from __future__ import annotations

from pathlib import Path


def run_sweep(config_path: str | Path) -> list[Path]:
    raise NotImplementedError("TODO(M2): sweeps and experiment registry")
