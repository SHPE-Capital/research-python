"""Config sweeps, seed ensembles and an experiment registry (M2).

A sweep config (`kind: sweep`, validated by `shpe_research.config.sweep.SweepConfig`) is a base
experiment plus a `grid:` of dotted-path overrides, for example::

    grid: {model.params.max_depth: [4, 8, 12], seed: [1, 2, 3]}

`run_sweep` expands the grid in listed order (last key varies fastest), runs every point with
`run_experiment`, and logs each run to MLflow (params = the overrides, metrics = every
`<split>.<metric>` the report produced, artifacts = the run directory). The default store is
`./mlruns` at the repository root (git-ignored); pass `tracking_uri` to use another.

Seed ensembles: put `seed` in the grid. `summarize` groups runs that differ only in seed and reports
mean, sample standard deviation and standard error per configuration, so the per-seed spread is always
visible. Selection works on seed means, so it never picks a single lucky seed. Averaging the seeds'
predictions into one deployed model is not done here (see `select_best`).

Selection (`select_best`) uses validation metrics only: a metric under `test.` is refused. Among
configurations whose mean is within one standard error of the best, the simpler one wins. Simplicity is
grid order: list each grid axis from simplest to most complex value; axes compare in `grid:` order.

Trial count: every MLflow run in the sweep's experiment is one attempt, kept even when it is dropped.
`trial_count` reads that total for E2's `deflated_sharpe`. Re-running a sweep adds attempts.

Timing: nothing here looks at data; each point's leakage guarantees are the pipeline's. The pipeline
scores and logs the test split as always, but selection cannot use it.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

SELECTION_SPLITS = ("validation",)


@dataclass
class SweepRun:
    """One executed grid point: its run directory, the grid overrides, metrics and simplicity order."""

    run_dir: Path
    overrides: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    order: tuple[int, ...] = field(default_factory=tuple)

    @property
    def config(self) -> dict[str, Any]:
        """The overrides without `seed`: runs with equal `config` are seeds of one configuration."""
        return {k: v for k, v in self.overrides.items() if k != "seed"}


def _set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for i, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or not isinstance(node.get(part), dict):
            raise KeyError(f"grid path {dotted!r}: {'.'.join(parts[: i + 1])!r} is not in the base config")
        node = node[part]
    node[parts[-1]] = value


def expand_grid(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Cartesian product of `grid` applied to deep copies of `base`.

    Returns `(overrides, config)` pairs in grid order, the last key varying fastest. Each path must run
    through existing mappings in `base` (only the final key may be new, e.g. a model param); otherwise
    KeyError. `base` is not mutated.
    """
    points: list[dict[str, Any]] = [{}]
    for key, values in grid.items():
        points = [{**p, key: v} for p in points for v in values]
    out = []
    for overrides in points:
        cfg = copy.deepcopy(base)
        for key, value in overrides.items():
            _set_path(cfg, key, value)
        out.append((overrides, cfg))
    return out


def _flatten(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        f"{split}.{name}": float(v)
        for split, block in metrics.items()
        for name, v in block.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    }


def default_tracking_uri() -> str:
    """The local `./mlruns` store at the repository root, as a file URI."""
    from shpe_research.config import REPO_ROOT

    return (REPO_ROOT / "mlruns").as_uri()


def execute_sweep(
    config_path: str | Path, out_root: str | Path | None = None, tracking_uri: str | None = None
) -> tuple[list[SweepRun], Any]:
    """Run every grid point and record it in MLflow; returns the runs and the parsed `SweepConfig`.

    Validates the sweep config and every expanded point before the first run, so a typo wastes nothing.
    Raises ValueError for a non-validation selection metric and NotImplementedError for optuna searches.
    """
    from shpe_research.config.experiment import ExperimentConfig as TypedConfig
    from shpe_research.config.sweep import SweepConfig
    from shpe_research.experiments.runner import ExperimentConfig, run_experiment

    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("sweeps log to MLflow: run `uv sync --extra ml`") from exc

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text())
    sweep = SweepConfig.model_validate(raw)
    if sweep.grid is None:
        raise NotImplementedError("optuna sweeps are not implemented; use a grid")
    split = sweep.metric.split(".", 1)[0]
    if split not in SELECTION_SPLITS:
        raise ValueError(f"metric {sweep.metric!r}: selection may only use {SELECTION_SPLITS} metrics, not {split!r}")

    base = raw["base"]
    points = expand_grid(base, sweep.grid)
    for _, cfg in points:
        TypedConfig.model_validate(cfg)  # fail on a bad grid value before anything runs
    axes = [k for k in sweep.grid if k != "seed"]

    # MLflow 3 refuses the file store unless opted in; the spec's default is the local ./mlruns folder.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri or default_tracking_uri())
    mlflow.set_experiment(str(base["name"]))
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    runs: list[SweepRun] = []
    for i, (overrides, cfg) in enumerate(points):
        cfg = {**cfg, "name": f"{cfg['name']}-{i:03d}"}
        result = run_experiment(ExperimentConfig(**cfg), out_root=out_root)
        order = tuple(sweep.grid[k].index(overrides[k]) for k in axes)
        runs.append(SweepRun(result.run_dir, overrides, result.metrics, order))
        with mlflow.start_run(run_name=result.run_dir.name):
            mlflow.log_params({**{k: str(v) for k, v in overrides.items()}, "sweep_point": str(i)})
            mlflow.set_tags({"sweep_config": path.name, "sweep_stamp": stamp})
            mlflow.log_metrics(_flatten(result.metrics))
            mlflow.log_artifacts(str(result.run_dir))
        log.info("sweep point %d/%d done: %s", i + 1, len(points), overrides)
    return runs, sweep


def run_sweep(
    config_path: str | Path, out_root: str | Path | None = None, tracking_uri: str | None = None
) -> list[Path]:
    """Expand the grid in `config_path`, run each point, log each to MLflow, return the run directories.

    Inputs: a sweep YAML (see configs/experiments/sweeps/hello_sweep.yaml), an optional runs directory and an
    optional MLflow tracking URI. Output: one run directory per grid point, in grid order. Run names get
    a `-NNN` suffix so points never share a directory.
    """
    runs, _ = execute_sweep(config_path, out_root, tracking_uri)
    return [r.run_dir for r in runs]


def _value(run: SweepRun, metric: str) -> float | None:
    split, name = metric.split(".", 1)
    v = run.metrics.get(split, {}).get(name)
    return float(v) if isinstance(v, int | float) and math.isfinite(v) else None


def summarize(runs: Sequence[SweepRun], metric: str) -> list[dict[str, Any]]:
    """One row per configuration (runs equal except for seed): mean, sd, se and n over seeds.

    `metric` is `<split>.<name>`, e.g. `validation.ic_mean`. Runs missing the metric are dropped; a
    configuration with no usable run is omitted. sd is the sample standard deviation (0 for one seed).
    """
    groups: dict[str, list[tuple[SweepRun, float]]] = {}
    for run in runs:
        v = _value(run, metric)
        if v is not None:
            groups.setdefault(repr(sorted(run.config.items())), []).append((run, v))
    rows = []
    for members in groups.values():
        vals = [v for _, v in members]
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        rows.append(
            {
                "config": members[0][0].config,
                "order": members[0][0].order,
                "mean": mean,
                "sd": sd,
                "se": sd / math.sqrt(n),
                "n_seeds": n,
                "values": vals,
            }
        )
    return rows


def select_best(runs: Sequence[SweepRun], metric: str, direction: str = "maximize") -> dict[str, Any]:
    """Pick a configuration by seed-mean VALIDATION metric; ties within one standard error go to the simpler.

    Returns the winning `summarize` row. Raises ValueError for a non-validation metric or when no run has
    the metric. A lower `order` (earlier in each grid list, axes in grid order) counts as simpler.
    Averaging the seeds' predictions into one deployed model is not implemented.
    """
    split = metric.split(".", 1)[0]
    if split not in SELECTION_SPLITS:
        raise ValueError(f"selection may only use {SELECTION_SPLITS} metrics, not {split!r} ({metric!r})")
    rows = summarize(runs, metric)
    if not rows:
        raise ValueError(f"no run has a usable {metric!r}")
    sign = 1.0 if direction == "maximize" else -1.0
    best = max(rows, key=lambda r: sign * r["mean"])
    near = [r for r in rows if sign * (best["mean"] - r["mean"]) <= best["se"]]
    return min(near, key=lambda r: (r["order"], -sign * r["mean"]))


def trial_count(tracking_uri: str, experiment: str) -> int:
    """Total attempts recorded in an MLflow experiment: the `n_trials` for `deflated_sharpe`."""
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return 0
    return len(client.search_runs([exp.experiment_id], max_results=50000))
