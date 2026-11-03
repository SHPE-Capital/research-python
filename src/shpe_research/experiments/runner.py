"""One config file -> one recorded run: a thin wrapper over pipeline/ (C1a).

The twelve stages live in `shpe_research.pipeline.stages`. This module keeps
the C0 names (`ExperimentConfig`, `RunResult`, `EVAL_SPLITS`, `load_bars`,
`build_dataset`, `run_experiment`) so existing imports keep working until C1h
replaces `ExperimentConfig`.

See configs/experiments/hello_model.yaml for the config shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shpe_research.contracts.dataset import Dataset
from shpe_research.experiments.sources import load_bars
from shpe_research.pipeline import stages
from shpe_research.pipeline.stages import EVAL_SPLITS, ExperimentConfig, RunResult

__all__ = ["EVAL_SPLITS", "ExperimentConfig", "RunResult", "build_dataset", "load_bars", "run_experiment"]


def build_dataset(cfg: ExperimentConfig, bars: pd.DataFrame) -> Dataset:
    """Features and labels joined at each sampled decision, split chronologically by session."""
    resolved = stages.plan(cfg)
    names = [f.name for f in resolved.features]
    bars = stages.session_filter(stages.universe(bars, cfg), cfg.session)
    feats = stages.transforms(stages.features(bars, resolved.features))
    cands = stages.decisions(feats, resolved.sampler)
    rows = stages.join(feats, cands, stages.labels(bars, resolved.label), names, resolved.label.name)
    return stages.fit_scheme(rows, names, resolved.label, resolved.target, resolved.scheme)


def run_experiment(cfg: ExperimentConfig, out_root: str | Path | None = None) -> RunResult:
    return stages.run_pipeline(cfg, out_root)
