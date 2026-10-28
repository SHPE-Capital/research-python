"""Acceptance tests for the evaluation tasks. Remove a test's skip marker in the PR that implements it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import HELLO


@pytest.mark.skip(reason="TODO(E1)")
def test_trading_metrics(tmp_path):
    from shpe_research.evaluation.trading import trading_metrics
    from shpe_research.experiments.runner import ExperimentConfig, run_experiment

    cfg = ExperimentConfig.from_yaml(HELLO)
    cfg.data.update(symbols=["AAA", "BBB", "CCC", "DDD"], start="2026-04-01", end="2026-06-30")
    run = run_experiment(cfg, out_root=tmp_path)
    preds = pd.read_parquet(run.run_dir / "predictions.parquet")
    metrics = trading_metrics(preds[preds["split"] == "validation"], cost_bps=(0, 5))
    assert {"turnover", "cost_adjusted", "by_group"} <= set(metrics)
    assert metrics["cost_adjusted"][5] < metrics["cost_adjusted"][0]


@pytest.mark.skip(reason="TODO(E2)")
def test_stats():
    from shpe_research.evaluation.stats import block_bootstrap, deflated_sharpe

    rng = np.random.default_rng(1)
    daily = rng.normal(0.0005, 0.01, 750)
    ci = block_bootstrap(daily, block_length=5, n=2000, seed=0)
    assert ci["low"] < daily.mean() < ci["high"]

    noise = rng.normal(0, 0.01, (100, 750))
    sharpes = noise.mean(axis=1) / noise.std(axis=1) * np.sqrt(252)
    assert deflated_sharpe(float(sharpes.max()), n_trials=100, n_obs=750) < 0.95
