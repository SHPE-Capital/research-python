"""Tests for evaluation/stats.py (E2): block bootstrap, deflated Sharpe, PBO."""

from __future__ import annotations

import numpy as np
import pytest

from shpe_research.evaluation.stats import (
    block_bootstrap,
    deflated_sharpe,
    paired_block_bootstrap,
    probability_of_backtest_overfitting,
)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    daily = np.random.default_rng(1).normal(0.0005, 0.01, 750)
    a = block_bootstrap(daily, block_length=5, n=500, seed=3)
    b = block_bootstrap(daily, block_length=5, n=500, seed=3)
    assert a == b
    assert a["low"] < daily.mean() < a["high"]
    assert a["block_length"] == 5 and a["n"] == 500


def test_bootstrap_wider_with_autocorrelation_and_longer_blocks():
    rng = np.random.default_rng(0)
    e = rng.normal(size=2000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.8 * x[i - 1] + e[i]
    w1 = block_bootstrap(x, 1, n=1000, seed=0)
    w20 = block_bootstrap(x, 20, n=1000, seed=0)
    assert (w20["high"] - w20["low"]) > 1.5 * (w1["high"] - w1["low"])


def test_bootstrap_validates_inputs():
    with pytest.raises(ValueError):
        block_bootstrap(np.array([]), 2)
    with pytest.raises(ValueError):
        block_bootstrap(np.arange(10.0), 0)
    with pytest.raises(ValueError):
        block_bootstrap(np.arange(10.0), 11)
    with pytest.raises(ValueError):
        block_bootstrap(np.array([1.0, np.nan, 2.0]), 1)


def test_paired_bootstrap_resamples_difference():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.01, 500)
    model = base + 0.002 + rng.normal(0, 0.001, 500)
    res = paired_block_bootstrap(model, base, block_length=5, n=1000, seed=0)
    assert res["low"] > 0
    assert res["mean"] == pytest.approx((model - base).mean())
    with pytest.raises(ValueError):
        paired_block_bootstrap(model, base[:-1], 5)


def test_deflated_sharpe_noise_best_of_100_not_significant():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (100, 750))
    sharpes = noise.mean(axis=1) / noise.std(axis=1) * np.sqrt(252)
    assert deflated_sharpe(float(sharpes.max()), n_trials=100, n_obs=750) < 0.95


def test_deflated_sharpe_monotone_in_trials_and_sharpe():
    p1 = deflated_sharpe(2.0, n_trials=1, n_obs=750)
    p100 = deflated_sharpe(2.0, n_trials=100, n_obs=750)
    assert 0 <= p100 < p1 <= 1
    assert deflated_sharpe(4.0, n_trials=100, n_obs=750) > p100
    assert deflated_sharpe(4.0, n_trials=100, n_obs=5000) > 0.99


def test_deflated_sharpe_single_trial_is_psr_against_zero():
    assert deflated_sharpe(0.0, n_trials=1, n_obs=500) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        deflated_sharpe(1.0, n_trials=0, n_obs=500)
    with pytest.raises(ValueError):
        deflated_sharpe(1.0, n_trials=5, n_obs=1)


def test_pbo_noise_is_near_half_and_planted_skill_is_low():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, (40, 640))
    assert 0.25 < probability_of_backtest_overfitting(noise, n_splits=8) < 0.75
    skilled = noise.copy()
    skilled[7] += 0.01  # one config with a persistent edge
    assert probability_of_backtest_overfitting(skilled, n_splits=8) < 0.1


def test_pbo_validates_inputs():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros(10))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((1, 100)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((4, 100)), n_splits=7)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((4, 6)), n_splits=8)
