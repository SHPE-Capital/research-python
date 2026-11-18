"""Tests for the D2 weighting policies and robust descriptor scaling."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from shpe_research.calibration.descriptor_scaling import fit_robust_scaler, transform
from shpe_research.calibration.weights import (
    _recency_raw,
    _similarity_raw,
    effective_sample_size,
    mix_uniform,
    recency_weights,
    similarity_weights,
    stage_weights,
    uniform_weights,
)
from shpe_research.config.momentum import WeightsSection

CFG = WeightsSection()
ROWS = np.array([[1, 10, 5], [2, 20, 5], [3, 30, 5], [4, 40, 5], [5, 50, 5]], dtype=float)
LOGGER = "shpe_research.calibration.weights"


def test_recency_hand_worked():
    w = mix_uniform(_recency_raw(3, 1.0), 0.2)
    np.testing.assert_allclose(w, [0.18095238095, 0.29523809524, 0.52380952381], rtol=1e-9)
    assert effective_sample_size(w) == pytest.approx(2.536231884057971)


def test_scaler_hand_worked():
    s = fit_robust_scaler(ROWS, iqr_floor=1e-12)
    np.testing.assert_allclose(s.medians, [3, 30, 5])
    np.testing.assert_allclose(s.iqrs, [2, 20, 0])
    assert s.kept.tolist() == [True, True, False]
    np.testing.assert_allclose(transform(s, ROWS[:1]), [[-1, -1]])


def test_scaler_rejects_non_finite():
    bad = ROWS.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_robust_scaler(bad, iqr_floor=1e-12)


def test_similarity_hand_worked():
    s = fit_robust_scaler(ROWS, iqr_floor=1e-12)
    cur = np.array([5.0, 50.0, 5.0])
    d2 = np.sum((transform(s, ROWS) - transform(s, cur[None, :])) ** 2, axis=1)
    np.testing.assert_allclose(d2, [8, 4.5, 2, 0.5, 0])
    w = mix_uniform(_similarity_raw(d2, 1.0), 0.2)
    np.testing.assert_allclose(w, [0.04645373, 0.07713864, 0.16962658, 0.31441947, 0.39236158], atol=1e-8)
    assert effective_sample_size(w) == pytest.approx(3.451981262251534)


@pytest.mark.parametrize("n", [63, 250, 1260])
def test_recency_hits_ess_target(n):
    r = recency_weights(n, CFG)
    assert abs(r.ess - 0.6 * n) <= 0.01 * n
    assert r.target_reached
    assert r.policy_used == "recency"
    assert r.weights.sum() == pytest.approx(1.0)


def test_similarity_hits_ess_target():
    rows = np.random.default_rng(0).normal(size=(250, 3))
    s = fit_robust_scaler(rows, iqr_floor=1e-12)
    r = similarity_weights(rows, np.array([0.5, -0.2, 1.0]), s, CFG)
    assert abs(r.ess - 0.6 * 250) <= 0.01 * 250
    assert r.target_reached
    assert r.policy_used == "similarity"


def test_scaling_never_sees_validation():
    rng = np.random.default_rng(1)
    search = rng.normal(size=(200, 3))
    val = rng.normal(size=(60, 3))
    cur = np.array([0.1, 0.2, 0.3])
    s = fit_robust_scaler(search, iqr_floor=1e-12)
    ws = similarity_weights(search, cur, s, CFG)
    wv = similarity_weights(val, cur, s, CFG)
    val2 = val * 100 + 7
    s2 = fit_robust_scaler(search, iqr_floor=1e-12)
    ws2 = similarity_weights(search, cur, s2, CFG)
    wv2 = similarity_weights(val2, cur, s2, CFG)
    np.testing.assert_array_equal(s.medians, s2.medians)
    np.testing.assert_array_equal(s.iqrs, s2.iqrs)
    assert ws.weights.tobytes() == ws2.weights.tobytes()
    assert not np.array_equal(wv.weights, wv2.weights)


def test_unreachable_target_uses_closest_and_logs(caplog):
    s = fit_robust_scaler(np.random.default_rng(2).normal(size=(50, 2)), iqr_floor=1e-12)
    cur = np.array([0.3, 0.3])
    rows = np.tile(cur, (20, 1))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        r = similarity_weights(rows, cur, s, CFG)
    np.testing.assert_allclose(r.weights, 1 / 20)
    assert r.target_reached is False
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_target_outside_bracket_uses_bound(caplog):
    cfg = WeightsSection(solver_bracket=(1e-3, 2e-3))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        r = recency_weights(100, cfg)
    assert r.parameter == pytest.approx(0.2)
    assert r.target_reached is False
    assert r.ess < 60
    msgs = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("60.0000" in m and f"{r.ess:.4f}" in m for m in msgs)


def test_invalid_falls_back_to_uniform():
    s = fit_robust_scaler(ROWS, iqr_floor=1e-12)
    cur = np.array([np.nan, 50.0, 5.0])
    r = similarity_weights(ROWS, cur, s, CFG)
    assert r.policy_used == "uniform"
    assert r.policy_requested == "similarity"
    assert r.fallback_reason == "non_finite_descriptors"
    np.testing.assert_array_equal(r.weights, np.full(5, 1 / 5))


def test_all_descriptors_dropped_falls_back():
    const = np.ones((10, 2))
    s = fit_robust_scaler(const, iqr_floor=1e-12)
    r = stage_weights("similarity", 10, CFG, stage_rows=const, current=np.ones(2), scaler=s)
    assert r.fallback_reason == "all_descriptors_zero_scale"
    assert r.policy_used == "uniform"


def test_uniform_policy():
    r = uniform_weights(5, CFG)
    assert r.ess == 5
    assert r.parameter is None
    assert stage_weights("uniform", 5, CFG).ess == 5


def test_similarity_requires_inputs():
    with pytest.raises(ValueError):
        stage_weights("similarity", 5, CFG)


def test_deterministic():
    a, b = recency_weights(250, CFG), recency_weights(250, CFG)
    assert np.array_equal(a.weights, b.weights)
    assert a.parameter == b.parameter
