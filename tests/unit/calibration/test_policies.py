from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.calibration.policies import (
    effective_sample_size,
    policy_weights,
    recency_weights,
    similarity_weights,
    uniform_weights,
)


def test_uniform_and_effective_sample_size():
    weights = uniform_weights(4)
    np.testing.assert_array_equal(weights, np.full(4, 0.25))
    assert effective_sample_size(weights) == pytest.approx(4.0)


def test_recency_half_life_and_cutoff_guard():
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    times = [cutoff - pd.Timedelta(days=20), cutoff - pd.Timedelta(days=10), cutoff]
    weights = recency_weights(times, cutoff, half_life_days=10)
    assert weights[2] == pytest.approx(2 * weights[1])
    assert weights[1] == pytest.approx(2 * weights[0])
    assert weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="after decision_cutoff"):
        recency_weights([cutoff + pd.Timedelta(seconds=1)], cutoff)


def test_similarity_favors_closest_regime_and_uses_constant_columns_safely():
    history = np.array([[0.0, 7.0], [1.0, 7.0], [4.0, 7.0], [8.0, 7.0]])
    weights = similarity_weights(history, np.array([4.1, 7.0]), bandwidth=1.0)
    assert int(np.argmax(weights)) == 2
    assert weights[2] > weights[1] > weights[0]
    assert weights.sum() == pytest.approx(1.0)


def test_policy_dispatch_validates_alignment_and_missing_values():
    with pytest.raises(ValueError, match="length"):
        policy_weights(
            "recency", n_observations=2, observation_times=["2026-01-01"], decision_cutoff="2026-02-01"
        )
    with pytest.raises(ValueError, match="NaN"):
        similarity_weights(np.array([[0.0], [np.nan]]), np.array([0.0]))
    with pytest.raises(ValueError, match="unknown policy"):
        policy_weights("future", n_observations=2)  # type: ignore[arg-type]


def test_similarity_is_invariant_to_future_data_not_passed_by_caller():
    history = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.4]])
    current = np.array([0.4, 0.2])
    first = similarity_weights(history, current)
    future = np.array([[1000.0, -1000.0]])
    second = similarity_weights(np.concatenate([history, future])[: len(history)], current)
    np.testing.assert_array_equal(first, second)
