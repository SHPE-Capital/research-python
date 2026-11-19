from __future__ import annotations

import numpy as np
import pytest

from shpe_research.calibration.objective import (
    AcceptanceRules,
    acceptance_checks,
    accepted,
    score_returns,
)


def test_weights_change_score_but_not_realized_path_diagnostics():
    returns = np.array([0.02, 0.01, 0.0, -0.02])
    original = returns.copy()
    uniform = score_returns(returns, np.ones(4))
    recent = score_returns(returns, np.array([0.05, 0.05, 0.1, 0.8]))
    assert recent.score < uniform.score
    assert recent.compounded_return == pytest.approx(uniform.compounded_return)
    assert recent.max_drawdown == pytest.approx(uniform.max_drawdown)
    assert recent.active_fraction == uniform.active_fraction == 0.75
    np.testing.assert_array_equal(returns, original)


def test_zero_return_days_count_and_constant_path_is_deterministic():
    flat = score_returns(np.zeros(10), np.ones(10))
    assert flat.score == 0.0
    assert flat.active_fraction == 0.0
    assert flat.compounded_return == 0.0
    positive = score_returns(np.full(10, 0.001), np.ones(10))
    assert np.isfinite(positive.score) and positive.score > 0


def test_acceptance_diagnostics_name_every_threshold():
    result = score_returns(np.array([0.01, 0.0, 0.01, -0.005]), np.ones(4))
    rules = AcceptanceRules(
        min_observations=4,
        min_ess=3.0,
        min_active_fraction=0.5,
        min_compounded_return=-0.1,
        max_drawdown=0.1,
    )
    checks = acceptance_checks(result, rules)
    assert {c.name for c in checks} == {
        "observations", "effective_sample_size", "active_fraction", "compounded_return", "max_drawdown"
    }
    assert accepted(result, rules)
    assert not accepted(result, AcceptanceRules(min_observations=20))


@pytest.mark.parametrize(
    "returns,weights,message",
    [
        (np.array([]), np.array([]), "non-empty"),
        (np.array([0.0, np.nan]), np.ones(2), "NaN"),
        (np.array([-1.0, 0.0]), np.ones(2), "<= -1"),
        (np.zeros(2), np.ones(3), "same length"),
    ],
)
def test_objective_rejects_invalid_or_misaligned_inputs(returns, weights, message):
    with pytest.raises(ValueError, match=message):
        score_returns(returns, weights)
