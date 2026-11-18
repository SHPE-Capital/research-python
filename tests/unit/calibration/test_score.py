"""Hand-worked tests for the common score J."""

from __future__ import annotations

import numpy as np
import pytest

from shpe_research.calibration.score import normalize_weights, score_j, score_terms

R = [0.01, -0.02, 0.0, 0.03]


def test_uniform_hand_worked() -> None:
    t = score_terms(R, None, downside_coeff=0.5)
    assert t.mean == pytest.approx(0.005, abs=1e-15)
    assert t.downside == pytest.approx(0.01, abs=1e-15)
    assert t.j == pytest.approx(0.0, abs=1e-15)


def test_weighted_hand_worked_and_normalized() -> None:
    for w in ([0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]):
        t = score_terms(R, w, downside_coeff=0.5)
        assert t.mean == pytest.approx(0.009)
        assert t.downside == pytest.approx(0.00894427190999916)
        assert t.j == pytest.approx(0.0045278640450004196)


def test_uniform_reproduces_plain_mean_and_downside() -> None:
    r = np.random.default_rng(0).normal(0.0005, 0.01, 250)
    expected = r.mean() - 0.5 * np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    assert score_j(r, None, downside_coeff=0.5) == pytest.approx(expected, abs=1e-15)


def test_coefficient_from_argument() -> None:
    assert score_j(R, None, downside_coeff=1.0) == pytest.approx(0.005 - 0.01, abs=1e-15)


def test_zero_days_count() -> None:
    base = score_terms(R, None, downside_coeff=0.5).mean
    padded = score_terms([*R, 0.0, 0.0], None, downside_coeff=0.5).mean
    assert padded != pytest.approx(base)
    assert padded == pytest.approx(0.02 / 6)


def test_normalize_weights_uniform_and_scaled() -> None:
    assert normalize_weights(None, 4) == pytest.approx([0.25] * 4)
    assert normalize_weights([1, 3], 2) == pytest.approx([0.25, 0.75])


@pytest.mark.parametrize(
    ("returns", "weights"),
    [
        ([], None),
        ([0.01, float("nan")], None),
        ([[0.01, 0.02]], None),
        ([0.01, 0.02], [1.0, -1.0]),
        ([0.01, 0.02], [0.0, 0.0]),
        ([0.01, 0.02], [1.0]),
        ([0.01, 0.02], [1.0, float("inf")]),
    ],
)
def test_invalid_inputs(returns: object, weights: object) -> None:
    with pytest.raises(ValueError):
        score_terms(returns, weights, downside_coeff=0.5)  # type: ignore[arg-type]
