"""Tests for the D4 bootstrap-ensemble surrogate and the usefulness metric."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from shpe_research.calibration.surrogate import Surrogate, encode, rank_usefulness
from shpe_research.config.momentum import SurrogateSection
from shpe_research.strategies.momentum.breakout.grid import (
    DEFAULT_DOMAIN,
    enumerate_configs,
    feature_vector,
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, (400, 18)).astype(float)
    coefs = rng.normal(size=18)
    y = x @ coefs + 0.1 * rng.normal(size=400)
    return x, y


def _fit(seed: int = 7, cfg: SurrogateSection | None = None) -> tuple[Surrogate, np.ndarray, np.ndarray]:
    x, y = _data()
    s = Surrogate(cfg or SurrogateSection(), seed).fit(x[:336], y[:336])
    return s, x, y


def test_fits_and_ranks_synthetic() -> None:
    s, x, y = _fit()
    pred = s.predict(x[336:])
    assert rank_usefulness(pred.mean, y[336:]) > 0.8


def test_same_seed_identical_different_seed_differs() -> None:
    a = _fit(7)[0].predict(_data()[0][336:])
    b = _fit(7)[0].predict(_data()[0][336:])
    c = _fit(8)[0].predict(_data()[0][336:])
    assert np.array_equal(a.mean, b.mean)
    assert np.array_equal(a.std, b.std)
    assert not np.array_equal(a.mean, c.mean)


def test_disagreement_nonnegative_and_informative() -> None:
    s, x, _ = _fit()
    pred = s.predict(x[336:])
    assert pred.mean.shape == pred.std.shape == (64,)
    assert (pred.std >= 0).all()
    assert pred.std.max() > 0


def test_hyperparameters_from_config() -> None:
    s, _, _ = _fit()
    assert len(s._models) == 5
    for m in s._models:
        assert m.max_iter == 200
        assert m.learning_rate == 0.05
        assert m.max_leaf_nodes == 15
        assert m.min_samples_leaf == 20
        assert m.l2_regularization == 1.0
        assert m.early_stopping is False
    s3, _, _ = _fit(cfg=SurrogateSection(n_models=3))
    assert len(s3._models) == 3


def test_refit_replaces() -> None:
    x, y = _data()
    s = Surrogate(SurrogateSection(), 7).fit(x[:200], y[:200])
    s.fit(x[200:], y[200:])
    fresh = Surrogate(SurrogateSection(), 7).fit(x[200:], y[200:])
    p, q = s.predict(x[:50]), fresh.predict(x[:50])
    assert np.array_equal(p.mean, q.mean)
    assert np.array_equal(p.std, q.std)


def test_rank_usefulness_hand_worked() -> None:
    assert rank_usefulness(np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 40.0])) == pytest.approx(1.0)
    assert rank_usefulness(np.array([1, 2, 3, 4.0]), np.array([40, 30, 20, 10.0])) == pytest.approx(-1.0)
    assert np.isnan(rank_usefulness(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    assert np.isnan(rank_usefulness(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))


def test_encode_uses_feature_vector() -> None:
    cfgs = list(itertools.islice(enumerate_configs(DEFAULT_DOMAIN), 3))
    out = encode(cfgs, DEFAULT_DOMAIN)
    assert out.dtype == np.float64
    assert out.shape[0] == 3
    assert out[0].tolist() == feature_vector(cfgs[0])


def test_predict_before_fit_raises() -> None:
    s = Surrogate(SurrogateSection(), 1)
    assert not s.fitted
    with pytest.raises(RuntimeError):
        s.predict(np.zeros((2, 18)))
