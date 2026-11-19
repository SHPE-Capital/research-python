import numpy as np
import pytest

from shpe_research.calibration.surrogate import SurrogateEnsemble


def test_surrogate_is_deterministic_and_reports_disagreement():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 5))
    y = x[:, 0] ** 2 - x[:, 1] + 0.1 * x[:, 2]
    query = rng.normal(size=(7, 5))

    a = SurrogateEnsemble(n_estimators=4, seed=11, min_samples_leaf=2).fit(x, y).predict(query)
    b = SurrogateEnsemble(n_estimators=4, seed=11, min_samples_leaf=2).fit(x, y).predict(query)

    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.std, b.std)
    assert a.mean.shape == a.std.shape == (7,)
    assert (a.std >= 0).all()


def test_surrogate_rejects_bad_inputs_and_predict_before_fit():
    model = SurrogateEnsemble(seed=1)
    with pytest.raises(RuntimeError):
        model.predict(np.ones((2, 3)))
    with pytest.raises(ValueError):
        model.fit(np.ones((1, 3)), np.ones(1))
    with pytest.raises(ValueError):
        SurrogateEnsemble(n_estimators=0)
