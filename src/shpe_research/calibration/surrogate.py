"""Bootstrap ensemble of gradient-boosted regressors used to rank candidate configs.

The ensemble yields a predicted score (mean) and a disagreement measure (std) per
candidate. Everything is seeded, so identical inputs and seed give identical output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from shpe_research.config.momentum import SurrogateSection
from shpe_research.strategies.momentum.breakout.grid import Config, GridDomain, feature_vector


@dataclass(frozen=True)
class SurrogatePrediction:
    """Ensemble output: ``mean`` and population ``std`` (ddof=0), each shape ``(m,)``."""

    mean: np.ndarray
    std: np.ndarray


class Surrogate:
    """Bootstrap ensemble of ``HistGradientBoostingRegressor`` models."""

    def __init__(self, cfg: SurrogateSection, seed: int) -> None:
        """Bind the ensemble settings and seed; nothing is fitted until ``fit``.

        Inputs: ``surrogate`` section (model count and hyperparameters) and the seed from which the
        bootstrap resamples and each model's random state are drawn. Timing: holds no data.
        """
        self._cfg = cfg
        self._seed = seed
        self._models: list[HistGradientBoostingRegressor] = []

    @property
    def fitted(self) -> bool:
        """True once ``fit`` has been called."""
        return bool(self._models)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Surrogate:
        """Fit ``n_models`` models on bootstrap resamples, replacing any previous fit.

        Inputs: ``X`` (n, d) features and ``y`` (n,) scores of already-evaluated
        configs, n >= 2. Output: self. Timing: uses only the rows passed, so the
        caller decides which evaluations are known at the fit date.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        if n < 2 or len(X) != n:
            raise ValueError("fit needs X and y with the same length >= 2")
        c = self._cfg
        models = []
        for i in range(c.n_models):
            idx = np.random.default_rng([self._seed, i]).integers(0, n, n)
            model = HistGradientBoostingRegressor(
                max_iter=c.max_iter,
                learning_rate=c.learning_rate,
                max_leaf_nodes=c.max_leaf_nodes,
                min_samples_leaf=c.min_samples_leaf,
                l2_regularization=c.l2_regularization,
                early_stopping=False,
                random_state=int(np.random.SeedSequence([self._seed, i]).generate_state(1)[0]),
            )
            models.append(model.fit(X[idx], y[idx]))
        self._models = models
        return self

    def predict(self, X: np.ndarray) -> SurrogatePrediction:
        """Predict mean score and model disagreement for candidate rows ``X`` (m, d).

        Output: ``SurrogatePrediction``; raises ``RuntimeError`` if not fitted.
        Timing: pure function of the fitted models; no data is read.
        """
        if not self._models:
            raise RuntimeError("Surrogate.predict called before fit")
        X = np.asarray(X, dtype=np.float64)
        preds = np.stack([m.predict(X) for m in self._models])
        return SurrogatePrediction(mean=preds.mean(axis=0), std=preds.std(axis=0, ddof=0))


def encode(configs: Sequence[Config], domain: GridDomain) -> np.ndarray:
    """Stack ``feature_vector`` of each config into a float64 (n, d) matrix.

    Inputs: configs and their grid domain. Output: feature matrix. Timing: depends
    only on the configs, not on any data.
    """
    return np.array([feature_vector(c, domain) for c in configs], dtype=np.float64)


def rank_usefulness(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation between predicted and next-batch actual scores.

    Inputs: equal-length arrays. Output: rho in [-1, 1], or nan if n < 3 or either
    side is constant. Timing: ``actual`` must be scores realised after the
    prediction was made; this function only compares them.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if len(predicted) < 3 or np.ptp(predicted) == 0 or np.ptp(actual) == 0:
        return float("nan")
    return float(spearmanr(predicted, actual).statistic)
