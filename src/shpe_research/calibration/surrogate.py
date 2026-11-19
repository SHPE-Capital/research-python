"""Deterministic surrogate ensemble for the finite momentum configuration grid.

The surrogate only prioritises configurations for an actual account replay.  Its
predictions are never accepted as observed performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class SurrogatePrediction:
    """Ensemble prediction mean and disagreement for candidate feature rows."""

    mean: np.ndarray
    std: np.ndarray


class SurrogateEnsemble:
    """Bootstrap ensemble of histogram gradient-boosting regressors.

    Inputs are configuration feature vectors and *actual* historical objective
    scores known at the calibration cutoff.  ``predict`` returns only proposal
    metadata; downstream selection must evaluate nominated configurations.
    """

    def __init__(
        self,
        n_estimators: int = 5,
        *,
        seed: int = 0,
        max_iter: int = 100,
        learning_rate: float = 0.08,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 5,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.model_params = {
            "max_iter": int(max_iter),
            "learning_rate": float(learning_rate),
            "max_leaf_nodes": int(max_leaf_nodes),
            "min_samples_leaf": int(min_samples_leaf),
        }
        self.models: list[HistGradientBoostingRegressor] = []

    def fit(self, features: np.ndarray, scores: np.ndarray) -> SurrogateEnsemble:
        """Fit on actual replay scores available at the current cutoff."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(scores, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("features must be 2-D and scores must be a matching 1-D array")
        if len(y) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("surrogate training needs at least two finite observations")
        rng = np.random.default_rng(self.seed)
        self.models = []
        for i in range(self.n_estimators):
            sample = rng.integers(0, len(y), size=len(y))
            model = HistGradientBoostingRegressor(random_state=self.seed + i, **self.model_params)
            model.fit(x[sample], y[sample])
            self.models.append(model)
        return self

    def predict(self, features: np.ndarray) -> SurrogatePrediction:
        """Return ensemble mean and disagreement for unevaluated feature rows."""
        if not self.models:
            raise RuntimeError("fit must be called before predict")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("features must be a finite 2-D array")
        predictions = np.vstack([model.predict(x) for model in self.models])
        return SurrogatePrediction(predictions.mean(axis=0), predictions.std(axis=0))
