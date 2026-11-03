"""Random forest models (M1a): a classifier and a regressor on the Model contract.

Trees need no feature scaling, so `scale = False`. Rows with missing features
are handled upstream by the dataset builder, not here. Classifier scores are
P(y = 1) in [0, 1]; regressor scores are predicted label values (unbounded).

Config:
    model:
      type: random_forest        # or random_forest_reg
      params:
        n_estimators: 500
        max_depth: 8             # ORB plan: 4 / 8 / 12
        min_samples_leaf: 50     # ORB plan: 20 / 50 / 100
        max_features: sqrt       # sqrt | null (all features)
        n_jobs: 1
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from shpe_research.contracts.model import register_model
from shpe_research.models.sklearn_base import SklearnModel


class _RandomForestBase(SklearnModel):
    scale = False

    def _forest_kwargs(self) -> dict:
        depth = self.params.get("max_depth", 8)
        return {
            "n_estimators": int(self.params.get("n_estimators", 500)),
            "max_depth": None if depth is None else int(depth),
            "min_samples_leaf": int(self.params.get("min_samples_leaf", 50)),
            "max_features": self.params.get("max_features", "sqrt"),
            "n_jobs": self.params.get("n_jobs", 1),
            "random_state": self.seed,
        }


@register_model
class RandomForestModel(_RandomForestBase):
    model_type = "random_forest"
    task = "classification"

    def make_estimator(self):
        return RandomForestClassifier(**self._forest_kwargs())


@register_model
class RandomForestRegressorModel(_RandomForestBase):
    model_type = "random_forest_reg"
    task = "regression"

    def make_estimator(self):
        return RandomForestRegressor(**self._forest_kwargs())
