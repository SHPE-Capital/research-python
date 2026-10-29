"""Logistic regression (C0): the hello-world model and the baseline every learned model must beat.

Config:
    model:
      type: logistic
      params: {C: 1.0}
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from shpe_research.contracts.model import register_model
from shpe_research.models.sklearn_base import SklearnModel


@register_model
class LogisticModel(SklearnModel):
    model_type = "logistic"
    task = "classification"

    def make_estimator(self):
        return LogisticRegression(
            C=float(self.params.get("C", 1.0)),
            max_iter=int(self.params.get("max_iter", 1000)),
            random_state=self.seed,
        )
