"""Base for models backed by a scikit-learn estimator.

Subclasses only implement `make_estimator()`. The estimator is wrapped in a
Pipeline whose StandardScaler (when `scale` is true) is fitted on the training
rows inside fit(), so scaling never sees validation or test data.

State is saved with joblib, which is pickle: only load model directories you
or the club produced.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shpe_research.contracts.model import Model

_STATE_FILE = "model.joblib"


class SklearnModel(Model):
    scale: ClassVar[bool] = True

    @abstractmethod
    def make_estimator(self):
        """A fresh, unfitted scikit-learn estimator built from self.params and self.seed."""

    def _pipeline(self) -> Pipeline:
        steps = [("scale", StandardScaler())] if self.scale else []
        return Pipeline([*steps, ("model", self.make_estimator())])

    def _fit(self, X: pd.DataFrame, y: pd.Series, weight: pd.Series) -> None:
        self.pipeline_ = self._pipeline()
        self.pipeline_.fit(X.to_numpy(), y.to_numpy(), model__sample_weight=weight.to_numpy())

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        values = X.to_numpy()
        if self.task == "classification":
            return self.pipeline_.predict_proba(values)[:, 1]
        return self.pipeline_.predict(values)

    def _save_state(self, directory: Path) -> None:
        joblib.dump(self.pipeline_, directory / _STATE_FILE)

    def _load_state(self, directory: Path) -> None:
        self.pipeline_ = joblib.load(directory / _STATE_FILE)
