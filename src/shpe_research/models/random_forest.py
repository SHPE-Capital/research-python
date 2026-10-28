"""TODO(M1a): random forest model.  Level: beginner.

Copy models/logistic.py. Build a RandomForestClassifier or RandomForestRegressor
(two registered classes, model_type "random_forest" and "random_forest_reg")
from params: n_estimators (500), max_depth (4/8/12 in the ORB plan),
min_samples_leaf (20/50/100), max_features ("sqrt" or None), n_jobs,
random_state=self.seed. Set scale = False (trees do not need scaling).

Done when:
    - tests/todo/test_models_todo.py::test_m1_models["random_forest"] passes
      (fit, predict, save, load round trip through the Model contract).
    - Imported in models/__init__.py, and an example config exists in configs/experiments/.
"""

from __future__ import annotations

from shpe_research.models.sklearn_base import SklearnModel


class RandomForestModel(SklearnModel):
    model_type = "random_forest"
    task = "classification"
    scale = False

    def make_estimator(self):
        raise NotImplementedError("TODO(M1a): random forest")
