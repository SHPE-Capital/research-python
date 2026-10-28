"""TODO(M1b): LightGBM gradient-boosted trees.  Level: beginner-intermediate.

Usually the strongest model for tabular features like ours. Needs the optional
extra: `uv sync --extra ml`.

Wrap lightgbm.LGBMClassifier / LGBMRegressor like models/logistic.py
(model_type "lightgbm" / "lightgbm_reg"). Params: n_estimators, learning_rate,
num_leaves, min_child_samples, subsample, colsample_bytree, reg_lambda,
random_state=self.seed, verbose=-1. Set scale = False. Import lightgbm inside
make_estimator so the base install keeps working without the extra.

Early stopping must use a slice of the TRAINING period (the latest sessions),
never the validation split; otherwise validation stops being an honest
estimate.

Done when:
    - tests/todo/test_models_todo.py::test_m1_models["lightgbm"] passes.
    - Imported in models/__init__.py (guarded so a missing lightgbm only disables this model).
"""

from __future__ import annotations

from shpe_research.models.sklearn_base import SklearnModel


class LightGBMModel(SklearnModel):
    model_type = "lightgbm"
    task = "classification"
    scale = False

    def make_estimator(self):
        raise NotImplementedError("TODO(M1b): LightGBM")
