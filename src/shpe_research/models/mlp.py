"""TODO(M1c): small feed-forward neural network.  Level: intermediate.

ORB plan defaults: hidden layers 16->8, 32->16 or 64->32, ReLU, one output,
Adam, learning rate 1e-3 or 3e-4, weight decay 1e-4 or 1e-3, batch size 64,
up to 200 epochs, patience 20 on a chronological stopping slice taken from
the END of the training period, restoring the best checkpoint. Training-only
standardization (inside the model).

Two acceptable routes:
    - sklearn.neural_network.MLPClassifier/MLPRegressor via SklearnModel (no new
      dependency; note its early_stopping picks a RANDOM validation slice, so
      implement the chronological slice yourself), or
    - PyTorch (`uv sync --extra nn`), implementing _fit/_predict/_save_state/
      _load_state directly on Model. Use the same three seeds as the RF and
      average them (see M2) rather than picking the luckiest.

Done when:
    - tests/todo/test_models_todo.py::test_m1_models["mlp"] passes.
    - The stopping slice is chronological and documented in the manifest params.
"""

from __future__ import annotations

from shpe_research.models.sklearn_base import SklearnModel


class MLPModel(SklearnModel):
    model_type = "mlp"
    task = "classification"

    def make_estimator(self):
        raise NotImplementedError("TODO(M1c): small neural network")
