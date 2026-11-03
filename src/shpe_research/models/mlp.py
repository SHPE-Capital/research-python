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

Implemented (M1c) on the scikit-learn route, so no new dependency.

Config:
    model:
      type: mlp
      params:
        hidden: [32, 16]          # ORB plan: [16, 8] / [32, 16] / [64, 32]; ReLU units
        learning_rate: 0.001      # 1e-3 or 3e-4
        weight_decay: 0.0001      # L2 penalty (sklearn `alpha`); 1e-4 or 1e-3
        batch_size: 64
        max_epochs: 200
        patience: 20              # epochs without a new best stopping loss
        stopping_fraction: 0.2    # share of training rows used as the stopping slice

Training data handling (all inside fit(), so nothing sees validation or test rows):
    - Rows are ordered by `decision_ts`; the LAST `stopping_fraction` of them
      (by time, not at random) form the stopping slice.
    - Fit rows whose `label_end_ts` is after the slice's first `decision_ts` are
      purged, so no fitting label is resolved inside the stopping period.
    - StandardScaler is fitted on the fit rows only. Sample weights are used for
      both the gradient steps and the stopping loss (weighted log loss).
    - The weights from the epoch with the lowest stopping loss are restored.
    - The manifest params gain `stopping_slice` (start, end, row counts, best
      epoch, epochs run, best loss) so the choice is auditable.

One instance trains one seed. Average several seeds (the RF's three) in the
experiment runner (M2) rather than picking the luckiest.
Missing features are rejected upstream by the Dataset contract.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shpe_research.contracts.dataset import Dataset
from shpe_research.contracts.model import register_model
from shpe_research.models.sklearn_base import SklearnModel


@register_model
class MLPModel(SklearnModel):
    """Small ReLU network; scores are P(y = 1) in [0, 1]."""

    model_type = "mlp"
    task = "classification"

    def _settings(self) -> dict[str, Any]:
        p = self.params
        cfg = {
            "hidden": [int(h) for h in p.get("hidden", [32, 16])],
            "learning_rate": float(p.get("learning_rate", 1e-3)),
            "weight_decay": float(p.get("weight_decay", 1e-4)),
            "batch_size": int(p.get("batch_size", 64)),
            "max_epochs": int(p.get("max_epochs", 200)),
            "patience": int(p.get("patience", 20)),
            "stopping_fraction": float(p.get("stopping_fraction", 0.2)),
        }
        if not cfg["hidden"] or min(cfg["hidden"]) < 1:
            raise ValueError("mlp: hidden must be a non-empty list of positive layer sizes")
        if not 0.0 < cfg["stopping_fraction"] <= 0.5:
            raise ValueError("mlp: stopping_fraction must be in (0, 0.5]")
        if cfg["max_epochs"] < 1 or cfg["patience"] < 1 or cfg["batch_size"] < 1:
            raise ValueError("mlp: max_epochs, patience and batch_size must be at least 1")
        return cfg

    def make_estimator(self) -> MLPClassifier:
        cfg = self._settings()
        return MLPClassifier(
            hidden_layer_sizes=tuple(cfg["hidden"]),
            activation="relu",
            solver="adam",
            alpha=cfg["weight_decay"],
            batch_size=cfg["batch_size"],
            learning_rate_init=cfg["learning_rate"],
            shuffle=True,
            random_state=self.seed,
        )

    def fit(self, dataset: Dataset) -> MLPModel:
        self._fit_meta = dataset.meta
        super().fit(dataset)
        self.manifest.params = {**self.params, "stopping_slice": self._stopping_info}
        return self

    def _fit(self, X: pd.DataFrame, y: pd.Series, weight: pd.Series) -> None:
        cfg = self._settings()
        meta = self._fit_meta
        order = np.argsort(meta["decision_ts"].to_numpy(), kind="stable")
        n = len(order)
        n_stop = max(1, int(round(cfg["stopping_fraction"] * n)))
        if n - n_stop < 2:
            raise ValueError(f"mlp: {n} training rows are too few for a stopping slice")
        stop_idx, early_idx = order[n - n_stop :], order[: n - n_stop]
        slice_start = meta["decision_ts"].iloc[stop_idx[0]]
        keep = (meta["label_end_ts"].iloc[early_idx] <= slice_start).to_numpy()
        fit_idx = early_idx[keep]
        if len(fit_idx) < 2:
            raise ValueError("mlp: no fit rows left after purging against the stopping slice")

        Xv, yv, wv = X.to_numpy(float), y.to_numpy(), weight.to_numpy(float)
        scaler = StandardScaler().fit(Xv[fit_idx])
        Xf, Xs = scaler.transform(Xv[fit_idx]), scaler.transform(Xv[stop_idx])
        clf = self.make_estimator()
        classes = np.array([0, 1])
        best_loss, best_epoch, best_clf, epochs_run = np.inf, -1, None, 0
        for epoch in range(cfg["max_epochs"]):
            clf.partial_fit(Xf, yv[fit_idx], classes=classes, sample_weight=wv[fit_idx])
            epochs_run = epoch + 1
            loss = log_loss(yv[stop_idx], clf.predict_proba(Xs)[:, 1], labels=classes, sample_weight=wv[stop_idx])
            if loss < best_loss - 1e-9:
                best_loss, best_epoch, best_clf = loss, epoch, copy.deepcopy(clf)
            elif epoch - best_epoch >= cfg["patience"]:
                break
        self.pipeline_ = Pipeline([("scale", scaler), ("model", best_clf)])
        self._stopping_info = {
            "kind": "chronological: last fraction of training rows by decision_ts",
            "fraction": cfg["stopping_fraction"],
            "start": slice_start.isoformat(),
            "end": meta["decision_ts"].iloc[stop_idx[-1]].isoformat(),
            "n_fit": int(len(fit_idx)),
            "n_purged": int(len(early_idx) - len(fit_idx)),
            "n_stopping": int(n_stop),
            "best_epoch": int(best_epoch + 1),
            "epochs_run": int(epochs_run),
            "best_loss": float(best_loss),
        }
