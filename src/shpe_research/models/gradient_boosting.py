"""LightGBM gradient-boosted trees (M1b).  Level: beginner-intermediate.

Usually the strongest model for tabular features like ours. Needs the optional
extra: `uv sync --extra ml`. Without it this module still imports and the
models are not registered (see models/__init__.py); building one raises
ImportError.

Config:
    model:
      type: lightgbm            # or lightgbm_reg for a regression label
      params:
        n_estimators: 300
        learning_rate: 0.05
        num_leaves: 31
        min_child_samples: 20
        subsample: 0.8          # bagging; LightGBM needs subsample_freq > 0 to use it
        subsample_freq: 1
        colsample_bytree: 0.8
        reg_lambda: 0.0
        early_stopping_rounds: 20      # optional; off by default
        early_stopping_fraction: 0.2   # share of the latest TRAINING rows held out

Early stopping uses the chronologically latest `early_stopping_fraction` of the
rows passed to fit() (by `decision_ts`), never the validation or test splits.
Those held-out rows are not used to fit the trees. With early stopping off
(the default) every training row is used.

Sample weights from the dataset are applied. Features must be NaN-free unless
the dataset allows NaN; LightGBM itself handles NaN natively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.contracts.dataset import Dataset
from shpe_research.contracts.model import register_model
from shpe_research.models.sklearn_base import SklearnModel

try:  # the `ml` extra is optional; only this module's models depend on it
    import lightgbm as _lgb
except ImportError:  # pragma: no cover - exercised only without the extra
    _lgb = None


class _LightGBMBase(SklearnModel):
    scale = False
    _estimator_name: str

    early_stopping_rows_: int = 0
    early_stopping_start_: pd.Timestamp | None = None

    def _lgb_params(self) -> dict:
        p = self.params
        return {
            "n_estimators": int(p.get("n_estimators", 300)),
            "learning_rate": float(p.get("learning_rate", 0.05)),
            "num_leaves": int(p.get("num_leaves", 31)),
            "min_child_samples": int(p.get("min_child_samples", 20)),
            "subsample": float(p.get("subsample", 1.0)),
            "subsample_freq": int(p.get("subsample_freq", 1 if float(p.get("subsample", 1.0)) < 1.0 else 0)),
            "colsample_bytree": float(p.get("colsample_bytree", 1.0)),
            "reg_lambda": float(p.get("reg_lambda", 0.0)),
            "n_jobs": int(p.get("n_jobs", 1)),
            "random_state": self.seed,
            "verbose": -1,
        }

    def make_estimator(self):
        if _lgb is None:
            raise ImportError("lightgbm is not installed; run `uv sync --extra ml`")
        return getattr(_lgb, self._estimator_name)(**self._lgb_params())

    def fit(self, dataset: Dataset):
        # _fit() only gets X, y and weight; keep the timestamps so the early-stopping
        # slice is the latest training rows, not whichever rows happen to be last.
        self._decision_ts = dataset.meta["decision_ts"]
        try:
            return super().fit(dataset)
        finally:
            del self._decision_ts

    def _fit(self, X: pd.DataFrame, y: pd.Series, weight: pd.Series) -> None:
        rounds = self.params.get("early_stopping_rounds")
        self.early_stopping_rows_ = 0
        self.early_stopping_start_ = None
        if not rounds:
            super()._fit(X, y, weight)
            return
        frac = float(self.params.get("early_stopping_fraction", 0.2))
        if not 0.0 < frac < 1.0:
            raise ValueError("early_stopping_fraction must be in (0, 1)")
        order = np.argsort(self._decision_ts.to_numpy(), kind="stable")
        n_stop = max(1, int(round(len(order) * frac)))
        if n_stop >= len(order):
            raise ValueError("not enough training rows to hold out an early-stopping slice")
        fit_idx, stop_idx = order[:-n_stop], order[-n_stop:]
        self.early_stopping_rows_ = n_stop
        self.early_stopping_start_ = self._decision_ts.iloc[stop_idx[0]]
        Xv, yv, wv = X.to_numpy(), y.to_numpy(), weight.to_numpy()
        est = self.make_estimator()
        est.fit(
            Xv[fit_idx], yv[fit_idx], sample_weight=wv[fit_idx],
            eval_set=[(Xv[stop_idx], yv[stop_idx])], eval_sample_weight=[wv[stop_idx]],
            callbacks=[_lgb.early_stopping(int(rounds), verbose=False)],
        )
        from sklearn.pipeline import Pipeline

        self.pipeline_ = Pipeline([("model", est)])


@register_model
class LightGBMModel(_LightGBMBase):
    model_type = "lightgbm"
    task = "classification"
    _estimator_name = "LGBMClassifier"


@register_model
class LightGBMRegressionModel(_LightGBMBase):
    model_type = "lightgbm_reg"
    task = "regression"
    _estimator_name = "LGBMRegressor"
