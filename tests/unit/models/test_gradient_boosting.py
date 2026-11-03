"""LightGBM model (M1b): regression twin, determinism, and train-only early stopping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from shpe_research.contracts.dataset import Dataset  # noqa: E402
from shpe_research.contracts.model import MODEL_REGISTRY, build_model, load_model  # noqa: E402


def _dataset(n: int = 400, binary: bool = True) -> Dataset:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    signal = X["a"] - 0.5 * X["b"] + rng.normal(scale=0.5, size=n)
    y = pd.Series((signal > 0).astype(int) if binary else signal, name="y")
    decision = pd.Series(
        pd.Timestamp("2026-04-01T14:00:00Z") + pd.to_timedelta(np.arange(n) * 30, unit="min")
    ).astype("datetime64[ns, UTC]")
    meta = pd.DataFrame(
        {
            "symbol": "AAA",
            "decision_ts": decision,
            "label_end_ts": decision + pd.Timedelta(minutes=5),
            "weight": 1.0,
            "split": "train",
            "era": decision.dt.strftime("%Y-%m-%d"),
        }
    )
    return Dataset(X=X, y=y, meta=meta, label_name="y")


def test_registered_keys():
    import shpe_research.models  # noqa: F401

    assert {"lightgbm", "lightgbm_reg"} <= set(MODEL_REGISTRY)


def test_same_seed_same_scores():
    ds = _dataset()
    a = build_model("lightgbm", seed=3, n_estimators=20, subsample=0.8, subsample_freq=1).fit(ds).predict(ds.X)
    b = build_model("lightgbm", seed=3, n_estimators=20, subsample=0.8, subsample_freq=1).fit(ds).predict(ds.X)
    np.testing.assert_allclose(a, b)


def test_regression_round_trip(tmp_path):
    ds = _dataset(binary=False)
    model = build_model("lightgbm_reg", seed=1, n_estimators=20).fit(ds)
    model.save(tmp_path / "m")
    np.testing.assert_allclose(load_model(tmp_path / "m").predict(ds.X), model.predict(ds.X))


def test_early_stopping_uses_latest_training_rows_only():
    ds = _dataset(n=600)
    model = build_model("lightgbm", seed=1, n_estimators=500, learning_rate=0.3,
                        early_stopping_rounds=5, early_stopping_fraction=0.2)
    model.fit(ds)
    est = model.pipeline_.named_steps["model"]
    assert est.n_estimators_ < 500
    # the held-out slice is the chronologically last 20% of the training rows
    assert model.early_stopping_rows_ == 120
    assert model.early_stopping_start_ == ds.meta["decision_ts"].sort_values().iloc[480]


def test_early_stopping_off_by_default():
    ds = _dataset()
    model = build_model("lightgbm", seed=1, n_estimators=15).fit(ds)
    assert model.pipeline_.named_steps["model"].n_estimators_ == 15
    assert model.early_stopping_rows_ == 0
