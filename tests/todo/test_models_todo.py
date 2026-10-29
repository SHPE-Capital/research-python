"""Acceptance tests for the model tasks. Remove a skip marker in the PR that implements it."""

from __future__ import annotations

import numpy as np
import pytest

from shpe_research.contracts.model import load_model
from shpe_research.experiments.runner import ExperimentConfig, build_dataset
from shpe_research.experiments.sources import load_bars
from tests.conftest import HELLO


@pytest.fixture(scope="module")
def dataset():
    cfg = ExperimentConfig.from_yaml(HELLO)
    cfg.data.update(symbols=["AAA", "BBB", "CCC"], start="2026-04-01", end="2026-05-29")
    return build_dataset(cfg, load_bars(cfg.data))


@pytest.mark.parametrize(
    ("module", "cls", "params"),
    [
        pytest.param("random_forest", "RandomForestModel", {"n_estimators": 50, "max_depth": 4},
                     marks=pytest.mark.skip(reason="TODO(M1a)"), id="random_forest"),
        pytest.param("gradient_boosting", "LightGBMModel", {"n_estimators": 50},
                     marks=pytest.mark.skip(reason="TODO(M1b)"), id="lightgbm"),
        pytest.param("mlp", "MLPModel", {"hidden": [16, 8]},
                     marks=pytest.mark.skip(reason="TODO(M1c)"), id="mlp"),
    ],
)
def test_m1_models(dataset, tmp_path, module, cls, params):
    import importlib

    model_cls = getattr(importlib.import_module(f"shpe_research.models.{module}"), cls)
    train = dataset.subset("train")
    model = model_cls(seed=1, **params).fit(train)
    scores = model.predict(dataset.subset("validation").X)
    assert scores.shape == (len(dataset.subset("validation")),)
    assert np.all((scores >= 0) & (scores <= 1))
    model.save(tmp_path / "m")
    np.testing.assert_allclose(load_model(tmp_path / "m").predict(train.X), model.predict(train.X))


@pytest.mark.skip(reason="TODO(M3)")
def test_onnx_export(dataset, tmp_path):
    import onnxruntime as ort

    from shpe_research.models.export_onnx import export_onnx
    from shpe_research.models.logistic import LogisticModel

    model = LogisticModel(seed=1).fit(dataset.subset("train"))
    golden = dataset.subset("validation").X.head(200)
    out = export_onnx(model, golden, tmp_path / "export")
    session = ort.InferenceSession(str(out / "model.onnx"))
    onnx_scores = session.run(None, {session.get_inputs()[0].name: golden.to_numpy(np.float32)})
    assert onnx_scores  # compare against model.predict(golden) within tolerance in your implementation
