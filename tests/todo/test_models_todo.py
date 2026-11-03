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
                     id="random_forest"),
        pytest.param("gradient_boosting", "LightGBMModel", {"n_estimators": 50},
                     marks=pytest.mark.skip(reason="TODO(M1b)"), id="lightgbm"),
        pytest.param("mlp", "MLPModel", {"hidden": [16, 8]}, id="mlp"),
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


def test_onnx_export(dataset, tmp_path):
    import json

    import onnxruntime as ort
    import pandas as pd

    from shpe_research.models.export_onnx import export_onnx
    from shpe_research.models.logistic import LogisticModel

    model = LogisticModel(seed=1).fit(dataset.subset("train"))
    golden = dataset.subset("validation").X.head(200)
    out = export_onnx(model, golden, tmp_path / "export")
    assert {p.name for p in out.iterdir()} >= {"model.onnx", "manifest.json", "golden.parquet"}
    session = ort.InferenceSession(str(out / "model.onnx"))
    onnx_scores = session.run(None, {session.get_inputs()[0].name: golden.to_numpy(np.float32)})
    assert onnx_scores
    expected = model.predict(golden)
    probs = onnx_scores[1]
    got = np.array([row[1] for row in probs]) if isinstance(probs, list) else np.asarray(probs)[:, 1]
    np.testing.assert_allclose(got, expected, atol=1e-5)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["feature_names"] == list(model.manifest.feature_names)
    assert len(manifest["feature_dtypes"]) == len(manifest["feature_names"])
    saved = pd.read_parquet(out / "golden.parquet")
    assert len(saved) == len(golden)
    np.testing.assert_allclose(saved["score"].to_numpy(), expected)


def test_onnx_export_fails_outside_tolerance(dataset, tmp_path):
    from shpe_research.models.export_onnx import export_onnx
    from shpe_research.models.logistic import LogisticModel

    model = LogisticModel(seed=1).fit(dataset.subset("train"))
    golden = dataset.subset("validation").X.head(50)
    with pytest.raises(ValueError, match="tolerance"):
        export_onnx(model, golden, tmp_path / "export", tolerance=-1.0)


def test_onnx_export_cli(tmp_path):
    from shpe_research.cli import main
    from shpe_research.experiments.runner import ExperimentConfig, run_experiment

    cfg = ExperimentConfig.from_yaml(HELLO)
    run = run_experiment(cfg, out_root=tmp_path).run_dir
    assert main(["export", str(run / "model")]) == 0
    assert (run / "export" / "model.onnx").exists()
    assert (run / "export" / "golden.parquet").exists()


def test_random_forest_regressor_round_trip(dataset, tmp_path):
    from shpe_research.models.random_forest import RandomForestRegressorModel

    train = dataset.subset("train")
    model = RandomForestRegressorModel(seed=1, n_estimators=20, max_depth=3).fit(train)
    assert model.predict(train.X).shape == (len(train),)
    model.save(tmp_path / "r")
    np.testing.assert_allclose(load_model(tmp_path / "r").predict(train.X), model.predict(train.X))
