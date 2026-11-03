"""MLP (M1c): chronological early-stopping slice, determinism, manifest documentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.contracts.model import ModelManifest, build_model, load_model
from shpe_research.experiments.runner import ExperimentConfig, build_dataset
from shpe_research.experiments.sources import load_bars
from tests.conftest import HELLO


@pytest.fixture(scope="module")
def train():
    cfg = ExperimentConfig.from_yaml(HELLO)
    cfg.data.update(symbols=["AAA", "BBB", "CCC"], start="2026-04-01", end="2026-05-29")
    return build_dataset(cfg, load_bars(cfg.data)).subset("train")


def test_registered_and_deterministic(train):
    a = build_model("mlp", seed=3, hidden=[8, 4], max_epochs=5).fit(train)
    b = build_model("mlp", seed=3, hidden=[8, 4], max_epochs=5).fit(train)
    np.testing.assert_allclose(a.predict(train.X), b.predict(train.X))


def test_stopping_slice_is_chronological_and_documented(train, tmp_path):
    model = build_model("mlp", seed=1, hidden=[8, 4], max_epochs=5, stopping_fraction=0.2).fit(train)
    info = model.manifest.params["stopping_slice"]
    decisions = train.meta["decision_ts"]
    slice_start = pd.Timestamp(info["start"])
    n_slice = info["n_stopping"]
    assert n_slice == pytest.approx(0.2 * len(train), abs=1)
    # the stopping rows are the latest n rows by decision_ts
    assert (decisions >= slice_start).sum() >= n_slice
    assert info["n_fit"] + info["n_purged"] + n_slice == len(train)
    assert info["epochs_run"] <= 5
    model.save(tmp_path / "m")
    assert ModelManifest.read(tmp_path / "m").params["stopping_slice"] == info
    np.testing.assert_allclose(load_model(tmp_path / "m").predict(train.X), model.predict(train.X))


def test_fit_rows_never_overlap_stopping_slice(train):
    model = build_model("mlp", seed=1, hidden=[8], max_epochs=2).fit(train)
    info = model.manifest.params["stopping_slice"]
    fit_rows = train.meta[train.meta["decision_ts"] < pd.Timestamp(info["start"])]
    kept = fit_rows[fit_rows["label_end_ts"] <= pd.Timestamp(info["start"])]
    assert len(kept) == info["n_fit"]


def test_rejects_bad_params(train):
    with pytest.raises(ValueError):
        build_model("mlp", stopping_fraction=0.9).fit(train)
