from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.contracts.dataset import Dataset
from shpe_research.contracts.model import build_model
from shpe_research.data.calendar import session_date
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.evaluation.importance import PermutationImportanceSection, permutation_importance


def _fit(seed: int = 3):
    """C1o `reversal` fixture: ret1 is the planted feature, ret30 a weak one, noise is pure noise."""
    bars = synthetic_bars(start="2026-03-02", end="2026-04-24", reversal={"ar": [-0.5]}, seed=seed)
    bars = bars.sort_values(["symbol", "ts"])
    logp = np.log(bars["close"])
    g = logp.groupby(bars["symbol"])
    bars["ret1"] = g.diff(1)
    bars["ret30"] = g.diff(30)
    bars["outcome"] = g.shift(-1) - logp
    bars["decision_ts"] = bars["ts"] + pd.Timedelta(minutes=1)
    bars["label_end_ts"] = bars["decision_ts"] + pd.Timedelta(minutes=1)
    rows = bars.dropna(subset=["ret1", "ret30", "outcome"])
    rows = rows[rows["ts"].dt.minute % 5 == 0].reset_index(drop=True)
    rows["noise"] = np.random.default_rng(0).normal(size=len(rows))
    era = session_date(rows["decision_ts"]).dt.strftime("%Y-%m-%d")
    days = sorted(era.unique())
    cut_a, cut_b = days[int(len(days) * 0.6)], days[int(len(days) * 0.8)]
    split = np.where(era < cut_a, "train", np.where(era < cut_b, "validation", "test"))
    meta = rows[["symbol", "decision_ts", "label_end_ts", "outcome"]].assign(weight=1.0, split=split, era=era)
    ds = Dataset(X=rows[["ret1", "ret30", "noise"]], y=(rows["outcome"] > 0).astype(int).rename("y"), meta=meta,
                 label_name="y")
    model = build_model("logistic", seed=0).fit(ds.subset("train"))
    return ds, model


def test_planted_feature_ranks_first():
    ds, model = _fit()
    table = permutation_importance(model, ds, split="validation", n_repeats=5, seed=0)
    assert table["mean_drop"].idxmax() == "ret1"
    assert table.loc["ret1", "mean_drop"] > 0


def test_noise_feature_near_zero():
    ds, model = _fit()
    table = permutation_importance(model, ds, split="validation", n_repeats=5, seed=0)
    row = table.loc["noise"]
    assert abs(row["mean_drop"]) <= 2 * row["std_drop"]


def test_does_not_mutate_inputs_or_refit():
    ds, model = _fit()
    before = ds.X.copy()
    scores = model.predict(ds.X.iloc[:50]).copy()
    permutation_importance(model, ds, split="validation", n_repeats=2, seed=1)
    pd.testing.assert_frame_equal(ds.X, before)
    np.testing.assert_array_equal(scores, model.predict(ds.X.iloc[:50]))


def test_permutes_within_era_on_validation_rows_only():
    ds, model = _fit()
    seen = []

    class Spy:
        task = model.task

        def predict(self, X):
            seen.append(X.copy())
            return model.predict(X)

    permutation_importance(Spy(), ds, split="validation", n_repeats=1, seed=0, features=["noise"])
    val = ds.subset("validation")
    permuted = seen[-1]
    assert permuted.index.equals(val.X.index)
    for _, idx in val.meta.groupby("era").groups.items():
        assert sorted(permuted.loc[idx, "noise"]) == sorted(val.X.loc[idx, "noise"])


def test_section_needs_bind_and_reports():
    ds, model = _fit()
    section = PermutationImportanceSection(n_repeats=2, seed=0)
    preds = ds.meta.assign(y=ds.y, score=model.predict(ds.X))
    metrics, _ = section.evaluate(preds, model.task)
    assert "note" in metrics["permutation_importance"]
    section.bind(ds, model)
    metrics, _ = section.evaluate(preds, model.task)
    assert set(metrics["permutation_importance"]) == {"ret1", "ret30", "noise"}


def test_registered():
    from shpe_research.pipeline.registry import build_report_section

    assert isinstance(build_report_section({"type": "permutation_importance"}), PermutationImportanceSection)
