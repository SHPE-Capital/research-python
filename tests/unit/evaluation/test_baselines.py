from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.data.calendar import session_date
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.evaluation.baselines import BaselinesSection, baseline_metrics


def _predictions(seed: int = 3) -> pd.DataFrame:
    """Validation-style predictions on the C1o `reversal` fixture; score = -past return (the planted signal)."""
    bars = synthetic_bars(start="2026-03-02", end="2026-04-10", reversal={"ar": [-0.5]}, seed=seed)
    bars = bars.sort_values(["symbol", "ts"])
    logp = np.log(bars["close"])
    g = logp.groupby(bars["symbol"])
    bars["past_return_5"] = g.diff(5)
    bars["outcome"] = g.shift(-5) - logp
    bars["decision_ts"] = bars["ts"] + pd.Timedelta(minutes=1)
    bars["label_end_ts"] = bars["decision_ts"] + pd.Timedelta(minutes=5)
    rows = bars.dropna(subset=["past_return_5", "outcome"])
    rows = rows[rows["ts"].dt.minute % 10 == 0].reset_index(drop=True)
    rows["era"] = session_date(rows["decision_ts"]).dt.strftime("%Y-%m-%d")
    rows["y"] = (rows["outcome"] > 0).astype(int)
    rows["score"] = -rows["past_return_5"]
    rows["weight"] = 1.0
    eras = sorted(rows["era"].unique())
    rows["split"] = np.where(rows["era"] < eras[len(eras) // 2], "train", "validation")
    return rows[["symbol", "decision_ts", "label_end_ts", "weight", "split", "era", "outcome", "y", "score",
                 "past_return_5"]]


def test_shuffled_label_near_zero():
    preds = _predictions()
    out = baseline_metrics(preds[preds["split"] == "validation"], "regression", which=("shuffled_label",), seed=0)
    assert abs(out["shuffled_label"]["ic_mean"]) < 0.05


def test_baselines_side_by_side_with_model():
    preds = _predictions()
    section = BaselinesSection(which=("zero", "last_value", "sign_of_past_return", "shuffled_label"))
    metrics, _ = section.evaluate(preds, "classification")
    cols = metrics["baselines"]
    assert set(cols) == {"model", "zero", "last_value", "sign_of_past_return", "shuffled_label"}
    assert cols["zero"]["n_rows"] == cols["model"]["n_rows"]  # same rows as the model
    assert cols["model"]["ic_mean"] > 0.05  # planted reversal: score = -past return


def test_sign_of_past_return_is_momentum_sign():
    preds = _predictions()
    val = preds[preds["split"] == "validation"]
    out = baseline_metrics(val, "regression", which=("sign_of_past_return",))
    # reversal fixture: the past-return sign (momentum) is anti-predictive
    assert out["sign_of_past_return"]["ic_mean"] < 0


def test_last_value_keeps_every_row():
    preds = _predictions()
    val = preds[preds["split"] == "validation"]
    out = baseline_metrics(val, "regression", which=("last_value",))
    assert out["last_value"]["n_rows"] == len(val)


def test_missing_past_return_column_is_reported_not_raised():
    preds = _predictions().drop(columns=["past_return_5"])
    out = baseline_metrics(preds, "regression", which=("sign_of_past_return",))
    assert "note" in out["sign_of_past_return"]


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        BaselinesSection(which=("nope",))


def test_registered():
    from shpe_research.pipeline.registry import build_report_section

    assert isinstance(build_report_section({"type": "baselines"}), BaselinesSection)
