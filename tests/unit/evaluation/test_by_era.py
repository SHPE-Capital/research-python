from __future__ import annotations

import pandas as pd

from shpe_research.evaluation.by_era import ByEraSection, by_era_table
from tests.unit.evaluation.test_baselines import _predictions


def test_era_counts():
    preds = _predictions()
    val = preds[preds["split"] == "validation"]
    table = by_era_table(val, "ic")
    assert len(table) == val["era"].nunique()
    assert set(table.index) == set(val["era"])
    assert table["n_rows"].sum() == len(val)


def test_section_summary():
    preds = _predictions()
    metrics, _ = ByEraSection(metric="ic").evaluate(preds, "classification")
    s = metrics["by_era"]
    assert s["n_eras"] == preds.loc[preds["split"] == "validation", "era"].nunique()
    assert 0.0 <= s["share_positive"] <= 1.0
    assert s["worst_value"] <= s["mean"]
    assert isinstance(s["worst_era"], str)


def test_hand_checked_share_positive():
    n = 6
    ts = pd.to_datetime(["2026-03-02T15:00Z"] * n + ["2026-03-03T15:00Z"] * n, utc=True)
    frame = pd.DataFrame({
        "decision_ts": ts, "symbol": list("ABCDEF") * 2, "split": "validation",
        "era": ["2026-03-02"] * n + ["2026-03-03"] * n,
        "score": list(range(n)) * 2,
        "outcome": list(range(n)) + list(range(n, 0, -1)),
    })
    frame["y"] = (frame["outcome"] > 0).astype(int)
    table = by_era_table(frame, "ic")
    assert table.loc["2026-03-02", "value"] == 1.0 and table.loc["2026-03-03", "value"] == -1.0
    s = ByEraSection(metric="ic").evaluate(frame, "regression")[0]["by_era"]
    assert s["share_positive"] == 0.5 and s["worst_era"] == "2026-03-03"


def test_registered():
    from shpe_research.pipeline.registry import build_report_section

    assert isinstance(build_report_section({"type": "by_era"}), ByEraSection)
