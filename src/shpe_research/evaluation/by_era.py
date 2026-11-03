"""Report section "by_era" (C1i): is the result stable across time?

One metric value per era, then the share of eras where it is positive and the worst era. An era is the `era`
column of the Dataset contract: the New York session date for intraday rows, the ISO week for daily rows. A
model whose mean is carried by two lucky eras shows up here as a low `share_positive` and a bad `worst_value`.

Input: a predictions frame with `decision_ts`, `era`, `score`, `y`, `outcome` and `split`. Every number is
computed from rows already scored out of sample; nothing is fitted. Eras with too few rows or a constant
score or outcome have no metric value (NaN); they still count in `n_eras` but not in `share_positive`, `mean`
or the worst era.

Metrics (higher is better for all of them, so "positive" and "worst" mean the same thing everywhere):
    ic          mean daily Spearman IC of score against `outcome` within the era (`report.daily_ic`)
    auc         ROC AUC of score against `y` minus 0.5 (classification)
    accuracy    accuracy of `score >= 0.5` against `y` minus the era's base rate (classification)
    neg_rmse    minus the RMSE of score against `y` (regression). Never positive, so `share_positive` is 0;
                read `mean` and `worst_value` instead.
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from shpe_research.evaluation.report import daily_ic

# The registry lives in `pipeline/`, a layer above `evaluation/`; resolved at import time as in `trading.py`.
register_report_section = importlib.import_module("shpe_research.pipeline.registry").register_report_section

METRICS = ("ic", "auc", "accuracy", "neg_rmse")


def metric_value(frame: pd.DataFrame, metric: str) -> float:
    """One higher-is-better number for `frame` (NaN when it cannot be computed).

    `frame` needs `decision_ts`, `score`, `y` and `outcome`. Timing: scores are the out-of-sample scores
    known at each row's `decision_ts`; `outcome` is the realised value after it.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {list(METRICS)}")
    if frame.empty:
        return float("nan")
    y = frame["y"].to_numpy()
    score = frame["score"].to_numpy()
    if metric == "ic":
        ic = daily_ic(frame)
        return float(ic.mean()) if len(ic) else float("nan")
    if metric == "auc":
        return float(roc_auc_score(y, score) - 0.5) if len(np.unique(y)) == 2 else float("nan")
    if metric == "accuracy":
        return float(((score >= 0.5) == (y > 0.5)).mean() - (y > 0.5).mean())
    return float(-np.sqrt(np.mean((y - score) ** 2)))


def by_era_table(frame: pd.DataFrame, metric: str = "ic") -> pd.DataFrame:
    """One row per era present in `frame`: index `era`, columns `n_rows` and `value` (NaN when undefined)."""
    rows = {era: {"n_rows": len(part), "value": metric_value(part, metric)} for era, part in frame.groupby("era")}
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["n_rows", "value"])
    table.index.name = "era"
    return table.sort_index()


def summarize_eras(table: pd.DataFrame) -> dict[str, Any]:
    """n_eras, n_valid, mean, share_positive, worst_era and worst_value from a `by_era_table`."""
    valid = table["value"].dropna()
    out: dict[str, Any] = {"n_eras": int(len(table)), "n_valid": int(len(valid))}
    if valid.empty:
        return out | {"mean": None, "share_positive": None, "worst_era": None, "worst_value": None}
    return out | {
        "mean": float(valid.mean()),
        "share_positive": float((valid > 0).mean()),
        "worst_era": str(valid.idxmin()),
        "worst_value": float(valid.min()),
    }


@register_report_section
class ByEraSection:
    """Report section "by_era": `metric` per era on `split` (validation by default) and its summary.

    Output: ({"by_era": {n_eras, n_valid, mean, share_positive, worst_era, worst_value, metric, split}}, {}).
    The per-era rows come from `by_era_table`. Opening `test` is the caller's choice, once.
    """

    section_type: ClassVar[str] = "by_era"

    def __init__(self, metric: str = "ic", split: str = "validation"):
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; choose from {list(METRICS)}")
        self.metric = metric
        self.split = split

    def evaluate(self, predictions: pd.DataFrame, task: str) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
        """Summary of the per-era metric on the chosen split; an empty split gives n_eras 0."""
        part = predictions[predictions["split"] == self.split]
        table = by_era_table(part, self.metric) if len(part) else pd.DataFrame(columns=["n_rows", "value"])
        return {"by_era": summarize_eras(table) | {"metric": self.metric, "split": self.split}}, {}

