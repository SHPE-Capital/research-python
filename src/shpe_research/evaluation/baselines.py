"""Report section "baselines" (C1i): better than what?

Scores the same rows with the same metrics (`report.evaluate`) as the model, using scores that carry no
model information. A model that does not clearly beat these has learned nothing worth trading.

    zero                 a constant score (0.5 for classification, 0.0 for regression): no information.
    last_value           per symbol, the most recent label already *known* at the row's decision time
                         (`label_end_ts <= decision_ts`), not the previous row's label, which may still be
                         unresolved. Rows with no known label yet score as "no information".
    sign_of_past_return  the sign of a past-return column of the predictions frame (momentum convention:
                         +1 after a rise). Needs a `past_return*` column in the frame; if there is none the
                         baseline reports a `note` instead of a number.
    shuffled_label       the label `y` permuted across rows within each era (seeded). It has the label's
                         marginal distribution and no relation to the row, so its IC is a noise floor.

Timing: every baseline uses only values known at the row's `decision_ts`, except `shuffled_label`, which is a
null control and uses labels by construction.

Input: predictions with `decision_ts`, `symbol`, `split`, `era`, `score`, `y` and `outcome`; `label_end_ts` is
needed for `last_value`. The section scores one split (validation by default) so the test split stays closed.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from shpe_research.evaluation.report import evaluate

register_report_section = importlib.import_module("shpe_research.pipeline.registry").register_report_section

BASELINES = ("zero", "last_value", "sign_of_past_return", "shuffled_label")


def _neutral(task: str) -> float:
    return 0.5 if task == "classification" else 0.0


def _as_score(values: pd.Series, task: str) -> pd.Series:
    """A signed value (return or sign) as a score: positive -> 1.0 and otherwise 0.0 for classification."""
    return (values > 0).astype(float) if task == "classification" else values.astype(float)


def _zero(frame: pd.DataFrame, task: str, **_: Any) -> pd.Series:
    return pd.Series(_neutral(task), index=frame.index)


def _last_value(frame: pd.DataFrame, task: str, **_: Any) -> pd.Series | str:
    if "label_end_ts" not in frame or "outcome" not in frame:
        return "needs label_end_ts and outcome columns"
    known = frame[["symbol", "label_end_ts", "outcome"]].sort_values("label_end_ts").rename(
        columns={"label_end_ts": "known_ts", "outcome": "last"}
    )
    left = frame[["symbol", "decision_ts"]].assign(_row=frame.index).sort_values("decision_ts")
    merged = pd.merge_asof(left, known, left_on="decision_ts", right_on="known_ts", by="symbol")
    last = merged.set_index("_row")["last"].reindex(frame.index)
    score = _as_score(last.fillna(0.0), task)
    return score.where(last.notna(), _neutral(task))


def _sign_of_past_return(frame: pd.DataFrame, task: str, past_return_col: str | None = None, **_: Any):
    col = past_return_col or next((c for c in frame.columns if str(c).startswith("past_return")), None)
    if col is None or col not in frame:
        return "no past_return column in the predictions frame"
    sign = np.sign(frame[col].astype(float)).fillna(0.0)
    return (sign + 1.0) / 2.0 if task == "classification" else sign


def _shuffled_label(frame: pd.DataFrame, task: str, seed: int = 0, **_: Any) -> pd.Series:
    rng = np.random.default_rng(seed)
    out = pd.Series(np.nan, index=frame.index)
    for _, idx in frame.groupby("era").groups.items():
        out.loc[idx] = rng.permutation(frame.loc[idx, "y"].to_numpy(dtype=float))
    return out


_BUILDERS = {
    "zero": _zero,
    "last_value": _last_value,
    "sign_of_past_return": _sign_of_past_return,
    "shuffled_label": _shuffled_label,
}


def baseline_metrics(
    frame: pd.DataFrame,
    task: str,
    which: Sequence[str] = BASELINES,
    seed: int = 0,
    past_return_col: str | None = None,
) -> dict[str, dict[str, Any]]:
    """`report.evaluate` metrics of each baseline on the rows of `frame`, keyed by baseline name.

    Inputs: `frame` as described in the module docstring (rows already restricted to the split being
    scored), `task` "classification" or "regression", `seed` for `shuffled_label`. A baseline that cannot be
    built from the available columns maps to {"note": reason}. Unknown names raise ValueError.
    """
    unknown = [w for w in which if w not in _BUILDERS]
    if unknown:
        raise ValueError(f"unknown baseline(s) {unknown}; choose from {list(BASELINES)}")
    out: dict[str, dict[str, Any]] = {}
    for name in which:
        score = _BUILDERS[name](frame, task, seed=seed, past_return_col=past_return_col)
        out[name] = {"note": score} if isinstance(score, str) else evaluate(frame.assign(score=score), task)
    return out


@register_report_section
class BaselinesSection:
    """Report section "baselines": the model and each baseline side by side on one split.

    Output: ({"baselines": {"model": metrics, "<baseline>": metrics, ...}}, {}). The render in `report.md`
    gets one column per entry. The same rows, labels and fold as the model are scored; only `score` changes.
    """

    section_type: ClassVar[str] = "baselines"

    def __init__(
        self,
        which: Sequence[str] = BASELINES,
        split: str = "validation",
        seed: int = 0,
        past_return_col: str | None = None,
    ):
        unknown = [w for w in which if w not in _BUILDERS]
        if unknown:
            raise ValueError(f"unknown baseline(s) {unknown}; choose from {list(BASELINES)}")
        self.which = tuple(which)
        self.split = split
        self.seed = seed
        self.past_return_col = past_return_col

    def evaluate(self, predictions: pd.DataFrame, task: str) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
        """Metrics of the model's own score and of each baseline; an empty split gives an empty block."""
        part = predictions[predictions["split"] == self.split]
        if part.empty:
            return {"baselines": {}}, {}
        block = {"model": evaluate(part, task)}
        block |= baseline_metrics(part, task, self.which, self.seed, self.past_return_col)
        return {"baselines": block}, {}
