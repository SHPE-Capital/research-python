"""Report section "permutation_importance" (C1i): which features carry the score?

For each feature, shuffle its values among the rows of the same era on the validation rows, score the
already-fitted model on the shuffled matrix, and record how much the metric falls (`drop = baseline -
permuted`). Repeated `n_repeats` times; the table has the mean and std of the drop per feature.

    * The model is never refitted; only `model.predict` is called, on a copy of the validation features.
    * Permuting inside an era keeps each session's level and volatility, so a feature cannot look important
      just because it separates calm days from wild ones.
    * Only the requested split's rows are used (validation by default), so the test split stays closed.
    * Importance is of the model, not of the world: correlated features share credit, and a feature the
      model ignores has a drop near 0, with either sign. A noise feature's mean drop should sit within about
      2 std of 0.

Timing: features are the values known at each row's `decision_ts`; the metric is computed on the realised
`outcome` and label `y` as in the report, so nothing beyond the validation rows' own data is touched.

Pipeline note: a report section only receives the predictions frame, which has no features and no model. This
section therefore needs `bind(dataset, model)` before `evaluate`; unbound it reports a note instead of numbers.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from shpe_research.evaluation.by_era import METRICS, metric_value

register_report_section = importlib.import_module("shpe_research.pipeline.registry").register_report_section


def _permute_within_era(values: np.ndarray, era_positions: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for pos in era_positions:
        out[pos] = values[rng.permutation(pos)]
    return out


def permutation_importance(
    model: Any,
    dataset: Any,
    split: str = "validation",
    n_repeats: int = 5,
    metric: str = "ic",
    seed: int = 0,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Drop in `metric` when one feature is permuted within era, per feature.

    Inputs: a fitted `model` (needs `predict(X)`), a `Dataset` (X, y, meta with `era`, `decision_ts` and
    `outcome`), the `split` to score, `n_repeats` shuffles per feature, `metric` (see `by_era.METRICS`, higher
    is better) and a `seed`. Output: DataFrame indexed by feature with `mean_drop`, `std_drop` (sample std over
    repeats, 0 when n_repeats is 1) and `baseline`. Inputs are not modified and the model is not refitted.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {list(METRICS)}")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    part = dataset.subset(split)
    if len(part) == 0:
        raise ValueError(f"no rows in split {split!r}")
    names = list(features) if features is not None else list(part.X.columns)
    meta = part.meta
    if "outcome" not in meta:
        raise ValueError("meta needs an 'outcome' column (the raw label value) for the metric")
    base_frame = pd.DataFrame(
        {"decision_ts": meta["decision_ts"], "y": part.y, "outcome": meta["outcome"]}, index=part.X.index
    )

    def score_of(X: pd.DataFrame) -> float:
        return metric_value(base_frame.assign(score=np.asarray(model.predict(X))), metric)

    baseline = score_of(part.X)
    era_positions = [np.asarray(pos) for pos in meta.groupby("era").indices.values()]
    rng = np.random.default_rng(seed)
    rows = {}
    for name in names:
        drops = []
        for _ in range(n_repeats):
            X = part.X.copy()
            X[name] = _permute_within_era(part.X[name].to_numpy(), era_positions, rng)
            drops.append(baseline - score_of(X))
        rows[name] = {
            "mean_drop": float(np.nanmean(drops)),
            "std_drop": float(np.nanstd(drops, ddof=1)) if n_repeats > 1 else 0.0,
            "baseline": baseline,
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["mean_drop", "std_drop", "baseline"]).rename_axis(
        "feature"
    )


@register_report_section
class PermutationImportanceSection:
    """Report section "permutation_importance": per-feature drop in `metric` on `split`, mean and std.

    Output: ({"permutation_importance": {feature: {"mean_drop": ..., "std_drop": ...}}}, {}). Call
    `bind(dataset, model)` first (see the module docstring); unbound it returns {"note": ...}.
    """

    section_type: ClassVar[str] = "permutation_importance"

    def __init__(self, n_repeats: int = 5, metric: str = "ic", seed: int = 0, split: str = "validation"):
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; choose from {list(METRICS)}")
        self.n_repeats = n_repeats
        self.metric = metric
        self.seed = seed
        self.split = split
        self._dataset: Any = None
        self._model: Any = None

    def bind(self, dataset: Any, model: Any) -> PermutationImportanceSection:
        """Attach the dataset (features) and fitted model this section permutes; returns self."""
        self._dataset, self._model = dataset, model
        return self

    def evaluate(self, predictions: pd.DataFrame, task: str) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
        """Importance table as nested dicts; a note when unbound or the split has no rows."""
        if self._dataset is None or self._model is None:
            return {"permutation_importance": {"note": "not bound to a dataset and model; call bind() first"}}, {}
        if not (self._dataset.meta["split"] == self.split).any():
            return {"permutation_importance": {"note": f"no rows in split {self.split!r}"}}, {}
        table = permutation_importance(
            self._model, self._dataset, self.split, self.n_repeats, self.metric, self.seed
        )
        block = {f: {"mean_drop": r["mean_drop"], "std_drop": r["std_drop"]} for f, r in table.iterrows()}
        return {"permutation_importance": block}, {}
