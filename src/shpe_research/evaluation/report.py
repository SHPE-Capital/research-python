"""Model report (C0 minimal version; E1 extends it).

Scores the same way for every model, so results are comparable:

    prediction quality   AUC and accuracy against the base rate (classification),
                         RMSE and R^2 (regression)
    ranking quality      daily information coefficient (IC): the Spearman
                         correlation between score and realized forward return
                         across that day's rows, then its mean, spread and
                         t-statistic across days
    economic shape       mean forward return by score decile

Prediction quality and trading value are different things: a model can rank
well and still lose money after costs. E1 adds turnover and cost-adjusted P&L.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

from shpe_research.data.calendar import session_date

MIN_ROWS_PER_DAY = 5


def daily_ic(frame: pd.DataFrame, score: str = "score", outcome: str = "outcome") -> pd.Series:
    """Spearman correlation of score and outcome for each New York session with enough rows."""
    day = session_date(frame["decision_ts"])
    ics = {}
    for d, group in frame.groupby(day):
        if len(group) >= MIN_ROWS_PER_DAY and group[score].nunique() > 1 and group[outcome].nunique() > 1:
            ics[d] = group[score].rank().corr(group[outcome].rank())
    return pd.Series(ics, dtype=float)


def decile_table(frame: pd.DataFrame, score: str = "score", outcome: str = "outcome") -> pd.DataFrame:
    """Row count and mean/median outcome per score decile (1 = lowest scores)."""
    deciles = pd.qcut(frame[score].rank(method="first"), 10, labels=range(1, 11))
    return frame.groupby(deciles, observed=True)[outcome].agg(["count", "mean", "median"]).rename_axis("decile")


def evaluate(frame: pd.DataFrame, task: str) -> dict[str, Any]:
    """Metrics for one split. `frame` has decision_ts, score, y and outcome (the raw forward return)."""
    y = frame["y"].to_numpy()
    score = frame["score"].to_numpy()
    metrics: dict[str, Any] = {"n_rows": len(frame), "n_days": int(session_date(frame["decision_ts"]).nunique())}
    if task == "classification":
        metrics["base_rate"] = float(y.mean())
        metrics["accuracy"] = float(accuracy_score(y, score >= 0.5))
        metrics["auc"] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y, score)))
        metrics["r2"] = float(r2_score(y, score))
    ic = daily_ic(frame)
    metrics["ic_days"] = len(ic)
    metrics["ic_mean"] = float(ic.mean()) if len(ic) else None
    metrics["ic_std"] = float(ic.std()) if len(ic) > 1 else None
    metrics["ic_t"] = (
        float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 1 and ic.std() > 0 else None
    )
    return metrics


def write_report(run_dir: Path, name: str, metrics: dict[str, dict], deciles: dict[str, pd.DataFrame]) -> Path:
    """report.json plus a readable report.md in `run_dir`."""
    run_dir = Path(run_dir)
    (run_dir / "report.json").write_text(
        json.dumps({"name": name, "metrics": metrics, "deciles": {k: v.reset_index().to_dict("records")
                                                                  for k, v in deciles.items()}},
                   indent=2, default=str)
    )
    lines = [f"# {name}", ""]
    keys = sorted({k for m in metrics.values() for k in m})
    lines += ["| metric | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for key in keys:
        cells = [_fmt(metrics[split].get(key)) for split in metrics]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    for split, table in deciles.items():
        lines += ["", f"## Forward return by score decile ({split})", "", "| decile | count | mean | median |",
                  "|---|---|---|---|"]
        lines += [f"| {d} | {int(r['count'])} | {r['mean']:.6f} | {r['median']:.6f} |" for d, r in table.iterrows()]
    lines += ["", "Fixture or not, treat the test column as opened once: tuning against it makes it validation."]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
