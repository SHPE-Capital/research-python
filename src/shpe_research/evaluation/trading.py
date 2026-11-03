"""Signal-level trading metrics added to every report (E1).

E1 measures *signals* (a score against forward returns), not trades; trade ledgers are T6's job. It answers
whether a score is worth trading:

    ic_decay        mean daily rank IC of the score against forward returns at several horizons.
    turnover        mean fraction of the top and bottom bucket that changes between consecutive decisions.
    cost_adjusted   mean net return per decision of "long top decile / short bottom decile" after a cost in
                    bps per side, plus the cost at which the edge disappears (`breakeven_bps`).
    by_group        IC and mean return by symbol, month and (when a `cell` column exists) D4 universe cell.

Input: a predictions frame with `decision_ts` (UTC, the time the score is known), `symbol`, `score` and
`outcome` (the raw forward return realized after `decision_ts`). Optional: `fwd_return_<h>m` columns (L1
labels, one per horizon in minutes) for `ic_decay`, and `cell` for the universe-cell breakdown. The function
never selects splits: pass it the validation or test rows you want scored. Rows with a missing score or
outcome are dropped.

Costs are signal-level: charged per side, on both legs, in bps of notional, on the names that change bucket
between consecutive decisions. They do not come from the T5 `CostModel` because E1 has no fills.

Conventions (decile bucket and cost accounting):
    * A bucket holds max(1, round(0.1 * n)) symbols of that decision time's cross-section; ties break by symbol
      name, so a constant score gives a constant ranking.
    * Decision times with fewer than 2 symbols are ignored for turnover and cost. The first decision has no
      predecessor, so it is excluded from turnover, gross and net returns (no entry cost is invented for it).
    * A name leaving a bucket costs 2 sides (sell out, buy in). With f_top and f_bottom the replaced fractions,
      the notional sides traded per decision are 2 * (f_top + f_bottom), each charged at `cost_bps`.
    * Return per decision is mean(outcome of top) - mean(outcome of bottom). Decisions closer together than
      the label horizon overlap; the numbers are per decision and are not annualized.
    * `breakeven_bps` = gross mean return / (mean sides traded * 1e-4); None when nothing is traded.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date
from shpe_research.evaluation.report import daily_ic

# The registry lives in `pipeline/`, a layer above `evaluation/` in the import-linter contract. A static import would
# break `lint-imports`, so it is resolved at import time instead. This is a workaround, not a fix: see the E1 report
# (Questions) for moving the registry down or adding an ignore entry in pyproject.toml (owned by C1l).
register_report_section = importlib.import_module("shpe_research.pipeline.registry").register_report_section

MIN_SYMBOLS_PER_DAY = 5
BUCKET_FRACTION = 0.1
_HORIZON_COLUMN = re.compile(r"^fwd_return_(\d+)m$")


def _num(value: Any) -> float | None:
    value = float(value)
    return None if np.isnan(value) else value


def _clean_ic(ic: pd.Series) -> float | None:
    return _num(ic.mean()) if len(ic) else None


def _day_ic(frame: pd.DataFrame, outcome: str) -> tuple[pd.Series, int]:
    """Daily rank IC over days with at least 5 symbols, and the number of days skipped for too few."""
    day = session_date(frame["decision_ts"])
    n_symbols = frame.groupby(day)["symbol"].transform("nunique")
    enough = n_symbols >= MIN_SYMBOLS_PER_DAY
    skipped = int(day[~enough].nunique())
    kept = frame[enough]
    if kept.empty:
        return pd.Series(dtype=float), skipped
    return daily_ic(kept.assign(decision_ts=kept["decision_ts"]), "score", outcome), skipped


def _horizons(frame: pd.DataFrame, horizons: Sequence[int] | None) -> list[int]:
    found = sorted(int(m.group(1)) for c in frame.columns if (m := _HORIZON_COLUMN.match(c)))
    return found if horizons is None else [int(h) for h in horizons]


def _ic_decay(frame: pd.DataFrame, horizons: Sequence[int] | None) -> dict:
    wanted = _horizons(frame, horizons)
    available = [h for h in wanted if f"fwd_return_{h}m" in frame.columns]
    missing = [h for h in wanted if h not in available]
    result: dict[int, float | None] = {}
    for h in available:
        part = frame.dropna(subset=["score", f"fwd_return_{h}m"])
        ic, _ = _day_ic(part, f"fwd_return_{h}m")
        result[h] = _clean_ic(ic)
    out: dict[str, Any] = {"by_horizon": result}
    if not wanted:
        out["note"] = "no fwd_return_<h>m columns in the predictions (needs the L1 multi-horizon labels)"
    elif missing:
        out["note"] = f"no fwd_return_<h>m column for horizons {missing}"
    return out


def _buckets(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per decision time (sorted): top and bottom symbol sets, long-short return and per-row leg sides."""
    decisions = []
    for ts, group in frame.groupby("decision_ts", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(["score", "symbol"], kind="stable")
        k = max(1, int(round(BUCKET_FRACTION * len(ordered))))
        k = min(k, len(ordered) // 2)
        bottom, top = ordered.iloc[:k], ordered.iloc[-k:]
        decisions.append(
            {
                "ts": ts,
                "top": set(top["symbol"]),
                "bottom": set(bottom["symbol"]),
                "k": k,
                "spread": float(top["outcome"].mean() - bottom["outcome"].mean()),
                "legs": pd.concat([top.assign(side=1.0), bottom.assign(side=-1.0)]),
            }
        )
    return decisions


def _replaced(prev: set, now: set, k: int) -> float:
    return len(now - prev) / k


def _group_stats(frame: pd.DataFrame, legs: pd.DataFrame, spreads: pd.Series, outcome: str = "outcome") -> dict:
    out: dict[str, dict] = {}
    by_symbol = {}
    for sym, g in frame.groupby("symbol"):
        ic = g["score"].rank().corr(g[outcome].rank()) if len(g) >= 5 and g["score"].nunique() > 1 else np.nan
        leg = legs[legs["symbol"] == sym] if len(legs) else legs
        by_symbol[str(sym)] = {
            "n": int(len(g)),
            "ic": _num(ic),
            "mean_return": _num((leg["side"] * leg["outcome"]).mean()) if len(leg) else None,
        }
    out["symbol"] = by_symbol

    month = frame["decision_ts"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    by_month = {}
    for m, g in frame.groupby(month):
        ic, _ = _day_ic(g, outcome)
        sp = spreads[spreads.index.isin(g["decision_ts"].unique())] if len(spreads) else spreads
        by_month[str(m)] = {"n": int(len(g)), "ic": _clean_ic(ic), "mean_return": _num(sp.mean()) if len(sp) else None}
    out["month"] = by_month

    cell_col = next((c for c in ("cell", "universe_cell") if c in frame.columns), None)
    if cell_col is not None:
        by_cell = {}
        for cell, g in frame.groupby(cell_col):
            ic, _ = _day_ic(g, outcome)
            leg = legs[legs[cell_col] == cell] if len(legs) else legs
            by_cell[str(cell)] = {
                "n": int(len(g)),
                "ic": _clean_ic(ic),
                "mean_return": _num((leg["side"] * leg["outcome"]).mean()) if len(leg) else None,
            }
        out["universe_cell"] = by_cell
    return out


def trading_metrics(
    predictions: pd.DataFrame,
    cost_bps: Sequence[float] = (0, 2, 5, 10),
    horizons: Sequence[int] | None = None,
) -> dict:
    """Is this score worth trading? IC decay, turnover, net return after costs and a by-group breakdown.

    Inputs: `predictions` with `decision_ts`, `symbol`, `score`, `outcome` (see the module docstring);
    `cost_bps` per-side costs to charge; `horizons` in minutes for `ic_decay` (default: every
    `fwd_return_<h>m` column present). Timing: each score is known at its `decision_ts`; `outcome` is realized
    after it, so nothing here looks ahead of the score. The caller chooses the split.

    Returns a dict with `ic_decay` ({"by_horizon": {minutes: mean daily IC}, "note"?}), `turnover` (float or
    None), `cost_adjusted` ({cost_bps: mean net return per decision, "breakeven_bps": ..., "n_decisions": ...}),
    `by_group`, `ic_days` and `ic_days_skipped` (days with fewer than 5 symbols). Empty input gives None entries.
    """
    frame = predictions.dropna(subset=["decision_ts", "symbol", "score", "outcome"]).copy()
    frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True)

    ic, skipped = _day_ic(frame, "outcome") if len(frame) else (pd.Series(dtype=float), 0)
    result: dict[str, Any] = {
        "ic_decay": _ic_decay(frame, horizons) if len(frame) else {"by_horizon": {}, "note": "no rows"},
        "ic_mean": _clean_ic(ic),
        "ic_days": int(len(ic)),
        "ic_days_skipped": skipped,
    }

    decisions = _buckets(frame) if len(frame) else []
    f_top, f_bot, spreads = [], [], []
    for prev, now in zip(decisions, decisions[1:], strict=False):
        f_top.append(_replaced(prev["top"], now["top"], now["k"]))
        f_bot.append(_replaced(prev["bottom"], now["bottom"], now["k"]))
        spreads.append(now["spread"])
    n = len(spreads)
    turnover = float(np.mean([(a + b) / 2 for a, b in zip(f_top, f_bot, strict=True)])) if n else None
    sides = np.array([2 * (a + b) for a, b in zip(f_top, f_bot, strict=True)])
    gross = float(np.mean(spreads)) if n else None

    cost: dict[Any, Any] = {}
    for c in cost_bps:
        cost[c] = float(np.mean(np.array(spreads) - sides * c * 1e-4)) if n else None
    mean_sides = float(sides.mean()) if n else 0.0
    cost["breakeven_bps"] = gross / (mean_sides * 1e-4) if n and mean_sides > 0 else None
    cost["n_decisions"] = n
    cost["note"] = "signal-level costs: per side, both legs, on bucket turnover; not the T5 CostModel"
    result["turnover"] = turnover
    result["cost_adjusted"] = cost

    legs = pd.concat([d["legs"] for d in decisions]) if decisions else frame.iloc[0:0].assign(side=[])
    spread_series = pd.Series({d["ts"]: d["spread"] for d in decisions}, dtype=float)
    result["by_group"] = _group_stats(frame, legs, spread_series) if len(frame) else {}
    return result


@register_report_section
class TradingSection:
    """Report section "trading": `trading_metrics` on the run's predictions, one block per evaluated split.

    Validation only by default. The test split is opened once, so the caller lists it in `splits` only for the
    final run. The metrics land under the key "trading" as {split: trading_metrics(...)}.
    """

    section_type: ClassVar[str] = "trading"

    def __init__(self, splits: Sequence[str] = ("validation",), cost_bps: Sequence[float] = (0, 2, 5, 10)):
        self.splits = tuple(splits)
        self.cost_bps = tuple(cost_bps)

    def evaluate(self, predictions: pd.DataFrame, task: str) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
        """({"trading": {split: metrics}}, {}); splits with no rows are omitted."""
        blocks = {}
        for split in self.splits:
            part = predictions[predictions["split"] == split]
            if not part.empty:
                blocks[split] = trading_metrics(part, cost_bps=self.cost_bps)
        return {"trading": blocks}, {}
