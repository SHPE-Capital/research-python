from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.evaluation.report import daily_ic
from shpe_research.features import PastReturn
from shpe_research.labels.forward_return import ForwardReturn, compute_labels
from shpe_research.validation.schemes import PurgedWalkForward, SingleScheme


def _rows() -> pd.DataFrame:
    bars = synthetic_bars(
        symbols=["AAA", "BBB", "CCC", "DDD"],
        start="2026-03-02",
        end="2026-05-29",
        reversal={"ar": [-0.2]},
    )
    feature = PastReturn(5)
    parts = [bars[bars["symbol"] == s].reset_index(drop=True) for s in bars["symbol"].unique()]
    feats = pd.concat(
        [
            pd.DataFrame({"symbol": b["symbol"], "decision_ts": b["ts_available"], "x": feature.compute(b)})
            for b in parts
        ]
    )
    label = compute_labels(bars, ForwardReturn(5))
    rows = feats.merge(label, on=["symbol", "decision_ts"]).dropna()
    rows = rows[rows["decision_ts"].dt.minute % 15 == 0].reset_index(drop=True)
    rows["era"] = session_date(rows["decision_ts"]).dt.strftime("%Y-%m-%d")
    rows["outcome"] = rows["fwd_return_5m"]
    return rows


def test_planted_signal_survives_purging():
    rows = _rows()
    scheme = PurgedWalkForward(n_folds=4, embargo="1 session", test_block=0.2)
    ics = []
    for fold in scheme.folds(rows):
        val = rows.iloc[fold.val_pos].assign(score=lambda d: -d["x"])  # reversal: fade the past return
        ics.append(daily_ic(val, "score", "outcome").mean())
    assert len(ics) == 4
    assert np.mean(ics) > 0


def _persistent_panel(n_sessions: int = 40, n_symbols: int = 800, horizon: int = 2, seed: int = 3) -> pd.DataFrame:
    """One row per symbol-session; the label sums `horizon` sessions of noise, so labels overlap across rows."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_sessions + horizon, tz="UTC") + pd.Timedelta(hours=21)
    noise = rng.normal(size=(n_sessions + horizon, n_symbols))
    label = sum(noise[i : i + n_sessions] for i in range(horizon))  # sessions x symbols
    rows = pd.DataFrame(
        {
            "symbol": np.tile(np.arange(n_symbols), n_sessions),
            "decision_ts": np.repeat(days[:n_sessions], n_symbols),
            "label_end_ts": np.repeat(days[horizon : horizon + n_sessions], n_symbols),
            "outcome": label.ravel(),
        }
    )
    rows["era"] = rows["decision_ts"].dt.strftime("%Y-%m-%d")
    return rows


def _val_ic(rows: pd.DataFrame, folds) -> float:
    """Pooled IC of "the symbol's latest train label" against the validation label (a model that memorises)."""
    scores, outcomes = [], []
    for fold in folds:
        last = rows.iloc[fold.train_pos].sort_values("decision_ts").groupby("symbol")["outcome"].last()
        val = rows.iloc[fold.val_pos]
        scores.append(val["symbol"].map(last).to_numpy())
        outcomes.append(val["outcome"].to_numpy())
    return float(np.corrcoef(np.concatenate(scores), np.concatenate(outcomes))[0, 1])


def test_leaky_label_caught():
    rows = _persistent_panel()
    single = _val_ic(rows, SingleScheme(test_block=0.2).folds(rows))
    purged = _val_ic(rows, PurgedWalkForward(n_folds=3, embargo="4 days", test_block=0.2).folds(rows))
    assert single > purged + 0.03  # the last train label overlaps the first validation session
    assert abs(purged) < 0.03  # purged, the label is pure noise
