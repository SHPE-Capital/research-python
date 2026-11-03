from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.evaluation.trading import trading_metrics


def _frame(rows):
    """rows: (decision_ts, symbol, score, outcome)."""
    df = pd.DataFrame(rows, columns=["decision_ts", "symbol", "score", "outcome"])
    df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True)
    df["split"] = "validation"
    return df


def test_ic_decay_planted():
    bars = synthetic_bars(start="2026-03-02", end="2026-04-10", reversal={"ar": [-0.5]}, seed=3)
    close = bars.pivot(index="ts", columns="symbol", values="close")
    logp = np.log(close)
    ret1 = logp.diff()
    out = {h: logp.shift(-h) - logp for h in (1, 4)}
    sample = close.index[::10]
    parts = []
    for ts in sample:
        for sym in close.columns:
            parts.append(
                {"decision_ts": ts, "symbol": sym, "score": -ret1.at[ts, sym], "split": "validation",
                 "outcome": out[1].at[ts, sym], "fwd_return_1m": out[1].at[ts, sym],
                 "fwd_return_4m": out[4].at[ts, sym]}
            )
    preds = pd.DataFrame(parts).dropna()
    decay = trading_metrics(preds, horizons=[1, 4])["ic_decay"]
    assert decay["by_horizon"][1] > decay["by_horizon"][4] > 0


def test_ic_decay_absent_without_columns():
    preds = _frame([("2026-03-02T15:00Z", s, i, i) for i, s in enumerate("ABCDEF")])
    decay = trading_metrics(preds)["ic_decay"]
    assert decay["by_horizon"] == {} and "note" in decay


def test_turnover_constant_score():
    rows = [(f"2026-03-02T15:{m:02d}Z", s, 1.0, 0.001 * i) for m in range(0, 50, 10) for i, s in enumerate("ABCDEF")]
    assert trading_metrics(_frame(rows))["turnover"] == 0.0


def test_breakeven():
    # two names, two decisions; the ranking flips, so both legs are replaced (turnover 1)
    rows = [
        ("2026-03-02T15:00Z", "A", 1.0, 0.01), ("2026-03-02T15:00Z", "B", 0.0, -0.01),
        ("2026-03-02T15:10Z", "A", 0.0, -0.005), ("2026-03-02T15:10Z", "B", 1.0, 0.005),
    ]
    m = trading_metrics(_frame(rows), cost_bps=(0, 25))
    assert m["turnover"] == pytest.approx(1.0)
    # gross 0.01 per decision after the first; 4 notional-sides of cost per decision -> 0.01 / 4 = 25 bps
    assert m["cost_adjusted"][0] == pytest.approx(0.01)
    assert m["cost_adjusted"]["breakeven_bps"] == pytest.approx(25.0)
    assert m["cost_adjusted"][25] == pytest.approx(0.0, abs=1e-12)


def test_by_group_and_skipped_days():
    rows = [("2026-03-02T15:00Z", s, i, i * 0.001) for i, s in enumerate("ABCDEF")]
    rows += [("2026-03-03T15:00Z", s, i, i * 0.001) for i, s in enumerate("AB")]  # < 5 symbols
    m = trading_metrics(_frame(rows))
    assert set(m["by_group"]) >= {"symbol", "month"}
    assert m["ic_days_skipped"] == 1
    assert m["by_group"]["month"]["2026-03"]["n"] == 8


def test_section_registered():
    from shpe_research.evaluation import trading  # noqa: F401
    from shpe_research.pipeline.registry import REPORT_SECTION_REGISTRY

    assert REPORT_SECTION_REGISTRY["trading"] is trading.TradingSection
