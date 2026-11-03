"""validate_candidates (C1b): every way a CandidateFrame can break the schema."""

from __future__ import annotations

import pandas as pd
import pytest

from shpe_research.contracts.base import ContractError, to_utc_ns
from shpe_research.contracts.candidates import validate_candidates


def minimal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "decision_ts": to_utc_ns(["2026-10-01T14:00:00Z", "2026-10-01T14:00:00Z"]),
            "side": ["long", "short"],
            "strategy_id": ["demo:abc", "demo:abc"],
        }
    )


def _exit_without_reason(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(action=["enter", "exit"], exit_reason=[None, None])


@pytest.mark.parametrize(
    ("mutate", "column"),
    [
        (lambda df: df.drop(columns="symbol"), "symbol"),
        (lambda df: df.drop(columns="decision_ts"), "decision_ts"),
        (lambda df: df.drop(columns="side"), "side"),
        (lambda df: df.drop(columns="strategy_id"), "strategy_id"),
        (lambda df: df.assign(side=["long", "flat"]), "side"),
        (lambda df: df.assign(action=["enter", "close"]), "action"),
        (_exit_without_reason, "exit_reason"),
        (lambda df: df.assign(decision_ts=df["decision_ts"].dt.tz_localize(None)), "decision_ts"),
        (lambda df: df.assign(decision_ts=df["decision_ts"].dt.tz_convert("America/New_York")), "decision_ts"),
        (lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True), "decision_ts"),
    ],
)
def test_validator_rejects(mutate, column):
    with pytest.raises(ContractError) as err:
        validate_candidates(mutate(minimal()))
    assert err.value.contract == "candidates"
    assert any(column in problem for problem in err.value.problems)


def test_validator_rejects_non_frame():
    with pytest.raises(ContractError):
        validate_candidates(None)


def test_duplicates_allow_other_actions():
    df = pd.concat([minimal().iloc[[0]], minimal().iloc[[0]]], ignore_index=True)
    df["action"] = ["enter", "exit"]
    df["exit_reason"] = [None, "time"]
    validate_candidates(df)


def test_validator_lists_every_problem():
    df = minimal().drop(columns="strategy_id").assign(side=["long", "flat"])
    with pytest.raises(ContractError) as err:
        validate_candidates(df)
    assert len(err.value.problems) >= 2


def test_validator_accepts_minimal():
    df = minimal()
    validate_candidates(df)  # the four required columns; action defaults to "enter"
    assert "action" not in df.columns
    validate_candidates(df.assign(action="enter"))
    validate_candidates(df.iloc[0:0])
