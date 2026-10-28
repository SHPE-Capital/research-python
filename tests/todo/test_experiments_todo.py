"""Acceptance tests for experiment and data tasks still open. Remove a skip marker in the PR that implements it."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.skip(reason="TODO(M2)")
def test_sweep(tmp_path):
    from shpe_research.experiments.sweep import run_sweep

    runs = run_sweep(Path(__file__).parents[2] / "configs/experiments/hello_sweep.yaml")
    assert len(runs) == 4


@pytest.mark.skip(reason="TODO(D6)")
def test_corporate_actions():
    from shpe_research.data.corporate_actions import fetch_corporate_actions

    assert fetch_corporate_actions(["AAPL"], "2020-08-01", "2020-09-30") is not None
