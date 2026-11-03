"""Acceptance tests for experiment and data tasks still open. Remove a skip marker in the PR that implements it."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.skip(reason="TODO(M2)")
def test_sweep(tmp_path):
    from shpe_research.experiments.sweep import run_sweep

    runs = run_sweep(Path(__file__).parents[2] / "configs/experiments/hello_sweep.yaml")
    assert len(runs) == 4

