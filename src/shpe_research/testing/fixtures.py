"""Fixtures for tests and notebooks: the planted-effects bars plus small helpers for the check kits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shpe_research.data.fixtures import synthetic_bars

__all__ = ["check_bars", "synthetic_bars", "synthetic_xy"]


def check_bars(symbols: tuple[str, ...] = ("AAA",), start: str = "2026-03-02", end: str = "2026-03-06") -> pd.DataFrame:
    """Five sessions of 1-minute bars (the default input of the check kits), with the fixture's fixed seed."""
    return synthetic_bars(symbols=symbols, start=start, end=end)


def synthetic_xy(
    task: str = "classification", n: int = 240, n_features: int = 3, seed: int = 7
) -> tuple[pd.DataFrame, pd.Series]:
    """Features `f0..` and a label with a weak linear dependence on `f0`: 0/1 for classification, float otherwise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    score = 0.8 * X["f0"] + rng.normal(scale=1.0, size=n)
    y = (score > 0).astype("int64") if task == "classification" else score.astype(float)
    return X, pd.Series(y, index=X.index, name="y")
