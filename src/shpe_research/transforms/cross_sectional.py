"""rank, zscore, winsorize, demean: per-timestamp transforms of one column.

Inputs: a frame with `decision_ts` and the column. Output: float64 Series aligned to the frame.
Timing: each row uses only rows with the same `decision_ts`, so only what is known at that decision.
NaN inputs stay NaN and do not count toward `min_count`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shpe_research.transforms.base import TIME_KEY, Transform, register_transform


def _grouped(frame: pd.DataFrame, column: str, min_count: int):
    """The column, its per-timestamp group, and a mask of rows in groups with enough usable values."""
    x = frame[column].astype("float64")
    group = x.groupby(frame[TIME_KEY])
    ok = group.transform("count") >= min_count
    return x, group, ok


@register_transform
@dataclass(frozen=True)
class Rank(Transform):
    """Cross-sectional rank (ties get the average rank); with `pct` the rank divided by the count, in (0, 1]."""

    transform_type = "rank"
    column: str
    pct: bool = True
    min_count: int = 5

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        _, group, ok = _grouped(frame, self.column, self.min_count)
        return group.rank(method="average", pct=self.pct).where(ok)


@register_transform
@dataclass(frozen=True)
class ZScore(Transform):
    """(x - mean) / std across symbols at each timestamp; std uses ddof=1. A zero std gives NaN."""

    transform_type = "zscore"
    column: str
    min_count: int = 5

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        x, group, ok = _grouped(frame, self.column, self.min_count)
        std = group.transform("std").replace(0.0, np.nan)
        return ((x - group.transform("mean")) / std).where(ok)


@register_transform
@dataclass(frozen=True)
class Winsorize(Transform):
    """Clip at the `lower` and `upper` cross-sectional quantiles of each timestamp."""

    transform_type = "winsorize"
    column: str
    lower: float = 0.01
    upper: float = 0.99
    min_count: int = 5

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        x, group, ok = _grouped(frame, self.column, self.min_count)
        lo, hi = group.transform("quantile", self.lower), group.transform("quantile", self.upper)
        return x.clip(lower=lo, upper=hi).where(ok)


@register_transform
@dataclass(frozen=True)
class Demean(Transform):
    """x minus its mean over the timestamp, or over the (timestamp, `by`) group when `by` names a column."""

    transform_type = "demean"
    column: str
    by: str | None = None

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        x = frame[self.column].astype("float64")
        keys = [frame[TIME_KEY]] + ([frame[self.by]] if self.by else [])
        return x - x.groupby(keys).transform("mean")
