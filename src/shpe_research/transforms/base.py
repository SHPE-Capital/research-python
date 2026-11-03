"""Cross-sectional transforms: stateless per-timestamp operations across symbols.

A transform takes the feature frame (one row per symbol and `decision_ts`) and returns one float64
column aligned to its index. It groups by `decision_ts` ONLY: no value from another timestamp is ever
used, so it cannot look ahead in time, and nothing is fitted across time (neutralization betas are
refit at every timestamp). NaN inputs are ignored; a timestamp with fewer than `min_count` usable rows
gives NaN for all its rows. Run it on the point-in-time universe (C1m): ranking against symbols not yet
listed would leak through the universe.

Register with `@register_transform`; build from config with `build_transform({"type": "rank", ...})`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

import pandas as pd

TIME_KEY = "decision_ts"
TRANSFORM_REGISTRY: dict[str, type[Transform]] = {}


def register_transform(cls: type[Transform]) -> type[Transform]:
    if cls.transform_type in TRANSFORM_REGISTRY:
        raise ValueError(f"transform_type {cls.transform_type!r} is already registered")
    TRANSFORM_REGISTRY[cls.transform_type] = cls
    return cls


class Transform(ABC):
    transform_type: ClassVar[str]

    column: str

    @property
    def name(self) -> str:
        """Output column name: the transform type prefixed to the input column."""
        return f"{self.transform_type}_{self.column}"

    @abstractmethod
    def compute(self, frame: pd.DataFrame) -> pd.Series:
        """float64 values aligned to `frame.index`; groups by `decision_ts` only."""


def build_transform(spec: dict[str, Any]) -> Transform:
    """A transform from a config entry like {"type": "rank", "column": "past_return_5m"}."""
    import shpe_research.transforms  # noqa: F401  (importing the package registers the built-ins)

    params = dict(spec)
    transform_type = params.pop("type")
    try:
        cls = TRANSFORM_REGISTRY[transform_type]
    except KeyError:
        raise ValueError(f"unknown transform {transform_type!r}; registered: {sorted(TRANSFORM_REGISTRY)}") from None
    return cls(**params)


def apply_transforms(frame: pd.DataFrame, transforms: Iterable[Transform]) -> pd.DataFrame:
    """A copy of `frame` with one new float64 column per transform. Each reads the original columns only."""
    out = frame.copy()
    for transform in transforms:
        if transform.column not in frame.columns:
            raise ValueError(f"transform {transform.name} needs column {transform.column!r}")
        out[transform.name] = transform.compute(frame).astype("float64")
    return out
