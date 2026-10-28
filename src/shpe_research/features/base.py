"""Feature contract and helpers.

A Feature turns ONE symbol's contract bars (sorted by ts) into one value per
bar. The value on a bar's row must be computable at that bar's ts_available
from that bar and earlier rows only. `assert_no_lookahead` checks exactly
that, and every feature's tests must call it.

Features are parametrized instances (e.g. PastReturn(window=5)); `name` must
encode the parameters so feature columns stay unique and self-describing.
Register a feature class with @register_feature so configs can build it:

    features:
      - {type: past_return, window: 5}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import pandas as pd

FEATURE_REGISTRY: dict[str, type[Feature]] = {}


def register_feature(cls: type[Feature]) -> type[Feature]:
    if cls.feature_type in FEATURE_REGISTRY:
        raise ValueError(f"feature_type {cls.feature_type!r} is already registered")
    FEATURE_REGISTRY[cls.feature_type] = cls
    return cls


class Feature(ABC):
    feature_type: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ("close",)

    @property
    @abstractmethod
    def name(self) -> str:
        """Column name, including parameters (e.g. "past_return_5m")."""

    @abstractmethod
    def compute(self, bars: pd.DataFrame) -> pd.Series:
        """One value per row of `bars` (one symbol, sorted by ts), aligned to bars.index."""


def build_feature(spec: dict[str, Any]) -> Feature:
    """A feature from a config entry like {"type": "past_return", "window": 5}."""
    import shpe_research.features  # noqa: F401  (importing the package registers the built-in features)

    params = dict(spec)
    feature_type = params.pop("type")
    try:
        cls = FEATURE_REGISTRY[feature_type]
    except KeyError:
        raise ValueError(f"unknown feature {feature_type!r}; registered: {sorted(FEATURE_REGISTRY)}") from None
    return cls(**params)


def compute_features(bars: pd.DataFrame, features: Iterable[Feature]) -> pd.DataFrame:
    """Features for every symbol in `bars`: symbol, ts, ts_available plus one column per feature."""
    features = list(features)
    frames = []
    for _, group in bars.groupby("symbol", sort=True):
        group = group.sort_values("ts")
        out = group[["symbol", "ts", "ts_available"]].copy()
        for feature in features:
            missing = [c for c in feature.requires if c not in group.columns]
            if missing:
                raise ValueError(f"feature {feature.name} needs columns {missing}")
            values = feature.compute(group)
            if not values.index.equals(group.index):
                raise ValueError(f"feature {feature.name} returned values not aligned to its input rows")
            out[feature.name] = values.astype(float)
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "ts_available", *[f.name for f in features]])
    return pd.concat(frames).reset_index(drop=True)


def assert_no_lookahead(feature: Feature, bars: pd.DataFrame, cutoffs: Iterable[int] | None = None) -> None:
    """Fail if any value of `feature` changes when later bars are removed.

    `bars` is one symbol's bars sorted by ts. The feature is computed on the
    full history and on prefixes ending at each cutoff; a causal feature gives
    identical values on every row the prefix contains.
    """
    bars = bars.reset_index(drop=True)
    full = feature.compute(bars).to_numpy(dtype=float)
    n = len(bars)
    if cutoffs is None:
        cutoffs = sorted({max(1, n // 7), n // 3, n // 2, (2 * n) // 3, n - 1} - {0})
    for k in cutoffs:
        prefix = feature.compute(bars.iloc[:k]).to_numpy(dtype=float)
        same = np.isclose(prefix, full[:k], rtol=1e-9, atol=1e-12, equal_nan=True)
        if not same.all():
            row = int(np.argmin(same))
            raise AssertionError(
                f"{feature.name} uses future data: row {row} is {full[row]!r} with the full history "
                f"but {prefix[row]!r} when bars after row {k - 1} are removed"
            )
