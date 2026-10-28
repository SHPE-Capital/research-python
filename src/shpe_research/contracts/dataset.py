"""Dataset contract: what every model trains and predicts on.

A Dataset is three aligned parts sharing one index:

    X     features, numeric, one column per feature, in a fixed order
    y     the label
    meta  at least: symbol, decision_ts (when the prediction is made),
          label_end_ts (when the label's outcome is fully known), weight,
          split ("train" | "validation" | "test" | "unassigned").
          Extra columns (e.g. the raw forward return behind a sign label)
          are allowed and travel with the rows.

label_end_ts is what lets a splitter purge training rows whose outcome
overlaps a later period. A row whose label ends at or before its decision is
a bug, and validation rejects it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from shpe_research.contracts.base import ContractError, check_columns

SPLITS = ("train", "validation", "test", "unassigned")

META_COLUMNS: dict[str, str] = {
    "symbol": "str",
    "decision_ts": "utc",
    "label_end_ts": "utc",
    "weight": "float",
    "split": "str",
}


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    label_name: str
    allow_nan_features: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def __len__(self) -> int:
        return len(self.X)

    def validate(self) -> None:
        """Raise ContractError listing every way this dataset breaks the contract."""
        problems = check_columns(self.meta, META_COLUMNS)
        if not (self.X.index.equals(self.y.index) and self.X.index.equals(self.meta.index)):
            problems.append("X, y and meta must share the same index")
        non_numeric = [c for c in self.X.columns if not pd.api.types.is_numeric_dtype(self.X[c])]
        if non_numeric:
            problems.append(f"non-numeric feature columns {non_numeric}")
        if self.y.isna().any():
            problems.append(f"{int(self.y.isna().sum())} rows have no label; drop them before building")
        if not self.allow_nan_features and self.X.isna().any().any():
            cols = self.X.columns[self.X.isna().any()].tolist()
            problems.append(f"NaN features in {cols}; drop rows or set allow_nan_features=True")
        if problems:
            raise ContractError("dataset", problems)
        if (self.meta["label_end_ts"] <= self.meta["decision_ts"]).any():
            problems.append("label_end_ts must be after decision_ts on every row")
        if (self.meta["weight"] < 0).any():
            problems.append("weights must be non-negative")
        unknown = set(self.meta["split"].unique()) - set(SPLITS)
        if unknown:
            problems.append(f"unknown split values {sorted(unknown)}")
        if problems:
            raise ContractError("dataset", problems)

    def subset(self, split: str) -> Dataset:
        """Rows of one split, as a Dataset."""
        mask = self.meta["split"] == split
        return replace(self, X=self.X[mask], y=self.y[mask], meta=self.meta[mask])

    def with_splits(self, splits: pd.Series) -> Dataset:
        """A copy whose meta.split comes from `splits` (aligned on the index)."""
        meta = self.meta.assign(split=splits.reindex(self.meta.index).astype(str))
        return replace(self, meta=meta)
