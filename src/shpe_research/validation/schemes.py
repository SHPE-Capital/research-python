"""Registered fit schemes: `single`, `walk_forward` and `purged_walk_forward` (C1f, ADR 0006).

A fit scheme decides which rows train a model and which score it. All three hold the
test block out first: the last `test_block` share of eras (sessions) is never in a
fold. Fold boundaries fall on era boundaries, so a session is never split between
train and validation. Each scheme implements `folds(meta) -> Iterator[Fold]`, where
`meta` has UTC `decision_ts` (when the row is decided), `label_end_ts` (when its
label is known) and, optionally, `era` (the session key; derived from `decision_ts`
when absent). Positions index `meta` as passed, not its index labels.

`purged_walk_forward` drops training rows whose `[decision_ts, label_end_ts]` overlaps
the validation interval and rows that start within `embargo` after it (the rule in
`validation/purged.py`, V1), and refuses an embargo shorter than the longest label.
`walk_forward` is the same fold geometry with purge and embargo off. `single` wraps
`split_by_day`: one fold, no purge.

Reference: Lopez de Prado, Advances in Financial Machine Learning, ch. 7.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from shpe_research.data.calendar import session_date
from shpe_research.validation.splits import split_by_day

# The registry lives in the higher `pipeline` layer; resolve it at runtime like events/ and evaluation/ do.
_registry = importlib.import_module("shpe_research.pipeline.registry")
register_fit_scheme = _registry.register_fit_scheme
FIT_SCHEME_REGISTRY = _registry.FIT_SCHEME_REGISTRY

_SESSIONS = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*sessions?\s*$", re.IGNORECASE)


def parse_embargo(value: str | pd.Timedelta | None) -> pd.Timedelta:
    """A `Timedelta` from None (zero), a Timedelta, "N session(s)" (N * 24 hours) or a pandas string.

    A session is approximated as 24 hours of calendar time, so a label that spans a weekend
    needs an embargo of that many calendar days. Raises ValueError for an unparsable string.
    """
    if value is None:
        return pd.Timedelta(0)
    if isinstance(value, pd.Timedelta):
        return value
    match = _SESSIONS.match(str(value))
    if match:
        return pd.Timedelta(days=float(match.group(1)))
    try:
        parsed = pd.Timedelta(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"cannot parse embargo {value!r}; use 'N session(s)' or a pandas Timedelta string") from exc
    if pd.isna(parsed):
        raise ValueError(f"cannot parse embargo {value!r}")
    return parsed


def _parse_share(value: float | str, name: str) -> float:
    share = float(value[:-1]) / 100 if isinstance(value, str) and value.strip().endswith("%") else float(value)
    if not 0 <= share < 1:
        raise ValueError(f"{name} must be in [0, 1), got {value!r}")
    return share


@dataclass(frozen=True)
class Fold:
    """One fit/score round. Positions index the `meta` passed to `folds`, ascending.

    `train_end` is the latest `label_end_ts` among the training rows (NaT if none);
    `val_start` the earliest `decision_ts` and `val_end` the latest `label_end_ts` of the
    validation rows; `embargo` the embargo applied. Validation rows are known at decision
    time only up to `val_start`; nothing in the fold uses data after `val_end`.
    """

    k: int
    train_pos: np.ndarray
    val_pos: np.ndarray
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    embargo: pd.Timedelta


def _utc(meta: pd.DataFrame, column: str) -> pd.Series:
    if column not in meta:
        raise ValueError(f"meta needs a {column!r} column")
    out = pd.to_datetime(meta[column], utc=True)
    if out.isna().any():
        raise ValueError(f"{column} must not contain missing values")
    return out


def _eras(meta: pd.DataFrame) -> np.ndarray:
    """Era key per row (session date string when `meta` has no `era` column)."""
    if "era" in meta:
        return meta["era"].astype(str).to_numpy()
    return session_date(_utc(meta, "decision_ts")).dt.strftime("%Y-%m-%d").to_numpy()


def _make_fold(k: int, meta: pd.DataFrame, train: np.ndarray, val: np.ndarray, embargo: pd.Timedelta) -> Fold:
    decision, label_end = _utc(meta, "decision_ts"), _utc(meta, "label_end_ts")
    train_end = label_end.iloc[train].max() if len(train) else pd.NaT
    return Fold(
        k=k,
        train_pos=np.sort(train),
        val_pos=np.sort(val),
        train_end=train_end,
        val_start=decision.iloc[val].min(),
        val_end=label_end.iloc[val].max(),
        embargo=embargo,
    )


class _Scheme:
    """Shared behaviour: test-block bookkeeping and the manifest entry."""

    scheme_type: ClassVar[str]
    test_block: float
    score_test: bool

    @property
    def test_scored(self) -> bool:
        """Whether the run may score the test block (`manifest.test_scored`); false unless `score_test`."""
        return bool(self.score_test)

    def test_mask(self, meta: pd.DataFrame) -> np.ndarray:
        """True for rows in the test block: the last `test_block` share of eras, assigned before any fold."""
        eras = _eras(meta)
        unique = np.sort(np.unique(eras))
        n_test = int(round(len(unique) * self.test_block))
        if self.test_block > 0:
            n_test = max(n_test, 1)
        return np.isin(eras, unique[len(unique) - n_test :]) if n_test else np.zeros(len(meta), dtype=bool)

    def folds(self, meta: pd.DataFrame) -> Iterator[Fold]:  # pragma: no cover - interface
        raise NotImplementedError

    def manifest_folds(self, meta: pd.DataFrame) -> list[dict[str, Any]]:
        """One JSON-ready dict per fold (boundaries as ISO strings, embargo as a Timedelta string)."""

        def iso(ts: pd.Timestamp) -> str | None:
            return None if pd.isna(ts) else ts.isoformat()

        return [
            {
                "fold": f.k,
                "train_end": iso(f.train_end),
                "val_start": iso(f.val_start),
                "val_end": iso(f.val_end),
                "embargo": str(f.embargo),
                "n_train": int(len(f.train_pos)),
                "n_val": int(len(f.val_pos)),
            }
            for f in self.folds(meta)
        ]


@dataclass(frozen=True)
class _WalkForwardBase(_Scheme):
    n_folds: int = 5
    embargo: Any = None
    test_block: Any = 0.2
    expanding: bool = True
    score_test: bool = False

    purge: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if int(self.n_folds) < 1:
            raise ValueError("n_folds must be at least 1")
        object.__setattr__(self, "n_folds", int(self.n_folds))
        object.__setattr__(self, "test_block", _parse_share(self.test_block, "test_block"))
        object.__setattr__(self, "embargo", parse_embargo(self.embargo) if self.purge else pd.Timedelta(0))

    def folds(self, meta: pd.DataFrame) -> Iterator[Fold]:
        """Yield `n_folds` folds over the development eras (everything outside the test block).

        Inputs: `meta` with `decision_ts`, `label_end_ts` and optionally `era`. Development eras are cut
        into `n_folds + 1` contiguous era blocks; fold k validates on block k + 1 and trains on the
        blocks before it (expanding) or on block k only. A decision at `val_start` sees only labels that
        ended before it once purged. Raises ValueError for missing timestamps, too few eras, or (purged
        only) an embargo shorter than the longest `label_end_ts - decision_ts`, naming
        `meta.attrs["label_name"]` (default "label").
        """
        decision, label_end = _utc(meta, "decision_ts"), _utc(meta, "label_end_ts")
        if self.purge:
            horizon = (label_end - decision).max()
            if horizon > self.embargo:
                name = meta.attrs.get("label_name", "label")
                raise ValueError(
                    f"embargo {self.embargo} is shorter than the longest horizon of label {name!r} ({horizon}); "
                    "raise the embargo"
                )
        eras = _eras(meta)
        dev = ~self.test_mask(meta)
        dev_eras = np.sort(np.unique(eras[dev]))
        if len(dev_eras) < self.n_folds + 1:
            raise ValueError(f"need at least {self.n_folds + 1} development eras for {self.n_folds} folds")
        blocks = np.array_split(dev_eras, self.n_folds + 1)
        dec, end = decision.to_numpy(), label_end.to_numpy()
        pad = self.embargo.to_timedelta64()
        for k in range(self.n_folds):
            val = np.flatnonzero(np.isin(eras, blocks[k + 1]))
            keep = np.concatenate(blocks[: k + 1]) if self.expanding else blocks[k]
            cand = np.flatnonzero(np.isin(eras, keep))
            if self.purge:
                val_start, val_end = dec[val].min(), end[val].max()
                overlap = (end[cand] >= val_start) & (dec[cand] <= val_end)
                embargoed = (dec[cand] > val_end) & (dec[cand] <= val_end + pad)
                cand = cand[~(overlap | embargoed)]
            yield _make_fold(k, meta, cand, val, self.embargo)


@register_fit_scheme
@dataclass(frozen=True)
class WalkForward(_WalkForwardBase):
    """Era-aligned walk-forward folds with no purge and no embargo (the baseline that leaks on long labels)."""

    scheme_type: ClassVar[str] = "walk_forward"
    purge: ClassVar[bool] = False


@register_fit_scheme
@dataclass(frozen=True)
class PurgedWalkForward(_WalkForwardBase):
    """Era-aligned walk-forward folds with purging and an embargo (López de Prado ch. 7); the default for candidates."""

    scheme_type: ClassVar[str] = "purged_walk_forward"
    purge: ClassVar[bool] = True


@dataclass(frozen=True)
class SingleScheme(_Scheme):
    """One chronological split by whole sessions (`split_by_day`), now with an explicit `test_block`.

    With `test_block` set, the last share of sessions is test and the rest is split train:validation
    in the ratio `train:validation`. One fold, no purge: only safe when no label crosses a session.
    This class replaces `SingleSplit` under the key "single" (same fields and `assign`, plus the new ones).
    """

    scheme_type: ClassVar[str] = "single"
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2
    gap_days: int = 0
    test_block: Any = None
    score_test: bool = False

    def __post_init__(self) -> None:
        if self.test_block is None:
            share = float(self.test)
        else:
            share = _parse_share(self.test_block, "test_block")
            rest = float(self.train) + float(self.validation)
            object.__setattr__(self, "train", (1 - share) * float(self.train) / rest)
            object.__setattr__(self, "validation", (1 - share) * float(self.validation) / rest)
            object.__setattr__(self, "test", share)
        object.__setattr__(self, "test_block", share)

    def assign(self, decision_ts: pd.Series) -> pd.Series:
        """Split name ("train", "validation", "test", "unassigned") per row, by whole sessions."""
        return split_by_day(
            decision_ts, train=self.train, validation=self.validation, test=self.test, gap_days=self.gap_days
        )

    def test_mask(self, meta: pd.DataFrame) -> np.ndarray:
        """True for rows `assign` puts in the test block."""
        return (self.assign(_utc(meta, "decision_ts")) == "test").to_numpy()

    def folds(self, meta: pd.DataFrame) -> Iterator[Fold]:
        """Yield the single train/validation fold; the test block is in neither."""
        names = self.assign(_utc(meta, "decision_ts")).to_numpy()
        train, val = np.flatnonzero(names == "train"), np.flatnonzero(names == "validation")
        if not len(train) or not len(val):
            raise ValueError("single split left no train or validation rows")
        yield _make_fold(0, meta, train, val, pd.Timedelta(days=int(self.gap_days)))


# C1a registered `SingleSplit` under "single"; replace it with the superset (same fields, adds test_block).
FIT_SCHEME_REGISTRY["single"] = SingleScheme
