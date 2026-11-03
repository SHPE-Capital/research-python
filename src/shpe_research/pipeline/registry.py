"""Registries for the pipeline's pluggable stages (same pattern as features/base.py).

    sampler       decisions: which bars become rows                     (events/, C1b)
    label         labels: the outcome each decision is scored on        (labels/, C1c)
    fit_scheme    fit_scheme: how rows are split into train/validation/test (validation/, C1f)
    report_section  evaluate: one block of the run report               (evaluation/, C1i)

Each `build_*` pops "type", imports the owning package so its built-ins
register, then raises ValueError listing the registered keys when the key is
unknown. C1a registers only what keeps the hello model working
(`every_n_minutes`, `forward_return`, `single`, `signal`); the builders add
these lazily and skip a key the owning package already registered, so later
packets can register the same classes in their own packages.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

SAMPLER_REGISTRY: dict[str, type] = {}
LABEL_REGISTRY: dict[str, type] = {}
FIT_SCHEME_REGISTRY: dict[str, type] = {}
REPORT_SECTION_REGISTRY: dict[str, type] = {}


def _decorator(registry: dict[str, type], key_attr: str) -> Callable[[type], type]:
    def register(cls: type) -> type:
        key = getattr(cls, key_attr)
        if key in registry:
            raise ValueError(f"{key_attr} {key!r} is already registered")
        registry[key] = cls
        return cls

    return register


register_sampler = _decorator(SAMPLER_REGISTRY, "sampler_type")
register_label = _decorator(LABEL_REGISTRY, "label_type")
register_fit_scheme = _decorator(FIT_SCHEME_REGISTRY, "scheme_type")
register_report_section = _decorator(REPORT_SECTION_REGISTRY, "section_type")


# ---- built-ins that keep the hello model working -------------------------------------------------


@dataclass(frozen=True)
class EveryNMinutes:
    """Decide on every bar whose minutes since the session open is a multiple of `minutes`."""

    sampler_type: ClassVar[str] = "every_n_minutes"
    minutes: int = 1

    def mask(self, decision_ts: pd.Series) -> pd.Series:
        from shpe_research.features.session.minute_of_session import minutes_since_open

        return (minutes_since_open(decision_ts) % int(self.minutes)) == 0


@dataclass(frozen=True)
class ForwardReturnLabel:
    """Log return from the decision bar's close to the close `horizon_minutes` later (same session)."""

    label_type: ClassVar[str] = "forward_return"
    horizon_minutes: int

    @property
    def name(self) -> str:
        return f"fwd_return_{self.horizon_minutes}m"

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """symbol, decision_ts, label_end_ts and the label column for every symbol in `bars`."""
        from shpe_research.labels.forward_return import ForwardReturn, compute_labels

        return compute_labels(bars, ForwardReturn(int(self.horizon_minutes)))


@dataclass(frozen=True)
class SingleSplit:
    """One chronological train/validation/test split by whole sessions."""

    scheme_type: ClassVar[str] = "single"
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2
    gap_days: int = 0

    def assign(self, decision_ts: pd.Series) -> pd.Series:
        """Split name per row."""
        from shpe_research.validation.splits import split_by_day

        return split_by_day(
            decision_ts, train=self.train, validation=self.validation, test=self.test, gap_days=self.gap_days
        )


@dataclass(frozen=True)
class SignalSection:
    """The C0 report: prediction quality, daily IC and deciles for each evaluated split."""

    section_type: ClassVar[str] = "signal"
    splits: tuple[str, ...] = ("train", "validation", "test")

    def evaluate(self, predictions: pd.DataFrame, task: str) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
        from shpe_research.evaluation import report

        metrics: dict[str, dict] = {}
        deciles: dict[str, pd.DataFrame] = {}
        for split in self.splits:
            part = predictions[predictions["split"] == split]
            if part.empty:
                continue
            metrics[split] = report.evaluate(part, task)
            deciles[split] = report.decile_table(part)
        return metrics, deciles


_BUILTINS: list[tuple[dict[str, type], str, type]] = [
    (SAMPLER_REGISTRY, "sampler_type", EveryNMinutes),
    (LABEL_REGISTRY, "label_type", ForwardReturnLabel),
    (FIT_SCHEME_REGISTRY, "scheme_type", SingleSplit),
    (REPORT_SECTION_REGISTRY, "section_type", SignalSection),
]


def _load(registry: dict[str, type], package: str) -> None:
    importlib.import_module(package)  # the owning package registers its own built-ins
    for reg, attr, cls in _BUILTINS:
        if reg is registry:
            reg.setdefault(getattr(cls, attr), cls)


def _build(registry: dict[str, type], package: str, kind: str, spec: dict[str, Any]) -> Any:
    _load(registry, package)
    params = dict(spec)
    key = params.pop("type")
    try:
        cls = registry[key]
    except KeyError:
        raise ValueError(f"unknown {kind} {key!r}; registered: {sorted(registry)}") from None
    return cls(**params)


def build_sampler(spec: dict[str, Any]) -> Any:
    return _build(SAMPLER_REGISTRY, "shpe_research.events", "sampler", spec)


def build_label(spec: dict[str, Any]) -> Any:
    return _build(LABEL_REGISTRY, "shpe_research.labels", "label", spec)


def build_fit_scheme(spec: dict[str, Any]) -> Any:
    return _build(FIT_SCHEME_REGISTRY, "shpe_research.validation", "fit scheme", spec)


def build_report_section(spec: dict[str, Any]) -> Any:
    return _build(REPORT_SECTION_REGISTRY, "shpe_research.evaluation", "report section", spec)
