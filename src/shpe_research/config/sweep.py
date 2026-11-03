"""Sweep config and the `load_config` entry point (C1h)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shpe_research.config.experiment import ExperimentConfig
from shpe_research.config.strategy import StrategySpec


class OptunaConfig(BaseModel):
    """Optuna search settings; every trial counts toward the trial count (AGENTS.md rule 3)."""

    model_config = ConfigDict(extra="forbid")

    n_trials: int = Field(gt=0)
    sampler: str = "tpe"
    seed: int = 0


class SweepConfig(BaseModel):
    """A base experiment plus either a `grid` of overrides or an `optuna` search."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["sweep"] = "sweep"
    base: ExperimentConfig
    grid: dict[str, list[Any]] | None = None
    optuna: OptunaConfig | None = None
    metric: str
    direction: Literal["maximize", "minimize"] = "maximize"

    @model_validator(mode="after")
    def _one_search(self) -> SweepConfig:
        if (self.grid is None) == (self.optuna is None):
            raise ValueError("give exactly one of 'grid' or 'optuna'")
        return self


def load_config(path: str | Path) -> ExperimentConfig | StrategySpec | SweepConfig:
    """Load and validate a YAML config; the top-level `kind` (experiment | strategy | sweep) picks the model.

    A missing `kind` means experiment. Raises pydantic `ValidationError` naming the dotted path of
    any unknown or invalid key, and ValueError for an unknown `kind`. Reads the file only.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    kind = raw.get("kind", "experiment")
    models = {"experiment": ExperimentConfig, "strategy": StrategySpec, "sweep": SweepConfig}
    if kind not in models:
        raise ValueError(f"{path}: unknown kind {kind!r}; expected one of {sorted(models)}")
    return models[kind].model_validate(raw)
