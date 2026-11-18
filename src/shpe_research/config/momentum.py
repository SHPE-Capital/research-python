"""Typed 0-G protocol config for the momentum strategy's weekly calibration (stream D slice).

This slice holds only the sections stream D reads: weights, score, search, surrogate,
validation, acceptance, cache, compute and walkforward. The sections `universe`, `data`,
`eligibility`, `grid`, `signals`, `fill`, `costs`, `exit`, `account`, `bootstrap` and
`quote_replay` are added by later 0-G work, and the grid domain joins the hash then.
Until then account settings are the `AccountConfig()` defaults (`TODO(MO-0G-rest)`).

The acceptance floors are provisional guide values pending a real-panel run and are
expected to rise with the panel size N.

Every model forbids unknown keys and is frozen. This module sits in the lowest import
layer: it imports nothing from `shpe_research`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

Policy = Literal["uniform", "recency", "similarity"]
Granularity = Literal["chunk", "session"]
DEFAULT_PROTOCOL_PATH: Path = Path(__file__).resolve().parents[3] / "configs" / "momentum" / "protocol.default.yaml"

_ID_RE = re.compile(r"^[a-z0-9_]+$")


class FrozenProtocolError(ValueError):
    """Raised when overrides are applied to a frozen protocol (or try to set `frozen`)."""


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WeightsSection(_Section):
    uniform_mix: float = 0.20
    ess_target_fraction: float = 0.60
    ess_tolerance: float = 0.01
    solver_bracket: tuple[float, float] = (1e-3, 1e4)
    solver_max_iter: int = 100
    iqr_floor: float = 1e-12
    descriptor_lookback: int = 20

    @model_validator(mode="after")
    def _validate(self) -> WeightsSection:
        _check(0 <= self.uniform_mix <= 1, "weights.uniform_mix must be in [0, 1]")
        _check(0 < self.ess_target_fraction <= 1, "weights.ess_target_fraction must be in (0, 1]")
        _check(self.ess_tolerance >= 0, "weights.ess_tolerance must be >= 0")
        lo, hi = self.solver_bracket
        _check(0 < lo < hi, "weights.solver_bracket must satisfy 0 < low < high")
        _check(self.solver_max_iter >= 1, "weights.solver_max_iter must be >= 1")
        _check(self.iqr_floor >= 0, "weights.iqr_floor must be >= 0")
        _check(self.descriptor_lookback >= 1, "weights.descriptor_lookback must be >= 1")
        return self


class ScoreSection(_Section):
    downside_coeff: float = 0.5

    @model_validator(mode="after")
    def _validate(self) -> ScoreSection:
        _check(self.downside_coeff >= 0, "score.downside_coeff must be >= 0")
        return self


class StrategyEntry(_Section):
    id: str
    base: str = "breakout"
    budget: int | None = None
    seeds: list[int] | None = None
    account_overrides: dict[str, float | int | str] = {}

    @model_validator(mode="after")
    def _validate(self) -> StrategyEntry:
        _check(bool(_ID_RE.match(self.id)), f"strategy id {self.id!r} must match [a-z0-9_]+")
        _check(self.budget is None or self.budget >= 1, "strategy budget must be >= 1")
        if self.seeds is not None:
            _check(
                bool(self.seeds) and len(set(self.seeds)) == len(self.seeds),
                "strategy seeds must be non-empty and unique",
            )
        return self


class SearchSection(_Section):
    budget: int = 2048
    seeds: list[int] = [1, 2, 3]
    initial_samples: int = 256
    population: int = 64
    population_max_subset_share: float = 0.25
    proposals_per_round: int = 1024
    mutation_share: float = 0.75
    recombination_share: float = 0.5
    max_proposal_attempts_factor: int = 20
    evals_per_round: int = 64
    evals_split: tuple[int, int, int] = (32, 16, 16)
    surrogate_min_results: int = 128
    finalists_per_seed: int = 12
    reference_equity: float = 100_000.0
    strategies: list[StrategyEntry] = [StrategyEntry(id="breakout")]

    @model_validator(mode="after")
    def _validate(self) -> SearchSection:
        _check(self.budget >= 1, "search.budget must be >= 1")
        _check(self.budget >= self.initial_samples, "search.budget must be >= search.initial_samples")
        _check(self.initial_samples >= 0, "search.initial_samples must be >= 0")
        _check(self.population >= 1, "search.population must be >= 1")
        _check(
            bool(self.seeds) and len(set(self.seeds)) == len(self.seeds),
            "search.seeds must be non-empty and unique",
        )
        for name in ("population_max_subset_share", "mutation_share", "recombination_share"):
            _check(0 <= getattr(self, name) <= 1, f"search.{name} must be in [0, 1]")
        _check(self.proposals_per_round >= 1, "search.proposals_per_round must be >= 1")
        _check(self.max_proposal_attempts_factor >= 1, "search.max_proposal_attempts_factor must be >= 1")
        _check(self.evals_per_round >= 1, "search.evals_per_round must be >= 1")
        _check(all(n >= 0 for n in self.evals_split), "search.evals_split entries must be >= 0")
        _check(
            sum(self.evals_split) == self.evals_per_round,
            "sum(search.evals_split) must equal search.evals_per_round",
        )
        _check(self.surrogate_min_results >= 0, "search.surrogate_min_results must be >= 0")
        _check(self.finalists_per_seed >= 0, "search.finalists_per_seed must be >= 0")
        _check(self.reference_equity > 0, "search.reference_equity must be > 0")
        ids = [s.id for s in self.strategies]
        _check(len(set(ids)) == len(ids), "search.strategies ids must be unique")
        return self


class SurrogateSection(_Section):
    n_models: int = 5
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0

    @model_validator(mode="after")
    def _validate(self) -> SurrogateSection:
        _check(self.n_models >= 1, "surrogate.n_models must be >= 1")
        _check(self.max_iter >= 1, "surrogate.max_iter must be >= 1")
        _check(self.learning_rate > 0, "surrogate.learning_rate must be > 0")
        _check(self.max_leaf_nodes >= 2, "surrogate.max_leaf_nodes must be >= 2")
        _check(self.min_samples_leaf >= 1, "surrogate.min_samples_leaf must be >= 1")
        _check(self.l2_regularization >= 0, "surrogate.l2_regularization must be >= 0")
        return self


class ValidationSection(_Section):
    sessions: int = 63

    @model_validator(mode="after")
    def _validate(self) -> ValidationSection:
        _check(self.sessions >= 1, "validation.sessions must be >= 1")
        return self


class AcceptanceSection(_Section):
    """Provisional floors (guide values) pending a real-panel run; expected to rise with N."""

    search_min_trades: int = 100
    search_min_sessions: int = 50
    validation_min_trades: int = 30
    validation_min_sessions: int = 20
    validation_min_j: float = 0.0  # strict: J > validation_min_j
    max_drawdown: float = 0.10  # pass if <=
    max_single_stock_share: float = 0.30  # pass if <=
    max_finalists: int = 36
    selection_bps: float = 10.0
    report_bps: float = 20.0

    @model_validator(mode="after")
    def _validate(self) -> AcceptanceSection:
        for name in (
            "search_min_trades",
            "search_min_sessions",
            "validation_min_trades",
            "validation_min_sessions",
            "max_finalists",
            "max_drawdown",
            "max_single_stock_share",
            "selection_bps",
            "report_bps",
        ):
            _check(getattr(self, name) >= 0, f"acceptance.{name} must be >= 0")
        return self


class CacheSection(_Section):
    path: str | None = None  # None -> <compute scratch>/momentum_cache.sqlite
    granularity: Granularity = "chunk"
    warmup_sessions: int = 20  # provisional; in the cache key
    busy_timeout_s: float = 30.0

    @model_validator(mode="after")
    def _validate(self) -> CacheSection:
        _check(self.warmup_sessions >= 0, "cache.warmup_sessions must be >= 0")
        _check(self.busy_timeout_s >= 0, "cache.busy_timeout_s must be >= 0")
        return self


class ComputeSection(_Section):
    workers: int = 1
    threads_per_worker: int = 1
    kernel: Literal["fast", "oracle"] = "fast"
    scratch_dir: str | None = None  # resolved by callers, not here
    max_wall_hours: float | None = 36.0

    @model_validator(mode="after")
    def _validate(self) -> ComputeSection:
        _check(self.workers >= 1, "compute.workers must be >= 1")
        _check(self.threads_per_worker >= 1, "compute.threads_per_worker must be >= 1")
        _check(self.max_wall_hours is None or self.max_wall_hours > 0, "compute.max_wall_hours must be > 0")
        return self


class WalkforwardSection(_Section):
    reserved_start: date | None = None  # no default date; the walk-forward task refuses to run while unset
    starting_cash: float = 100_000.0
    policies: list[Policy] = ["uniform", "recency", "similarity"]

    @model_validator(mode="after")
    def _validate(self) -> WalkforwardSection:
        _check(self.starting_cash > 0, "walkforward.starting_cash must be > 0")
        return self


class MomentumProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    frozen: bool = False
    weights: WeightsSection = WeightsSection()
    score: ScoreSection = ScoreSection()
    search: SearchSection = SearchSection()
    surrogate: SurrogateSection = SurrogateSection()
    validation: ValidationSection = ValidationSection()
    acceptance: AcceptanceSection = AcceptanceSection()
    cache: CacheSection = CacheSection()
    compute: ComputeSection = ComputeSection()
    walkforward: WalkforwardSection = WalkforwardSection()


@dataclass(frozen=True)
class ResolvedProtocol:
    protocol: MomentumProtocol
    overrides: tuple[str, ...]
    source: str  # path string or "<defaults>"
    hash: str  # protocol_hash(protocol)


def protocol_hash(p: MomentumProtocol) -> str:
    """64-hex SHA-256 over every field of the protocol (canonical JSON)."""
    blob = json.dumps(p.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def apply_overrides(p: MomentumProtocol, overrides: Sequence[str]) -> MomentumProtocol:
    """Apply `"a.b.c=value"` overrides (values parsed as YAML); validate the whole model once."""
    if not overrides:
        return p
    if p.frozen:
        raise FrozenProtocolError("protocol is frozen; overrides are refused")
    data = copy.deepcopy(p.model_dump())
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"override {item!r} must look like 'section.field=value'")
        if key == "frozen" or key.startswith("frozen."):
            raise FrozenProtocolError("overrides may not set 'frozen'; edit the YAML to freeze")
        parts = key.split(".")
        node: Any = data
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"unknown override path {key!r}")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ValueError(f"unknown override path {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return MomentumProtocol.model_validate(data)


def load_protocol(path: str | Path | None = None, overrides: Sequence[str] = ()) -> ResolvedProtocol:
    """Load a protocol YAML (or the defaults when `path` is None) and apply overrides."""
    if path is None:
        base, source = MomentumProtocol(), "<defaults>"
    else:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        base, source = MomentumProtocol.model_validate(raw or {}), str(path)
    resolved = apply_overrides(base, overrides)
    return ResolvedProtocol(resolved, tuple(overrides), source, protocol_hash(resolved))


def dump_protocol_yaml(p: MomentumProtocol) -> str:
    return yaml.safe_dump(p.model_dump(mode="json"), sort_keys=False)
