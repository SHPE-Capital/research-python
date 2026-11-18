"""Group-aware evolutionary search over the breakout grid (MO-D3).

A seeded ask/tell loop: a broad initial sample, then rounds of mutation/recombination and broad
proposals, of which ``evals_per_round`` are selected (at random before the surrogate is ready,
by the surrogate's predicted mean and disagreement afterwards). Every proposal, selection,
evaluation and rule adjustment is written to an audit trail. All randomness comes from
``np.random.default_rng(seed)`` (plus the per-round surrogate seed), ties sort by ``config_id``,
so the same inputs give a byte-identical audit trail. Every tunable is a protocol field.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, Protocol

import numpy as np

from shpe_research.calibration.surrogate import Surrogate, encode, rank_usefulness
from shpe_research.config.momentum import SearchSection, SurrogateSection
from shpe_research.strategies.momentum.breakout.grid import Config, GridDomain, Number, config_id, is_valid

SearchStatus = Literal["running", "complete", "pool_exhausted", "incomplete_time_limit"]
AUDIT_SCHEMA = "search_audit.v1"
_MIN_FIT_ROWS = 2  # Surrogate.fit needs at least two rows (structural, not a tunable)
_EVO_ORIGINS = ("mutation", "recombination")


@dataclass(frozen=True)
class EvalOutcome:
    """Result of evaluating one config at the search stage."""

    config_id: str
    score: float  # search-stage J under the policy's search weights
    feasible: bool  # passed acceptance.search_eligibility
    reasons: tuple[str, ...] = ()
    cache_hit: bool = False  # True if no chunk replay was needed


class SearchEvaluator(Protocol):
    """Scores a batch of configs on the search window (one outcome per config, any order)."""

    def evaluate(self, configs: Sequence[Config]) -> list[EvalOutcome]:
        """One outcome per config (any order). Timing: search-window data known at the cutoff only."""
        ...


@dataclass(frozen=True)
class ScoredConfig:
    """A feasible evaluated config and its search score."""

    config: Config
    config_id: str
    score: float


@dataclass(frozen=True)
class AuditRecord:
    """One audit line; ``round`` 0 is the initial sample."""

    kind: Literal["proposal", "selection", "evaluation", "adjustment", "surrogate", "status", "finalists"]
    seed: int
    round: int
    data: dict[str, Any]

    def to_json(self) -> str:
        """Return the record as one canonical JSON line.

        Output: ``{"schema": "search_audit.v1", "kind", "seed", "round", **data}`` with sorted keys.
        ``data`` holds only JSON-safe values (a non-finite score is written as null by
        ``SearchRun.tell``), so the line parses under strict JSON. Pure; depends only on the record.
        """
        return json.dumps(
            {"schema": AUDIT_SCHEMA, "kind": self.kind, "seed": self.seed, "round": self.round, **self.data},
            sort_keys=True,
        )


@dataclass(frozen=True)
class SeedResult:
    """Outcome of one seed's search."""

    seed: int
    status: SearchStatus
    finalists: tuple[ScoredConfig, ...]  # <= finalists_per_seed, feasible only, by (-score, config_id)
    logical_evaluations: int
    actual_evaluations: int  # outcomes with cache_hit False
    surrogate_usefulness: tuple[float, ...]  # one per round where a prediction existed (nan allowed)
    audit: tuple[AuditRecord, ...]


@dataclass(frozen=True)
class _Proposal:
    config: Config
    config_id: str
    origin: str
    parents: tuple[str, ...]


def _half_up(x: float) -> int:
    """Round half up: ``floor(x + 0.5)``."""
    return math.floor(x + 0.5)


def _subsets(domain: GridDomain) -> list[tuple[str, ...]]:
    """Valid component subsets in canonical order (same order as ``grid.enumerate_configs``)."""
    optional = [n for n in domain.component_names if n not in domain.required]
    out = []
    for k in range(domain.min_components, domain.max_components + 1):
        extra = k - len(domain.required)
        if 0 <= extra <= len(optional):
            out.extend(tuple(sorted((*domain.required, *c))) for c in combinations(optional, extra))
    return out


def _pick(rng: np.random.Generator, items: Sequence[Any], k: int) -> list[Any]:
    """``k`` items uniformly without replacement, in draw order."""
    if k <= 0:
        return []
    return [items[i] for i in rng.choice(len(items), size=k, replace=False)]


def _nearest(value: Number, vals: Sequence[Number]) -> Number:
    """The domain value equal to ``value``, else the nearest one (lower on ties)."""
    for v in vals:
        if v == value:
            return v
    return min(vals, key=lambda v: (abs(v - value), v))


def _fix_ema(params: dict[str, Number], domain: GridDomain, rng: np.random.Generator) -> None:
    fast, slow = params["ema.fast"], params["ema.slow"]
    if slow > fast:
        return
    dom = dict(domain.params_of("ema"))
    if bigger := [s for s in dom["slow"] if s > fast]:
        params["ema.slow"] = min(bigger)
    elif smaller := [f for f in dom["fast"] if f < slow]:
        params["ema.fast"] = max(smaller)
    else:
        pairs = [(f, s) for f in dom["fast"] for s in dom["slow"] if s > f]
        if not pairs:
            raise ValueError("domain has no EMA pair with slow > fast")
        params["ema.fast"], params["ema.slow"] = pairs[int(rng.integers(len(pairs)))]


def repair(subset: Iterable[str], params: Mapping[str, float], domain: GridDomain, rng: np.random.Generator) -> Config:
    """Map an arbitrary (subset, params) draft onto a valid config of ``domain``.

    Inputs: draft component names and flat ``component.param`` values; ``rng`` supplies the random
    choices. Rules: drop unknown components, add required ones, add (too few) or drop (too many)
    random optional components, fill missing params with a random domain value, snap off-domain
    values to the nearest domain value (lower on ties), drop params of inactive components, and fix
    EMA ``slow <= fast``. Output: a ``Config`` with ``is_valid(config, domain)``. Timing: pure
    function of its arguments; reads no data.
    """
    names = domain.component_names
    optional = [c for c in names if c not in domain.required]
    chosen = {c for c in subset if c in names} | set(domain.required)
    while len(chosen) < domain.min_components:
        free = [c for c in optional if c not in chosen]
        if not free:
            raise ValueError("domain cannot reach min_components")
        chosen.add(free[int(rng.integers(len(free)))])
    while len(chosen) > domain.max_components:
        active = [c for c in optional if c in chosen]
        if not active:
            raise ValueError("required components exceed max_components")
        chosen.remove(active[int(rng.integers(len(active)))])
    out: dict[str, Number] = {}
    for comp in names:
        if comp in chosen:
            for p, vals in domain.params_of(comp):
                key = f"{comp}.{p}"
                out[key] = _nearest(params[key], vals) if key in params else vals[int(rng.integers(len(vals)))]
    if "ema" in chosen:
        _fix_ema(out, domain, rng)
    return Config.make(domain.version, chosen, out)


def _draw_unseen(
    domain: GridDomain, subset: tuple[str, ...], seen: AbstractSet[str], rng: np.random.Generator
) -> tuple[str, Config] | None:
    """A uniformly random valid config of ``subset`` whose ID is not in ``seen``, or None if none is left.

    Scans the subset's parameter grid in a random permutation and returns the first unseen valid
    config, which is uniform over the unseen ones without hashing the whole subset.
    """
    keys: list[str] = []
    value_lists: list[tuple[Number, ...]] = []
    for comp in domain.component_names:
        if comp in subset:
            for p, vals in domain.params_of(comp):
                keys.append(f"{comp}.{p}")
                value_lists.append(vals)
    size = math.prod(len(v) for v in value_lists)
    for flat in rng.permutation(size):
        rest, params = int(flat), {}
        for key, vals in zip(keys, value_lists, strict=True):
            rest, idx = divmod(rest, len(vals))
            params[key] = vals[idx]
        cfg = Config.make(domain.version, subset, params)
        if is_valid(cfg, domain) and (cid := config_id(cfg)) not in seen:
            return cid, cfg
    return None


def sample_broad(domain: GridDomain, n: int, rng: np.random.Generator, exclude: AbstractSet[str]) -> list[Config]:
    """Draw up to ``n`` distinct valid configs whose IDs are not in ``exclude``.

    Subsets are visited round-robin from a random offset in canonical order; parameters are uniform
    over the subset's domain, then ``repair``ed. When a draw collides, the config is drawn uniformly
    from the subset's unseen configs instead (the same distribution as redrawing until new); a subset
    with none left is skipped. Output: list of fewer than ``n`` configs only when the domain minus
    ``exclude`` is exhausted. Timing: pure function of its arguments; reads no data.
    """
    subsets = _subsets(domain)
    out: list[Config] = []
    if n <= 0 or not subsets:
        return out
    seen = set(exclude)
    exhausted: set[tuple[str, ...]] = set()
    i = int(rng.integers(len(subsets)))
    while len(out) < n and len(exhausted) < len(subsets):
        sub = subsets[i % len(subsets)]
        i += 1
        if sub in exhausted:
            continue
        params = {
            f"{c}.{p}": vals[int(rng.integers(len(vals)))]
            for c in domain.component_names
            if c in sub
            for p, vals in domain.params_of(c)
        }
        cfg = repair(sub, params, domain, rng)
        cid = config_id(cfg)
        if cid in seen:
            drawn = _draw_unseen(domain, sub, seen, rng)
            if drawn is None:
                exhausted.add(sub)
                continue
            cid, cfg = drawn
        seen.add(cid)
        out.append(cfg)
    return out


def _mutate(cfg: Config, domain: GridDomain, rng: np.random.Generator) -> Config:
    """One mutation: change one active param or toggle one optional component, then repair."""
    params = cfg.as_dict()
    subset = set(cfg.subset)
    genes: list[tuple[str, str, tuple[Number, ...]]] = [
        (comp, f"{comp}.{p}", vals)
        for comp in domain.component_names
        if comp in subset
        for p, vals in domain.params_of(comp)
    ]
    genes += [(comp, "", ()) for comp in domain.component_names if comp not in domain.required]
    comp, key, vals = genes[int(rng.integers(len(genes)))]
    if key:
        others = [v for v in vals if v != params[key]]
        if others:
            params[key] = others[int(rng.integers(len(others)))]
    elif comp in subset:
        subset.remove(comp)
    else:
        subset.add(comp)  # repair draws its params at random
    return repair(subset, params, domain, rng)


def _recombine(a: Config, b: Config, domain: GridDomain, rng: np.random.Generator) -> Config:
    """Subset from A or B by coin; each param from a parent that has it (coin if both); then mutate."""
    subset = a.subset if rng.integers(2) == 0 else b.subset
    pa, pb = a.as_dict(), b.as_dict()
    params: dict[str, Number] = {}
    for comp in domain.component_names:
        if comp in subset:
            for p, _ in domain.params_of(comp):
                key = f"{comp}.{p}"
                if key in pa and key in pb:
                    params[key] = pa[key] if rng.integers(2) == 0 else pb[key]
                elif key in pa or key in pb:
                    params[key] = pa.get(key, pb.get(key))
    return _mutate(repair(subset, params, domain, rng), domain, rng)


def _round_seed(seed: int, r: int) -> int:
    return int(np.random.SeedSequence([seed, r]).generate_state(1)[0])


class SearchRun:
    """Seeded ask/tell search for one seed.

    ``ask`` returns the next batch to evaluate (``[]`` once finished); ``tell`` must return outcomes
    for exactly that batch. ``budget`` and ``max_wall_seconds`` override the section's budget and
    add a wall-clock stop checked at each ``ask`` (round boundary).

    Attributes:
        budget: Logical evaluations allowed (cache hits count).
        status: ``running`` until the search stops.
        proposals: The latest round's proposals in proposal order (initial sample or evolution plus
            broad), before selection; read-only, for inspection and tests.
    """

    def __init__(
        self,
        domain: GridDomain,
        search: SearchSection,
        surrogate: SurrogateSection,
        seed: int,
        *,
        budget: int | None = None,
        max_wall_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start a search for one seed.

        Inputs: the grid domain, the search and surrogate sections, the seed (the only randomness
        source, ``np.random.default_rng(seed)``), an optional budget override (>= 1, else
        ValueError), an optional wall-clock limit in seconds and the clock it is read from (its value
        now is the start time). Nothing is proposed or evaluated until ``ask``. Timing: holds no data.
        """
        self.budget = search.budget if budget is None else budget
        if self.budget < 1:
            raise ValueError("budget must be >= 1")
        self.status: SearchStatus = "running"
        self._domain, self._search, self._sur_cfg, self._seed = domain, search, surrogate, seed
        self._max_wall, self._clock = max_wall_seconds, clock
        self._start = clock()
        self._rng = np.random.default_rng(seed)
        self._round = 0
        self._results: dict[str, tuple[Config, EvalOutcome]] = {}
        self._pending: dict[str, Config] = {}
        self._pending_pred: dict[str, float] | None = None
        self._n_train = 0
        self._logical = 0
        self._actual = 0
        self._usefulness: list[float] = []
        self._audit: list[AuditRecord] = []
        self.proposals: tuple[Config, ...] = ()  # the latest round's proposals

    @property
    def done(self) -> bool:
        """True once the search has stopped (any status other than ``running``)."""
        return self.status != "running"

    def _log(self, kind: Any, rnd: int | None = None, **data: Any) -> None:
        self._audit.append(AuditRecord(kind, self._seed, self._round if rnd is None else rnd, data))

    def ask(self) -> list[Config]:
        """Next batch of configs to evaluate, in selection order; ``[]`` when finished.

        Checks the wall clock first (round boundary). Raises ``RuntimeError`` if the previous batch
        has not been told. Timing: uses only outcomes already told.
        """
        if self._pending:
            raise RuntimeError("ask called before the previous batch was told")
        if self.done:
            return []
        if self._max_wall is not None and self._clock() - self._start >= self._max_wall:
            self._finish("incomplete_time_limit")
            return []
        remaining = self.budget - self._logical
        if self._round == 0 and self._search.initial_samples > 0:
            selected = self._initial(remaining)
        else:
            selected = self._evolve(remaining)
        if not selected:
            self._finish("pool_exhausted")
            return []
        self._pending = {config_id(c): c for c in selected}
        self._round += 1
        return selected

    def _initial(self, remaining: int) -> list[Config]:
        want = self._search.initial_samples
        n = min(want, remaining)
        configs = sample_broad(self._domain, n, self._rng, frozenset(self._results))
        self.proposals = tuple(configs)
        for c in configs:
            self._log("proposal", config_id=config_id(c), origin="initial", parents=[])
        if n < want:
            self._log("adjustment", rule="budget_truncated", requested=want, got=n)
        if len(configs) < n:
            self._log(
                "adjustment",
                rule="pool_exhausted" if not configs else "proposal_shortfall",
                requested=n,
                got=len(configs),
            )
        for c in configs:
            self._log("selection", config_id=config_id(c), reason="initial", pred_mean=None, pred_std=None)
        return configs

    def _population(self) -> list[Config]:
        s = self._search
        ranked = sorted(self._results.items(), key=lambda kv: (-kv[1][1].score, kv[0]))
        feasible = [cfg for _, (cfg, o) in ranked if o.feasible]
        infeasible = [cfg for _, (cfg, o) in ranked if not o.feasible]
        cap = math.floor(s.population * s.population_max_subset_share)
        pop: list[Config] = []
        leftover: list[Config] = []
        per_subset: Counter[tuple[str, ...]] = Counter()
        for cfg in feasible:
            if len(pop) < s.population and per_subset[cfg.subset] < cap:
                pop.append(cfg)
                per_subset[cfg.subset] += 1
            else:
                leftover.append(cfg)
        return pop + (leftover + infeasible)[: s.population - len(pop)]

    def _evo(self, k: int, pop: list[Config], taken: set[str], attempts: list[int], cap: int) -> list[_Proposal]:
        out: list[_Proposal] = []
        while pop and len(out) < k and attempts[0] < cap:
            attempts[0] += 1
            if len(pop) >= 2 and self._rng.random() < self._search.recombination_share:
                i, j = (int(x) for x in self._rng.choice(len(pop), size=2, replace=False))
                child = _recombine(pop[i], pop[j], self._domain, self._rng)
                origin, parents = "recombination", (config_id(pop[i]), config_id(pop[j]))
            else:
                i = int(self._rng.integers(len(pop)))
                child = _mutate(pop[i], self._domain, self._rng)
                origin, parents = "mutation", (config_id(pop[i]),)
            cid = config_id(child)
            if cid not in taken:
                taken.add(cid)
                out.append(_Proposal(child, cid, origin, parents))
        return out

    def _propose(self) -> list[_Proposal]:
        s = self._search
        total = s.proposals_per_round
        n_evo = _half_up(total * s.mutation_share)
        pop = self._population()
        taken = set(self._results)
        attempts, cap = [0], s.max_proposal_attempts_factor * total
        props = self._evo(n_evo, pop, taken, attempts, cap)
        broad = sample_broad(self._domain, total - len(props), self._rng, taken)
        for c in broad:
            cid = config_id(c)
            taken.add(cid)
            props.append(_Proposal(c, cid, "broad", ()))
        if len(props) < total:
            props += self._evo(total - len(props), pop, taken, attempts, cap)
        self.proposals = tuple(p.config for p in props)
        for p in props:
            self._log("proposal", config_id=p.config_id, origin=p.origin, parents=list(p.parents))
        got_evo = sum(p.origin in _EVO_ORIGINS for p in props)
        if got_evo != n_evo and props:
            self._log("adjustment", rule="share_rebalanced", requested=n_evo, got=got_evo)
        if not props:
            self._log("adjustment", rule="pool_exhausted", requested=total, got=0)
        elif len(props) < total:
            self._log("adjustment", rule="proposal_shortfall", requested=total, got=len(props))
        return props

    def _evolve(self, remaining: int) -> list[Config]:
        props = self._propose()
        if not props:
            return []
        s = self._search
        slots = min(s.evals_per_round, remaining)
        feas = sorted((cid, cfg, o.score) for cid, (cfg, o) in self._results.items() if o.feasible)
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        self._pending_pred = None
        if len(feas) >= max(s.surrogate_min_results, _MIN_FIT_ROWS):
            model = Surrogate(self._sur_cfg, _round_seed(self._seed, self._round))
            model.fit(encode([f[1] for f in feas], self._domain), np.array([f[2] for f in feas], dtype=np.float64))
            pred = model.predict(encode([p.config for p in props], self._domain))
            mean = {p.config_id: float(m) for p, m in zip(props, pred.mean, strict=True)}
            std = {p.config_id: float(d) for p, d in zip(props, pred.std, strict=True)}
            self._n_train = len(feas)
        ids = [p.config_id for p in props]
        if len(props) <= slots:
            picks = [(cid, "pool_remainder") for cid in ids]
        else:
            if slots < s.evals_per_round:
                self._log("adjustment", rule="budget_truncated", requested=s.evals_per_round, got=slots)
            picks = self._pick_surrogate(ids, mean, std, slots) if mean else self._pick_random(props, slots)
        by_id = {p.config_id: p.config for p in props}
        for cid, reason in picks:
            self._log("selection", config_id=cid, reason=reason, pred_mean=mean.get(cid), pred_std=std.get(cid))
        if mean:
            self._pending_pred = {cid: mean[cid] for cid, _ in picks}
        return [by_id[cid] for cid, _ in picks]

    def _pick_random(self, props: list[_Proposal], slots: int) -> list[tuple[str, str]]:
        evo = [p.config_id for p in props if p.origin in _EVO_ORIGINS]
        broad = [p.config_id for p in props if p.origin not in _EVO_ORIGINS]
        q_evo = min(_half_up(self._search.evals_per_round * self._search.mutation_share), slots, len(evo))
        q_broad = min(slots - q_evo, len(broad))
        q_evo = min(slots - q_broad, len(evo))
        return [(c, "pre_surrogate_evolution") for c in _pick(self._rng, evo, q_evo)] + [
            (c, "pre_surrogate_broad") for c in _pick(self._rng, broad, q_broad)
        ]

    def _pick_surrogate(
        self, ids: list[str], mean: dict[str, float], std: dict[str, float], slots: int
    ) -> list[tuple[str, str]]:
        quotas, left = [], slots
        for q in self._search.evals_split:
            quotas.append(min(q, left))
            left -= quotas[-1]
        top = sorted(ids, key=lambda c: (-mean[c], c))[: quotas[0]]
        top_set = set(top)
        rest = [c for c in ids if c not in top_set]
        dis = sorted(rest, key=lambda c: (-std[c], c))[: quotas[1]]
        dis_set = set(dis)
        rest = [c for c in rest if c not in dis_set]
        rnd = _pick(self._rng, rest, min(quotas[2], len(rest)))
        return [(c, "top_predicted") for c in top] + [(c, "disagreement") for c in dis] + [(c, "random") for c in rnd]

    def tell(self, outcomes: Sequence[EvalOutcome]) -> None:
        """Record outcomes for exactly the last ``ask`` batch (any order).

        Each outcome counts one logical evaluation (cache hits included); outcomes with
        ``cache_hit`` False also count as actual. Scores: a NaN score, or a non-finite score on a
        feasible outcome, raises ``ValueError``; an infeasible outcome may carry ``-inf`` (invalid
        accounting), which the audit records as ``score: null`` so every line is strict JSON. The
        round's surrogate usefulness compares predictions with scores of the feasible outcomes only
        (ledger ruling P8-2). Raises ``ValueError`` if the IDs do not match the batch; nothing is
        recorded when a check fails. Timing: outcomes must be search-stage results known at the
        calibration cutoff.
        """
        if not self._pending:
            raise RuntimeError("tell called without a pending ask")
        ids = [o.config_id for o in outcomes]
        if len(ids) != len(set(ids)) or set(ids) != set(self._pending):
            raise ValueError("outcomes must cover exactly the last ask")
        for o in outcomes:
            if math.isnan(o.score) or (o.feasible and not math.isfinite(o.score)):
                raise ValueError(f"invalid score {o.score!r} for {o.config_id} (feasible={o.feasible})")
        asked_round = self._round - 1
        ordered = sorted(outcomes, key=lambda o: o.config_id)
        for o in ordered:
            self._log(
                "evaluation",
                asked_round,
                config_id=o.config_id,
                score=float(o.score) if math.isfinite(o.score) else None,
                feasible=bool(o.feasible),
                reasons=list(o.reasons),
                cache_hit=bool(o.cache_hit),
                logical_index=self._logical,
            )
            self._logical += 1
            self._actual += not o.cache_hit
            self._results[o.config_id] = (self._pending[o.config_id], o)
        if self._pending_pred is not None:
            feasible = [o for o in ordered if o.feasible]
            u = rank_usefulness(
                np.array([self._pending_pred[o.config_id] for o in feasible]), np.array([o.score for o in feasible])
            )
            self._usefulness.append(u)
            self._log("surrogate", asked_round, n_train=self._n_train, usefulness=None if math.isnan(u) else u)
        self._pending, self._pending_pred = {}, None
        if self._logical >= self.budget:
            self._finish("complete")

    def _finalists(self) -> tuple[ScoredConfig, ...]:
        feasible = sorted((-o.score, cid, cfg) for cid, (cfg, o) in self._results.items() if o.feasible)
        return tuple(ScoredConfig(cfg, cid, -neg) for neg, cid, cfg in feasible[: self._search.finalists_per_seed])

    def _finish(self, status: SearchStatus) -> None:
        self.status = status
        fin = self._finalists()
        self._log("finalists", config_ids=[f.config_id for f in fin], scores=[f.score for f in fin])
        self._log("status", status=status, logical=self._logical, actual=self._actual)

    def result(self) -> SeedResult:
        """Snapshot of the run: status, finalists, evaluation counts, usefulness and audit trail.

        Callable at any time; ``status`` is ``running`` until the search stops. Timing: reflects
        only outcomes already told.
        """
        return SeedResult(
            seed=self._seed,
            status=self.status,
            finalists=self._finalists(),
            logical_evaluations=self._logical,
            actual_evaluations=self._actual,
            surrogate_usefulness=tuple(self._usefulness),
            audit=tuple(self._audit),
        )


def run_search(
    domain: GridDomain,
    search: SearchSection,
    surrogate: SurrogateSection,
    seed: int,
    evaluator: SearchEvaluator,
    **kw: Any,
) -> SeedResult:
    """Drive a ``SearchRun`` with ``evaluator`` until it stops; return its ``SeedResult``.

    ``kw`` is passed to ``SearchRun`` (``budget``, ``max_wall_seconds``, ``clock``). Timing: the
    evaluator must score each config on search-window data known at the cutoff.
    """
    run = SearchRun(domain, search, surrogate, seed, **kw)
    while configs := run.ask():
        run.tell(evaluator.evaluate(configs))
    return run.result()
