"""Tests for the D3 group-aware evolutionary search (ask/tell loop and audit trail)."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from shpe_research.calibration.search import (
    EvalOutcome,
    SearchRun,
    SeedResult,
    repair,
    run_search,
    sample_broad,
)
from shpe_research.config.momentum import SearchSection, SurrogateSection, load_protocol
from shpe_research.strategies.momentum.breakout.grid import (
    DEFAULT_DOMAIN,
    Config,
    GridDomain,
    config_id,
    feature_vector,
    is_valid,
)

SMALL = SearchSection(
    budget=192,
    initial_samples=64,
    population=16,
    proposals_per_round=128,
    evals_per_round=32,
    evals_split=(16, 8, 8),
    surrogate_min_results=48,
    finalists_per_seed=5,
)
SMALL_SUR = SurrogateSection(max_iter=30, min_samples_leaf=5)


class LinearEvaluator:
    """Deterministic score: a fixed linear function of the feature vector."""

    def __init__(
        self,
        domain: GridDomain = DEFAULT_DOMAIN,
        feasible: Callable[[Config], bool] = lambda c: True,
        cache_hit: bool = False,
    ) -> None:
        self.domain = domain
        self.feasible = feasible
        self.cache_hit = cache_hit
        self.calls: list[str] = []

    def evaluate(self, configs: Sequence[Config]) -> list[EvalOutcome]:
        out = []
        for c in configs:
            v = np.array(feature_vector(c, self.domain))
            w = np.linspace(-1.0, 1.0, len(v))
            cid = config_id(c)
            self.calls.append(cid)
            ok = self.feasible(c)
            out.append(EvalOutcome(cid, float(v @ w), ok, () if ok else ("search_floor",), cache_hit=self.cache_hit))
        return out


def _records(res: SeedResult, kind: str, rnd: int | None = None) -> list[dict]:
    return [r.data for r in res.audit if r.kind == kind and (rnd is None or r.round == rnd)]


def _run_rounds(run: SearchRun, ev: LinearEvaluator, rounds: int) -> None:
    for _ in range(rounds):
        run.tell(ev.evaluate(run.ask()))


def _small_domain() -> GridDomain:
    return GridDomain.from_mapping(
        {
            "version": "t-small",
            "components": {"breakout": {"n": [10, 20], "b": [0]}, "vwap": {"e": [1, 2, 3]}},
            "required": ["breakout"],
            "min_components": 1,
            "max_components": 2,
        }
    )


def test_initial_samples_distinct_cover_subsets() -> None:
    rng = np.random.default_rng(0)
    configs = sample_broad(DEFAULT_DOMAIN, 256, rng, frozenset())
    ids = [config_id(c) for c in configs]
    assert len(configs) == 256
    assert len(set(ids)) == 256
    assert all(is_valid(c, DEFAULT_DOMAIN) for c in configs)
    assert len({c.subset for c in configs}) == 14


def test_sample_broad_respects_exclude() -> None:
    first = sample_broad(DEFAULT_DOMAIN, 50, np.random.default_rng(1), frozenset())
    excl = frozenset(config_id(c) for c in first)
    again = sample_broad(DEFAULT_DOMAIN, 50, np.random.default_rng(1), excl)
    assert not excl & {config_id(c) for c in again}


def test_repair_cases() -> None:
    rng = np.random.default_rng(0)
    c = repair(["adx"], {"adx.period": 14, "adx.q": 20}, DEFAULT_DOMAIN, rng)
    assert "breakout" in c.subset and is_valid(c, DEFAULT_DOMAIN)

    five = repair(["breakout", "adx", "ema", "rvol", "vwap"], {}, DEFAULT_DOMAIN, rng)
    assert len(five.subset) <= 4 and "breakout" in five.subset and is_valid(five, DEFAULT_DOMAIN)

    n33 = repair(["breakout", "vwap"], {"breakout.n": 33, "breakout.b": 0.1, "vwap.e": 2}, DEFAULT_DOMAIN, rng)
    assert n33.as_dict()["breakout.n"] == 40

    tie = repair(["breakout", "vwap"], {"breakout.n": 30, "breakout.b": 0.1, "vwap.e": 2}, DEFAULT_DOMAIN, rng)
    assert tie.as_dict()["breakout.n"] == 20  # nearest, lower on ties

    unknown = repair(["breakout", "vwap", "bogus"], {"breakout.n": 10, "ema.fast": 5}, DEFAULT_DOMAIN, rng)
    assert unknown.subset == ("breakout", "vwap") and is_valid(unknown, DEFAULT_DOMAIN)

    dom = GridDomain.from_mapping(
        {
            "version": "t-ema",
            "components": {"breakout": {"n": [10]}, "ema": {"fast": [5, 30], "slow": [20, 35]}},
            "required": ["breakout"],
            "min_components": 2,
            "max_components": 2,
        }
    )
    e = repair(["breakout", "ema"], {"breakout.n": 10, "ema.fast": 30, "ema.slow": 20}, dom, rng)
    assert e.as_dict()["ema.slow"] == 35 and e.as_dict()["ema.fast"] == 30


def test_proposals_distinct_valid_with_shares() -> None:
    run = SearchRun(DEFAULT_DOMAIN, SearchSection(), SurrogateSection(), seed=3)
    ev = LinearEvaluator()
    _run_rounds(run, ev, 1)
    run.ask()
    res = run.result()
    props = _records(res, "proposal", 1)
    ids = [p["config_id"] for p in props]
    assert len(ids) == 1024 and len(set(ids)) == 1024
    assert not set(ids) & set(ev.calls)
    origins = Counter(p["origin"] for p in props)
    assert origins["mutation"] + origins["recombination"] == 768
    assert origins["broad"] == 256
    assert origins["mutation"] > 0 and origins["recombination"] > 0
    assert [config_id(c) for c in run.proposals] == ids
    assert all(is_valid(c, DEFAULT_DOMAIN) for c in run.proposals)


def test_pre_surrogate_split() -> None:
    run = SearchRun(DEFAULT_DOMAIN, SearchSection(), SurrogateSection(), seed=2)
    ev = LinearEvaluator(feasible=lambda c: False)
    _run_rounds(run, ev, 1)
    asked = run.ask()
    reasons = Counter(s["reason"] for s in _records(run.result(), "selection", 1))
    assert len(asked) == 64
    assert reasons == {"pre_surrogate_evolution": 48, "pre_surrogate_broad": 16}


def test_surrogate_split() -> None:
    run = SearchRun(DEFAULT_DOMAIN, SearchSection(), SurrogateSection(max_iter=30), seed=2)
    ev = LinearEvaluator()
    _run_rounds(run, ev, 1)
    run.ask()
    sel = _records(run.result(), "selection", 1)
    assert Counter(s["reason"] for s in sel) == {"top_predicted": 32, "disagreement": 16, "random": 16}
    assert all(s["pred_mean"] is not None and s["pred_std"] is not None for s in sel)


def test_budget_counts_cache_hits() -> None:
    res = run_search(DEFAULT_DOMAIN, SMALL, SMALL_SUR, 4, LinearEvaluator(cache_hit=True))
    assert res.status == "complete"
    assert res.logical_evaluations == 192
    assert res.actual_evaluations == 0


def test_deterministic_per_seed() -> None:
    a = run_search(DEFAULT_DOMAIN, SMALL, SMALL_SUR, 5, LinearEvaluator())
    b = run_search(DEFAULT_DOMAIN, SMALL, SMALL_SUR, 5, LinearEvaluator())
    c = run_search(DEFAULT_DOMAIN, SMALL, SMALL_SUR, 6, LinearEvaluator())
    ja = [r.to_json() for r in a.audit]
    assert ja == [r.to_json() for r in b.audit]
    assert a.finalists == b.finalists
    assert ja != [r.to_json() for r in c.audit]
    first = json.loads(ja[0])
    assert first["schema"] == "search_audit.v1" and first["seed"] == 5


def test_audit_complete() -> None:
    ev = LinearEvaluator(feasible=lambda c: "adx" in c.subset)
    res = run_search(DEFAULT_DOMAIN, SMALL, SMALL_SUR, 7, ev)
    evaluated = [e["config_id"] for e in _records(res, "evaluation")]
    assert len(evaluated) == res.logical_evaluations == len(ev.calls) == 192
    proposed = {p["config_id"] for p in _records(res, "proposal")}
    selected = [s["config_id"] for s in _records(res, "selection")]
    assert set(evaluated) <= proposed
    assert sorted(selected) == sorted(evaluated)
    assert [e["logical_index"] for e in _records(res, "evaluation")] == list(range(192))
    assert len(res.surrogate_usefulness) == len(_records(res, "surrogate")) >= 1
    status = _records(res, "status")
    assert status == [{"status": "complete", "logical": 192, "actual": 192}]


def test_pool_exhausted_small_domain() -> None:
    dom = _small_domain()
    ev = LinearEvaluator(domain=dom)
    res = run_search(dom, SMALL, SMALL_SUR, 1, ev, budget=64)
    assert res.status == "pool_exhausted"
    assert res.logical_evaluations == 8
    assert len(ev.calls) == len(set(ev.calls)) == 8
    assert any(a["rule"] == "pool_exhausted" for a in _records(res, "adjustment"))


def test_no_config_evaluated_twice() -> None:
    ev = LinearEvaluator()
    run_search(DEFAULT_DOMAIN, SMALL, SMALL_SUR, 8, ev)
    assert len(ev.calls) == len(set(ev.calls)) == 192


def test_finalists_top_feasible() -> None:
    ev = LinearEvaluator(feasible=lambda c: "ema" not in c.subset)
    res = run_search(DEFAULT_DOMAIN, SMALL, SMALL_SUR, 9, ev)
    evals = {e["config_id"]: e for e in _records(res, "evaluation")}
    fin = res.finalists
    assert 0 < len(fin) <= SMALL.finalists_per_seed
    assert all(evals[f.config_id]["feasible"] and "ema" not in f.config.subset for f in fin)
    assert [(-f.score, f.config_id) for f in fin] == sorted((-f.score, f.config_id) for f in fin)
    best = sorted((-e["score"], cid) for cid, e in evals.items() if e["feasible"])[: len(fin)]
    assert [cid for _, cid in best] == [f.config_id for f in fin]
    assert _records(res, "finalists") == [{"config_ids": [f.config_id for f in fin], "scores": [f.score for f in fin]}]


def test_time_limit_incomplete() -> None:
    ticks = itertools.count(0, 10)
    res = run_search(
        DEFAULT_DOMAIN, SMALL, SMALL_SUR, 1, LinearEvaluator(), max_wall_seconds=25, clock=lambda: float(next(ticks))
    )
    assert res.status == "incomplete_time_limit"
    assert res.logical_evaluations < SMALL.budget
    kinds = [r.kind for r in res.audit]
    assert kinds[-1] == "status"
    assert _records(res, "status")[0]["status"] == "incomplete_time_limit"


def test_tell_order_independent() -> None:
    def go(shuffle: bool) -> list[str]:
        run = SearchRun(DEFAULT_DOMAIN, SMALL, SMALL_SUR, seed=11)
        ev = LinearEvaluator()
        rng = np.random.default_rng(99)
        while not run.done:
            outcomes = ev.evaluate(run.ask())
            if shuffle:
                outcomes = [outcomes[i] for i in rng.permutation(len(outcomes))]
            run.tell(outcomes)
        return [r.to_json() for r in run.result().audit]

    assert go(False) == go(True)


def test_tell_must_cover_last_ask() -> None:
    run = SearchRun(DEFAULT_DOMAIN, SMALL, SMALL_SUR, seed=1)
    outcomes = LinearEvaluator().evaluate(run.ask())
    try:
        run.tell(outcomes[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("partial tell accepted")


def test_yaml_only_dev_comparison() -> None:
    p = load_protocol(None, ["search.budget=8192", "search.seeds=[1,2,3,4,5]"]).protocol
    assert SearchRun(DEFAULT_DOMAIN, p.search, p.surrogate, p.search.seeds[0]).budget == 8192
    assert len(p.search.seeds) == 5


PLANTED_SUBSET = ("adx", "breakout", "rvol")
PLANTED = {
    "breakout.n": 40,
    "breakout.b": 0.10,
    "adx.period": 14,
    "adx.q": 25,
    "rvol.w": 5,
    "rvol.v": 1.5,
}


class PlantedEvaluator:
    """Always feasible; score 0 at the planted config, negative distance elsewhere."""

    def evaluate(self, configs: Sequence[Config]) -> list[EvalOutcome]:
        out = []
        for c in configs:
            params = c.as_dict()
            dist = 0.0
            for key, target in PLANTED.items():
                comp, p = key.split(".")
                if comp in c.subset:
                    vals = dict(DEFAULT_DOMAIN.params_of(comp))[p]
                    dist += abs(vals.index(params[key]) - vals.index(target))
            dist += 2 * len(set(c.subset) ^ set(PLANTED_SUBSET))
            dist += len(set(c.subset) - set(PLANTED_SUBSET))
            out.append(EvalOutcome(config_id(c), -dist, True))
        return out


def test_finds_planted_optimum() -> None:
    planted = Config.make(DEFAULT_DOMAIN.version, PLANTED_SUBSET, PLANTED)
    assert is_valid(planted, DEFAULT_DOMAIN)
    res = run_search(DEFAULT_DOMAIN, SearchSection(), SurrogateSection(), 1, PlantedEvaluator())
    assert res.status == "complete"
    assert res.finalists[0].config_id == config_id(planted)
