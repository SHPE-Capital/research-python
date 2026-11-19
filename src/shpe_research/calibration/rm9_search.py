"""One common evolutionary candidate bank; true replay observations teach a surrogate, never replace validation."""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from shpe_research.calibration.surrogate import SurrogateEnsemble
from shpe_research.strategies.momentum.breakout.grid import (
    DEFAULT_DOMAIN,
    Config,
    config_id,
    enumerate_configs,
    feature_vector,
)


class Grid:
    def __init__(self):
        self.configs = list(enumerate_configs())
        self.indices = {c: i for i, c in enumerate(self.configs)}
        self.f = np.asarray([feature_vector(c) for c in self.configs], dtype="f4")
        self.groups = defaultdict(list)
        for i, c in enumerate(self.configs):
            self.groups[c.subset].append(i)

    def broad(self, n, rng):
        groups = list(self.groups.values())
        out = []
        for g in groups:
            if len(out) == n:
                break
            out.append(int(rng.choice(g)))
        if len(out) < n:
            remain = np.setdiff1d(np.arange(len(self.configs)), out)
            out.extend(rng.choice(remain, n - len(out), replace=False).tolist())
        return [(i, "stratified-initial") for i in out]

    def propose(self, n, observed, scores, p, cut, round_id):
        rng = np.random.default_rng([p["seed"], cut, round_id])
        if not len(observed):
            return self.broad(n, rng)
        used = set(observed)
        remain = np.array([i for i in range(len(self.configs)) if i not in used], dtype=int)
        if len(remain) < n or len(scores) != len(observed) or not np.isfinite(scores).all():
            raise ValueError("invalid candidate budget or observed score")
        parents = np.array(observed)[np.argsort(-scores, kind="stable")[: max(2, int(np.sqrt(len(scores))))]]
        pool = {}
        for _ in range(4 * p["pool"]):
            if len(pool) >= p["pool"] // 2:
                break
            left, right = (self.configs[int(k)] for k in rng.choice(parents, 2))
            a, b = left.as_dict(), right.as_dict()
            subset = set(left.subset if rng.random() < 0.5 else right.subset)
            if rng.random() < 0.3:
                gene = str(rng.choice(["adx", "ema", "rvol", "vwap"]))
                subset.symmetric_difference_update([gene])
                if not 2 <= len(subset) <= 4:
                    subset = set(left.subset)
            genes = {}
            for name in sorted(subset):
                for key, vs in DEFAULT_DOMAIN.params_of(name):
                    k = f"{name}.{key}"
                    possible = [v[k] for v in (a, b) if k in v]
                    val = possible[int(rng.integers(len(possible)))] if possible else vs[int(rng.integers(len(vs)))]
                    if rng.random() < 0.28:
                        val = vs[int(np.clip(vs.index(val) + rng.choice([-1, 1]), 0, len(vs) - 1))]
                    genes[k] = val
            idx = self.indices[Config.make(DEFAULT_DOMAIN.version, tuple(subset), genes)]
            if idx not in used:
                pool[idx] = "evolution+surrogate"
        for i in rng.choice(remain, min(len(remain), max(n, p["pool"])), replace=False):
            pool.setdefault(int(i), "broad+surrogate")
        idx = np.array(list(pool))
        with threadpool_limits(limits=1):
            model = SurrogateEnsemble(
                n_estimators=p["surrogate_trees"],
                seed=int(rng.integers(2**30)),
                max_iter=p["surrogate_iter"],
                min_samples_leaf=3,
            )
            model.fit(self.f[observed], scores)
            pred = model.predict(self.f[idx])
        rank = idx[np.lexsort((idx, -(pred.mean + p["ucb"] * pred.std)))].tolist()
        random_n = int(np.floor(n * p["explore"] + rng.random()))
        chosen = [(int(i), "exploration") for i in rng.choice(remain, random_n, replace=False)]
        have = {i for i, _ in chosen}
        chosen.extend((i, pool[i]) for i in rank if i not in have)
        return chosen[:n]


class Bank:
    """SQLite stores immutable proposal batches, actual config-week paths, and atomic completed weekends.

    The in-memory matrix is a read-through index. A process killed before a transaction commits
    retries only that batch. The seed, cohort boundaries and parents deterministically reconstruct
    search intelligence; previous scores are recomputed whenever history expands.
    """

    def __init__(self, path, capacity, plans):
        self.db = sqlite3.connect(path, timeout=90)
        self.db.execute("PRAGMA journal_mode=DELETE")  # portable one-writer SQLite, including shared storage
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS trial(slot INTEGER PRIMARY KEY, idx INTEGER UNIQUE, cut INTEGER, round INTEGER,
                                         origin TEXT, config_id TEXT);
        CREATE TABLE IF NOT EXISTS cell(slot INTEGER, week INTEGER, returns BLOB, trades BLOB,
                                        PRIMARY KEY(slot,week)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS decision(cut INTEGER PRIMARY KEY, body TEXT NOT NULL);
        """)
        self.offset = np.r_[0, np.cumsum([len(x["days"]) for x in plans])]
        self.r = np.full((capacity, self.offset[-1]), np.nan)
        self.trade = np.zeros((capacity, self.offset[-1]), dtype="i4")
        self.done = np.zeros((capacity, len(plans)), bool)
        self.idx = [x[0] for x in self.db.execute("select idx from trial order by slot")]
        for slot, w, r, t in self.db.execute("select slot,week,returns,trades from cell"):
            self._add(slot, w, np.frombuffer(r, "<f8"), np.frombuffer(t, "<i4"))

    def _add(self, s, w, r, t):
        cut = slice(self.offset[w], self.offset[w + 1])
        if (
            len(r) != cut.stop - cut.start
            or len(t) != len(r)
            or not np.isfinite(r).all()
            or np.any(np.asarray(r) <= -1)
        ):
            raise ValueError("invalid replay path")
        self.r[s, cut] = r
        self.trade[s, cut] = t
        self.done[s, w] = True

    def cohort(self, grid, cut, num, builder):
        rows = list(self.db.execute("select slot from trial where cut=? and round=? order by slot", (cut, num)))
        if rows:
            return [v[0] for v in rows]
        picks = builder()
        slots = []
        with self.db:
            for idx, source in picks:
                s = len(self.idx)
                self.db.execute(
                    "insert into trial values(?,?,?,?,?,?)", (s, idx, cut, num, source, config_id(grid.configs[idx]))
                )
                slots.append(s)
                self.idx.append(idx)
        return slots

    def store(self, w, rows):
        with self.db:
            for slot, item in rows:
                r = np.asarray(item["r"], "<f8")
                t = np.asarray(item["trades"], "<i4")
                self.db.execute("insert into cell values(?,?,?,?)", (slot, w, r.tobytes(), t.tobytes()))
                self._add(slot, w, r, t)


def weighted_scores(matrix, weights):
    if not np.isfinite(matrix).all():
        raise ValueError("incomplete historical return path; cannot score")
    mean = matrix @ weights
    variance = np.maximum((matrix * matrix) @ weights - mean**2, 0.0)
    return mean / np.maximum(np.sqrt(variance), 1e-12) * np.sqrt(252.0)
