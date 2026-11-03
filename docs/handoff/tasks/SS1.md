# SS1: Pair selection

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Adv | SS2, SS3 | W7, stat-arb studies | `strategy-builder` (review: `leakage-auditor`) |

## Goal

Choose candidate pairs from a universe on a formation window only, with
several interchangeable methods, and report how many pairs were tested so the
multiple-testing cost is visible.

## Owns (exclusive)

- `src/shpe_research/strategies/statarb/selection.py`
- `tests/unit/strategies/statarb/test_selection.py`

## Reads

- [teams/stat-arb.md](../../teams/stat-arb.md)
- [SS2.md](SS2.md), [SS3.md](SS3.md)
- [workshop/examples/07-unsupervised-pairs.md](../../workshop/examples/07-unsupervised-pairs.md)

## Interface

`select_pairs(prices, formation: (start, end), method, **params) -> PairSet`.
`PairSet` rows: `leg1`, `leg2`, `score`, `beta`, `half_life_bars`,
`coint_pvalue`, `cluster`; attrs `n_tested`, `method`, `formation`.

| `method` | Candidate pairs | Ranked by |
|---|---|---|
| `all_pairs` | Every pair (optionally within sector) | Engle-Granger p-value |
| `distance` | Every pair | Sum of squared differences of normalised prices (Gatev, Goetzmann and Rouwenhorst, 2006) |
| `correlation` | Every pair | Return correlation |
| `pca_cluster` | Pairs within an OPTICS cluster of PCA return loadings | Engle-Granger p-value |

Shared params: `max_pairs`, `min_half_life_bars`, `max_half_life_bars`,
`pvalue_max`, `hedge` (an SS2 key).

## Behaviour

- Uses prices inside `formation` only; a test truncates after `end` and checks
  nothing changes.
- Every method fills `beta` (SS2), `half_life_bars` (SS3) and `coint_pvalue`.
- `n_tested` counts every pair evaluated, before filters.
- `pca_cluster`: standardised returns, k components (default 5), OPTICS
  (`min_samples` default 3) on loadings; noise points are excluded.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/strategies/statarb/test_selection.py` | `test_planted_pairs_found` | On `synthetic_bars(cointegrated_pairs=[...])` both planted pairs rank in the top 3 for `all_pairs` |
| | `test_trap_pair_rejected` | The correlated, non-cointegrated trap pair fails `pvalue_max` |
| | `test_formation_only` | Truncation after `formation.end` changes nothing |
| | `test_pca_cluster_ari` | On `synthetic_bars(factors={"market": 1, "sectors": 3})` OPTICS clusters match sectors with ARI ≥ 0.8 |
| | `test_n_tested` | `all_pairs` on n symbols reports n(n−1)/2 |

## Out of scope

Trading the pairs (SS4), hedge-ratio and OU maths (SS2, SS3).

## Done when

Tests pass and W7 can call each method.
