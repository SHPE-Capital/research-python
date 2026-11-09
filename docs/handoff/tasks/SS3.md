# SS3: Ornstein-Uhlenbeck fit and half-life

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 | Int | C1a | SS1, SS4, W7, W9 | `strategy-builder` (review: `trade-management-analyst`) |

## Goal

Fit a mean-reverting (OU / AR(1)) model to a spread and report its half-life,
so pairs can be screened by speed and holding limits can be set in half-lives.

## Owns (exclusive)

- OU functions in `src/shpe_research/features/cross_asset/pair_math.py`
  (`fit_ou`, `spread_zscore`; shared with F7 and T6)
- `src/shpe_research/strategies/statarb/ou.py`: re-exports plus
  strategy-only helpers (half-life screen used by SS1)
- `tests/unit/features/cross_asset/test_ou.py`

## Reads

- [teams/stat-arb.md](../../teams/stat-arb.md)
- [T6.md](T6.md) (`ou_half_life` in `analysis/holding.py` calls this)

## Interface

`fit_ou(spread) -> OUFit(kappa, mu, sigma, half_life_bars, r2, n)`;
`spread_zscore(spread, window_bars, min_obs=30) -> pd.Series`.

## Behaviour

- Fit Δs_t = a + b s_{t−1} + ε by OLS; κ = −b, μ = −a / b,
  half-life = ln 2 / κ bars. If b ≥ 0, half-life is `inf`.
- `spread_zscore` matches the platform: rolling mean and sample standard
  deviation over `window_bars` of past values; z = 0 when std = 0; NaN until
  `min_obs`.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/features/cross_asset/test_ou.py` | `test_half_life_recovered` | Planted half-life 12 and 30 bars → within 30% |
| | `test_random_walk_inf` | Random walk gives a large or `inf` half-life |
| | `test_zscore_zero_std` | Constant spread → z = 0 |
| | `test_zscore_past_only` | Truncation after t leaves z at t unchanged |

## Out of scope

Hedge ratios (SS2), trading rules (SS4).

## Done when

Tests pass, T6's `ou_half_life` delegates to `pair_math.fit_ou`, and
`strategies/statarb/ou.py` contains no copy of the fit.
