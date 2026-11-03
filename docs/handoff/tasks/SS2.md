# SS2: Hedge ratios

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 | Int | C1a | SS1, SS4, W7, P9 evidence | `strategy-builder` (review: `platform-parity`) |

## Goal

Estimate the hedge ratio β for a pair with interchangeable methods, where the
`rolling_ols` method reproduces the platform's `pairs_trading` v6 numbers.

## Owns (exclusive)

- Hedge-ratio functions in `src/shpe_research/features/cross_asset/pair_math.py`:
  `fixed`, `rolling_ols`, `engle_granger` (shared with F7; SS3 adds OU
  functions to the same file, the F7-F8 packet only imports it)
- `src/shpe_research/strategies/statarb/__init__.py` (SS2 merges first in
  Wave 1; SS3 rebases on it)
- `src/shpe_research/strategies/statarb/hedge.py`: the `@register_hedge`
  registry, re-exports of the `pair_math.py` functions, and the strategy-only
  `kalman` and `tls` methods
- `tests/unit/features/cross_asset/test_hedge.py`, `tests/unit/strategies/statarb/test_hedge.py`

## Reads

- [teams/stat-arb.md](../../teams/stat-arb.md)
- [design/platform-parity.md](../../design/platform-parity.md)
- Platform: `backend/src/strategies/pairs/{pairsTypes,pairsConfig}.ts`

## Interface

`@register_hedge` keyed on `hedge_type`; `hedge_ratio(p1, p2, method, **params) -> pd.Series`.

| Key | Params | Notes |
|---|---|---|
| `fixed` | `beta` (1.0) | Platform default |
| `rolling_ols` | `window_ms` (14.4e6), `recalc_bars` (5) | OLS of p1 on p2 over the window, recomputed every `recalc_bars`; platform parity |
| `kalman` | `delta`, `obs_var` | Time-varying β; not exportable until P9 |
| `tls` | `window_ms` | Total least squares; not exportable until P9 |

`engle_granger(p1, p2) -> (stat, pvalue)`: no-lag Dickey-Fuller on OLS
residuals with MacKinnon p-values, as the platform computes it.

## Behaviour

- Prices are raw (not log), matching the platform spread `p1 − β p2`. A
  `price_space: log` option exists for research and is not exportable (P9).
- `rolling_ols` uses bars strictly before t; β at t is the last recomputed value.
- β updates only while the pair is cointegrated at the platform's threshold
  (0.05); otherwise the previous β is kept.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/features/cross_asset/test_hedge.py` | `test_fixed` | Constant series |
| | `test_rolling_ols_recovers_beta` | On a planted pair with β 1.2, estimate within 0.05 |
| | `test_rolling_ols_past_only` | Truncation after t leaves β at t unchanged |
| | `test_engle_granger_matches_platform` | p-value equals a platform test vector to 1e-6 |
| | `test_beta_frozen_when_not_cointegrated` | β constant through a non-cointegrated stretch |
| `tests/unit/strategies/statarb/test_hedge.py` | `test_kalman_tracks_drift` | On a β drifting 1.0 → 1.4, Kalman error below rolling OLS error |
| | `test_registry_keys` | `fixed`, `rolling_ols`, `kalman`, `tls` all resolve |

## Out of scope

OU fit (SS3), pair selection (SS1), Kalman export (P9).

## Done when

Tests pass, `pair_math.py` holds every estimator that a feature needs, and
`strategies/statarb/hedge.py` imports from it rather than copying it
(`features/` never imports `strategies/`).
