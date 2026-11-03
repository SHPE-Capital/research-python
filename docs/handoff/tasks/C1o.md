# C1o: Synthetic fixture options

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 0 | Int–Adv | C1a, migration PR | C1b, C1d, C1e, every S task, W0–W9 | `repo-architect` (review: `leakage-auditor`) |

## Goal

Extend `synthetic_bars` with opt-in planted effects so every strategy family
and every workshop example has a fixture where the right answer is known.
Defaults stay exactly as they are.

## Owns (exclusive)

- `src/shpe_research/data/fixtures.py` (new keyword-only options and one private `_plant_<effect>` function per effect; existing signature unchanged)
- `src/shpe_research/experiments/sources.py` (`_fixture` passthrough only, until C1g replaces it)
- `tests/unit/data/test_fixture_effects.py`

## Reads

- `src/shpe_research/data/fixtures.py` (`synthetic_bars(symbols=DEFAULT_SYMBOLS, start="2026-03-02", end="2026-08-28", seed=7, signal_strength=0.3)`)
- `src/shpe_research/experiments/sources.py` (`_fixture` passes only `symbols`, `start`, `end`, `seed`, `signal_strength`)
- [workshop/README.md](../../workshop/README.md) and the example specs for the pass bands

## Interface

The existing signature is unchanged. New keyword-only options, all default
`None` (off):

| Option | Shape | Planted effect | Used by |
|---|---|---|---|
| `momentum` | `{breakout_prob, drift_bps, drift_minutes}` | On chosen sessions, price breaks the opening-range high and drifts up for `drift_minutes` | C1b, SM1, SM2 |
| `orb_rvol` | `{rvol_threshold, drift_bps, drift_minutes, context: "SPY"}` | After the first close above the range high H, a positive 30-minute drift that is zero when relative volume is below `rvol_threshold` and positive above it; adds an `SPY` context symbol | W4, SM1 |
| `reversal` | `{ar: [phi_1, ...], scale}` | AR(p) on 1-minute returns with negative coefficients | SR1–SR3, W2, W9 |
| `cointegrated_pairs` | `[{beta, half_life_bars}]`, plus `trap: true` | Pairs whose spread `p1 − beta·p2` is an OU process with that half-life; `trap` adds a pair that is correlated but not cointegrated | SS1–SS4, W7, W9 |
| `vol_clustering` | `{persistence}` | GARCH-like daily variance with the given persistence, applied to intraday returns | C1c, C1e, W5 |
| `factors` | `{market, sectors}` | Returns = β·market + sector factor + idiosyncratic; writes `beta` and `sector` into fixture metadata | C1d, W7, W8 |
| `xs_alpha` | `{ic, contaminate: {beta, sector}}` | A daily cross-sectional score with information coefficient `ic` against next-period residual return; `contaminate` mixes in β and sector exposure | W8, T9 |
| `null_symbols` | `int` or list of names (`N01…`) | Pure random-walk symbols with no effect | W6, W7 |
| `daily` | `{years, n_symbols, xs_momentum: true}` | Daily bars, at least 3 years and 50 symbols; `xs_momentum` plants a 12-1 cross-sectional momentum effect | C1e, W3, W8, SM3 |

Every option that needs a span (HAR uses 22 sessions of history plus a test
block) extends `end` automatically and documents the minimum span.

`experiments/sources._fixture` passes every key it receives to
`synthetic_bars`, not just the five it passes today.

## Behaviour

- With no new options, output is byte-identical to today for any seed
  (the hello model golden and every existing test are unchanged).
- Effects compose: `factors` plus `xs_alpha` plus `null_symbols` is valid.
- Planted parameters are returned through `df.attrs["planted"]` so tests can
  compare estimates with truth.
- All randomness flows from `seed`.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/data/test_fixture_effects.py` | `test_defaults_unchanged` | Hash of `synthetic_bars()` equals the pre-change hash |
| | `test_reversal_recovered` | OLS of r_t on r_{t−1} recovers phi_1 within 0.05 |
| | `test_pair_half_life` | Estimated OU half-life of each planted pair within 30% of `half_life_bars` |
| | `test_trap_not_cointegrated` | Engle-Granger p-value of the trap pair > 0.1 |
| | `test_vol_persistence` | Lag-1 autocorrelation of daily RV > 0.5 at persistence 0.95 |
| | `test_factors_beta` | Regression β per symbol within 0.1 of `attrs["planted"]["beta"]` |
| | `test_xs_alpha_ic` | Mean daily rank IC within 0.02 of `ic` |
| | `test_xs_momentum` | Top-minus-bottom 12-1 decile spread > 0 over the sample |
| | `test_orb_rvol_conditional` | Mean 30-minute post-breakout return above the RVOL threshold > below it |
| | `test_sources_passthrough` | `{type: fixture, reversal: {...}}` in a config reaches `synthetic_bars` |

## Out of scope

The features, samplers and strategies that read these effects.

## Done when

Tests pass, existing tests and goldens are unchanged, and every option in the
table is documented in the `synthetic_bars` docstring with its minimum span.
