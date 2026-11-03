# SR3: Residual fade

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 3 | Adv | F7/F8, T1 (SR2 for shared code) | reversals studies, P7 evidence | `strategy-builder` (review: `leakage-auditor`) |

## Goal

Fade the part of a stock's move that the market (SPY) and its sector do not
explain, so the strategy does not take market beta disguised as reversal.

## Owns (exclusive)

- `src/shpe_research/strategies/reversal/residual.py`
- `configs/strategies/reversal/residual_fade.yaml`
- `tests/unit/strategies/reversal/test_residual.py`

## Reads

- [F7-F8.md](F7-F8.md) (market return, beta and residual features on C1d context symbols)
- [SR2.md](SR2.md), [teams/reversals.md](../../teams/reversals.md)

## Interface

Registered as `residual_fade`. Params as SR2 plus `beta_window_bars`,
`residualize_on` (`spy`\|`spy_sector`). Output adds `residual_return` and
`beta` columns to the candidates.

## Behaviour

- Residual return at t = r_t − β̂ × r_SPY,t, with β̂ from bars before t only (F8).
- Stretch is computed on the cumulative residual since the anchor.
- Same exits and stop handling as SR2.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/strategies/reversal/test_residual.py` | `test_passes_check_strategy` | C1k check passes |
| | `test_beta_uses_past_only` | Truncation after t leaves β̂ at t unchanged |
| | `test_market_move_no_signal` | A pure market move (residual 0) gives no candidate |
| | `test_factor_fixture` | On `synthetic_bars(factors=...)` the mean beta of candidates is near 0 |

## Out of scope

Feature computation (F7/F8), platform implementation (P7).

## Done when

Tests pass and a study can compare SR1, SR2 and SR3 on the same fixture.
