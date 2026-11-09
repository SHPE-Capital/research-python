# SS4: Pairs candidate generator (platform parity)

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Adv | SS2, SS3, T1 | T7 goldens, W7, W9, stat-arb studies | `strategy-builder` (review: `platform-parity`) |

## Goal

Reproduce the platform's `pairs_trading` v6 strategy in research, trade by
trade, so the stat-arb team can study changes against an exact baseline.

## Owns (exclusive)

- `src/shpe_research/strategies/statarb/pairs.py`
- `configs/strategies/statarb/pairs.yaml`
- `tests/unit/strategies/statarb/test_pairs.py`

## Reads

- [teams/stat-arb.md](../../teams/stat-arb.md)
- [design/platform-parity.md](../../design/platform-parity.md)
- [SS2.md](SS2.md), [SS3.md](SS3.md)
- Platform: `backend/src/strategies/pairs/{pairsTypes,pairsConfig}.ts`, `tests/strategy/*.test.ts`

## Interface

Registered as `pairs`; exports as `pairs_trading`. Params (research name →
platform default):

| Param | Default |
|---|---|
| `rolling_window_ms` | 3.6e6 |
| `cooldown_ms` | 60000 |
| `hedge` | `fixed`, β 1 |
| `entry_z` / `exit_z` / `stop_z` | 2 / 0.5 / 4 |
| `max_hold_ms` | 86.4e6 |
| `min_obs` | 30 |
| `notional_usd` | 5000 |
| `ols_window_ms` / `recalc_bars` | 14.4e6 / 5 (when `hedge: rolling_ols`) |
| `coint_pvalue` | 0.05 |
| `hysteresis` | 0.15 |

Emits one candidate row per pair action with `leg1`, `leg2`, `beta`,
`leg1_qty`, `leg2_qty`, `zscore`, `trigger_label`
(`z_score_entry_long`\|`z_score_entry_short`\|`z_score_<exit type>`).

## Behaviour

- Spread on raw prices; z from SS3 `spread_zscore`.
- Exits are checked before entries, in order: `stop_loss` (|z| ≥ stop),
  `max_hold_exit`, `exit` (|z| ≤ exit). These are signal exits (next open).
- Entry when |z| ≥ entry and the cointegration gate passes; hysteresis 0.15
  as the platform applies it. The gate blocks entries only.
- `leg1_qty = floor(notional / p1)`, `leg2_qty = floor(leg1_qty × β)`.
- `session_end` is False.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/strategies/statarb/test_pairs.py` | `test_passes_check_strategy` | C1k check passes |
| | `test_exit_before_entry` | A bar with |z| ≥ stop on an open trade exits, does not re-enter |
| | `test_leg_qty_floor` | Qty formula on known prices |
| | `test_gate_blocks_entries_only` | Gate off → no entry, open trade still exits |
| | `test_planted_pair_profitable_gross` | On a planted pair gross P&L > 0 (seeded) |
| `tests/integration/golden/pairs_trading/` | T7 goldens | All six sessions pass, including the coint flip |

## Out of scope

Selection (SS1), Kalman/TLS export (P9).

## Done when

Unit tests and all `pairs_trading` goldens pass.
