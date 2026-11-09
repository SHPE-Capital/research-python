# SM1: Opening-range breakout candidate generator

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Int | C1b, T1 (F4 optional) | W4, W9, momentum studies | `strategy-builder` (review: `leakage-auditor`) |

## Goal

Emit one long candidate per symbol per session when price first closes above
the opening-range high, with the stop and entry levels written into the
candidate, so T1 can simulate any exit policy without strategy-specific code.

## Owns (exclusive)

- `src/shpe_research/strategies/momentum/__init__.py`
- `src/shpe_research/strategies/momentum/orb.py`
- `configs/strategies/momentum/orb.yaml`
- `tests/unit/strategies/momentum/test_orb.py`

## Reads

- [teams/momentum.md](../../teams/momentum.md)
- [design/trade-layer.md](../../design/trade-layer.md)
- [C1b.md](C1b.md) (`range_breakout` event), [T1.md](T1.md) (`entry_order: stop`, `stop_rule: price`)
- [F4.md](F4.md) (opening-range features), Platform: `backend/src/strategies/momentum/index.ts`

## Interface

Registered as `orb`. Params:

| Param | Values | Default |
|---|---|---|
| `range_minutes` | 5, 15, 30 | 15 |
| `entry_cutoff` | `HH:MM` | `11:30` |
| `confirm_rule` | `close_above` \| `stop_at_high` | `close_above` |
| `stop_rule` | `range_mid` \| `range_low` \| `atr` \| `none` | `range_low` |
| `stop_atr_mult` | float | 1.0 |
| `min_rvol` | float or None | None |
| `direction` | `long` \| `both` | `long` |

Output: a `CandidateFrame` with `side`, `decision_ts`, `ts_available`,
`stop_price`, `entry_price` (= range high for `stop_at_high`), `range_high`,
`range_low`, `rvol` (when F4 is merged).

## Behaviour

- The range is built from bars whose `ts_available` is at or before
  09:30 + `range_minutes`; no candidate before the range closes.
- `close_above`: decision at the first completed bar that closes above the
  range high; T1 enters at the next open. `stop_at_high`: decision when the
  range completes, with `entry_order: stop` at `entry_price`.
- At most one candidate per symbol per session; none after `entry_cutoff`.
- Stop levels are computed from range values and ATR on completed bars only.
- Platform equivalent: P8. Until P8 merges, T7 refuses export.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/strategies/momentum/test_orb.py` | `test_passes_check_strategy` | `check_strategy` (C1k) passes |
| | `test_no_candidate_before_range_closes` | No `decision_ts` before 09:30 + `range_minutes` |
| | `test_one_per_session` | Max one candidate per symbol per date |
| | `test_stop_levels` | `range_mid` = (H + L) / 2; `range_low` = L |
| | `test_orb_rvol_fixture` | On `synthetic_bars(orb_rvol=...)` mean 30-minute forward return is higher for `rvol` above the planted level (hypothesis check, seeded) |

## Out of scope

Exit simulation (T1), meta-labeling (W4), platform implementation (P8).

## Done when

Tests pass and `configs/strategies/momentum/orb.yaml` runs end to end on the fixture.
