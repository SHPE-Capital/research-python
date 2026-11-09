# SM2: EMA crossover baseline

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 | Beg | C1a | momentum studies (baseline) | `strategy-builder` (review: `leakage-auditor`) |

## Goal

A simple, well-understood intraday momentum baseline that every momentum study
compares against. It is a good first strategy for a new team member.

## Owns (exclusive)

- `src/shpe_research/strategies/momentum/ema_cross.py`
- `configs/strategies/momentum/ema_cross.yaml`
- `tests/unit/strategies/momentum/test_ema_cross.py`

## Reads

- [teams/momentum.md](../../teams/momentum.md)
- `src/shpe_research/events/` (C1a protocol)
- Platform indicators: `backend/src/services/indicators/*.ts` (EMA seeding)

## Interface

Registered as `ema_cross`. Params `fast` (default 9), `slow` (21),
`direction` (`long`\|`both`). Emits `action = entry` on a cross and
`action = exit` on the opposite cross.

## Behaviour

- EMA seeded with the SMA of the first `n` closes, matching the platform
  indicator; values are NaN until seeded.
- Cross detected on completed bars: `fast[t] > slow[t]` and `fast[t−1] <= slow[t−1]`.
- No candidates in the first `slow` bars of each session unless the EMA is
  carried across sessions (`carry: true`, default false).

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/strategies/momentum/test_ema_cross.py` | `test_passes_check_strategy` | C1k check passes |
| | `test_ema_seed_matches_platform` | First value equals the SMA of the first `n` closes |
| | `test_cross_on_completed_bars` | Truncation after t leaves candidates up to t unchanged |
| | `test_entry_exit_alternate` | Entries and exits alternate per symbol |

## Out of scope

Platform export (no `ema_cross` platform type; T7 refuses it).

## Done when

Tests pass and the config runs on the fixture.
