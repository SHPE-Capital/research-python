# SR1: Bar-fade baseline (platform parity)

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Int | C1a, T1 | T7 goldens, W9, reversals studies | `strategy-builder` (review: `platform-parity`) |

## Goal

Reproduce the platform's `minute_reversal` strategy (v2) exactly in research,
so the reversals team has a parity-checked baseline and T7 has its first
exportable strategy.

## Owns (exclusive)

- `src/shpe_research/strategies/reversal/__init__.py`
- `src/shpe_research/strategies/reversal/bar_fade.py`
- `configs/strategies/reversal/bar_fade.yaml`
- `tests/unit/strategies/reversal/test_bar_fade.py`

## Reads

- [teams/reversals.md](../../teams/reversals.md)
- [design/platform-parity.md](../../design/platform-parity.md)
- Platform: `backend/src/strategies/minuteReversal/*`, `config/strategyCatalog.ts`,
  `tests/strategy/*.test.ts`

## Interface

Registered as `bar_fade`; exports as `minute_reversal`. Params with platform
defaults: `qty` 1, `max_position` 1, `cooldown_ms` 0, `max_bar_gap_ms`
180000, `regular_hours_only` true.

Emits rows with `action` (`entry`\|`exit`), `side`, `qty` and
`trigger_label` (`minute_up_sell`\|`minute_down_buy`).

## Behaviour

- Up bar (`close > prev close`) → sell; down bar → buy; `close == prev` → nothing.
- A gap to the previous bar longer than `max_bar_gap_ms` resets `prev`.
- Regular hours: minute ≥ 09:30 and minute + 1 < 16:00 New York.
- The generator tracks an assumed-filled position: with `max_position` 1, a
  sell signal while long emits an exit row (and an entry row for the short if
  the platform flips); it never emits more exposure than `max_position`.
- `session_end` is False in the config; the platform does not flatten.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/strategies/reversal/test_bar_fade.py` | `test_passes_check_strategy` | C1k check passes |
| | `test_direction_rule` | Up → sell, down → buy, flat → none |
| | `test_gap_reset` | A 4-minute gap produces no signal on the next bar |
| | `test_regular_hours` | No signal on the 15:59 bar |
| | `test_position_cap` | Never more than `max_position` open |
| `tests/integration/golden/minute_reversal/` | T7 goldens | All five sessions pass |

## Out of scope

Improved reversal signals (SR2, SR3), platform changes.

## Done when

Unit tests and all `minute_reversal` goldens pass.
