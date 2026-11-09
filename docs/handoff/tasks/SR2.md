# SR2: Stretch fade

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Int–Adv | T1, F2, F3 | W9, P7 evidence, reversals studies | `strategy-builder` (review: `leakage-auditor`) |

## Goal

Fade moves that are large relative to recent volatility and far from an anchor
(VWAP or an EMA), rather than every up or down bar. This is the reversals
team's main research strategy and the evidence for P7.

## Owns (exclusive)

- `src/shpe_research/strategies/reversal/stretch.py`
- `configs/strategies/reversal/stretch_fade.yaml`
- `tests/unit/strategies/reversal/test_stretch.py`

## Reads

- [teams/reversals.md](../../teams/reversals.md)
- [F2.md](F2.md) (session features: VWAP distance), [F3.md](F3.md) (volatility)
- [platform-tasks.md P7](../../platform-tasks.md#p7-reversal-strategy-family)

## Interface

Registered as `stretch_fade`. Params mirror the P7 field list in snake_case:
`anchor` (`vwap`\|`ema`), `ema_bars`, `vol_lookback_bars`,
`entry_threshold_sigma`, `exit_at_sigma`, `max_hold_bars`,
`disaster_stop_sigma`, `trade_window` (`HH:MM-HH:MM`). `residual_vs_spy` is
SR3's.

## Behaviour

- Stretch = (close − anchor) / (σ × close) with σ from F3 on completed bars.
- Entry when |stretch| ≥ `entry_threshold_sigma`, opposite to the stretch.
- Signal exit when |stretch| ≤ `exit_at_sigma` or after `max_hold_bars`.
- The disaster stop is written to `stop_price` at `disaster_stop_sigma`; T1
  runs it with `stop_rule: price`.
- Expected outcome (hypothesis): net expectancy improves over SR1 at the
  platform default cost; the study must show it with a calibrated T5 model.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/strategies/reversal/test_stretch.py` | `test_passes_check_strategy` | C1k check passes |
| | `test_threshold_symmetry` | A stretch of +k and −k give opposite sides |
| | `test_no_entry_outside_window` | No candidate outside `trade_window` |
| | `test_stop_price_written` | `stop_price` = entry ± `disaster_stop_sigma` × σ × price |
| | `test_reversal_fixture` | On `synthetic_bars(reversal=...)` gross expectancy > 0 (seeded) |

## Out of scope

Market residuals (SR3), platform implementation (P7).

## Done when

Tests pass and the config runs end to end; T7 refuses export naming P7.
