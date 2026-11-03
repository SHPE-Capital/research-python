# C1c: Label registry

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 (`triple_barrier` after T1) | Int | C1a; T1 for `triple_barrier` only | L2, W4, W5, SM1 studies | `repo-architect` (review: `leakage-auditor`) |

## Goal

Make labels pluggable and give every label a `label_end_ts`, so purged
validation (V1, C1f) works for any horizon.

## Owns (exclusive)

- `src/shpe_research/labels/__init__.py`, `labels/base.py`
- `src/shpe_research/labels/forward_return.py` (registration only; behaviour unchanged)
- `src/shpe_research/labels/realized_var.py`, `labels/triple_barrier.py`
- `tests/unit/labels/test_label_registry.py`, `tests/unit/labels/test_realized_var.py`,
  `tests/unit/labels/test_triple_barrier.py`

`labels/multi_horizon.py` stays L1's file; C1c only registers it through the
L1 function once L1 merges (one line in `labels/__init__.py`).

## Reads

- [design/pipeline.md](../../design/pipeline.md) (Labels stage), [design/trade-layer.md](../../design/trade-layer.md) (why labels reuse the simulator)
- `src/shpe_research/labels/forward_return.py`, `labels/multi_horizon.py`
- `src/shpe_research/trade/simulate.py`, `trade/types.py` (T1)

## Interface

`labels/base.py`: `Label` ABC with `label_type: ClassVar[str]` and
`compute(bars, decisions) -> pd.DataFrame` returning columns
`symbol`, `decision_ts`, `label`, `label_end_ts`, plus label-specific extras.

| Key | Params | Label | `label_end_ts` | Lands |
|---|---|---|---|---|
| `forward_return` | `horizon_minutes` | Existing C0 label | Existing | Wave 1 |
| `multi_horizon` | L1's params, `target` (which horizon is `label`) | L1 columns | Bar time of the chosen horizon | when L1 merges |
| `forward_realized_var` | `horizon_sessions=1`, `bar_minutes=5`, `log=True` | Realized variance of session d+1 (sum of squared `bar_minutes` log returns over the regular session), log if `log` | Close of session d+1 (its last regular bar's `ts_available`) | Wave 1 (W5) |
| `triple_barrier` | `exit_policy`, `fill_model`, `cost_model`, `target` (`net_return`\|`sign`\|`hit_target`) | Calls `trade.simulate.simulate_trades` on the decisions | `exit_ts` from the ledger | after T1 |

`decisions` is a `CandidateFrame` (or the `every_n_minutes` frame).

## Behaviour

- `label_end_ts > decision_ts` for every row, always.
- A label that needs bars past the data end is NaN, not truncated: the last
  rows of the sample have NaN labels and `join` drops them.
- `forward_realized_var` for a decision on session d never reads a bar from
  session d; it starts at d+1's 09:30 bar. Decisions on the last session in
  the data get NaN.
- `triple_barrier` contains no exit logic of its own. Unfilled entries give a
  NaN label with `exit_reason = unfilled` kept as an extra column.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/labels/test_label_registry.py` | `test_keys_registered` | `forward_return`, `forward_realized_var` resolve; `triple_barrier` resolves once T1 merges |
| | `test_end_after_decision[key]` | `label_end_ts > decision_ts` on the fixture for each key |
| `tests/unit/labels/test_realized_var.py` | `test_hand_case` | Two hand-built sessions: label equals the hand sum of squared 5-minute log returns of day 2 |
| | `test_planted_persistence` | On C1o `vol_clustering{persistence: 0.95}`, corr(log RV_d, label) > 0.5 |
| | `test_truncation_invariance` | Labels for decisions before the cut are unchanged when data after d+1 is removed |
| `tests/unit/labels/test_triple_barrier.py` | `test_matches_simulator` | Label equals `simulate_trades(...).net_return` row for row |
| | `test_end_is_exit_ts` | `label_end_ts == exit_ts` |

## Out of scope

Exit logic (T1), multi-horizon math (L1), the purged splitter (V1).

## Done when

The three Wave 1 keys pass their tests and the hello model golden is
unchanged. `triple_barrier` lands as a follow-up PR after T1, or in this PR if
T1 has merged first.
