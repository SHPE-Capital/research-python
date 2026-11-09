# C1k: Contributor check kits

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 | Int | C1a, C1o | C1j, every new feature, model, label and strategy | `repo-architect` (review: `leakage-auditor`) |

## Goal

Give every component type one function that runs the standard contract,
leakage and reproducibility checks, so a contributor's test file is three lines.

## Owns (exclusive)

- `src/shpe_research/testing/checks.py`, `testing/fixtures.py`
- `tests/unit/testing/test_checks.py`

## Reads

- `src/shpe_research/features/testing.py` (`assert_no_lookahead`)
- `src/shpe_research/data/fixtures.py` (`synthetic_bars`), `contracts/`
- [design/testing.md](../../design/testing.md)

## Interface

| Function | Checks |
|---|---|
| `check_feature(feature, bars=None)` | Output columns match `output_columns`; index aligned; `assert_no_lookahead`; deterministic; no NaN after warm-up |
| `check_model(model, X, y, seed=7)` | Same seed gives same predictions; save/load round-trips; ONNX export (if supported) matches within 1e-5; `task` ClassVar consistent with output |
| `check_label(label, bars, decisions)` | `label_end_ts > decision_ts`; truncation invariance; NaN at the data end, not truncated |
| `check_strategy(generator, bars)` | Output passes `validate_candidates`; every `decision_ts` is a bar's `ts_available`; truncation invariance; deterministic |

`testing/fixtures.py` re-exports `synthetic_bars` so tests import from the package.

## Behaviour

- Each check raises `AssertionError` with a message naming the failed check
  and the first offending row.
- Default `bars` is `synthetic_bars()` with its fixed signature.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/testing/test_checks.py` | `test_check_feature_passes_builtins[key]` | Every registered feature passes |
| | `test_check_feature_catches_centered_mean` | A centered rolling mean fails the lookahead check |
| | `test_check_model_passes_builtins[key]` | Every registered model passes |
| | `test_check_label_catches_truncation` | A label that truncates at the data end fails |
| | `test_check_strategy_catches_future_ts` | A generator emitting a ts with no bar fails |

## Out of scope

Workshop pass bands (W0), golden reports (C1a).

## Done when

Tests pass and existing feature tests are rewritten to use `check_feature`
where that shortens them.
