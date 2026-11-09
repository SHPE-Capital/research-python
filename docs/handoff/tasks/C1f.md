# C1f: Fit schemes

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Adv | C1a, V1 | W2, W5, every study that reports walk-forward results | `repo-architect` (review: `leakage-auditor`) |

## Goal

Replace the single train/validation/test split with registered fit schemes:
`single`, `walk_forward` and `purged_walk_forward` with an embargo, all of
which hold out the test block first.

## Owns (exclusive)

- `src/shpe_research/validation/schemes.py`
- `tests/unit/validation/test_schemes.py`, `tests/integration/test_fit_schemes.py`

## Reads

- [adr/0006-fit-schemes-purged-walk-forward.md](../../adr/0006-fit-schemes-purged-walk-forward.md)
- `src/shpe_research/validation/purged.py` (V1), `validation/splits.py`
- `src/shpe_research/pipeline/registry.py`, `pipeline/stages.py` (C1a)

## Interface

Config (validated by C1h):

```yaml
fit_scheme:
  type: purged_walk_forward     # single | walk_forward | purged_walk_forward
  n_folds: 5
  embargo: "1 session"          # or a pandas Timedelta string
  test_block: 20%               # last share of sessions, held out first
  expanding: true
```

This replaces the `split: {method: purged_walk_forward}` wording in the V1
stub docstring.

Each scheme (`@register_fit_scheme`) implements
`folds(meta) -> Iterator[Fold]`, where `Fold` has `k`, `train_pos`,
`val_pos`, `train_end`, `val_start`, `val_end`, `embargo`.

| Key | Wraps |
|---|---|
| `single` | Existing `split_by_day` (C1a registers; C1f only adds `test_block`) |
| `walk_forward` | `purged_walk_forward` with purge and embargo disabled |
| `purged_walk_forward` | V1 `purged_walk_forward(meta, n_folds, embargo, expanding)` |

## Behaviour

- The test block (last `test_block` share of eras) is assigned `test` before
  any fold is built and is never in a fold. `manifest.test_scored` stays false
  unless the config sets `score_test: true`.
- Folds are era-aware: fold boundaries fall on era boundaries.
- `embargo` must be at least the longest `label_end_ts − decision_ts` in the
  dataset; the scheme raises if it is shorter and names the offending label.
- The model stage fits one model per fold and writes `model/fold_<k>/`;
  predictions carry `fold`. Report sections aggregate over folds.
- Fold boundaries and embargo go into `manifest.json` (`folds`).

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/validation/test_schemes.py` | `test_test_block_untouched` | No fold's train or val positions include a test-block row |
| | `test_purge_no_overlap` | For every fold, no train row's `[decision_ts, label_end_ts]` overlaps the val interval |
| | `test_embargo_too_short_raises` | A 1-hour embargo with multi-day labels raises |
| | `test_era_boundaries` | Every fold boundary equals an era start |
| `tests/integration/test_fit_schemes.py` | `test_planted_signal_survives_purging` | On C1o `reversal`, mean val IC over folds > 0 with purged walk-forward |
| | `test_leaky_label_caught` | A label with 2-session horizon and `single` scheme shows higher val IC than purged walk-forward with embargo (statistical marker) |

## Out of scope

The splitter itself (V1). Combinatorial purged CV is a later task.

## Done when

The hello model runs with each of the three schemes, the tests pass, and the
manifest records folds.
