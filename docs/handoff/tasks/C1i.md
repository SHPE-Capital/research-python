# C1i: Baselines, permutation importance and by-era report sections

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Int | C1a, E1 | W3 (optional), every study report | `research-analyst` (review: `leakage-auditor`) |

## Goal

Make every report answer "better than what?" and "which features carry it?"
and "is it stable across time?" with registered report sections.

## Owns (exclusive)

- `src/shpe_research/evaluation/baselines.py`, `evaluation/importance.py`, `evaluation/by_era.py`
- `tests/unit/evaluation/test_baselines.py`, `test_importance.py`, `test_by_era.py`

## Reads

- `src/shpe_research/evaluation/signal.py`, `evaluation/trading.py` (E1)
- `src/shpe_research/pipeline/registry.py` (`register_report_section`)
- [design/pipeline.md](../../design/pipeline.md) (Evaluate and Record stages)

## Interface

Registered report sections:

| Key | Params | Output |
|---|---|---|
| `baselines` | `which` (`zero`, `last_value`, `sign_of_past_return`, `shuffled_label`) | Same metrics as the model for each baseline, side by side |
| `permutation_importance` | `n_repeats=5`, `metric`, `seed` | Drop in the metric per feature on validation, mean and std |
| `by_era` | `metric` | Metric per era, share of positive eras, worst era |

## Behaviour

- Baselines use the same rows, folds and labels as the model.
- Permutation importance permutes within era, on validation rows only, and
  never refits.
- `by_era` uses the `era` column (C1a); daily eras are ISO weeks.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/evaluation/test_baselines.py` | `test_shuffled_label_near_zero` | Shuffled-label IC within ±0.05 on the fixture |
| `tests/unit/evaluation/test_importance.py` | `test_planted_feature_ranks_first` | On C1o `reversal`, the planted feature has the largest drop |
| | `test_noise_feature_near_zero` | A pure-noise feature's mean drop is within 2 std of 0 |
| `tests/unit/evaluation/test_by_era.py` | `test_era_counts` | One row per era present in validation |

## Out of scope

Signal metrics themselves (E1), trade diagnostics (T6).

## Done when

Tests pass and the three sections render in the hello model report.
