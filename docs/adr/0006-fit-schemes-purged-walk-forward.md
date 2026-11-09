# 0006. Fit schemes, purged walk-forward and embargo

- Status: Proposed
- Date: 2026-11-02
- Tasks: V1, C1f

## Context

The C0 runner fits once on a chronological `split_by_day` (train 0.6,
validation 0.2, test 0.2, `gap_days` 0). That is fine for one-minute labels
that end inside their session. It is not enough once candidates replace
fixed-interval sampling (ADR 0004): labels overlap, holds can span sessions,
and a single split gives one noisy estimate. V1 already specifies
`purged_walk_forward(meta, n_folds=5, embargo=None, expanding=True)` in
`validation/purged.py`, following López de Prado, *Advances in Financial
Machine Learning* (2018), chapter 7. `Dataset` already requires
`label_end_ts > decision_ts`, which is the interval purging needs.

## Decision

We will make the fit scheme a registered, configurable stage
(`@register_fit_scheme`) with three schemes:

| Key | Behaviour |
|---|---|
| `single` | Today's `split_by_day`. |
| `walk_forward` | Train on a window, score the next block, roll forward; expanding or rolling; refit per fold. |
| `purged_walk_forward` | Walk-forward, dropping training rows whose `[decision_ts, label_end_ts]` overlaps the scored block, and rows that start within `embargo` after it. |

Rules for all schemes: the final test block is reserved first and scored once;
every prediction is out of sample and carries its `fold`; fold boundaries fall
on era (session) boundaries; the default embargo is at least the longest label
horizon in the config. `purged_walk_forward` is the default for any config
whose decisions come from events or strategy candidates.

Details: [pipeline.md](../design/pipeline.md#fit-schemes).

## Consequences

- Validation estimates are less optimistic and come with a spread across
  folds.
- Runs take longer (one fit per fold). The fixture keeps CI fast.
- Every label must report its true `label_end_ts`; `check_label` tests this,
  because a wrong end time leaks silently.
- Each fold is one more trial for the deflated Sharpe ratio. The run manifest
  records the trial count.

## Alternatives considered

- **Random k-fold.** Leaks future information through overlapping labels and
  autocorrelation; it gives inflated scores on financial series.
- **Single split with a gap.** Kept as `single`, but one split gives one noisy
  estimate and wastes data.
- **Combinatorial purged cross-validation (also López de Prado, 2018).** Gives
  many backtest paths, useful for overfitting statistics. Deferred: it can be
  added later as another registered fit scheme without changing this design.
