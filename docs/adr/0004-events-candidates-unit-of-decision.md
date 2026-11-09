# 0004. Events and candidates as the unit of decision

- Status: Proposed
- Date: 2026-11-02
- Tasks: C1a, C1b, SM1, SR1, SS4

## Context

The C0 runner makes one row per symbol every `sample_every_minutes` minutes.
Real strategies do not act on a clock. ORB acts on a breakout. The platform
`minuteReversal` acts on each completed bar whose close differs from the
previous close. `PairsStrategy` acts when the spread z-score crosses a
threshold. Meta-labeling, deciding which of a strategy's signals to take,
needs rows that are exactly the strategy's signals, with their side.

## Decision

We will make the decision point, not the bar, the unit of a dataset row. Every
row comes from a `CandidateFrame` with required columns `symbol`,
`decision_ts`, `side` and `strategy_id`, plus optional columns holding what the
generator knew at `decision_ts`. Three sources produce one:

- samplers registered with `@register_sampler` (today's every-N-minutes
  becomes one of them);
- generic events in `events/`: CUSUM filter, threshold crossings, session open;
- strategy `CandidateGenerator`s in `strategies/`, registered with
  `@register_strategy`.

The column schema lives in `contracts/candidates.py`. `events/types.py`
defines the `CandidateGenerator` protocol and re-exports `CandidateFrame`.
`decision_ts` is always a bar's `ts_available`. Filters mark candidates as
skipped with a reason; they do not delete them.

Details: [pipeline.md](../design/pipeline.md#decision-points-samplers-events-and-candidates),
[trade-layer.md](../design/trade-layer.md).

## Consequences

- One dataset shape serves signal studies, meta-labeling and backtests.
- Candidates can be compared one to one with platform signals (P1 logs
  platform candidates), which is the basis of parity.
- Skip reasons let a study report how many candidates each filter removed.
- Rows are no longer evenly spaced, so overlapping labels are common. Purging
  ([0006](0006-fit-schemes-purged-walk-forward.md)) becomes the default, not an
  option.

## Alternatives considered

- **Keep fixed-interval sampling and add a "signal fired" feature.** Rows where
  the strategy would not act dilute training and make meta-labeling results
  meaningless.
- **Strategies emit trades directly.** Mixes the decision with filters, sizing
  and fills, so none of them can be studied on its own.
- **Event-based sampling only (CUSUM), as in López de Prado (2018).** Kept as
  one source, but strategy candidates are what the platform trades, so they
  must be first-class.
