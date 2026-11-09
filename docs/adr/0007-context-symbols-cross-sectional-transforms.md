# 0007. Context symbols and cross-sectional transforms

- Status: Proposed
- Date: 2026-11-02
- Tasks: C1d, F7, F8, C1m

## Context

`Feature.compute` today sees one symbol's bars at a time
(`features/base.py:compute_features` loops per symbol). Several planned
features need other symbols: the market return (F8) needs SPY, the pair spread
(F7) needs the second leg, the residual fade (SR3) needs SPY for beta.
tasks.md lists F7 and F8 as depending on each other for this reason. The
cross-sectional ranker (workshop example 3) and portfolio neutralization
(example 8) need operations across all symbols at one timestamp, which a
per-symbol feature cannot express. `experiments/sources.py` can already load a
snapshot's `context` role.

## Decision

We will add two mechanisms.

**Context symbols.** The `data:` config lists them separately
(`context_symbols: [SPY]`). Their bars are passed to features that declare
they need them. Context symbols never become dataset rows and never enter the
trade universe unless also listed as trade symbols. Context values obey the
same `ts_available` rule as the symbol's own bars.

**Cross-sectional transforms** in `transforms/` (rank 3): `rank`, `zscore`,
`winsorize`, `demean`, `neutralize`. They run after per-symbol features and
before the join, each on the rows that share one `decision_ts`, with a
`min_count` below which they return NaN. They run on the point-in-time
universe from the security master (C1m), not on whatever symbols a snapshot
contains.

C1d extends the feature contract and bumps `CONTRACT_VERSION`. Details:
[pipeline.md](../design/pipeline.md#context-symbols).

## Consequences

- F7 and F8 build on one contract extension instead of each inventing one.
- Transforms cannot look ahead in time, but can leak through the universe;
  tying them to the security master closes that gap.
- Minute bars are sparse for less liquid names, so cross-sectional statistics
  at minute frequency can be noisy; `min_count` makes that visible instead of
  silent.

## Alternatives considered

- **Join SPY as an extra column in the bar frame.** Works for one symbol, but
  breaks for pair counterparts that differ per row and hides the
  `ts_available` alignment.
- **Cross-sectional operations inside features.** A feature would need the
  whole universe, breaking the per-symbol no-lookahead check
  (`assert_no_lookahead`).
- **Transforms inside the model.** Fit-on-train processors belong there, but a
  cross-sectional rank is stateless per timestamp and is needed by
  non-model studies too.
