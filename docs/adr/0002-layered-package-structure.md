# 0002. Layered package structure and import rules

- Status: Proposed
- Date: 2026-11-02
- Tasks: C1a, C1l, migration PR (Wave 0)

## Context

`src/shpe_research/` today has flat packages (`contracts`, `data`, `features`,
`labels`, `models`, `validation`, `evaluation`, `experiments`) and no rule
about which may import which. The planned additions (`pipeline`, `events`,
`transforms`, `trade`, `strategies`, `portfolio`, `analysis`, `deploy`,
`testing`, `scaffold`) make import cycles and hidden coupling likely,
especially with several worktrees merging at once. Look-ahead bugs are also
easier to hide when, for example, a feature can import evaluation code.

## Decision

We will organise the package in ranked layers. A module may import only from
its own rank or lower ranks.

| Rank | Packages |
|---|---|
| 8 | `cli` |
| 7 | `experiments`, `scaffold` |
| 6 | `pipeline`, `deploy`, `testing` |
| 5 | `strategies`, `portfolio`, `analysis` |
| 4 | `evaluation`, `models`, `validation`, `events`, `labels` |
| 3 | `features`, `transforms`, `trade` |
| 2 | `data` |
| 1 | `contracts`, `config` |

import-linter enforces the rules in CI (C1l) with three contracts: a layers
contract for the table; an independence contract between
`strategies.momentum`, `strategies.reversal` and `strategies.statarb`; and a
forbidden contract that stops library code from importing `testing`. Studies
(`studies/`) and workshop notebooks sit outside the package and may import
anything. The full tree, the reason for each rank, and the import-linter
configuration are in [architecture.md](../design/architecture.md).

## Consequences

- Cycles become CI failures instead of runtime surprises.
- A column schema used across ranks lives at the lowest rank that needs it.
  `CandidateFrame` lives in `contracts/candidates.py` because `trade/` (rank 3)
  consumes it and `events/` (rank 4) produces it.
- Team strategy code cannot import another team's code. Shared logic has to be
  promoted to `trade/`, `features/` or `analysis/`, which is the intended
  pressure.
- Existing modules move once (tests into `unit/` and `integration/`, study
  folders into `studies/<team>/`) in a Wave 0 PR, before parallel work starts.

## Alternatives considered

- **No rules, code review only.** Does not scale to parallel worktrees and
  agents; reviewers miss transitive imports.
- **A separate installable package per layer.** Strong isolation, but release
  overhead a research team does not need. import-linter gives the same
  guarantee inside one package.
- **One folder per strategy with everything inside.** Duplicates fill, sizing
  and validation logic per strategy, which is how research and the platform
  drift apart.
