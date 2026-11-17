# 0013. Weekly calibration layer above trade and strategies

- Status: Proposed
- Date: 2026-11-13
- Tasks: 0-C

## Context

The momentum strategy picks a configuration each week by searching a grid, simulating an account
for every candidate, and weighting past weeks. That work needs the trade toolkit (`trade/`) and the
signal definitions (`strategies/`), and it orchestrates them; neither of those may depend on it.
Separately, [0009](0009-research-platform-parity.md) says the platform defines semantics and
research follows. For this strategy the account and signal semantics are new, and they are
validated in research first.

## Decision

1. A new package `shpe_research.calibration` holds weekly calibration (search, weighting, config
   selection). It sits **above** `trade/` and `strategies/` in the import-linter layers, below
   `pipeline | deploy | testing`. It may import both; they must not import it.
2. The account simulator I/O contract lives in `trade/account_types.py` (dataclasses and the
   `simulate_account(panel, mask, cfg)` signature only), so both layers share one contract.
3. For this strategy, **research defines the semantics and the platform ports them**: the
   research code is the reference and the platform is checked against golden fixtures. This is an
   explicit exception to ADR 0009 and applies only to the momentum strategy.

## Consequences

- Dependencies stay one-directional; the linter enforces it once the layer is registered.
- The platform port must track research changes, so semantic changes need a fixture update.
- Other strategies keep the ADR 0009 direction.

## Alternatives considered

- **Put calibration inside `strategies/momentum`.** Rejected: it would force `strategies` to
  depend on the search machinery and blur the layer.
- **Keep the account types in `calibration/`.** Rejected: `trade/` would then need to import
  upward.
