# 0011. Pluggable data sources and a security master

- Status: Proposed
- Date: 2026-11-02
- Tasks: C1g, C1m, D6

## Context

Data today comes from Alpaca through `data/` downloaders (bars D1, audit D2,
quotes and trades D3) into frozen, checksummed snapshots (D5). The universe
builder (D4) draws a seeded liquidity-by-volatility universe with a dev and
held-out split. Planned work needs more sources: daily bars and fundamentals
for the cross-sectional ranker, macro series (FRED) for regime features,
filings (EDGAR) for event studies, and local files for the workshop fixture.
Cross-sectional studies also need to know which symbols were tradable on each
date; using today's symbol list for past dates causes survivorship bias.

## Decision

- **Source registry.** `data/sources/` holds one module per source
  (`alpaca`, `file`, `fred`, `edgar`), each implementing a common source
  protocol and registered by key. Every source writes the same snapshot layout
  that D5 freezes and verifies, so experiments read snapshots and never call a
  source directly. Credentials come from environment variables only.
- **Security master.** `data/security_master.py` records, per symbol and date
  range: listing status, exchange, asset type, and corporate-action identifiers
  (splits, symbol changes, delistings; split records come from D6,
  `data/corporate_actions.py`). It answers "which symbols were in the
  universe on date d". The universe builder, cross-sectional transforms
  ([0007](0007-context-symbols-cross-sectional-transforms.md)) and the
  portfolio layer all use it.

The exact protocol is in the C1g and C1m packets
([C1g](../handoff/tasks/C1g.md), [C1m](../handoff/tasks/C1m.md)).

## Consequences

- Adding a source does not touch experiments or features.
- Survivorship bias in cross-sectional studies becomes something the code
  prevents, not something each researcher must remember.
- The security master needs a data source for delistings and symbol changes.
  Alpaca's asset list covers current listings only, so historical coverage is
  partial until a better source is chosen; reports flag universes built without
  full history.
- Snapshots stay the single input format, so checksums and the verifier still
  apply.

## Alternatives considered

- **One downloader module per source with its own output format.** Each
  experiment would need source-specific loading code.
- **A vendor data platform (e.g. a paid survivorship-free database).** Best
  coverage, but a cost decision for later; the source registry lets one be
  added as another key.
- **No security master; filter by data availability.** A symbol with no bars
  on a date looks the same as a delisted one, and the universe silently
  follows hindsight.
