# Architecture decision records

An ADR records one decision that shapes the repo: what was decided, why, and
what it costs. ADRs are short and are not edited after they are accepted; a
later ADR supersedes an earlier one instead. The format and the reasons for
keeping ADRs are in [0001](0001-record-architecture-decisions.md).

When an ADR and another document disagree, the ADR wins (see
[../README.md](../README.md)). Fix the other document, or write a new ADR.

## Index

| # | Title | Status | Design doc |
|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Proposed | — |
| [0002](0002-layered-package-structure.md) | Layered package structure and import rules | Proposed | [architecture.md](../design/architecture.md) |
| [0003](0003-pipeline-stages-and-registries.md) | Pipeline stages and registries | Proposed | [pipeline.md](../design/pipeline.md) |
| [0004](0004-events-candidates-unit-of-decision.md) | Events and candidates as the unit of decision | Proposed | [pipeline.md](../design/pipeline.md), [trade-layer.md](../design/trade-layer.md) |
| [0005](0005-strategies-and-shared-trade-toolkit.md) | Strategies plus a shared trade toolkit | Proposed | [trade-layer.md](../design/trade-layer.md) |
| [0006](0006-fit-schemes-purged-walk-forward.md) | Fit schemes, purged walk-forward and embargo | Proposed | [pipeline.md](../design/pipeline.md) |
| [0007](0007-context-symbols-cross-sectional-transforms.md) | Context symbols and cross-sectional transforms | Proposed | [pipeline.md](../design/pipeline.md) |
| [0008](0008-typed-configs-pydantic.md) | Typed configs with Pydantic | Proposed | [pipeline.md](../design/pipeline.md) |
| [0009](0009-research-platform-parity.md) | Research to platform parity and config export | Proposed | [platform-parity.md](../design/platform-parity.md) |
| [0010](0010-workshop-notebooks-jupytext.md) | Workshop notebooks as jupytext files with a solutions branch | Proposed | [testing.md](../design/testing.md), [workshop README](../workshop/README.md) |
| [0011](0011-pluggable-data-sources-security-master.md) | Pluggable data sources and a point-in-time security master | Proposed | [architecture.md](../design/architecture.md) |
| [0012](0012-docs-diataxis-mkdocs.md) | Docs with Diátaxis and mkdocs-material | Proposed | — |

## Process

1. Copy [0000-template.md](0000-template.md) to `NNNN-short-title.md` with the
   next free number.
2. Open a PR with the ADR alone, status Proposed. Discussion happens on the PR.
3. The lead merges it as Accepted, or closes it as Rejected and keeps the file
   with that status so the reasoning survives.
4. To reverse a decision, write a new ADR that supersedes the old one, and set
   the old one's status to "Superseded by NNNN". That status line is the only
   edit allowed to an accepted ADR.

All ADRs above are Proposed. Wave 0 starts with the lead signing them off (see
[../handoff/README.md](../handoff/README.md)).
