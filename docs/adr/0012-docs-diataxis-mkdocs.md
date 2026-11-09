# 0012. Docs organised by Diátaxis, built with mkdocs

- Status: Proposed
- Date: 2026-11-02
- Tasks: C1n

## Context

Documentation is spread across `docs/` (workshop, design, ADRs, team guides,
handoff packets, research method) and module docstrings. Readers have
different needs: a workshop participant wants a guided path, a team member
wants to know how to add a feature, a reviewer wants the API, and a new lead
wants the reasons behind the design. Without a structure these get mixed in
one page each and nothing is easy to find. Broken links between docs go
unnoticed.

## Decision

We will organise the docs site with the Diátaxis framework and build it with
mkdocs-material:

| Diátaxis type | Where |
|---|---|
| Tutorials | `docs/workshop/` and the `workshop/` notebooks ([0010](0010-workshop-notebooks-jupytext.md)) |
| How-to guides | `docs/teams/`, `docs/research/` and how-to pages (add a feature, add a strategy, run a sweep) |
| Reference | API pages generated from docstrings with mkdocstrings; config schemas from the Pydantic models ([0008](0008-typed-configs-pydantic.md)) |
| Explanation | `docs/design/` and `docs/adr/` |

`docs/README.md` stays the canonical map for agents and contributors.
`docs/handoff/` is working material for implementation and is excluded from
the published navigation. CI runs `mkdocs build --strict`, so a broken link or
missing page fails the build. Docs dependencies are in a `docs` dependency
group.

## Consequences

- Each page has one job, and readers know where to look.
- Docstrings become part of the published reference, so public functions need
  them; ruff's docstring rules can enforce this per package.
- Strict builds catch broken links early, including links to packets that do
  not exist yet; those pages need a stub until the packet is written.
- One more CI step and dependency group.

## Alternatives considered

- **Plain Markdown on GitHub, no site.** No build step, but no API reference,
  no search and no link checking.
- **Sphinx with autodoc.** Mature and strong for API docs, but reStructuredText
  or MyST configuration is heavier than mkdocs for a mostly-Markdown repo.
- **A wiki.** Separate from the code, so it drifts and is not reviewed in PRs.
