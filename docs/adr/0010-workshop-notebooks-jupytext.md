# 0010. Workshop notebooks as jupytext percent files

- Status: Proposed
- Date: 2026-11-02
- Tasks: W0–W9 (see [docs/workshop](../workshop/README.md))

## Context

The workshop teaches ML for stock research through worked examples that use
this repo's pipeline. Each example needs a narrative notebook, exercises and
reference solutions. `.ipynb` files store outputs and metadata as JSON, which
makes diffs unreadable, merge conflicts frequent and code review of the
teaching material impractical. The examples also go stale silently when the
library changes unless something runs them.

## Decision

- Each example lives in `workshop/NN_slug/` with `notebook.py` (narrative and
  worked code) and `exercises.py` (tasks with `TODO` cells), both as jupytext
  percent-format Python files. Participants open them as notebooks through
  jupytext; `.ipynb` files are not committed.
- Examples import from the `shpe_research` package and run configs from `configs/`;
  they do not copy library code.
- Solutions live on the `workshop-solutions` branch, so participants do not see
  them by default.
- CI runs every `notebook.py` against the planted-effects fixture in
  `tests/workshop/`; the solutions branch runs `exercises.py` too. Each example
  has a time budget.
- Notebook dependencies (jupyter, jupytext, plotting) are in a `workshop`
  dependency group, not in the core package.

## Consequences

- Workshop material is reviewed and diffed like code.
- Library changes that break an example fail CI on the same PR.
- Running on the fixture keeps CI fast and offline; examples on real data
  stay manual and are documented as such.
- Keeping two branches in sync is a cost; the solutions branch is rebased on
  `main` before each workshop.

## Alternatives considered

- **Commit `.ipynb` with outputs stripped (nbstripout).** Diffs improve but
  remain JSON; reviews are still hard.
- **Paired `.ipynb` and `.py`.** Two files per notebook drift apart.
- **Solutions in the same branch under `solutions/`.** Simpler, but
  participants see answers before attempting the exercises.
- **Quarto or Jupyter Book source.** Good for publishing, but adds a toolchain
  for a teaching repo that already plans a mkdocs site
  ([0012](0012-docs-diataxis-mkdocs.md)).
