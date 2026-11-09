# C1n: Documentation site

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 3 | Beg | C1a and the docs written by the eight writers | none | `repo-architect` (review: lead) |

## Goal

Build `docs/` into a browsable site with API pages generated from docstrings,
and fail CI on broken links.

## Owns (exclusive)

- `mkdocs.yml`
- `docs/api/` (generated stubs only)
- The `docs` dependency group in `pyproject.toml` (`mkdocs-material`, `mkdocstrings[python]`)
- The `mkdocs build --strict` step in `.github/workflows/ci.yml`

## Reads

- [docs/README.md](../../README.md) (navigation order), [adr/0012-docs-diataxis-mkdocs.md](../../adr/0012-docs-diataxis-mkdocs.md)
- Every file under `docs/`

## Interface

- Nav sections: Start here, Design, ADRs, Teams, Workshop, Handoff, Tasks, API.
- `docs/api/<package>.md` contains a single `::: shpe_research.<package>` directive per package.

## Behaviour

- Relative links in the existing docs work unchanged; no doc is rewritten for the site.
- Links to packets that do not exist yet fail the strict build; C1n lands after
  the writers' docs and after every linked packet file exists.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| CI | `mkdocs build --strict` | Exits 0 |
| CI | link check | No broken relative link under `docs/` |

## Out of scope

Hosting and deployment of the site.

## Done when

`mkdocs build --strict` passes in CI and `mkdocs serve` shows every section.
