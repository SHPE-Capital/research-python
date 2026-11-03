# C1l: Import-layer enforcement

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 0 | Beg–Int | C1a, migration PR | none (CI gate for all later PRs) | `repo-architect` (review: lead) |

## Goal

Enforce the package rank order in CI so lower layers never import higher ones.

## Owns (exclusive)

- The `[tool.importlinter]` section of `pyproject.toml` and the `lint` dev dependency on `import-linter`
- The `lint-imports` step in `.github/workflows/ci.yml`
- `tests/unit/test_layers.py`

## Reads

- [design/architecture.md](../../design/architecture.md) (ranks)
- [adr/0002-layered-package-structure.md](../../adr/0002-layered-package-structure.md)

## Interface

A `layers` contract, highest first:

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

Plus a `forbidden` contract: `shpe_research.trade` must not import
`shpe_research.events`.

## Behaviour

- Packages in the same rank may import each other only where the contract
  lists them as independent=false; default is independent.
- `labels` (rank 4) may import `trade` (rank 3) for `triple_barrier`.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/test_layers.py` | `test_lint_imports_passes` | `lint-imports` exits 0 on the tree |
| | `test_violation_detected` | A temp module in `features` importing `models` makes `lint-imports` fail |

## Out of scope

Moving modules (migration PR).

## Done when

CI runs `lint-imports` and it passes on main.
