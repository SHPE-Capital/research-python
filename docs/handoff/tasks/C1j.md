# C1j: `shpe new` scaffolding framework and templates

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Beg–Int | C1a, C1k | T8 (strategy template) | `repo-architect` (review: lead) |

## Goal

Give contributors one command that creates a correctly placed, registered and
tested component from a template, so new work starts in the right shape.

## Owns (exclusive)

- `src/shpe_research/scaffold/__init__.py`, `scaffold/new.py`, `scaffold/templates/feature/`,
  `scaffold/templates/model/`, `scaffold/templates/study/`
- The `new` subcommand in `src/shpe_research/cli.py`
- `tests/unit/scaffold/test_new.py`

T8 owns `scaffold/templates/strategy/` and the spec card; C1j only provides
the framework it plugs into.

## Reads

- [design/architecture.md](../../design/architecture.md) (layout and ranks)
- `src/shpe_research/features/base.py`, `models/base.py`
- `src/shpe_research/testing/` (C1k check kits)

## Interface

```
shpe new feature <subpackage>/<name>   # features/<subpackage>/<name>.py + test using check_feature
shpe new model <name>                  # models/<name>.py + test using check_model
shpe new study <team>/<id>-<slug>      # studies/<team>/<id>-<slug>/{README.md, config.yaml, notes.md}
shpe new strategy <team>/<name>        # registered by T8
```

- Template kinds are discovered from `scaffold/templates/<kind>/`; each holds
  files with `{{name}}`, `{{class_name}}`, `{{team}}` placeholders and a
  `template.toml` listing targets.
- `--dry-run` prints the paths without writing.

## Behaviour

- Refuses to overwrite an existing file.
- Rejects a feature subpackage that is not one of the migration-PR
  subpackages, and a team that is not momentum, statarb or reversals.
- Generated tests pass as generated (the template ships a trivial but valid
  implementation marked `TODO`).

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/scaffold/test_new.py` | `test_feature_scaffold_passes_check` | Generated feature in a tmp tree passes `check_feature` |
| | `test_model_scaffold_passes_check` | Generated model passes `check_model` |
| | `test_study_scaffold_layout` | The three study files exist under the team path |
| | `test_refuses_overwrite` | Second run exits non-zero and changes nothing |
| | `test_unknown_kind_lists_kinds` | Error lists the template kinds found |

## Out of scope

Strategy template and spec card (T8). Docs site (C1n).

## Done when

Tests pass and `shpe new feature returns/demo --dry-run` prints the expected paths.
