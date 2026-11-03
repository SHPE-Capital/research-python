# 0008. Typed configs with Pydantic

- Status: Proposed
- Date: 2026-11-02
- Tasks: C1h, T7, T8

## Context

`ExperimentConfig` in `experiments/runner.py` is a dataclass built from YAML.
Nested sections (`data`, `features`, `label`, `model`, `split`) are plain
dicts, so a misspelled key is ignored and the default is used without warning.
Registry builders (`build_feature`, `build_model`) check the type key but not
the parameters. `StrategySpec` will feed `export_platform_config`, which must
not export a config with a silently dropped field.

## Decision

We will define every config in `config.py` and `strategies/spec.py` as
Pydantic models with `extra="forbid"`. Each registered component may declare a
Pydantic params model; the builder validates `{type: ..., **params}` against it
and lists the registered keys when the type is unknown. Configs are validated
in full before any data loads. The resolved config, with defaults filled in, is
written to the run directory.

Pydantic becomes a core dependency (added in the Wave 0 dependency pass).

## Consequences

- Typos and wrong types fail at load, with the field path in the message.
- The JSON schema of each config can be generated for editor completion and
  for the docs site.
- `StrategySpec` validation is the first half of the parity guarantee: a spec
  that validates and exports names only fields the platform accepts.
- One more dependency, and each component author writes a small params model.
  `shpe new` (C1j) generates it.

## Alternatives considered

- **Dataclasses with manual checks.** What exists today; checks are
  incomplete and spread across builders.
- **Hydra or OmegaConf.** Strong at composing and overriding configs, but adds
  its own runtime and conventions; Pydantic covers validation, and sweeps
  (M2) handle overrides.
- **JSON Schema files.** Language-neutral, but a second source of truth next
  to the Python types.
