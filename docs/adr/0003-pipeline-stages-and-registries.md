# 0003. Pipeline stages and registries

- Status: Proposed
- Date: 2026-11-02
- Tasks: C1a, C1c, C1f, C1i

## Context

`experiments/runner.py:build_dataset` hard-codes the pipeline: session filter,
features, a `ForwardReturn` label only (any other label type raises), sampling
every N minutes, one `split_by_day`. Features and models already use
registries: `FEATURE_REGISTRY` with `@register_feature` in `features/base.py`,
and `@register_model` in `contracts/model.py`. Without the same pattern for
the other parts, every new label, sampler or split method is another branch in
the runner, and parallel tasks all edit the same file.

## Decision

We will split the pipeline into stages in `pipeline/`: load, universe, session
filter, features, transforms, decisions, labels, join, fit scheme, model,
evaluate, record. Each stage is a function from frames to frames. Only load and
record touch the file system.

Every pluggable part uses a registry with the `FEATURE_REGISTRY` pattern: a
module-level dict, a decorator that rejects duplicate keys, and a builder that
takes `{type: <key>, ...params}` from config. `@register_label`,
`@register_sampler`, `@register_fit_scheme` and `@register_report_section` live
in `pipeline/registry.py`; `@register_strategy` lives in
`strategies/registry.py`. After C1a the runner dispatches through the
registries, and no other task edits it.

Stage details: [pipeline.md](../design/pipeline.md).

## Consequences

- A new label, sampler, fit scheme or report section is a new file plus one
  import line. Worktrees do not conflict on a shared file.
- Configs name components by stable keys, so moving a module does not break
  configs.
- Registration happens on import, so each package's `__init__.py` must import
  its modules. The `check_*` kits verify registration.
- Each stage can be tested on the fixture without the runner.

## Alternatives considered

- **Keep the runner and add branches.** Simple today, but every parallel task
  would conflict on one file and the runner would keep growing.
- **sklearn `Pipeline` end to end.** Fits the model stage, but not event
  sampling, labels with their own end times, or purged folds that need
  metadata the sklearn API does not carry. Fit-on-train processors still use
  sklearn inside the model.
- **A workflow framework (Kedro, Prefect, Airflow).** Adds infrastructure and
  concepts for a team that runs experiments locally. The stage functions can be
  wrapped by one later if needed.
