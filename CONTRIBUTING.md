# Contributing

## Claiming a task

1. Pick an open task in [docs/tasks.md](docs/tasks.md). Its row lists the files
   to work in, what it depends on, and what "done" means.
2. Open a GitHub issue titled `<task id>: <task name>` and assign yourself, so
   nobody else starts the same work.
3. Branch from `main` (`git checkout -b <your-name>/<task id>`), implement, and
   open a PR that references the issue.

Stub files contain a `TODO(<task id>)` docstring describing the contract to
satisfy, and tasks with acceptance tests have them in `tests/todo/`, marked
skipped. Remove the skip marker as part of your PR; the task is done when those
tests pass.

## Rules every PR follows

- **No future data.** A value computed for time `t` may only use information
  available at `t`. Bars become available at `ts_available`, not `ts`. Every
  feature must pass `assert_no_lookahead` (see `shpe_research/features/base.py`).
- **Never commit data, keys or run outputs.** `data/`, `runs/`, `snapshots/`
  and `.env` are git-ignored. Do not put keys in notebooks, logs or screenshots.
- **Scale inside the model.** Normalization statistics (means, standard
  deviations, scalers) are fit on training rows only, inside the model's
  pipeline, never during ingestion.
- **Record every attempt.** Results you discard still count toward how many
  things were tried, which matters when judging whether a result is luck.
- **Tests and lint pass**: `uv run ruff check` and `uv run pytest`.
- Someone other than the author reviews each PR.
