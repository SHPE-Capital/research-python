# research-python

SHPE Capital's research system: market data, features, labels, models and
evaluation for the club's strategy teams. It runs independently of the trading
platform. The platform only consumes finished model artifacts.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for you.

```bash
git clone https://github.com/SHPE-Capital/research-python.git
cd research-python
uv sync
uv run pytest
uv run shpe run configs/experiments/hello_model.yaml
```

The last command runs the hello model end to end on synthetic bars, with no
keys or downloads needed. It builds features and a 30-minute forward-return
label, trains a logistic regression on the first 60% of sessions, and writes a
report to `runs/<timestamp>-hello_model/`. The fixture contains a small,
deliberate signal, so expect a validation AUC slightly above 0.5. A much
higher number on fixture data means something is leaking future data.

## How the pieces fit

```
data/         calendar, downloads, store, audit, universe, snapshots   -> bars contract
features/     point-in-time features                                   -> feature columns
labels/       outcomes that start after the decision                   -> label column
validation/   chronological splits                                     -> split per row
models/       anything implementing the Model contract                 -> scores + manifest
evaluation/   the same report for every model
experiments/  one YAML config -> one recorded run
```

Every arrow is a contract in [docs/contracts.md](docs/contracts.md). Code that
takes and returns those shapes works with everyone else's code.

## Where to start

- [docs/tasks.md](docs/tasks.md): the task board. Pick an open task and claim it.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to claim a task and what a PR needs.
