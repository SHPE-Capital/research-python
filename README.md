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

## Real market data

**Without Alpaca keys:** ask the research lead for the current workshop
snapshot (a `.zip`), then check it and point a config at it:

```bash
uv run shpe data snapshot verify path/to/workshop-v0.zip
# in your experiment config:
#   data: {source: snapshot, path: path/to/workshop-v0.zip, role: dev}
```

**With your own keys** (copy `.env.example` to `.env` and fill it in):

```bash
uv run shpe data bars --symbols SPY,KO --from 2026-10-01 --to 2026-10-27        # 1-minute bars
uv run shpe data audit --symbols SPY,KO --from 2026-10-01 --to 2026-10-27       # coverage and data problems
uv run shpe data ticks --symbols KO --from 2026-10-27 --to 2026-10-27 \
    --start-time 09:30 --end-time 10:00                                          # quotes and trades
uv run shpe data tick-summary --symbols KO --from 2026-10-27 --to 2026-10-27   # per-minute spread, imbalance, signed volume
uv run shpe data universe --config configs/universe/workshop_v0.yaml --download
uv run shpe data snapshot build --name workshop-v0 --universe data/universes/workshop_v0.yaml \
    --from 2026-07-01 --to 2026-09-30 --extra-symbols SPY
uv run shpe data verify                                                          # re-check stored checksums
```

Downloads are resumable: re-running a command skips what is already stored.
Data lives in `./data` (or `RESEARCH_DATA_DIR`) and snapshots in
`./snapshots`. Both are git-ignored. Snapshots contain Alpaca market data, so
share them only with members, never publicly.

## Where to start

- [docs/tasks.md](docs/tasks.md): the task board. Pick an open task and claim it.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to claim a task and what a PR needs.
