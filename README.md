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
```

## Where to start

- [docs/tasks.md](docs/tasks.md): the task board. Pick an open task and claim it.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to claim a task and what a PR needs.
