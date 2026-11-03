# Workshop examples

Runnable material for the ML-for-stock-research workshop. The spec, schedule
and rubric are in [docs/workshop/README.md](../docs/workshop/README.md); setup
is [docs/workshop/setup.md](../docs/workshop/setup.md). This page explains the
layout and the conventions every example follows ([ADR 0010](../docs/adr/0010-workshop-notebooks-jupytext.md)).

## Layout

One folder per example, `workshop/NN_slug/` (for example `01_leakage_lab`):

```
workshop/NN_slug/
  README.md          # learning goals, run commands, exercise list
  notebook.py        # jupytext percent format: the walkthrough
  exercises.py       # functions participants fill in (they raise NotImplementedError)
  configs/*.yaml     # ExperimentConfig / StrategySpec files
workshop/_common/    # display helpers and the real-data guard (no research logic)
tests/workshop/      # test_NN_slug.py per example, plus the shared harness
```

`.ipynb` files are not committed.

## Running a notebook

```bash
uv run python workshop/NN_slug/notebook.py            # as a script
uv sync --group workshop                              # jupytext + JupyterLab
uv run --group workshop jupyter lab                   # open the .py as a notebook
```

Notebooks run top to bottom on the synthetic fixture, offline, with no API keys.
A notebook that imports `workshop._common` needs the repo root on `sys.path`; start
the notebook with:

```python
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(next(p for p in [_here, *_here.parents] if (p / "workshop" / "_common").is_dir())))
```

(The test harness sets `PYTHONPATH` itself.)

### Conventions

- Each section is a short explanation, a runnable cell and a question for participants.
- Fixed seeds. Every number is net of costs wherever trading is involved.
- Exercises are marked `# EXERCISE` and raise `NotImplementedError`. The notebook
  catches that and prints that the exercise is pending, so the starter version runs
  top to bottom.
- A cell that needs the frozen snapshot sits behind the guard:

  ```python
  from workshop._common import real_data_enabled

  if real_data_enabled():      # SHPE_WORKSHOP_REAL_DATA=1
      ...                      # snapshot cells, role: dev only
  ```

  Without the variable the cell is skipped. Never put keys or snapshot data in a notebook.
- Library code a notebook needs goes in `src/shpe_research/` through a normal task. If it
  is missing, link its task ID and use a clearly labelled stand-in in the notebook.

## Solutions branch

`main` holds the starter versions. The filled-in `exercises.py` files live on the
**`workshop-solutions`** branch, at the same paths. Rules:

- Solutions are never merged into `main`.
- Keep the branch current by merging `main` into it before each workshop.
- Facilitators share it after the session.
- CI on that branch sets `SHPE_WORKSHOP_SOLUTIONS` so the full path is tested.

## Tests and the harness

Each example has `tests/workshop/test_NN_slug.py`, marked `@pytest.mark.workshop`
(applied automatically under `tests/workshop/`; add `@pytest.mark.slow` to a test over
60 seconds). Each example test must finish in under 5 minutes on a laptop.

```bash
uv run pytest tests/workshop            # starters on main
uv run pytest -m workshop
uv run pytest -m "not workshop"         # skip them
```

Helpers in `tests/workshop/`:

| Helper | Use |
|---|---|
| `run_notebook(example, solutions=None, env=None, timeout=300)` in `conftest.py` | Runs `notebook.py` as a script in a scratch copy of the folder. `example` is a path or a slug. Returns a `CompletedProcess`; assert on `returncode`, `stdout`. |
| `Band(metric, low, high, seeds=(0, 1, 2, 3, 4))` and `assert_in_band(fn, band)` in `bands.py` | Runs `fn(seed)` per seed and requires every value in `[low, high]`. On failure the message lists every seed's value. |

Solutions swap: if `solutions=` is given, or `SHPE_WORKSHOP_SOLUTIONS` points at a
solution module (a file, or a folder containing `exercises.py`), the scratch copy's
`exercises.py` is replaced by it. The repo files are never modified.
`SHPE_WORKSHOP_REAL_DATA` is cleared for every run unless the test sets it through `env`.

Set pass bands from the spread over several seeds, never from one lucky seed, and say in
the test comment which planted effect (see `src/shpe_research/data/fixtures.py`) the band
comes from.
