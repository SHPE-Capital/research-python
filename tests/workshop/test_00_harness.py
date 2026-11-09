"""W0 acceptance tests: the workshop harness (bands, notebook runner, real-data guard)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.workshop.bands import Band, assert_in_band
from tests.workshop.conftest import run_notebook

pytestmark = pytest.mark.workshop

TOY_NOTEBOOK = '''\
# %% [markdown]
# Toy notebook used by the harness tests.

# %%
import exercises
from workshop._common import real_data_enabled

# %%
try:
    answer = exercises.solve()
    print(f"ANSWER={answer}")
except NotImplementedError:
    print("EXERCISE pending")

# %%
if real_data_enabled():
    print("REAL-DATA CELL RAN")
else:
    print("real-data cell skipped")
'''

STUB = '''\
def solve():
    # EXERCISE
    raise NotImplementedError
'''

SOLUTION = '''\
def solve():
    return 42
'''


@pytest.fixture
def toy_example(tmp_path: Path) -> Path:
    folder = tmp_path / "99_toy"
    folder.mkdir()
    (folder / "notebook.py").write_text(TOY_NOTEBOOK, encoding="utf-8")
    (folder / "exercises.py").write_text(STUB, encoding="utf-8")
    return folder


def test_band_all_seeds():
    values = {0: 0.5, 1: 0.6, 2: 0.9, 3: 0.55, 4: 0.52}
    band = Band("auc", 0.4, 0.7)
    assert band.seeds == (0, 1, 2, 3, 4)

    with pytest.raises(AssertionError) as err:
        assert_in_band(lambda seed: values[seed], band)  # only seed 2 is outside
    message = str(err.value)
    for seed, value in values.items():
        assert f"seed {seed}" in message and f"{value}" in message
    assert "auc" in message

    ok = assert_in_band(lambda seed: values[seed] if seed != 2 else 0.65, band)
    assert ok == [0.5, 0.6, 0.65, 0.55, 0.52]


def test_band_rejects_nan():
    with pytest.raises(AssertionError):
        assert_in_band(lambda seed: float("nan"), Band("x", 0.0, 1.0, seeds=(1, 2)))


def test_notebook_runs_as_script(toy_example: Path, tmp_path: Path):
    stub = run_notebook(toy_example)
    assert stub.returncode == 0, stub.stderr
    assert "EXERCISE pending" in stub.stdout

    solution = tmp_path / "solution_exercises.py"
    solution.write_text(textwrap.dedent(SOLUTION), encoding="utf-8")
    solved = run_notebook(toy_example, solutions=solution)
    assert solved.returncode == 0, solved.stderr
    assert "ANSWER=42" in solved.stdout
    # the swap happens in a copy: the original stub is untouched
    assert "NotImplementedError" in (toy_example / "exercises.py").read_text(encoding="utf-8")


def test_solutions_from_environment(toy_example: Path, tmp_path: Path, monkeypatch):
    solution = tmp_path / "sol.py"
    solution.write_text(SOLUTION, encoding="utf-8")
    monkeypatch.setenv("SHPE_WORKSHOP_SOLUTIONS", str(solution))
    assert "ANSWER=42" in run_notebook(toy_example).stdout


def test_real_data_guard(toy_example: Path, monkeypatch):
    monkeypatch.delenv("SHPE_WORKSHOP_REAL_DATA", raising=False)
    off = run_notebook(toy_example)
    assert off.returncode == 0, off.stderr
    assert "real-data cell skipped" in off.stdout
    assert "REAL-DATA CELL RAN" not in off.stdout

    on = run_notebook(toy_example, env={"SHPE_WORKSHOP_REAL_DATA": "1"})
    assert "REAL-DATA CELL RAN" in on.stdout
