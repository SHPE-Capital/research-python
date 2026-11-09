# Participant setup

Do this before session 1. It takes about 15 minutes, most of it downloads.
No Alpaca keys are needed for any session.

## 1. Install uv

[uv](https://docs.astral.sh/uv/) manages Python and the dependencies. It
installs Python 3.12 for you, so you don't need to install Python yourself.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal afterwards so `uv` is on your `PATH`.

## 2. Clone and check the install

```bash
git clone https://github.com/SHPE-Capital/research-python.git
cd research-python
uv sync
uv run pytest
uv run shpe run configs/experiments/hello_model.yaml
```

The last command runs the hello model end to end on synthetic bars and writes
a report to `runs/<timestamp>-hello_model/report.md`. **Check:** the
validation AUC should be slightly above 0.5. The fixture contains a small,
deliberate signal; a much higher number means something leaks future data.

## 3. Optional extras, by example

| Install | Adds | Needed for |
|---|---|---|
| `uv sync` | numpy, pandas, pyarrow, scikit-learn, exchange-calendars | Part 1, example 1 |
| `uv sync --extra ml` | LightGBM, ONNX, MLflow | Examples 2–9 |
| `uv sync --extra nn` | PyTorch (large download) | The LSTM and PyTorch MLP in example 2 only |
| `uv sync --extra ml --extra nn` | Both | Everything |
| `uv sync --group workshop` | jupytext, JupyterLab | Opening the notebooks as notebooks (added by W0) |

`uv sync` removes extras you don't name, so list every extra you want in a
single command.

On macOS, LightGBM needs the OpenMP runtime: `brew install libomp`.

## 4. Data

**Fixture (every session):** generated on demand; nothing to download. To
write it to a file for your own experiments:

```bash
uv run shpe fixtures            # writes data/fixtures/synthetic_1min.parquet
```

**Workshop snapshot (from example 1 onwards):** ask the research lead for
the current snapshot zip, put it under `snapshots/` (git-ignored) and verify
it:

```bash
uv run shpe data snapshot verify snapshots/workshop-v0.zip
```

Point an experiment config at it:

```yaml
data: {source: snapshot, path: snapshots/workshop-v0.zip, role: dev}
```

Use `role: dev` only. Held-out names are opened by the facilitator, once,
at the end of an example.

The snapshot contains Alpaca market data. Keep it on your machine or the
club's private drive, and never commit, upload or share it publicly.

## 5. Notebooks

Workshop notebooks are plain `.py` files in jupytext percent format under
`workshop/NN_slug/`. Run one as a script:

```bash
uv run python workshop/01_leakage_lab/notebook.py
```

Or open it as a notebook in JupyterLab once W0 has added the `workshop`
group: `uv run --group workshop jupyter lab`. VS Code's Python extension
also runs `# %%` cells directly.

## 6. Git

Create a GitHub account if you don't have one and ask to be added to the
SHPE-Capital organisation. Every session ends with a pull request, so check
that you can push a branch:

```bash
git checkout -b <your-name>/setup-check
git push -u origin <your-name>/setup-check
```

Delete the branch afterwards.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Open a new terminal; on Windows, sign out and back in |
| `uv run pytest` fails on a fresh clone | Run `uv sync` again, then report it in the workshop channel with the full output |
| LightGBM import error on macOS | `brew install libomp`, then `uv sync --extra ml` |
| Snapshot verify reports a checksum mismatch | Download the zip again; never edit files inside a snapshot |
| Times look 4–5 hours off | You are looking at UTC. All stored times are UTC; convert to `America/New_York` for display only |
