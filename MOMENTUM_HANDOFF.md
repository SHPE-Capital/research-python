# SHPE Capital | Momentum research handoff

**Start here, Christopher and Juan.** This is the research program for choosing which momentum settings to run each week. It uses the club's existing Python strategy grid, indicators, calendar, account simulator, and weekly platform contract. It adds the walk-forward experiment, reusable archive, adaptive benchmark, and evidence checks. It sends **zero broker orders**.

## Run it on a laptop

1. Extract the entire handoff ZIP to a short local path, for example `C:\SHPE`. Open the **research-python** folder inside it. With an existing Git checkout, follow the separate worktree instructions below instead. Do not use the older research bundle.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed. Open a terminal in the folder. In Windows Command Prompt, run **`RUN_MOMENTUM_DEMO.cmd`** (in PowerShell, **`.\RUN_MOMENTUM_DEMO.cmd`**). The first install needs the internet for Python 3.12 and locked dependencies. The demonstration itself uses no network, Alpaca keys, GPU, or club account. A fresh interpreter/Numba cache can make the first run slower.
3. Success means the final line says **PASS**. Open **`runs/rm9-demo/job/results.html`** for the readable result, **`report.json`** for the numbers, **`verification.json`** for checks, and **`weekly-configs/0007-uniform.json`** for an actual decision. Keep the terminal open if a stage fails; the runbook explains the messages.

The launcher opens the output folder in File Explorer on success; set `RM9_NO_OPEN=1` in unattended environments. The equivalent commands work on macOS or Linux, or in a Windows terminal with forward slashes:

```sh
uv sync --locked --python 3.12
uv run --locked python scripts/momentum/rm9.py demo --workers 2
uv run --locked python scripts/momentum/rm9.py verify --mode demo --deep
```

**Expected, not a performance target:** 12 historical fixture weeks, 9 stocks targeted each week, 4 initial search weeks + 3 validation weeks, then 5 previously unseen weeks. That produces 15 policy decisions, 26 actually tried settings and 286 configuration-week reference replays. Prices have deliberate trends so entries **and** exits are exercised. Positive demo returns have no investment meaning. Running the same command on a completed folder verifies/reuses it; for an independent rerun choose a new `--root`.

## What Juan receives

- **[Research guide](studies/momentum/rm9-weekly-calibration/LOCKED_RESEARCH_PROTOCOL.md)** — the question, exact windows and eligibility, a dated example, the search budget, and how to interpret a result.
- **[Execution audit](studies/momentum/rm9-weekly-calibration/EXECUTION_RULE_AUDIT.md)** — indicators, entry, sizing, sampled SIP execution costs, exits, missing data, and what is still only a model.
- **[Juan's operator guide](studies/momentum/rm9-weekly-calibration/JUAN_RUNBOOK.md)** — source data, a tiny real SIP probe, reduced real pilot, staging, Slurm, monitoring, restart, troubleshooting, verification and platform parity. This is also the file to give Claude.
- The ZIP's **evidence/** folder contains results from an executed synthetic run, one real-width synthetic benchmark, and test/restart evidence. These files are outside the repo source on purpose. The exact inputs for the live-data experiment are not included in the ZIP.

A laptop can prove the workflow. The decision to spend cluster time requires (1) a real-data pilot, (2) enough eligible, historically identifiable stocks and quote coverage, (3) measured capacity on Juan's allocation. No investment result or platform integration is implied by passing the demo.

## Put the reviewed changes into the existing GitHub repo

The download has **two ways to use the same source**: run `research-python/` immediately, or apply the included `momentum-research.patch` to a clean worktree of [SHPE-Capital/research-python](https://github.com/SHPE-Capital/research-python). Do not apply the patch on top of the already-patched ZIP source. There is nothing to commit on the original branch until you do this yourself.

From the root of your **existing Git clone**, on Windows (replace the example ZIP location if different):

```powershell
git fetch origin
git worktree add -b christopher/momentum-research-handoff ../research-python-momentum 19bbaee34561ecce7797e26e4e556739a8a10027
cd ../research-python-momentum
git apply --check C:/SHPE/momentum-research.patch
git apply C:/SHPE/momentum-research.patch
git status --short
.\RUN_MOMENTUM_DEMO.cmd
uv run --locked pytest tests/unit/calibration tests/integration/test_momentum_rm9_handoff.py
uv run --locked ruff check
uv run --locked lint-imports
```

If the branch name already exists locally, use an unused branch name in both Git commands. The worktree command uses the exact upstream revision used to build this handoff; no need to overwrite a teammate's branch or update files in your current checkout. If you get a patch conflict, stop and check the base with `git rev-parse HEAD`; do not force it. Quote paths that include spaces.

**Only after review**, run the following in the new, clean worktree:

```powershell
git add -- MOMENTUM_HANDOFF.md RUN_MOMENTUM_DEMO.cmd docs/tasks.md scripts/momentum/rm9.py scripts/momentum/rm9_scale.py scripts/momentum/hpg/rm9.sbatch src/shpe_research/calibration src/shpe_research/data/rm9_io.py src/shpe_research/data/rm9_market.py src/shpe_research/data/rm9_universe.py src/shpe_research/trade/account_fast.py studies/momentum/rm9-weekly-calibration tests/unit/calibration tests/integration/test_momentum_rm9_handoff.py
git diff --cached --stat
git diff --cached --check
git commit -m "Add reproducible weekly momentum calibration and compute handoff"
git push -u origin christopher/momentum-research-handoff
```

Open a pull request **from that branch to main**. Have Juan review the sampled-cost fill contract and reproduce the pilot before merging. Never add `.env`, the data archive, `runs/`, Python environments or broker keys. The optional handoff ZIP itself does not belong in the repository.
