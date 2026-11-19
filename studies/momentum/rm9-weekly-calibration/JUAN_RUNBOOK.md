# Juan's runbook | Human or Claude operator

**Your deliverable:** a verified research report and reproducible weekly decisions. No code in this runner submits brokerage orders. A healthy job, a favorable backtest and an independent platform replay are three different results. Do not use one as evidence for another.

## 1. Prove one installation, then one real feed

Use Python 3.12 via `uv sync --locked --python 3.12` in the repo root. Install dependencies where downloads are allowed. On HiPerGator use one approved CPU allocation on a single node; there is no GPU or MPI path here. Choose persistent output/storage with working POSIX locks, atomic rename and SQLite fsync. Set a writable Numba cache if the install is read-only: `export NUMBA_CACHE_DIR="/your/writable/numba-cache"`. Do not put the only checkpoint on temporary local storage that disappears with the job.

1. Run the [laptop demo](../../../MOMENTUM_HANDOFF.md). It must end in PASS. No keys.
2. On an approved machine with access to the team's historical SIP data, supply `ALPACA_DATA_KEY` and `ALPACA_DATA_SECRET` through its secret environment or an untracked local `.env`. Do not distribute keys in the handoff, commit them, send them to a bot, or copy them to the compute node. The existing HTTP client shares a rate limiter within that process; reserve headroom for other applications using the account. `ALPACA_DATA_REQUESTS_PER_MINUTE` controls the process budget. The pull is historical REST, not simultaneous live WebSocket subscriptions.
3. Test one symbol and a short sample before a large acquisition:

```sh
uv run --locked python scripts/momentum/rm9.py probe --mode pilot
uv run --locked python scripts/momentum/rm9.py plan --mode pilot
uv run --locked python scripts/momentum/rm9.py stage --mode pilot --approximate-master --pull-workers 2
uv run --locked python scripts/momentum/rm9.py audit-data --mode pilot --deep
uv run --locked python scripts/momentum/rm9.py benchmark --mode pilot --workers 2
uv run --locked python scripts/momentum/rm9.py run --mode pilot
uv run --locked python scripts/momentum/rm9.py verify --mode pilot --deep
```

Probe tests SPY bars and a ten-second quote window; check positive counts and zero orders. Pilot uses 12 weeks, a target of 9 stocks, and a small set of recognizable candidates when using the approximate directory. It uses the real price/quote schema and actual fund accounting. It may legitimately choose cash. If a pilot's quote strata lack coverage, stop and inspect rather than imputing spreads. Pilot is engineering evidence, not a smaller primary market result.

## 2. Build the real archive once

Prefer an operator-supplied dated instrument file with provenance and coverage of dead names, new listings and symbol changes (including the pre-study warmup and quote sample). The real universe is selected from this candidate directory each weekend. The directory must be **broader than 1,000 names** if 1,000 can qualify after screens. One row is a dated alias of a stable security, not necessarily a distinct ticker. Inclusive dates; no overlapping symbol or identity intervals. Use an honest round-lot share count. Example shape only (fictional):

```csv
asset_id,symbol,start,end,kind,exchange,lot
example-id-1,EXA,2008-01-01,2019-03-14,common,NYSE,100
example-id-1,EXB,2019-03-15,2026-12-31,common,NYSE,100
example-id-2,EXC,2022-06-01,2026-12-31,common,NASDAQ,100
```

```sh
uv run --locked python scripts/momentum/rm9.py plan --mode full
uv run --locked python scripts/momentum/rm9.py probe --mode full
uv run --locked python scripts/momentum/rm9.py stage --mode full --master /actual/path/instruments.csv --archive /actual/path/rm9-data --pull-workers 2
uv run --locked python scripts/momentum/rm9.py audit-data --mode full --archive /actual/path/rm9-data --deep
```

Without a dated master, replace the `--master FILE` arguments with the explicit `--approximate-master`. This queries active and inactive equities, excludes obvious non-common names, and treats a ticker as a literal historical symbol. It cannot reconstruct the full historical listings directory, distinguish every security type, or recover every delisting/rename/round lot. Its outputs are labeled approximate. Do not present them as survivorship-free. If the archive never attains a full 1,000-stock week, staging refuses to seal. If it sometimes falls short, the counts are disclosed; review them before calling the run representative.

Acquisition gets raw and split-adjusted **daily** history for candidate identities; it fetches **minute** symbol-months needed for selected names, lookbacks and eligibility checks, plus a pre-study quote sample. It caches compressed Parquet + SHA-256 and seals the instrument file, weekly schedule, quote table and raw inventory. A retry reuses complete units. Keep the entire sealed archive, including `*.sha.json` alongside Parquet, for repeatable replays. Each new candidate can need old bars; deleting them mid-study defeats reuse. Already-sealed archives perform compute without HTTP. `audit-data --deep` checks all input bytes, including after a transfer. Restricted market data are not shipped with this handoff.

Before a decade pull, inspect pilot requests, bytes, seconds, storage quotas, API allowance, and historical entitlements; test on the actual acquisition/storage path. At 1,000 × 2,514 × 390 the full width is roughly **980 million stock-minutes** before early closes or missing sessions; this is a scale illustration, not a billed quote or an exact request count. Prepared numerical arrays for 522 full-width weeks alone are provisioned at roughly **39 GB decimal**, plus raw data, SQLite history, Numba/temp/build space and backups. Node “free disk” can exceed your user/group quota: confirm the quota yourself. No raw download time is counted toward the 48-hour compute window.

## 3. Freeze the real compute budget

Run `benchmark` where `run` will execute, with its actual cores, RAM and filesystem. It measures feature builds, serial and parallel replays, surrogate refits, reads and committed writes. It sees Slurm limits, CPU affinity and cgroup limits where exposed. Inspect `job/machine-profile.json`: `machine`, `timings`, `storage_probe`, `overhead_seconds`, `trials`, `selected` and `status`. Wait for **READY** and a non-null budget. The number of workers measured fastest is frozen; adding threads can be slower. The archived price histories and the three scientific arms never change according to a computer's speed.

A benchmark is an estimate. Data density, file-cache evictions and busy shared storage can change throughput. The runner does not assume every weekly backtest lasts 48/(number of weekends) hours; it assigns equal evaluation opportunity per later weekend and measures the actual stage progression. Slurm wall-clock limits are **per submitted job**; the application separately caps cumulative active study time at 48 hours. If savings permit a bigger candidate budget later, start a **new** preregistered run, don't change a budget after seeing results.

## 4. Quick Linux/HiPerGator handoff

Use [UF's scheduling guide](https://docs.rc.ufl.edu/quickstart/scheduling/) and [job script examples](https://docs.rc.ufl.edu/scheduler/sample_job_scripts/) for current local rules. Juan supplies an allocation authorized for SHPE, site software/module setup, job time, quota, account/QoS/partition if required, and durable paths. The checked-in shell script is a template, not a claim to have submitted anything on HiPerGator. The phase commands also run directly on a Linux workstation or Windows terminal without Slurm.

From the **repo root** on the cluster, set real values, activate/install the locked environment, and create the log folder *before* sbatch. Below, shell variables in CAPITALS for cores, memory and wall-time must be set to Juan's approved values. For example, `WALLTIME` has Slurm's `HH:MM:SS` shape; do not request resources you don't have. Use the same node class and CPU/memory **request** for benchmark and full run.

```bash
export RM9_PYTHON="$PWD/.venv/bin/python"
export RM9_RUN="/absolute/durable/path/rm9-full-01"
export RM9_ARCHIVE="/absolute/durable/path/rm9-data"
export RM9_MODE=full
mkdir -p "$RM9_RUN/logs"
# Set CORES, MEMORY, WALLTIME first (for instance in your local shell).
# Include --account / --qos / --partition below only as required by your allocation.
submit_rm9 () {
  sbatch --chdir="$PWD" --cpus-per-task="$CORES" --mem="$MEMORY" --time="$WALLTIME" \
    --output="$RM9_RUN/logs/%x-%j.out" --export=ALL scripts/momentum/hpg/rm9.sbatch
}
export RM9_PHASE=audit-data RM9_DEEP=1
submit_rm9
# After PASS, submit a separate benchmark job:
export RM9_PHASE=benchmark
submit_rm9
# After READY and inspection, start the full run:
export RM9_PHASE=run
submit_rm9
# Only after COMPLETE:
export RM9_PHASE=verify RM9_DEEP=1
submit_rm9
```

Run the phases **one at a time**, checking each result before the next. If desired, use `RM9_JOB_MINUTES` to finish a batch and exit before the scheduler time limit; leave a safety margin. The template requests a USR1 warning 180 seconds before the job ends. Signals request a checkpoint at the next boundary; a crash can lose the current batch or partial prepared week, never a committed decision. Slurm hard termination between heartbeats may miss up to roughly the 15-second heartbeat interval of active time. Keep one writer per output path. No automatic requeue, no GPU and no keys in the job.

The workstation equivalent after staging is `python scripts/momentum/rm9.py benchmark --mode full --root RUN --archive ARCHIVE`, then `run`, then `verify --deep` using the same arguments and locked Python. To rehearse graceful stopping, pass `--pause-after 1` once; run again without the flag.

## 5. Monitor and diagnose

```sh
uv run --locked python scripts/momentum/rm9.py status --mode full --root /actual/path/rm9-full-01
```

This reads atomic status files while a job is running. On the cluster also check the Slurm output file and scheduler status. `job/progress.json` is updated at least around every 15 seconds while computing: stage, active/remaining time, RSS, sampled peak, and during backfills completed versus total reference cells and an ETA. The ETA is a **reference-replay estimate**, not a guaranteed finish time. Absence of new completed cells during compilation/refitting is normal; a stale heartbeat plus a terminated scheduler job is not. `progress.jsonl` is the timeline. Guard `clock.json`, `machine-profile.json`, `archive.sqlite3`, prepared weeks and manifests against manual edits.

| State / exit | What it means and what to do |
|---|---|
| NOT_STARTED / READY / RUNNING | No compute yet / benchmark admitted a frozen workload / one current writer. Do not launch a second writer. |
| PAUSED (75) | Graceful interruption, requested demo checkpoint or per-job limit. Submit `run` again, same repo, same archive, same root and adequate allocation. A pending benchmark restarts if its profile was not yet sealed. |
| NO_FIT (3) | Even the minimum frozen budget does not meet forecast time/memory/storage. Inspect per-budget values and quota; use a **new run root** with an approved larger/faster allocation, reusing the archive. Do not silently shrink the scientific scope or rerun until signals look attractive. |
| BUDGET_EXHAUSTED (4) | Partial study reached the cumulative cap. No final policy comparison; keep diagnostics. A longer or smaller protocol is a separately identified experiment. |
| FAILED / exception (1) | See last error and scheduler log. 401/403: operator checks SIP permissions privately. Quote strata missing: inspect the historical master, sampled dates and feed. DATA_QUALITY_STOP: inspect listing intervals and missing RTH coverage. UNRESOLVED_EXIT: inspect the halt, end-of-session tape and corporate action. Never invent a price or discard a stock after seeing its result. |
| Hash mismatch, no disk, OOM or scheduler kill | Do not delete the evidence. Restore the matching sealed inputs or transfer again and `audit-data --deep`. If only a job interruption, resume in place. If code/data/protocol must change, use a **new root** and record the reason. A timing profile from one host is not a benchmark for another. |
| COMPLETE, then verification PASS | Research calculations reconcile. Read market findings separately. If verify fails, preserve evidence; a stale older PASS is not a substitute for the current command's exit code. |
| Ledger compare FAIL (5) | See research-only/platform-only entries and differing fields. Repair the adapter, not the historical evidence; do not deploy. Two empty ledgers do not establish parity. |

## 6. Inspect and hand over the result

- `archive/archive.json` (or external archive root): date coverage, width/shortfalls, source limits, hashed schedules. `quote-costs.json`: sampled windows, observed buckets, realized cutoff. `schedule/NNNN.json`: membership, gate reasons, regime inputs. Keep raw market data private.
- `job/run-manifest.json`, `inputs.json`, `provenance.json`, `machine-profile.json`: versioned code/data/protocol IDs, worker count and frozen search budget. The source hash covers actual Python contents, not just the last Git commit. A portable checkout without Git still has a source hash.
- `job/weekly-configs/NNNN-{uniform,recency,similarity}.json`: the decision **before** that week, including cash and acceptance checks, ticker IDs, cutoff, search/validation periods and account. `job/trials.csv`: actual nominations and origin. `archive.sqlite3`: complete cached reference cells and forward decision records, including finalists and rejection counts.
- `job/daily.csv`: date-aligned funded returns and costs, base and stress. `job/weekly.csv`: one row per policy-week for plots and paper tables, including selected configuration, equity, gross/net/stress returns, costs, trades, cash weeks, ambiguity and concentration. `results.html`: offline human-readable results. `report.json`: separate primary and development results, inactivity, costs, drawdowns, trials and paired intervals. `output-seal.json` and `verification.json` check outputs. The source checkout contains no market returns.
- Do not read a simulated “winner” into a demo. In a full run check net absolute results, stress, drawdown, participation, concentration, gaps and both interval widths. If no rule clearly beats uniform, report that. Check the limitations of the instrument source and sampled costs before extrapolating. A scientifically inconclusive experiment can still be a technical success.

**Platform boundary:** Juan owns the SHPE platform adapter, not the research runner. Export the research ledger for a nonempty forward week (zero-based index, full first forward = 117; demo = 7). Generate the same week's ledger independently through the platform using the exported config, same inputs, `platform-costs.json`, same starting equity, and the same warmup/previous-day ranks. The sidecar is mandatory for sampled quote-mode execution. Then:

```sh
uv run --locked python scripts/momentum/rm9.py ledger --mode full --root RUN --archive ARCHIVE --week 117 --policy uniform
uv run --locked python scripts/momentum/rm9.py compare-ledgers --research-ledger RUN/job/ledger-117-uniform.csv --platform-ledger PLATFORM.csv
```

Match exact stable asset ID, entry/exit UTC, integer quantity and exit reason; price tolerance is 0.5 basis point relative, net P&L tolerance one cent. Include normal days, halts/missing bars when present, and an early close in the independent replay review. A passing comparison on one nonempty week is a first compatibility check, not proof every date matches. Do not submit actual orders from the archived historical configs.

**If Claude takes over debugging:** first read this file, the protocol and the execution audit. Ask for the exact command, status code, sanitized log tail, machine profile and input/seal IDs. Reproduce with the smallest failing slice or the offline demo. Preserve the original run; never “fix” a return, fingerprint, threshold or date to turn a check green. Rerun the relevant regression and the public demo, and state exactly what was and was not verified.
