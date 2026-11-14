# Momentum compute resources (E6 intake)

Task E6. Spec: plan section 13 "Initial machines to benchmark". **No allocation
has been confirmed** for any host below except the reference machine. Fields
marked `TBD` are to be filled in by the host owner; do not guess them.

Benchmarks run natively, not in Docker (the compose CPU cap skews timings). Real
numbers come from the E1 scale matrix; stand-in replay numbers are never C4
numbers.

## Reference machine (measured 2026-11-13)

| Field | Value |
|---|---|
| CPU | AMD Ryzen 7 3700X, 8 cores / 16 threads |
| RAM | 15.9 GB total; only about 2.4 GB free with the usual apps open (2.4 to 3.4 GB observed) |
| Local scratch | 2 TB NVMe, about 86 GB free (80 to 86 GB observed) |
| GPU | RTX 3070 Ti (unused by this plan) |
| Wall-time limit | None (owner's machine; not available while in use) |
| Queue delay | None |
| Credit limit | None |
| Market data may be stored | Yes, outside the repo (`data/` is not committed) |
| Launch a checkpointed job | Native `uv run python scripts/momentum/bench.py ...`; checkpoint via the D6 week-chunk cache so a restart resumes at the last finished week |

Fit:

- N = 500 over 5 years (about 5 GB panel): fits, but close with ~2.4 GB free
  RAM. Close other apps or memory-map the panel.
- N = 1,000 over 10 years (about 20 GB panel): does not fit in RAM. Needs
  chunked memory-mapping (A6 week-chunk panel) or a larger-RAM host.
- Worker count is bounded by RAM per worker, not by the 16 threads; take the
  per-worker peak from E1 before choosing a pool size.

## Candidate host: AWS account

Status: no account or credits confirmed.

| Field | Value |
|---|---|
| Account owner / contact | TBD |
| Instance type(s) considered | TBD |
| Usable cores (vCPU) | TBD |
| RAM | TBD |
| Local scratch (instance store / EBS size) | TBD |
| Wall-time limit | TBD (none for on-demand; spot interruption notice if spot) |
| Queue delay | TBD |
| Credit limit / budget cap | TBD |
| Market data may be stored here | TBD (check data licence and Alpaca terms) |
| Launch a checkpointed job | TBD (command, checkpoint location, resume procedure) |

## Candidate host: university HPC

Status: no allocation confirmed.

| Field | Value |
|---|---|
| Cluster / contact / sponsor | TBD |
| Scheduler (e.g. Slurm) and partition | TBD |
| Usable cores per job | TBD |
| RAM per job / per node | TBD |
| Local scratch (path, size, purge policy) | TBD |
| Wall-time limit per job | TBD |
| Queue delay (typical and worst observed) | TBD |
| Credit / core-hour limit | TBD |
| Market data may be stored here | TBD (check cluster data policy) |
| Launch a checkpointed job | TBD (submit script, requeue/resume behaviour, checkpoint dir) |

## Candidate hosts: teammate laptops

One row per laptop. Copy the table for each teammate.

| Field | Value |
|---|---|
| Owner | TBD |
| CPU (cores / threads) | TBD |
| RAM total / typically free | TBD |
| Local scratch free | TBD |
| Wall-time limit (sleep, battery, availability window) | TBD |
| Queue delay | n/a |
| Credit limit | n/a |
| Market data may be stored here | TBD (disk encryption, who else has access) |
| Launch a checkpointed job | TBD |

## Recommendation

Pending host intake and the E1 scale-matrix numbers. Provisional split, to be
confirmed once the fields above are filled in:

- **Research run** (batch, checkpointed, high core and RAM, finite duration):
  a host with enough RAM for the chosen N and history, ideally university HPC
  or AWS if the scale matrix shows the reference machine's RAM is binding;
  otherwise the reference machine overnight with the panel memory-mapped.
- **Always-on paper service** (small, steady, low-latency, reliable uptime): a
  separate small always-on host (for example a small AWS instance). It must
  not be a laptop or the reference machine, and it does not need the research
  panel.

Decision owner: Person 1. Record the final choice and date here.
