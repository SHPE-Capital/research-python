#!/usr/bin/env bash
#SBATCH --job-name=shpe-rm9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@180
# Juan supplies an approved CPU/memory/time request, cwd, and output path when submitting.
# No account, QoS, partition, module name, scratch path, or secrets are assumed.
set -euo pipefail
: "${RM9_RUN:?absolute durable output path required}"
: "${RM9_ARCHIVE:?absolute path to sealed archive required}"
: "${RM9_PYTHON:?absolute path to locked environment Python required}"
: "${RM9_MODE:?demo, pilot or full required}"
: "${RM9_PHASE:?audit-data, benchmark, run or verify required}"
for path in "$RM9_RUN" "$RM9_ARCHIVE" "$RM9_PYTHON"; do
  [[ "$path" == /* ]] || { echo "Run, archive and Python paths must be absolute." >&2; exit 2; }
done
case "$RM9_MODE" in demo|pilot|full) ;; *) exit 2;; esac
case "$RM9_PHASE" in audit-data|benchmark|run|verify) ;; *) exit 2;; esac
[[ -f scripts/momentum/rm9.py && -x "$RM9_PYTHON" && -r "$RM9_ARCHIVE/archive.json" ]] ||
  { echo "Check --chdir, Python and sealed archive." >&2; exit 2; }
opts=()
if [[ "$RM9_PHASE" == run || "$RM9_PHASE" == benchmark ]]; then
  [[ -z "${RM9_JOB_MINUTES:-}" ]] || opts+=(--job-minutes "$RM9_JOB_MINUTES")
  [[ -z "${SLURM_CPUS_PER_TASK:-}" ]] || opts+=(--workers "$SLURM_CPUS_PER_TASK")
fi
if [[ "$RM9_PHASE" == verify || "$RM9_PHASE" == audit-data ]]; then
  [[ "${RM9_DEEP:-0}" != 1 ]] || opts+=(--deep)
fi
umask 077
# Research runs from the sealed archive; compute jobs do not need API keys.
unset ALPACA_DATA_KEY ALPACA_DATA_SECRET
# Replace the batch shell so Slurm's batch-only early warning reaches Python directly.
exec "$RM9_PYTHON" -u scripts/momentum/rm9.py "$RM9_PHASE" --mode "$RM9_MODE" \
  --root "$RM9_RUN" --archive "$RM9_ARCHIVE" "${opts[@]}"
