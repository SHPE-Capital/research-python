#!/bin/bash
# A1 staged pull on HiPerGator (CPU-only, 2 CPUs; rate-limit bound, ~11 h estimated for 500x5).
# Prereqs (once): repo + env under /blue/dream_team/$USER, a symbols file, and ~/.alpaca_env (chmod 600) with
#   export ALPACA_DATA_KEY=...   export ALPACA_DATA_SECRET=...
# Submit from the repo root:  sbatch --export=ALL,SYMBOLS=/blue/dream_team/$USER/momentum/symbols_500.txt scripts/momentum/hpg/pull_500x5.sbatch
# Resumable: resubmit the same command after a timeout or failure.
#SBATCH --job-name=mo-pull-500x5
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.log
set -eo pipefail  # no -u: conda activate reads unset vars
: "${SYMBOLS:?set SYMBOLS to a one-ticker-per-line file}"
ROOT=/blue/dream_team/$USER/momentum/store
mkdir -p "$ROOT"
module load conda
conda activate /blue/dream_team/$USER/envs/momentum
source ~/.alpaca_env
python scripts/momentum/pull_archive.py --stage 500x5 --root "$ROOT" --symbols-file "$SYMBOLS" --max-gb 60
