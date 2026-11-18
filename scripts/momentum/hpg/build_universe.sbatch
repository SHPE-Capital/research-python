#!/bin/bash
# Build the 1,500-ticker pool on HiPerGator (needs ~/.alpaca_env). Output: $OUT/symbols_1500.txt
#SBATCH --job-name=mo-universe
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
set -eo pipefail
module load conda
conda activate /blue/dream_team/$USER/envs/momentum
source ~/.alpaca_env
M=/blue/dream_team/$USER/momentum
export RESEARCH_DATA_DIR=$M/data
OUT=$M/universe_1500
mkdir -p "$OUT"
cd $M/research-python
python scripts/momentum/build_interim_universe.py --config configs/universe/momentum_pool1500_v0.yaml --out "$OUT" --download --workers 2
cp "$OUT/symbols_500.txt" "$OUT/symbols_1500.txt"
grep -vc '^#' "$OUT/symbols_1500.txt"
