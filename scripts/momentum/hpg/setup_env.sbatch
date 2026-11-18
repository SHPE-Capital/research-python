#!/bin/bash
# One-time: build the conda env for the momentum pull and calibration. Run from the logs/ parent dir.
#SBATCH --job-name=mo-setup
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.log
set -eo pipefail
module load conda
ENV=/blue/dream_team/$USER/envs/momentum
SRC=/blue/dream_team/$USER/momentum/research-python
if [ ! -d "$ENV" ]; then
  conda create -y -p "$ENV" python=3.12 pip
fi
conda activate "$ENV"
pip install --no-cache-dir -e "$SRC" numba psutil pytest
python -c "import shpe_research, numba, pandas, pyarrow; print('import ok', numba.__version__)"
