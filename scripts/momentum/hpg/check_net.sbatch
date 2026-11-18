#!/bin/bash
# Does a compute node reach Alpaca's data API? Sends NO credentials: a 401/403 means reachable.
#SBATCH --job-name=mo-netcheck
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=0:05:00
#SBATCH --output=logs/%x_%j.log
set -eo pipefail
hostname
for url in https://data.alpaca.markets/v2/stocks/bars https://paper-api.alpaca.markets/v2/clock; do
  printf '%s -> ' "$url"
  curl -s -o /dev/null -m 20 -w '%{http_code}\n' "$url" || echo "UNREACHABLE"
done
