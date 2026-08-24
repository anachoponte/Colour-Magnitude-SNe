#!/bin/bash
# Full colour-magnitude production run on BlueBEAR (SLURM).
#
#   sbatch scripts/run_full.sh                      # all 10 band pairs, serial (~hours)
#   sbatch scripts/run_full.sh --n-cadence 50000    # smaller/faster run
#
# To parallelise one band pair per array task (~10x wall-clock), uncomment the
# --array line below and submit the same script; each task writes its own
# <pair>.pkl into a shared results/run_<jobid> directory.
#
#SBATCH --job-name=cmsne-full
#SBATCH --output=cmsne-full-%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
##SBATCH --array=0-9        # <-- uncomment for one-pair-per-task parallel mode

set -euo pipefail

# --- Python environment -----------------------------------------------------
# Needs the cmsne deps (numpy scipy pandas matplotlib astropy scikit-learn
# sncosmo) AND opsimsummaryv2 installed. Override CONDA_BASE / CMSNE_ENV if your
# setup differs (e.g. a BEAR-apps Miniforge module instead of ~/miniconda3).
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CMSNE_ENV:-cmsne-env}"

# --- OpSim database (v5.3.5 baseline) ---------------------------------------
: "${CMSNE_OPSIM_DB:?set CMSNE_OPSIM_DB to a baseline_v5.3.5_10yrs.db before submitting}"
export CMSNE_OPSIM_DB

cd "$(dirname "$0")/.."
echo "host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)  db=$CMSNE_OPSIM_DB"

# All 10 band pairs, in the package's canonical order.
PAIRS=(g-r g-i g-z g-y r-i r-z r-y i-z i-y z-y)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Array mode: this task handles exactly one band pair, into a shared dir.
    PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
    OUT="results/run_${SLURM_ARRAY_JOB_ID}"
    echo "array task ${SLURM_ARRAY_TASK_ID} -> pair ${PAIR}"
    exec python scripts/run_full.py --pairs "$PAIR" --out "$OUT" "$@"
else
    # Serial mode: all pairs in one process.
    exec python scripts/run_full.py "$@"
fi
