#!/bin/bash
# Full colour-magnitude production run on BlueBEAR (BEAR-apps module + venv).
#
#   sbatch scripts/run_full_bluebear.sh                    # all 10 pairs, serial
#   sbatch scripts/run_full_bluebear.sh --n-cadence 50000  # smaller run
#
# For one-pair-per-task parallelism (~10x wall-clock), uncomment the --array line;
# each task writes its own <pair>.pkl into a shared results/run_<jobid> dir.
#
#SBATCH --job-name=cmsne-full
#SBATCH --output=cmsne-full-%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint=icelake        # match the venv architecture (built on icelake)
#SBATCH --mail-type=END,FAIL
##SBATCH --array=0-9

set -euo pipefail

# --- BEAR-apps 2021b stack (numpy/scipy/pandas/astropy/matplotlib/sklearn) -----
module purge
module load bluebear
module load bear-apps/2021b
module load Python/3.9.6-GCCcore-11.2.0
module load SciPy-bundle/2021.10-foss-2021b
module load astropy/5.0.4-foss-2021b
module load matplotlib/3.4.3-foss-2021b
module load scikit-learn/1.0.1-foss-2021b

# --- project venv (sncosmo / opsimsummaryv2 / healpy / iminuit / pyarrow) -------
source /rds/projects/s/smithgp-lensed-transients/andres/virtual-environments/cmsne-icelake/bin/activate

# --- OpSim database (v5.3.2, the baseline the original run used) ----------------
export CMSNE_OPSIM_DB="${CMSNE_OPSIM_DB:-/rds/projects/s/smithgp-lensed-transients/andres/opsim_data/baseline_v5.3.2_10yrs.db}"

cd "$(dirname "$0")/.."
echo "host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)  db=$CMSNE_OPSIM_DB"

PAIRS=(g-r g-i g-z g-y r-i r-z r-y i-z i-y z-y)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
    OUT="results/run_${SLURM_ARRAY_JOB_ID}"
    echo "array task ${SLURM_ARRAY_TASK_ID} -> pair ${PAIR}"
    exec python scripts/run_full.py --pairs "$PAIR" --out "$OUT" "$@"
else
    exec python scripts/run_full.py "$@"
fi
