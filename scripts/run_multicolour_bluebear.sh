#!/bin/bash
# All-band photometry run on BlueBEAR (BEAR-apps module + cmsne-icelake venv).
#
#   sbatch --array=0-9 scripts/run_multicolour_bluebear.sh          # one class per task
#   sbatch scripts/run_multicolour_bluebear.sh --n 20000            # all classes, serial
#
#SBATCH --job-name=cmsne-mc
#SBATCH --output=slurm_logs/mc-%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint=icelake
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge
module load bluebear
module load bear-apps/2021b
module load Python/3.9.6-GCCcore-11.2.0
module load SciPy-bundle/2021.10-foss-2021b
module load astropy/5.0.4-foss-2021b
module load matplotlib/3.4.3-foss-2021b
module load scikit-learn/1.0.1-foss-2021b
source /rds/projects/s/smithgp-lensed-transients/andres/virtual-environments/cmsne-icelake/bin/activate

export CMSNE_OPSIM_DB="${CMSNE_OPSIM_DB:-/rds/projects/s/smithgp-lensed-transients/andres/opsim_data/baseline_v5.3.2_10yrs.db}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "host=$(hostname)  pwd=$(pwd)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)  db=$CMSNE_OPSIM_DB"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    OUT="results/mc_${SLURM_ARRAY_JOB_ID}"
    echo "array task ${SLURM_ARRAY_TASK_ID}"
    exec python scripts/run_multicolour.py --class-index "${SLURM_ARRAY_TASK_ID}" --out "$OUT" "$@"
else
    exec python scripts/run_multicolour.py "$@"
fi
