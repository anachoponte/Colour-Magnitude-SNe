#!/bin/bash
# Dual-schema run on BlueBEAR (bare Python module + cmsne-icelake-modern venv).
#
# Uses cmsne-icelake-modern, not cmsne-icelake: that venv's own pip-installed
# packages (modern scikit-learn/numpy/scipy/sncosmo) are shadowed at runtime
# if any of the SciPy-bundle/astropy/matplotlib/scikit-learn BEAR modules are
# loaded, because `module load` prepends their site-packages to PYTHONPATH
# ahead of the venv's own -- so only the bare Python module is loaded here.
#
#   sbatch --array=0-9 scripts/run_dual_schema_bluebear.sh          # one class per task
#   sbatch scripts/run_dual_schema_bluebear.sh --n 20000            # all classes, serial
#
#SBATCH --job-name=cmsne-dual
#SBATCH --output=slurm_logs/dual-%A_%a.out
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
source /rds/projects/s/smithgp-lensed-transients/andres/virtual-environments/cmsne-icelake-modern/bin/activate

export CMSNE_OPSIM_DB="${CMSNE_OPSIM_DB:-/rds/projects/s/smithgp-lensed-transients/andres/opsim_data/baseline_v5.3.2_10yrs.db}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "host=$(hostname)  pwd=$(pwd)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)  db=$CMSNE_OPSIM_DB"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    OUT="results/dual_${SLURM_ARRAY_JOB_ID}"
    echo "array task ${SLURM_ARRAY_TASK_ID}"
    exec python scripts/run_dual_schema.py --class-index "${SLURM_ARRAY_TASK_ID}" --out "$OUT" "$@"
else
    exec python scripts/run_dual_schema.py "$@"
fi
