#!/bin/bash
#SBATCH --job-name=cmsne-cmp-mag
#SBATCH --output=slurm_logs/cmp-mag-%j.out
#SBATCH --time=02:00:00
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

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "host=$(hostname)  pwd=$(pwd)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"

DATA_DIR="${1:-results/dual_59297152}"
OUT_FILE="${2:-results/comparison_snapshot_vs_peakmag.json}"

echo "Comparing Snapshot Magnitude vs Peak Magnitude on $DATA_DIR..."
python -u scripts/compare_mag_modes.py --data-dir "$DATA_DIR" --out-file "$OUT_FILE" --pairs gr,ri,iz "${@:3}"
echo "Done!"
