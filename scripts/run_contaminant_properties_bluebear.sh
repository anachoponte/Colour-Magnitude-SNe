#!/bin/bash
#SBATCH --job-name=cmsne-contam
#SBATCH --output=slurm_logs/contam-%j.out
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
OUT_DIR="${2:-docs/figures/purity_investigation}"

echo "Running Contaminant Properties & Secondary-Cut Investigation on $DATA_DIR..."
for TARGET in all_lensed lensed_ia lensed_sn1bc lensed_sn2l lensed_sn2n lensed_sn2p; do
    echo "=== target=$TARGET ==="
    python -u scripts/investigate_contaminant_properties.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --target "$TARGET"
done
echo "Done!"
