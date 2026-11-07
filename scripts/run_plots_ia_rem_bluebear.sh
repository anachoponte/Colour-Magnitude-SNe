#!/bin/bash
#SBATCH --job-name=cmsne-plt-ia
#SBATCH --output=slurm_logs/plt-ia-%j.out
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
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
DATA_DIR="${1:-results/dual_59297152}"

python -u scripts/plot_evolving_boundaries.py \
    --data-dir "$DATA_DIR" \
    --out-dir results/figures_snapshot_size_alpha \
    --target lensed_ia \
    --mag-mode snapshot \
    --pairs ri,iz,zy \
    --weight-encoding size_alpha \
    --n-sources 120 \
    --n-resample 150 \
    --also-plot-errors
echo "Done ri,iz,zy!"
