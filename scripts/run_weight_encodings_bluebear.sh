#!/bin/bash
#SBATCH --job-name=cmsne-weights
#SBATCH --output=slurm_logs/weights-%j.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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

echo "1. Generating weight encoding comparison figures (size, colour, alpha, size_alpha)..."
python -u scripts/plot_weight_encodings.py \
    --data-dir "$DATA_DIR" \
    --out-dir results/figures_weight_encodings \
    --pair gr \
    --mag-mode snapshot \
    --n-sources 150

echo "2. Re-running lensed_Ia evolving boundaries (snapshot, distinguishable) without lensed CC background contamination..."
python -u scripts/plot_evolving_boundaries.py \
    --data-dir "$DATA_DIR" \
    --out-dir results/figures_snapshot_distinguishable \
    --target lensed_ia \
    --mag-mode snapshot \
    --pairs gr \
    --n-sources 150 \
    --n-resample 150

echo "3. Re-running lensed_Ia purity & sweet-spot investigation without lensed CC background contamination..."
python -u scripts/investigate_purity_tradeoff.py \
    --data-dir "$DATA_DIR" \
    --out-dir results/figures_purity_investigation \
    --target lensed_ia

echo "All tasks complete!"
