#!/bin/bash
#SBATCH --job-name=cmsne-plots
#SBATCH --output=slurm_logs/plots-%j.out
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
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

echo "=== 1. Generating Snapshot Distinguishable + Error-Bar Plots for ALL 5 Band Pairs (ug,gr,ri,iz,zy) ==="
python -u scripts/plot_evolving_boundaries.py \
    --data-dir "$DATA_DIR" \
    --out-dir results/figures_snapshot_size_alpha \
    --target lensed_ia \
    --mag-mode snapshot \
    --weight-encoding size_alpha \
    --n-sources 120 \
    --n-resample 150 \
    --also-plot-errors

python -u scripts/plot_evolving_boundaries.py \
    --data-dir "$DATA_DIR" \
    --out-dir results/figures_snapshot_size_alpha \
    --target all_lensed \
    --mag-mode snapshot \
    --weight-encoding size_alpha \
    --n-sources 120 \
    --n-resample 150 \
    --also-plot-errors

echo "=== 2. Generating Standard Matched-Peakmag Plots (Size+Opacity + High-Contrast Palette + Error Bars) ==="
python -u scripts/plot_evolving_boundaries.py \
    --data-dir "$DATA_DIR" \
    --out-dir results/figures_standard_size_alpha \
    --target lensed_ia \
    --mag-mode matched \
    --weight-encoding size_alpha \
    --n-sources 150 \
    --n-resample 150 \
    --also-plot-errors

python -u scripts/plot_evolving_boundaries.py \
    --data-dir "$DATA_DIR" \
    --out-dir results/figures_standard_size_alpha \
    --target all_lensed \
    --mag-mode matched \
    --weight-encoding size_alpha \
    --n-sources 150 \
    --n-resample 150 \
    --also-plot-errors

echo "=== 3. Generating Individual Lensed Subtype Plots (sn1bc, sn2l, sn2n, sn2p) for gr & ri ==="
for SUB in lensed_sn1bc lensed_sn2l lensed_sn2n lensed_sn2p; do
    python -u scripts/plot_evolving_boundaries.py \
        --data-dir "$DATA_DIR" \
        --out-dir results/figures_subtypes_size_alpha \
        --target "$SUB" \
        --mag-mode snapshot \
        --pairs gr,ri \
        --weight-encoding size_alpha \
        --n-sources 120 \
        --n-resample 150 \
        --also-plot-errors
done

echo "All standard + error-bar size+opacity figures complete!"
