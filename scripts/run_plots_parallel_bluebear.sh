#!/bin/bash
#SBATCH --job-name=cmsne-plt-par
#SBATCH --output=slurm_logs/plt-par-%A_%a.out
#SBATCH --array=1-4
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

case "${SLURM_ARRAY_TASK_ID}" in
  1)
    echo "Task 1: Snapshot all_lensed (ug,gr,ri,iz,zy)"
    python -u scripts/plot_evolving_boundaries.py \
        --data-dir "$DATA_DIR" \
        --out-dir results/figures_snapshot_size_alpha \
        --target all_lensed \
        --mag-mode snapshot \
        --weight-encoding size_alpha \
        --n-sources 120 \
        --n-resample 150 \
        --also-plot-errors
    ;;
  2)
    echo "Task 2: Matched standard lensed_ia (ug,gr,ri,iz,zy)"
    python -u scripts/plot_evolving_boundaries.py \
        --data-dir "$DATA_DIR" \
        --out-dir results/figures_standard_size_alpha \
        --target lensed_ia \
        --mag-mode matched \
        --weight-encoding size_alpha \
        --n-sources 150 \
        --n-resample 150 \
        --also-plot-errors
    ;;
  3)
    echo "Task 3: Matched standard all_lensed (ug,gr,ri,iz,zy)"
    python -u scripts/plot_evolving_boundaries.py \
        --data-dir "$DATA_DIR" \
        --out-dir results/figures_standard_size_alpha \
        --target all_lensed \
        --mag-mode matched \
        --weight-encoding size_alpha \
        --n-sources 150 \
        --n-resample 150 \
        --also-plot-errors
    ;;
  4)
    echo "Task 4: Individual lensed subtypes (sn1bc, sn2l, sn2n, sn2p) for gr,ri"
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
    ;;
esac
echo "Array task ${SLURM_ARRAY_TASK_ID} complete!"
