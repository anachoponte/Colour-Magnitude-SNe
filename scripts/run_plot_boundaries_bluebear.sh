#!/bin/bash
# Boundary plotting run on BlueBEAR (BEAR-apps module + cmsne-icelake venv).
#
#   sbatch scripts/run_plot_boundaries_bluebear.sh results/dual_<jobid> [results/figures_dir] [extra args...]
#
#SBATCH --job-name=cmsne-plot-boundaries
#SBATCH --output=slurm_logs/plot-boundaries-%j.out
#SBATCH --time=04:00:00
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

DATA_DIR="${1:?Usage: sbatch scripts/run_plot_boundaries_bluebear.sh <data-dir> [out-dir]}"
OUT_DIR="${2:-results/figures_$(basename "$DATA_DIR")_band_matched}"

mkdir -p "$OUT_DIR"
echo "Plotting evolving boundaries from $DATA_DIR into $OUT_DIR..."
python scripts/plot_evolving_boundaries.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --target lensed_ia --mag-mode matched "${@:3}"
python scripts/plot_evolving_boundaries.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --target all_lensed --mag-mode matched "${@:3}"
echo "Done! Figures written to $OUT_DIR"
