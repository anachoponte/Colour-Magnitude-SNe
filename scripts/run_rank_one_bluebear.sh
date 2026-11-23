#!/bin/bash
#SBATCH --job-name=cmsne-rank1
#SBATCH --output=slurm_logs/rank1-%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --constraint=icelake
#SBATCH --mail-type=END,FAIL

# One ranked-candidate target per job, so each gets its own memory allocation.
#
# WHY THIS EXISTS: run_rank_candidates_bluebear.sh loops all six targets in a
# single job, which was fine on the 100k-per-class baseline but not on the 15x
# brute-force data. Measured on job 62740406 (results/brute15_62368493):
# target=lensed_ia alone peaked at 60 GB RSS (488k signal + 499k background
# events held in memory), i.e. 94% of a 64 GB request. all_lensed pools ALL
# FIVE lensed classes as signal (TARGET_SIGNAL_CLASSES in
# investigate_contaminant_properties.py) with no subsampling -- 7.45 GB of
# pickles vs 2.6 GB for lensed_Ia -- so it needs roughly 2x that peak and
# would be OOM-killed inside the shared 64 GB job. Splitting also lets the
# six targets run concurrently instead of serially at ~3h each.
#
# Override --mem on the sbatch line for the big targets:
#   sbatch --mem=240G scripts/run_rank_one_bluebear.sh all_lensed <DATA> <OUT>

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

TARGET="${1:?usage: run_rank_one_bluebear.sh TARGET [DATA_DIR] [OUT_DIR]}"
DATA_DIR="${2:-results/brute15_62368493}"
OUT_DIR="${3:-docs/figures/purity_investigation/brute15}"
mkdir -p "$OUT_DIR"

echo "host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
echo "target=$TARGET  data=$DATA_DIR  out=$OUT_DIR  mem=${SLURM_MEM_PER_NODE:-?}M"
exec python -u scripts/rank_candidates.py \
    --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --target "$TARGET"
