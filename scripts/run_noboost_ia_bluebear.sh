#!/bin/bash
#SBATCH --job-name=cmsne-nbia
#SBATCH --output=slurm_logs/nbia-%j.out
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --constraint=icelake
#SBATCH --mail-type=END,FAIL

# Clean uniform-proposal regeneration of lensed_Ia and unlensed_Ia.
#
# In results/dual_59297152 these are the ONLY two classes that flagged sparse
# cells and so took the boosted path (verified by diag_baseline_uniformity.py,
# job 62294411). 15% of their attempts were therefore non-uniform, which
# violates the assumption spike_proposal_ceiling.py makes when it derives eps
# from kept-event bin counts. The other eight classes had n_boost = 0 and are
# genuinely uniform, so only these two need re-measuring.
#
# --no-boost sets pilot_frac=1.0: every attempt is a uniform draw. That makes
# this run both (a) a clean basis for the spike's eps, and (b) the persisted
# efficiency estimate the detection-aware proposal needs, which is what lets it
# escape spending 85% of its budget on a uniform pilot.

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
# NOTE: the sibling wrappers default to baseline_v5.3.2_10yrs.db, which is the
# DB results/dual_59297152 was generated with (per its manifest.json) but which
# NO LONGER EXISTS on disk -- only v5.3.5 is present. Any regeneration therefore
# uses a different survey cadence and is NOT directly comparable to that baseline
# on absolute rates. Flagged rather than silently substituted.
export CMSNE_OPSIM_DB="${CMSNE_OPSIM_DB:-/rds/projects/s/smithgp-lensed-transients/andres/opsim_data/baseline_v5.3.5_10yrs.db}"
echo "host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)  db=$CMSNE_OPSIM_DB"

OUT="${1:-results/noboost_ia_$SLURM_JOB_ID}"
N="${2:-100000}"
mkdir -p "$OUT"

for IDX in 0 1; do     # 0 = lensed_Ia, 1 = unlensed_Ia
    echo "=== class-index $IDX ==="
    python -u scripts/run_dual_schema.py --class-index "$IDX" --n "$N" \
        --no-boost --out "$OUT"
done
echo "Done! -> $OUT"
