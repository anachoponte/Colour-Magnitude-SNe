#!/bin/bash
# 15x brute-force production run: enough background ESS to measure a 0.1% FPR
# to ~20% Poisson precision.
#
#   sbatch --array=0-9 scripts/run_brute15_bluebear.sh
#
# WHY 15x.  The measurable-FPR floor is MIN_SAMPLE_FOR_STATS / ESS_background.
# The dual_59297152 baseline gave calibration-split background ESS = 1856, so a
# floor of 10/1856 = 0.54% -- which is exactly where the clean operating points
# stopped. ESS scales linearly with N at fixed proposal, so:
#
#     events at threshold   Poisson err   ESS needed (0.1% FPR)   x baseline
#            10                32%               10,000              5.4x
#            25                20%               25,000             13.5x
#           100                10%              100,000             54x
#
# 15x (rounded up from 13.5x) targets ~25 effective background events at a 0.1%
# FPR cut: ~20% Poisson error, so the leading digit of the quoted FPR is real.
#
# PRECEDENT.  The strict end of this literature operates at ~10 events and does
# not quote an error bar at all: the Strong Lens Finding Challenge (Metcalf
# et al. 2019, A&A 625 A119) uses TPR_10 -- the TPR at fewer than TEN false
# positives in 100,000 -- as its headline metric, and HOLISMOKES XXI Part II
# (lensed SNe Ia, LSST-like) quotes its TPR at an FPR of "O(10^-4)" on ~80,000
# background events, i.e. ~8 surviving events, with no stated uncertainty.
# Going to 25 events clears that precedent; going to 100 (54x) would cost 4x
# more for a number that changes no follow-up triage decision.
#
# CONFIGURATION.  Boosting stays ON at the baseline pilot_frac=0.85. The
# n_attempted weight deficit that made it wrong (commit 0489253) is fixed, and
# holding the proposal fixed is what makes the ESS-proportional-to-N scaling
# above valid. Changing the proposal and N together would confound both.
#
# COSTS (extrapolated from job 59297152: 16-29 min/class, 5.0-5.6 GB, 569 MB):
#   wall clock ~7.2 h (10-way array, all classes in parallel)
#   ~55 CPU-hours, ~8.5 GB output
# Memory is the binding constraint, not time -- the 6 h in the sibling wrappers
# was self-imposed, the icelake partitions allow 30 days.

#SBATCH --job-name=cmsne-b15
#SBATCH --output=slurm_logs/b15-%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
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

# NOTE: results/dual_59297152 was generated with baseline_v5.3.2_10yrs.db, which
# NO LONGER EXISTS on disk -- only v5.3.5 is present. This run therefore uses a
# different survey cadence and its ABSOLUTE rates are not directly comparable to
# that baseline. Every class is regenerated here, so the run is internally
# consistent; the affected Finding tables get recomputed, not compared.
export CMSNE_OPSIM_DB="${CMSNE_OPSIM_DB:-/rds/projects/s/smithgp-lensed-transients/andres/opsim_data/baseline_v5.3.5_10yrs.db}"

N="${CMSNE_N:-1500000}"
OUT="results/brute15_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
mkdir -p "$OUT" slurm_logs

echo "host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
echo "db=$CMSNE_OPSIM_DB  n=$N  out=$OUT  task=${SLURM_ARRAY_TASK_ID:-serial}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    exec python -u scripts/run_dual_schema.py --class-index "${SLURM_ARRAY_TASK_ID}" \
        --n "$N" --out "$OUT" "$@"
else
    exec python -u scripts/run_dual_schema.py --n "$N" --out "$OUT" "$@"
fi
