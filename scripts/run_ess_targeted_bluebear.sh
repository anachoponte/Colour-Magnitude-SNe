#!/bin/bash
#SBATCH --job-name=cmsne-esst
#SBATCH --output=slurm_logs/esst-%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --constraint=icelake
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL

# ESS-targeted generation: every reported target resolvable at the 0.1% FPR
# budget, with per-class budgets instead of one flat N.
#
# WHY NOT A FLAT MULTIPLE. The 15x run (results/brute15_62368493) scaled all ten
# classes equally. Measured on its ranked-candidate output, that left ESS wildly
# uneven against the MIN_SAMPLE_FOR_STATS=10 bar -- worst calib ESS over the five
# decision epochs, at the 0.1% budget:
#
#     target          signal ESS   background ESS
#     lensed_ia            73.0          17.9      <- 7x more signal than needed
#     lensed_sn1bc         10.4          10.6
#     lensed_sn2n           3.6          29.5      <- thin
#     lensed_sn2p           2.5          18.1      <- thin
#     lensed_sn2l           3.4          17.0      <- thin
#
# Background clears everywhere at 0.1%, so it needs no scaling beyond a cushion;
# the shortfall is signal, and only for three CC subtypes. A flat multiple buys
# lensed_ia margin it already has while leaving sn2p short. Budgets below scale
# each class by max(1, 13/worst_ESS) -- target 13, not 10, so the WORST epoch
# still clears the bar rather than landing exactly on it.
#
# 0.01% is deliberately NOT targeted: it would need background x4.3 and
# lensed_sn2l x6.7 (10M attempts alone, for a class with ~4 events per decade),
# taking the run to 55M. Reporting 0.1% with honest ESS and marking 0.01%
# unresolved already exceeds the field precedent (Metcalf+2019 TPR_10;
# HOLISMOKES XXI quotes O(1e-4) on ~80k background with no uncertainty at all).
#
# ESS scales linearly with N only at FIXED proposal, so boosting stays at the
# baseline pilot_frac=0.85 -- do not add --no-boost here. If boost flagging
# shifts at these larger N, re-measure before trusting the scale factors.

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

# NOTE: v5.3.2 (which results/dual_59297152 used) is gone from disk; v5.3.5 is a
# different cadence, so absolute rates here are NOT comparable to that baseline.
export CMSNE_OPSIM_DB="${CMSNE_OPSIM_DB:-/rds/projects/s/smithgp-lensed-transients/andres/opsim_data/baseline_v5.3.5_10yrs.db}"

# Indexed by --class-index, i.e. run_dual_schema.py's CLASSES order:
#   0 lensed_Ia      1 unlensed_Ia    2 lensed_sn1bc   3 lensed_sn2l
#   4 lensed_sn2n    5 lensed_sn2p    6 unlensed_sn1bc 7 unlensed_sn2l
#   8 unlensed_sn2n  9 unlensed_sn2p
N_MAP=(1500000 1840000 1880000 5740000 5420000 7800000 1840000 1840000 1840000 1840000)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
N="${N_MAP[$IDX]}"
OUT="results/esstarget_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
echo "class-index=$IDX  n=$N  out=$OUT  db=$CMSNE_OPSIM_DB"
exec python -u scripts/run_dual_schema.py --class-index "$IDX" --n "$N" --out "$OUT" "$@"
