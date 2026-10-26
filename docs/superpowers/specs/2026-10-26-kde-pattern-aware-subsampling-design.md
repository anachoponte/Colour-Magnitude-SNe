# KDE Pattern-Aware Subsampling — Design

## Problem

Re-running Manual Validation (after PR #36's adaptive-bandwidth fix landed)
against real BlueBEAR production data (`results/mc_58818280`, 52,780 signal
`lensed_Ia` / 141,428 background events) found that the bright+red corner's
`P(lensed)` still failed the success criterion (best achievable across a
`kde_min_ess` sweep was ≈0.12, not >0.9), and a much more severe problem
surfaced: **0 of 40,000 posterior-surface grid points had blend weight
`alpha > 0.99`** — the mechanism that is supposed to leave the dense "bulk"
untouched and trusted to the tree was engaging almost everywhere, not just
in genuinely sparse tails.

### Root cause

`ColourClassifier._store_kde_training_data` calls
`_weighted_subsample(Xs, ws, kde_max_events)` on the raw feature matrix
*before* any NaN-row filtering. On real data, **95% of signal events and
90% of background events have a NaN in `gr` or `peakmag`** (most simulated
events are not measured in both). `_weighted_subsample` draws rows
proportional to weight, blind to which columns are finite, so the
`kde_max_events=4000` cap is drawn almost entirely from rows that
`_kde_subset_for_pattern` then discards the moment it filters to the
all-finite pattern. Confirmed directly against real data:

```
_kde_Xs.shape=(4000, 2)   _kde_Xb.shape=(4000, 2)        <- the raw capped draw
finite over BOTH columns: signal 65/4000 (1.6%), background 361/4000 (9%)
```

Only 65 signal rows and 361 background rows survive to serve the
all-finite pattern — versus 2,623 and 13,993 rows that actually exist in
the full population. The local KDE at nearly every query point is working
off a population roughly 40–60x smaller than `kde_max_events` implies,
which is why local ESS (and hence the tree-vs-KDE blend weight `alpha`) is
tiny almost everywhere, not just in the tail.

This is independent of, and precedes, both PR #35 (the local density-ratio
blend) and PR #36 (adaptive bandwidth widening) — both of those work
correctly; this bug is in `_store_kde_training_data`/`_weighted_subsample`'s
original call site, which has never accounted for missing-feature rows.

## Chosen approach: filter per pattern, then cap per pattern

Stop subsampling globally at `fit()` time. Store the full (positive-weight)
population instead, and apply `_weighted_subsample`'s `kde_max_events` cap
*lazily*, once per distinct missing-data pattern, inside
`_kde_subset_for_pattern` — filtering to that pattern's finite rows first,
then capping. Each pattern spends the budget only on rows it can actually
use, instead of competing with every other pattern (including rows no
pattern can use at all) for one shared, order-dependent draw.

**This does not distort relative transient-class rates.** The
weight-proportional draw + mass-preserving rescale in `_weighted_subsample`
depends only on each row's weight value, never on which other columns
happen to be finite. Moving the NaN filter earlier changes *which pool* is
drawn from, not the mechanism that preserves each class's relative
representation within that pool. If anything, a pattern's subsample
becomes a *more* faithful rate-weighted sample (drawn from thousands of
real candidates instead of a few dozen noise-dominated survivors).

**Semantic change, explicitly accepted:** `kde_max_events` no longer bounds
the total size of the stored KDE training set — it bounds each pattern's
own subsample independently. Total memory/compute can scale with however
many distinct missing-data patterns real queries actually hit. On this
project's real data (checked directly), the number of patterns that matter
in practice is small (a handful), so this is not a practical concern; the
full default feature set (5 colours + peakmag) is used in production, so
this was confirmed to be the right scope to design for, not an
over-generalization for the narrower `keys=['gr']` validation setup used so
far in this investigation.

## Where it lives

`cmsne/classifier.py`, two functions:

- `_store_kde_training_data`: drop the `_weighted_subsample` calls; store
  `Xs`/`ws`/`Xb`/`wb` filtered only to positive weight (unconditional, as
  before). Bandwidths are now computed from this full population rather
  than a capped, NaN-contaminated sample — `_scott_bandwidths`/
  `_weighted_std` already filter NaN per-column, so this is safe and more
  statistically grounded than before, not a new risk.
- `_kde_subset_for_pattern`: after the existing finite-row filter, call
  `_weighted_subsample(Xs[finite_s], ws[finite_s], self.kde_max_events)`
  (and the background equivalent) before caching. Cache shape, callers
  (`_blend_with_local_density`, `_local_density_ratio`'s `ess_ceiling_*`
  args from the prior perf fix) are unaffected — this only changes what
  feeds the cache, not its shape or how it's consumed.

No other function signature changes. `_local_density_ratio`,
`_blend_weight`, `_widened_kernel_values`, `fit()`, `__init__` are
untouched.

## Testing

- The two existing `kde_max_events` regression tests (checking the cap
  against `_kde_Xs` directly) are rewritten to check the per-pattern result
  from `_kde_subset_for_pattern` instead, since `_kde_Xs` is no longer
  capped.
- New test reproducing the real bug's mechanism at small scale: 950/1000
  rows missing one column, 50 with both finite, `kde_max_events=10`. The
  all-finite pattern reaches its full cap of 10 (drawn only from the 50
  genuinely-usable rows), and a second, disjoint pattern reaches its own
  cap of 10 independently — demonstrating patterns no longer starve each
  other of a shared budget.

## Real-data validation

Re-run Manual Validation once this lands: re-check the bright+red corner
(success criterion unchanged, P(lensed) > 0.9), re-check that
backward-compatibility's "bulk" grid points now actually reach
`alpha > 0.99`, regenerate the boundary figure, re-measure timing (now
touching a much larger per-pattern pool before each cap, though still
bounded by `kde_max_events` same as before).

## Validation results (confirmed against real BlueBEAR data, commit `1fe0eb3`)

**This fix is confirmed.** Re-running the real-data validation
(`results/mc_58818280`) with it in place shows a direct, measurable
correction of the diagnosed bug:

- Bulk-point local ESS at the default `ess_threshold=10, log_width=0.3`
  rose from ~2–19 (broken, pre-fix) to **~5–185**. `alpha` at individual
  bulk points now takes sensible, varied values (e.g. 0.99 at
  `gr=0.5, peakmag=20`; 0.79 at `gr=1.0, peakmag=21.5`) instead of being
  uniformly near zero everywhere.
- Grid-wide backward compatibility: **304/40,000 points now reach
  `alpha > 0.99`**, up from 0/40,000 (which previously crashed the
  validation script on an empty-array reduction). The mechanism no longer
  engages indiscriminately everywhere.
- Timing overhead is acceptable: 36.8s (blended+widened) vs 17.1s
  (tree-only) on a 200x200 grid, 2.2x.

**What this fix does NOT resolve, and was never expected to:** the
bright+red corner's `P(lensed) > 0.9` success criterion (from PR #35's
original spec) still fails (grid min=0.0002, median=0.0005, max=0.9997).
At the four corner test points, fixed-bandwidth local ESS is still tiny
(`ess_sig` 1.18–5.05, `ess_bg` 1.00–1.91) even after this fix -- consistent
with this investigation's earlier finding that only ~13-15 raw `lensed_Ia`
events land in that exact box out of 100,000 attempted. This fix recovers
the *representative* sample for patterns that have one; it cannot
manufacture real events that were never simulated into an extreme tail.

More notably, sweeping `kde_min_ess` at the corner shows the local density
estimate *changes sign* once widening sees a real neighbourhood rather
than the single nearest point: at `kde_min_ess=1` (no widening), two of
the four corner points give `p_kde≈0.9999` (favouring signal); at
`kde_min_ess≥5` (real widening engages), the same points drop to
`p_kde≈0.0006–0.0036` (favouring background). This reads as a genuine
question about `lensed_Ia`'s Monte Carlo sampling density in this extreme
magnification/redshift regime relative to its true rate, not a defect in
this fix or in PR #36's widening mechanism -- tracked separately as
issue #37 (see that issue for detail). The subsampling fix in this spec is
considered complete and validated on its own terms; the corner's unmet
criterion is a separate, open question.
