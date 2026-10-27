# Uncertainty-Aware Noise Injection for ColourClassifier — Design

## Problem

Investigating issue #37 (why the bright+red corner's `P(lensed)` still
fails its >0.9 success criterion even after the adaptive-bandwidth fix,
PR #36, and the KDE pattern-aware subsampling fix, commit `1fe0eb3`) found
that the properly-balanced signal/background density ratio in that region
genuinely favours background (~0.001-0.005, not a bug) — but that this
ratio is itself dominated by a single `unlensed_sn2p` draw at `z=0.0033`
with `gr=6.16`, a colour that is not physically plausible for a Type IIP
SN (plateau-phase g-r is normally ~0.5-1).

Traced to `cmsne/multicolour.py::band_photometry`: detection is a bare
"is this magnitude finite" cut at single-visit depth, with no
signal-to-noise floor. A colour built from a magnitude sitting right at
that depth, paired against a well-measured magnitude in the other band,
produces exactly this kind of inflated, physically-implausible colour —
and every point is currently treated by the classifier as an exact,
noiseless measurement regardless of how close to the detection limit it
is. One such point can single-handedly dominate a local density estimate,
exactly as observed.

`cmsne/lightcurve.py` already computes and stores `obs_mag_error`/
`obs_snr` per observation (`Lightcurve.obs_mag_error`) — this information
exists during simulation but is discarded before it reaches a colour
feature.

## Chosen approach: Monte Carlo noise-injection replication

For each event, draw `n_uncertainty_replicates` (`K`) noisy copies of its
feature vector from `N(x_i, sigma_i)` per dimension (independent across
dimensions), each carrying `weight / K` so total rate mass is preserved
exactly. Feed this expanded, smeared population into both the tree
(`HistGradientBoostingClassifier` + calibration) and the KDE
(`_store_kde_training_data`/`_kde_subset_for_pattern`) unchanged — neither
needs new training-time machinery, they just see a bigger, blurred-out
input instead of sharp point estimates. The same mechanism applies
symmetrically at query time: perturb-and-average a query event's own
replicas through the existing `probability()`/`score()` pipeline.

This was chosen over two alternatives discussed and rejected:
- **Exact analytic bandwidth convolution for KDE only** (`h_eff =
  sqrt(h_global^2 + sigma_i^2)`), with a separate noise-injection pass for
  the tree. Cheaper and deterministic for the KDE side, but requires two
  different mechanisms (and two different test suites) instead of one,
  and doesn't reuse the already-tested KDE kernel code as directly.
- **Query-time-only averaging**, with no training-side change at all.
  Cheapest, but does not address the actual issue #37 finding, which is
  about a spurious *training* point dominating a density estimate, not a
  noisy query.

## Where it lives

**New fields, `cmsne/multicolour.py`:**
- `band_photometry` additionally returns each band's magnitude error at
  the same nearest-detection index it already selects for the magnitude
  itself (no new epoch-selection logic, just also reading
  `lightcurve.obs_mag_error[nearest]`).
- `event_multicolour` gains `{colour}_err = sqrt(err_a**2 + err_b**2)`
  for every adjacent-band colour and its `_d` (evolved) counterpart,
  plus `peakmag_err` (error of whichever observation achieved the
  brightest-observed magnitude) and `peakmag_firstdet_err` (error of the
  first-detection observation).
- **`peakmag_fit_err` is explicitly out of scope.** `peakmag_fit` is a
  parabola-fit quantity, not a single observation; proper uncertainty
  propagation would need the fit's own covariance, a bigger problem than
  this plan takes on, and it isn't this project's default magnitude
  feature. A classifier configured with `magnitude_feature='peakmag_fit'`
  will have nothing to perturb that one feature with, and noise injection
  falls back to point-estimate behaviour for it alone.

**New helpers, `cmsne/classifier.py`:**
- `colour_feature_sigma_matrix(events, keys, extra)` — mirrors
  `colour_feature_matrix`, reading `f'{key}_err'`/`f'{feature}_err'` from
  each event dict into a matrix with the same column order and shape as
  the feature matrix it corresponds to. Missing/absent `_err` fields
  produce `NaN` (consistent with a missing feature value).
- `_replicate_with_uncertainty(X, sigma, weights, dims, n_replicates,
  seed=0)` — the noise-injection primitive (see Mechanics).
- `_grouped_effective_sample_size(g, event_id, n_original)` — Kish's ESS
  computed after summing each original event's replicas back together
  (see Mechanics — this is the fix for the ESS-inflation problem below).

**Two different injection points, for cost reasons** — the tree and the
KDE are deliberately NOT replicated the same way or at the same stage:
- **Tree side** replicates the full balanced `Xs`/`Xb` once, in `fit()`,
  right before the tree-training call (cost: `K ×` the full population,
  `~50k-150k` events/class — a real, bounded multiplier on `fit()`'s
  tree-training wall-clock, re-measured directly in Manual Validation
  rather than estimated here, same posture as every prior timing concern
  in this investigation).
- **KDE side replicates *after* `_kde_subset_for_pattern`'s existing
  filter-then-cap step**, applied only to the small (`kde_max_events`-
  sized) set of *original* events that survive it — not before. Doing it
  before would reintroduce a milder version of the bug fixed in commit
  `1fe0eb3`: the `kde_max_events` budget would be spent on redundant
  replicas of the same handful of original events instead of genuinely
  diverse ones. This keeps KDE-side cost bounded and predictable
  (`kde_max_events × K` rows per pattern, not
  `(raw population size) × K`).
- **Query time:** `probability()`/`score()` expand each query event into
  its own `K` replicas (same helper), run the *unmodified* existing
  pipeline on the expanded batch, then average the per-replica results
  back to one value per original query event.

## Mechanics

### `_replicate_with_uncertainty`

```
_replicate_with_uncertainty(X, sigma, weights, dims, n_replicates, seed=0):
    # Auto-skip: no real uncertainty data -> zero extra compute, exact
    # backward compatibility with every existing fixture/population.
    if n_replicates <= 1 or not np.any(np.isfinite(sigma[:, dims]) & (sigma[:, dims] > 0)):
        return X, weights, np.arange(len(X))

    rng = np.random.default_rng(seed)
    n = len(X)
    X_rep = np.repeat(X, n_replicates, axis=0)
    event_id = np.repeat(np.arange(n), n_replicates)
    sigma_rep = np.repeat(sigma, n_replicates, axis=0)
    for d in dims:
        perturbable = np.isfinite(sigma_rep[:, d]) & (sigma_rep[:, d] > 0)
        X_rep[perturbable, d] += rng.normal(0.0, sigma_rep[perturbable, d])
    w_rep = np.repeat(weights, n_replicates) / n_replicates
    return X_rep, w_rep, event_id
```

- Per-dimension, per-row: a finite, positive `sigma` perturbs; `NaN`/
  non-positive `sigma` leaves that dimension exactly as-is for every
  replica of that row (consistent with how a missing feature is already
  handled elsewhere — nothing to perturb with means no perturbation,
  not an error).
- `weights / n_replicates` makes `sum(w_rep[event_id == i]) ==
  weights[i]` exactly, for every original event `i` — this is what keeps
  relative transient-class rate representation untouched (same invariant
  established for the subsampling fix: the mechanism depends only on
  each row's own weight, never on anything else about it).

### The ESS-inflation problem, and `_grouped_effective_sample_size`

Kish's ESS is `(sum(w))**2 / sum(w**2)`. Naively calling
`effective_sample_size` on `K` replicas of one real observation (each at
weight `w/K`) gives `(w)**2 / (K * (w/K)**2) = K` — the formula cannot
distinguish `K` correlated copies of one uncertain measurement from `K`
independent ones. Left uncorrected, this would silently inflate
`local_ess_sig`/`local_ess_bg` and every ESS check inside
`_widened_kernel_values` by whatever `K` happens to be, re-introducing
exactly the kind of fragile, incidental-choice-sensitive behaviour this
investigation has already spent two rounds removing.

```
_grouped_effective_sample_size(g, event_id, n_original):
    g_grouped = np.bincount(event_id, weights=g, minlength=n_original)
    return effective_sample_size(g_grouped)
```

- Applies wherever ESS is computed **on kernel-evaluated values**:
  `local_ess_sig`/`local_ess_bg` in `_local_density_ratio`, and the
  scale-dependent ESS checks inside `_widened_kernel_values`.
- Does **not** apply to the `S`/`B` probability-ratio sums themselves — a
  flat sum over replicas already equals a sum over grouped-then-summed
  values (addition is associative), so no grouping is needed there.
- Does **not** apply to the per-class widen/skip "ceiling" check
  (`effective_sample_size` of raw training weights) either — since each
  event's replica weights sum back to its exact original weight, that
  ceiling is identical whether computed on the replicated or the
  original, un-replicated weight array, so it can simply use the
  original array directly and skip replication-awareness entirely.

### Carrying `event_id` through the KDE pattern cache

`_kde_subset_for_pattern`'s cache currently returns a 6-tuple (`Xs, ws,
Xb, wb, ess_ceiling_sig, ess_ceiling_bg`, added in commit `1fe0eb3`).
Rather than grow this into an awkward, ever-longer positional tuple, this
plan replaces it with a small, explicitly-named container (e.g. a
`NamedTuple`) carrying `Xs, ws, Xb, wb, ess_ceiling_sig, ess_ceiling_bg,
event_id_s, event_id_b, n_orig_s, n_orig_b` — a deliberate, in-scope
cleanup of a structure that was already starting to strain, consistent
with improving code this plan is touching anyway rather than letting the
positional-tuple pattern keep growing. `_local_density_ratio` and
`_widened_kernel_values` gain the corresponding `event_id`/`n_original`
parameters needed to call `_grouped_effective_sample_size` instead of
`effective_sample_size` directly.

### Query-time averaging

`probability()` (and `score()`, for consistency — both ultimately
evaluate the same tree/KDE machinery on query features): expand every
query event into its own `K` replicas via `_replicate_with_uncertainty`
(using that event's own `*_err` fields), run the existing, *unmodified*
per-event pipeline on the expanded batch, then average each original
event's `K` results back to one value — a batch-expand-then-group-average
wrapper around the existing method bodies, not a change to what those
bodies compute. `rank()` calls `score()` internally and needs no separate
change to inherit this. Likewise `TrajectoryClassifier` (PR #14/#15),
which subclasses `ColourClassifier` and already inherits the density
blend and adaptive widening "with no extra code" per its existing test —
the same inheritance applies here automatically.

## New parameter

`ColourClassifier(..., n_uncertainty_replicates=DEFAULT_N_UNCERTAINTY_REPLICATES)`.
No separate on/off flag is needed: `_replicate_with_uncertainty` already
auto-skips (zero extra compute) whenever no real `sigma` data is present
for the relevant dimensions — which is true of every existing test
fixture and every already-generated population, so the entire existing
PR #35/#36/subsampling-fix test suite is expected to pass completely
unchanged. Starting default: **provisional**, like every other blend
constant in this investigation (`ess_threshold`, `log_width`,
`kde_min_ess`) — checked against the real, regenerated BlueBEAR data as
part of this plan's own Manual Validation step, not guessed in the
abstract.

## Backward compatibility

Covered above: the auto-skip path makes this a strict no-op (identical
output, no extra compute) for any population or test fixture lacking
`*_err` fields, which includes every existing test and every
already-generated population. A direct regression test asserts the full
pre-existing suite passes unchanged.

## Performance

Two different cost profiles, both deferred to be re-measured directly in
Manual Validation (same posture as every prior performance question in
this investigation):

- **Tree training**: `K ×` the full population size feeding
  `HistGradientBoostingClassifier`/`CalibratedClassifierCV` — a real,
  potentially significant multiplier on `fit()`'s wall-clock (Manual
  Validation's own tree-only baseline was ~17s per fit+score on a 200x200
  grid; this will be re-measured with replication included, not
  estimated here).
- **KDE evaluation**: bounded by `kde_max_events × K` rows per pattern
  (not the raw population size), since replication happens *after* the
  per-pattern cap, reusing the exact budget the subsampling fix already
  established.
- **Combined overhead**: by the time this lands, the posterior surface
  carries three stacked costs relative to the original tree-only
  baseline (density blend, adaptive widening, noise injection). Manual
  Validation reports the *total* combined overhead, not just this
  increment, since what matters for real use (e.g. `rank()`-ing a
  candidate list) is the full cost, not any one piece of it.
- **`probability()`/`rank()` query-side cost scales ~K², not K**: each
  query event is replicated into `K` query-replicas
  (`_blend_with_local_density`), and each of those `K` replicas is then
  evaluated against a KDE training subset that is *itself* `K`-times
  larger (`_kde_subset_for_pattern`'s own replication) — `K` query
  replicas × a `K`-times-larger kernel sum each, not `K` of either alone.
  Measured ~30x slower at the default `K=8` on a 9k-event/100-query
  workload (second-pass review finding, 2026-10-27) — a real cost worth
  knowing before `rank()`-ing a large real candidate list, not a defect
  in the design (both replication points are individually necessary; see
  Mechanics above).

**A subtler statistical effect, not a bug:** projecting independently-
perturbed cumulative-minimum columns (trajectory `peakmag{suffix}`,
see `TrajectoryClassifier.monotonic_column_groups`) back onto the
monotonic manifold via `np.minimum.accumulate` introduces a mild
systematic *downward* bias on later-grid-step `peakmag` replicas
relative to the unperturbed value — the minimum of several independent
noisy draws skews lower (brighter) than any single draw. This is the
statistically *correct* behaviour (it is exactly what would happen if
every underlying epoch's magnitude were perturbed and the running
minimum recomputed from scratch), not an artefact of this
implementation's shortcut of perturbing the already-derived cumulative
column directly — but it is real and non-obvious enough to flag here
(second-pass review finding, 2026-10-27).

## Testing and validation plan

**Unit tests** (`tests/test_cmsne.py`, house style):
- `_replicate_with_uncertainty` is a strict no-op (identical `X`/
  `weights`, no RNG draws) when `sigma` is entirely non-finite/non-
  positive over the relevant `dims`.
- Weight preservation: for every original event, the sum of its
  replicas' weights equals its original weight exactly (direct check,
  not inferred).
- Perturbation: with a large, finite `sigma`, replicas of the same row
  measurably scatter around the original value (seeded for determinism).
- The ESS-inflation regression: one event with large `sigma`, `K`
  replicas -- `_grouped_effective_sample_size` returns ≈1, not ≈K (a
  direct test that the specific bug described above is absent, not just
  an inference from the code).
- A synthetic reproduction of the issue #37 finding at small scale: one
  high-`sigma` background point dominating a local density estimate
  *without* noise injection (mirroring the real `gr=6.16` event), shown
  to smear out appropriately and shift the ratio *with* it.
- Full backward-compat run: the complete pre-existing test suite
  (PR #35, #36, the subsampling fix) passes unchanged.
- A sanity check that `HistGradientBoostingClassifier` still produces
  calibrated, sane output when trained on a replicated/perturbed input
  (not a deep ML-quality test — standard sklearn machinery fed a
  different-shaped input).

**Real-data validation** (BlueBEAR-scale, in scope for this plan, not
deferred — unlike every prior spec in this investigation):
1. Regenerate the Monte Carlo population at the same scale as
   `results/mc_58818280` (10 classes, ~50k-150k events each) using the
   updated `event_multicolour`, so real `*_err` fields exist. This is the
   single most expensive step in this entire plan -- likely a multi-hour
   BlueBEAR job, matching the scale of prior full-population generation
   runs in project memory.
2. Re-check the specific smoking-gun mechanism: does the regenerated
   population's analogous extreme-colour event (if one recurs) now get
   appropriately smeared rather than dominating a local density estimate
   outright?
3. Re-run the full Manual Validation sequence (corner check, bulk
   backward-compatibility, boundary figure, timing), now sweeping
   `n_uncertainty_replicates` as a fourth tuning axis alongside
   `ess_threshold`/`log_width`/`kde_min_ess`.
4. Re-measure total combined timing overhead (see Performance).

## Validation results (confirmed against real BlueBEAR data, job `59044740`)

**Implementation is confirmed correct; it does not rescue the bright+red
corner, and the reason is now understood.** All unit tests pass (305/305,
see `tests/test_cmsne.py`'s own synthetic issue #37 reproduction), and the
mechanism is proven present and working on the real regenerated population
(`results/mc_59044740`, 10 classes, 52,780-141,428 events/class, same
`--n 100000` scale as `mc_58818280`). Every finite colour/magnitude in the
regenerated population carries a finite, positive `*_err` (spot-checked
directly: 100% coverage across all 10 classes).

**The literal original smoking-gun event is still there, and still barely
moved.** The exact `unlensed_sn2p` draw this investigation's "Problem"
section was built on recurs in the regenerated population (`gr=6.160`,
`peakmag=13.5`, `z=0.00330`, same identity as the original finding). Its
real, propagated `gr_err` is **0.0073** — only ~5% of the `gr` KDE
bandwidth (`0.1382`, Scott's rule). Noise-injecting it with 500 replicas at
its own real `σ` changes its kernel contribution at its own location by
**0.13%** (ratio 0.9987). The hand-picked synthetic unit test used `σ=5.0`
(36× the bandwidth) specifically to demonstrate the mechanism clearly; the
real LSST single-visit photometric error model near the detection limit
never produces an uncertainty anywhere near that large, because a colour's
error is the quadrature sum of two per-band errors and LSST single-visit
magnitude errors are O(0.01-0.3 mag), not unbounded as a measurement
approaches the depth cut. The original hypothesis — "a point at the
detection limit is treated as noiseless when it should carry large
uncertainty" — is correct in direction but the real uncertainty magnitude
turns out to be far too small, relative to the KDE's own bandwidth, to
matter.

**The actual corner-dominating mechanism is different from the original
hypothesis, at the specific grid points tested.** Direct 2D kernel-distance
inspection at the corner query `(gr=2.0, peakmag=16.0)` found the nearest
real background training point (`gr=2.084`, `peakmag=16.462`,
`gr_err=0.0283`) carries a weight (`1.999e-05`) roughly **5 orders of
magnitude larger** than its neighbors (`~1e-13`-`1e-14`) — an
importance-sampling-boosted rare-cell draw, not a near-detection-limit
measurement. Its own `gr_err/bandwidth` ratio (`0.20`) is larger than the
`gr=6.16` event's but still modest; a single overwhelmingly-weighted point
survives noise injection's smearing by design (replication redistributes
*where* its mass sits locally, not *how much* total mass it carries). This
is a distinct statistical fragility — sample-weight concentration from
importance-sampling boosting — that measurement-uncertainty smearing was
never intended to address.

**Four-axis sweep (`ess_threshold`/`log_width`/`kde_min_ess`/
`n_uncertainty_replicates`) on the real data:**
- `n_uncertainty_replicates` swept at `[1, 4, 8, 16, 32]` (1 = explicit
  no-injection baseline) at the four corner test points: `p_blended` moves
  by at most ~0.001 absolute across the entire sweep (e.g. `gr=2.0,
  peakmag=16.0`: 0.0031 → 0.0031 → 0.0031 → 0.0030 → 0.0031 for
  replicates 1/4/8/16/32) — consistent with the small real `σ`/bandwidth
  ratios found above. The other three axes reproduce the SAME qualitative
  behaviour already characterised in the adaptive-bandwidth and
  subsampling-fix investigations (widening engages and can flip the
  corner's sign once `kde_min_ess` crosses ~5; `ess_threshold`/`log_width`
  control how much weight the blend gives the KDE side).
- **Bright+red corner (`P(lensed) > 0.9`) re-check: FAILS**, exactly as
  before noise injection — `min=0.0002, median=0.0005, max=0.9997` over
  3000 grid points, matching the pre-noise-injection subsampling-fix
  validation's own corner numbers to the precision reported there
  (`docs/superpowers/specs/2026-10-26-kde-pattern-aware-subsampling-design.md`;
  see "Provenance" below for why this is "matching," not "byte-identical").
  This failure was already understood to be driven by genuine `lensed_Ia`
  Monte Carlo sparsity in that extreme region (~13-15 raw events out of
  100,000 attempted, per that investigation) — not something noise
  injection, which only smears *measurement* uncertainty, could ever have
  been expected to fix.
- **Bulk backward-compatibility**: 427/40,000 grid points reach
  `alpha > 0.99` (up from 304/40,000 in the subsampling-fix validation on
  the OLD population — not a regression, just a different random
  population draw at larger scale); `max|blended-tree|` in that bulk
  region is 0.0031, `mean` 0.0014 — both still effectively zero, confirming
  noise injection is a true no-op in the dense bulk exactly as designed.
  34,868/40,000 points have `alpha<0.5` (mechanism actively engaging) —
  this population's signal/background overlap is evidently sparser almost
  everywhere outside the dense core than the earlier, smaller `mc_58818280`
  population; not investigated further here as it is outside this plan's
  scope (noise injection itself, not the blend-weight tuning).
- **Boundary figure**: `results/boundary_gr_noise_injection.png` written
  (gr-band heatmap, same style as the adaptive-bandwidth investigation's
  own figure).
- **Timing** (200×200 grid): tree-only baseline 96.6s; blend+widening with
  `n_uncertainty_replicates=1` (no injection) 44.2s; blend+widening+
  injection at the default `n_uncertainty_replicates=8` 215.1s (2.2× the
  tree-only baseline, 4.9× the no-injection blend). The no-injection blend
  case being *faster* than the tree-only baseline is unexplained by this
  investigation (plausibly `CalibratedClassifierCV`'s isotonic-fit
  variance across runs) and not pursued further — the number that matters
  for real use, the full combined overhead, is the 2.2× figure.

**Provenance: which commit these numbers came from, and what changed
underneath them since.** The numbers above were first committed at
`febe808`. Five behaviour-changing commits landed in `fit()` AFTER
`febe808` (all part of the same whole-branch review cycle, before this
plan merged):

- `0771971` — batch-order-independent replication (content-seeded RNG;
  unrelated to the numbers above) + a first attempt at controlling
  `HistGradientBoostingClassifier`'s `early_stopping` (disable it
  whenever replication occurs). Did NOT touch `min_samples_leaf` at all.
- `34a8b13` — the `self.threshold` calibration-basis fix (unrelated to
  the numbers above; affects `predict()`'s operating point, not
  `probability()`/`score()`'s values).
- `3db02d0` — the trajectory monotonicity + shared-offset-grouping fix
  (see Mechanics above). A true no-op for the static colour-magnitude
  schema used throughout this validation (every group is a singleton, no
  `monotonic_groups` are ever set), so it changed `fit()`'s code path but
  not its behaviour for the numbers reported here.
- `d29558a` — revised the `early_stopping` fix to decide from the
  *original* (pre-replication) row count rather than "did replication
  happen", so `n_uncertainty_replicates=1` and `>1` get the same
  `early_stopping` *flag* for the same data. Also where `min_samples_leaf`
  scaling (`20 * K`) was FIRST introduced.
- `7298fb9` — found `d29558a`'s flag-matching insufficient:
  `HistGradientBoostingClassifier`'s internal train/validation split has
  no concept of "replica siblings", so at `K>1` near-duplicate replicas
  can land on both sides of that split regardless of the flag, letting
  validation loss track training loss almost exactly and silently
  defeating the stop-on-plateau criterion — `n_iter_` can diverge toward
  `max_iter` at `K>1` even with `early_stopping=True` on both sides,
  while `K=1` (no replication, no such leakage) genuinely stops early.
  Fixed by precomputing `n_iter_` from a one-time reference fit on the
  *original* population, then fitting the replicated data with
  `early_stopping` explicitly off and `max_iter` pinned to that exact
  value — verified on a GENUINELY discriminating fixture (one that first
  needed correcting itself: an earlier, overly-separable fixture passed
  under BOTH the broken and the fixed code identically, so it could not
  tell them apart; the corrected fixture proves discrimination directly,
  by reconstructing the old flag-only logic on the same replicated data
  and confirming it diverges before checking the actual fix does not).
  Known, documented limitation under `calibrate=True` (the default):
  `CalibratedClassifierCV`'s own cv folds each train on a different
  subset, so `K=1` has no single canonical `n_iter_` to match in the
  first place — the exact-match guarantee is specific to
  `calibrate=False`; see the `KNOWN LIMITATION` comment in `fit()`.

**What this means for the numbers above.** At `febe808` (and `d29558a`),
`fit()` had no `min_samples_leaf` control at all (sklearn's own default,
20, applied to the *replicated* row count), and the `n_iter_`-divergence
mechanism `7298fb9` fixes was present in principle at every `K>1` in the
sweep — not as an asymmetry introduced later against `K=1` (an earlier,
inaccurate framing in this plan's own PR description said exactly that;
it has been corrected), but as an issue that could in principle have
affected every `K>1` config in the originally-reported sweep equally,
since `K=1` alone never replicates and was never exposed to it.
**Empirically, on this real population, it had no observable effect**:
re-running the sweep with `7298fb9`'s actual fix (job `59203583`) shows
`n_iter_=300` (i.e. `early_stopping` never actually engages) at *every*
swept `K` — `1, 4, 8, 16, 32` alike — so neither the originally-reported
numbers nor the fixed ones were ever shaped by early-stopping divergence
on this specific population/feature combination; the mechanism exists
and is now closed, but this dataset never exercised it. `min_samples_leaf`
*does* scale with `K` as designed (`20, 80, 160, 320, 640` at
`K=1,4,8,16,32`) — a real, intentional capacity control, not a bug, and
not previously documented in this section.

**Re-validated numbers (job `59203583`, commit `7298fb9`), compared
against the originally-reported ones (job `59049686`, commit `febe808`):**
corner `min=0.000206` vs `0.0002`, `median=0.000478` vs `0.0005` — matching
at the ~4-decimal precision either run can be expected to reproduce at
(not byte-identical: everything here is deterministic at fixed seeds, but
the surrounding code is not the same code any more -- content-seeded RNG
reseeding (`0771971`), `min_samples_leaf` going from sklearn's default to
`20 * K` (`d29558a`), and `early_stopping` going from on to off with
`max_iter` pinned (`7298fb9`) all change WHICH rows get perturbed how and
WHICH rows HGB holds back for its own internal validation, which is
sufficient on its own to move a result by a few units in the 4th decimal
even though nothing about this specific re-run was left to chance). The
`n_uncertainty_replicates`
sweep values likewise match to within ~0.0001 absolute at every `K`
tested. Timing came back at `224.5s` for this re-run (vs `215.1s`
originally, vs `274.9s` on an intermediate, now-superseded re-check under
`d29558a`'s flag-only fix) — all three in the same ~215-275s band,
consistent with ordinary BlueBEAR load variance across separate,
untimed, single runs rather than a reproducible trend; not pursued
further, same posture as the no-injection-blend-faster-than-tree-only
anomaly already noted above. **Conclusion stands, now on accurate
footing: no correction to the headline validation result is needed.**

**Conclusion.** The feature is implemented and tested correctly (unit and
integration), is a true no-op wherever no uncertainty data exists, and does
measurably smear the few real training points whose propagated uncertainty
is large relative to the KDE bandwidth. On this real LSST-cadence
population, however, those uncertainties are uniformly too small relative
to the bandwidth to move the bright+red corner's headline number, and the
corner's actual dominant failure mode at the specific grid points tested is
importance-sampling weight concentration, not measurement noise. Issue #37
remains open for a different reason than originally hypothesised — its
resolution (if pursued) likely lies in rate-boosting/subsampling of
extreme-weight draws rather than in measurement-uncertainty modelling.
This plan's own scope (uncertainty-aware noise injection) is complete and
validated on its own terms.
