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
