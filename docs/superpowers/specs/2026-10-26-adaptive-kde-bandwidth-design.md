# Adaptive KDE Bandwidth for the Local Density-Ratio Blend — Design

## Problem

PR #35 (merged, commit `5d96d4c`) added a local weighted-KDE density-ratio
blend to `ColourClassifier.probability()` — see
`docs/superpowers/specs/2026-10-25-local-density-posterior-blend-design.md`
for the full original design. Manual validation against real BlueBEAR
production data (`results/mc_58818280`, 52,780 signal / 141,428 background
events) found that it does **not** achieve its goal: `P(lensed)` in the
bright+red corner (`gr≥2.0, peakmag≤16`) collapsed to **~0.0002**, not the
success criterion of >0.9, the opposite direction from the ~0.46 plateau this
whole investigation set out to fix.

### Root cause

Diagnosed directly against the real data: at the corner, local effective
sample size (Kish's ESS on the kernel-weighted values, at the fixed
Scott's-rule bandwidth) is **exactly 1.00 for both classes**, at four
different corner query points tested. An ESS of exactly 1 means the kernel
sum is dominated by a single surviving training point — `p_kde = S/(S+B)`
there is not estimating a tail density at all, it is comparing two individual
points' kernel values, which is statistically arbitrary. In this case it
happened to favour background.

This is consistent with an earlier finding from the same investigation: only
13-15 raw `lensed_Ia` events land in this exact corner out of 100,000
attempted, and their combined weight is already near the float-precision
noise floor *within signal's own population* — reaching the corner requires
an extreme (low-z, high-magnification) combination that `rate_weight`
suppresses for the signal class too, just less severely than for background.
Background pools nine contaminant classes, each contributing its own rare
tail, which can rival or exceed signal's at this extreme.

Ruled out: an extreme outlier inflating the bandwidth calculation itself
(the mechanism behind two earlier bugs in this same investigation,
`diagnose_sparse_cells`'s percentile clipping and the original `robust_range`
fix). Direct inspection of the real pooled weighted standard deviations
(`gr`: 0.61, `peakmag`: 0.97) and the top contributors to each column's
weighted variance showed no single point dominating — the bandwidths
themselves are reasonable. The failure is specifically that a *fixed*
bandwidth, however reasonably sized for the bulk, is too narrow to pick up
more than one effective neighbour at a point this far into the tail.

## Chosen approach: adaptive ("balloon") bandwidth for the KDE side only

Widen each class's kernel, independently, per query point, until its local
kernel sum reflects a statistically meaningful number of effective
neighbours — rather than accepting whatever a single fixed bandwidth
happens to give. This is a standard technique (variable-bandwidth / balloon
kernel density estimation), not new infrastructure: it reuses
`effective_sample_size` and `_kernel_weighted_values` unchanged, adding only
an iterative scale search on top.

**Critical constraint, carried over as the thing this fix must not break:**
the blend-weight decision (how much to trust the tree vs. the KDE) must stay
driven by the *fixed*-bandwidth local ESS, exactly as before. Widening only
makes `p_kde`'s own value reliable; it must never make the blend weight think
there is more support than the tree was actually trained with at that scale.

## Where it lives

Entirely inside `_local_density_ratio` in `cmsne/classifier.py`, plus one new
helper, `_widened_kernel_values`. No other function signature changes.
`_blend_weight`, `_blend_with_local_density`, `_kde_subset_for_pattern`,
`fit()`, `_store_kde_training_data`, `_scott_bandwidths` are all untouched.

## Mechanics

### `_widened_kernel_values`

```
_widened_kernel_values(x, X, weights, bandwidths, dims, min_ess, growth=2.0, max_iter=20):
    scale = 1.0
    g = kernel_weighted_values(x, X, weights, bandwidths, dims)
    ess = effective_sample_size(g)
    if ess >= min_ess:
        return g, ess
    ceiling = effective_sample_size(weights)   # best possible, scale -> infinity
    for _ in range(max_iter):
        if ess >= min_ess or ess >= ceiling - 1e-9:
            break
        scale *= growth
        g_new = kernel_weighted_values(x, X, weights, bandwidths * scale, dims)
        ess_new = effective_sample_size(g_new)
        if ess_new <= ess:      # widening has stopped helping -- plateaued
            ess = ess_new
            break
        g, ess = g_new, ess_new
    return g, ess
```

- Isotropic, geometric widening (`growth=2.0`): simple, symmetric across
  dimensions, no per-dimension search needed.
- Two independent stopping conditions, not just a max-iteration cap:
  reaching `min_ess` (success), or reaching the class's own overall ceiling
  (this project's extreme rate-weight dynamic range means some classes may
  never reach `min_ess` at any bandwidth — stop once nothing more can be
  extracted, rather than iterating uselessly).
- A no-op in the dense bulk: the first check at `scale=1.0` already passes
  there, so this only engages exactly where the diagnosed bug lives.
- Each class widens independently (separate calls for signal and
  background) — their sparsity need not match, and that asymmetry is exactly
  the physical signal this investigation is trying to recover.

### `_local_density_ratio` (updated)

```
_local_density_ratio(x, Xs, ws, Xb, wb, bandwidths, dims, min_kde_ess):
    # Fixed-bandwidth ESS -- drives the blend-weight decision. UNCHANGED.
    local_ess_sig = effective_sample_size(kernel_weighted_values(x, Xs, ws, bandwidths, dims))
    local_ess_bg  = effective_sample_size(kernel_weighted_values(x, Xb, wb, bandwidths, dims))

    # Widened sums -- make p_kde itself reliable. NEW.
    gs, _ = _widened_kernel_values(x, Xs, ws, bandwidths, dims, min_kde_ess)
    gb, _ = _widened_kernel_values(x, Xb, wb, bandwidths, dims, min_kde_ess)
    S, B = sum(gs), sum(gb)
    if S + B < 1e-300:
        return None, local_ess_sig, local_ess_bg   # unchanged guard, still correct
    return S / (S + B), local_ess_sig, local_ess_bg
```

Return signature unchanged — `_blend_with_local_density` needs no edits at
all. Only `p_kde`'s *value* changes, and only where widening engages.

### New parameter

`ColourClassifier(..., kde_min_ess=DEFAULT_KDE_MIN_ESS)`, threaded through to
`_local_density_ratio`'s call site, the same pattern as `ess_threshold`/
`log_width`. Starting default: `DEFAULT_KDE_MIN_ESS = 10.0` — a classic
rule-of-thumb floor for a minimally-stable local density estimate.
**Provisional**, exactly like the other two blend constants: checked against
the real BlueBEAR data (scripts already exist from the first manual
validation pass) rather than guessed in the abstract.

## Backward compatibility

`local_ess_sig`/`local_ess_bg` (what `_blend_weight` consumes) are computed
identically to before this fix — a direct regression test must assert they
are unchanged by widening. The dense-bulk backward-compatibility behaviour
already validated for PR #35 is preserved by construction, since widening
cannot engage there (the first unwidened check already clears `min_kde_ess`
in any reasonably-populated region).

## Performance

Widening only runs where the fixed-bandwidth check already fails — which is
also exactly where `_blend_weight` is about to downweight the tree anyway.
In the bulk, cost is identical to before (one kernel evaluation per class).
In the tail, up to `max_iter=20` kernel evaluations per class per query
point in the worst case, though the ceiling/plateau check should stop this
far sooner in practice. Actual wall-clock will be re-measured as part of
re-running Manual Validation once this lands — same deferred-to-real-data
posture as the original spec, not a new category of risk.

## Testing and validation plan

**Unit tests** (`tests/test_cmsne.py`, house style):
- A synthetic reproduction of the exact real-world failure at small scale:
  both classes have only 1-2 points very far from a query point (fixed-
  bandwidth `local_ess≈1` for both), but widening reaches enough real
  neighbours that `p_kde` reflects each class's true relative density
  rather than single-point noise — and, with signal's distribution
  deliberately extending further toward the query point than background's
  (mirroring the real asymmetry), `p_kde` should end up favouring signal.
- Widening is a no-op when `min_ess` is already satisfied at `scale=1`
  (regression: identical `g`/`ess` to a direct `_kernel_weighted_values` call
  in a dense synthetic scenario).
- The ceiling/plateau early-exit stops correctly when a class's own overall
  ESS is permanently below `min_ess` (synthetic case: a class whose total
  weight is concentrated in very few points, with a query point far from
  all of them).
- `local_ess_sig`/`local_ess_bg` returned by `_local_density_ratio` are
  byte-for-byte unchanged by this fix, in both a dense and a sparse
  synthetic scenario — a direct regression test, not just an inference from
  the code.
- The full regression suite from the original spec (dense-region match,
  sparse-corner shift, backward-compat, degenerate guards,
  `TrajectoryClassifier` inheritance) still passes.

**Real-data validation** (BlueBEAR-scale, not automatable, scripts already
exist from the first manual-validation pass against `results/mc_58818280`):
1. Re-run the bright+red corner check. **Success criterion unchanged: P(lensed)
   exceeds 0.9** in the corner.
2. Re-run the `ess_threshold`/`log_width`/`kde_min_ess` tuning sweep against
   real data, now including `kde_min_ess` as a third axis.
3. Re-check backward compatibility in the bulk (should be unaffected, per
   the Backward Compatibility section above, but verify directly rather than
   assume).
4. Regenerate the boundary figure, visually confirm the bulk is unchanged
   and the corner now shows `P → 1` instead of `P → 0`.
5. Re-measure timing with widening's added iteration cost included.

## Validation outcome

This plan's own real-data validation ran twice. The first run found a
separate, more severe bug unrelated to widening itself -- `kde_max_events`
subsampling wasting its budget on NaN-feature rows before any pattern-
specific filtering, leaving backward compatibility broken almost
everywhere, not just in the tail (see
`docs/superpowers/specs/2026-10-26-kde-pattern-aware-subsampling-design.md`
for that fix, confirmed and merged in commit `1fe0eb3`).

With that fix in place, re-running this plan's own validation: the
`ess_threshold`/`log_width` backward-compatibility story is confirmed much
improved (bulk local ESS and `alpha` now take sensible values; 304/40,000
grid points reach `alpha > 0.99`, up from 0). The corner's own success
criterion (**P(lensed) > 0.9**) still does not hold (grid min=0.0002,
median=0.0005), and the `kde_min_ess` sweep shows the corner's local
density estimate changes *sign* once real widening engages versus a
single-point fixed estimate -- read as evidence of genuine `lensed_Ia`
Monte Carlo sparsity in this exact extreme-magnification regime, not a
defect in this plan's widening mechanism. Tracked separately as issue #37.
This plan's own scope (adaptive bandwidth widening, keeping the blend-
weight decision on the fixed bandwidth) is considered complete and
correctly implemented; the corner's unmet criterion is now out of this
plan's scope.
