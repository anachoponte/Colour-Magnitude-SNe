# Local Density-Ratio Posterior Blend — Design

## Problem

`ColourClassifier` (`cmsne/classifier.py`) computes `P(lensed)` via a
`HistGradientBoostingClassifier` wrapped in isotonic `CalibratedClassifierCV`
(`method='isotonic', cv=3`), used directly by `cmsne.contours.posterior_surface`
and inherited unchanged by `TrajectoryClassifier`. In sparse/extreme regions of
colour-magnitude space — e.g. very bright **and** very red — that are physically
near-impossible for unlensed events but merely rare for lensed events, the fitted
posterior does not approach `P(lensed) ≈ 1` the way the true Bayesian density
ratio should. There is no physical explanation for a bright, red transient other
than lensing magnification; the posterior should reflect that regardless of how
rarely such an event is drawn.

### Why the sparse-corner importance-sampling feature (PR #33/#34) doesn't fix this

This project recently shipped adaptive pilot+boost importance sampling
(`cmsne/importance_sampling.py`, PR #33, bias-fixed before merge) plus a fix for
`diagnose_sparse_cells`'s bin-edge computation (PR #34, this same session,
commit `ac689e1`). Both are correct and merged, but a real BlueBEAR production
run (`results/mc_58818280`, boosted, vs `results/mc_58819823`, `--no-boost`
baseline, both `--n 100000`, all 10 classes) showed they do not solve this
problem:

- `diagnose_sparse_cells` computes its bin edges via `cmsne.contours.robust_range`
  (rate-weighted 1st–99th percentile) — necessary to exclude zero-weight
  extreme-outlier artifacts (the PR #34 bug), but it also excludes the bright+red
  corner itself: for `lensed_Ia` at `n=100,000`, only 13–15 raw events land in
  `gr≥2.0, peakmag≤16` at all (combined rate-weight ≈2.7×10⁻¹² against a total of
  ≈1.1×10⁻⁴) — far outside the class's own 1st–99th percentile range. The corner
  gets no bin, so it is never flagged, so boosting never reaches it.
- Empirically: `P(lensed)` in that corner is **0.4558 (no-boost) → 0.4576
  (boosted)** — a fitted `posterior_surface` on `gr` vs `peakmag`, signal =
  `lensed_Ia`, background = the other 9 classes, explicit grid range `(-1.5, 3.5)
  × (13, 25)` to actually reach the corner. Essentially unchanged, nowhere near 1.
- Root mechanism: `HistGradientBoostingClassifier` is a tree ensemble — like the
  isotonic calibration layered on top of it, trees are piecewise-constant and
  repeat their most-extreme leaf's value for any point beyond the training data's
  support. Swapping the calibration method alone would not fix this — the base
  tree itself also plateaus out there.

This spec addresses the *actual* root cause (a model whose score and
calibration both plateau outside training support), independent of what feeds
training data into that model.

## Scope decision

A fully general fix was chosen over pre-registering specific boxes (e.g.
hardcoding "bright+red" as a box): the mechanism must work for **any** point
`ColourClassifier.probability()` is asked to evaluate, not only ones registered
in advance.

## Chosen approach: local weighted-KDE density-ratio blend

Considered and rejected:
- **Calibration-only fix** (e.g. sigmoid/Platt calibration instead of isotonic,
  or a parametric tail extrapolation on the raw tree score): cheaper, but the
  base `HistGradientBoostingClassifier` itself is piecewise-constant and
  plateaus past its training leaves' support, independent of the calibration
  layer. This would move the plateau value but cannot be shown to approach 1.
- **Replace the tree+isotonic model entirely** with a global KDE density-ratio
  estimator: theoretically the cleanest single mechanism, but KDE degrades in
  the 6-D feature space (5 adjacent colours + magnitude) where the tree was
  specifically chosen for beating simpler models on this problem, and would
  require re-validating every existing headline number (55.6%/88.8% colour-first,
  boundary studies) against a fundamentally different model, for a problem that
  is only in the tail. Not recommended.

**Chosen:** keep the tree+isotonic model as the primary estimator (preserves its
proven bulk performance), and blend it with a local weighted-KDE density-ratio
estimate, with the blend weight driven by how much local effective sample
support the training data has near the query point. This reuses the already-
tested `cmsne.importance_sampling.effective_sample_size` function (Kish's ESS)
unchanged, generalized from a binned statistic to a pointwise, kernel-weighted
one, and extends the weighted-KDE machinery already built for PR #33/#34's
latent-space proposals to the classifier's observable feature space.

## Where it lives

Entirely inside `ColourClassifier` (`cmsne/classifier.py`), in `fit()` and
`probability()`. Because `TrajectoryClassifier` subclasses `ColourClassifier`
and inherits `fit`/`score`/`probability`/`rank`/`predict`/`recovery_rate`
unchanged (only the feature schema differs — see its docstring), and because
`posterior_surface` instantiates `ColourClassifier` directly, all three
consumers (`posterior_surface`, `ColourClassifier` direct use,
`TrajectoryClassifier`) get the fix automatically. No other file needs to
change for the fix itself (test files aside).

**Deliberate scope boundary:** `score()` (and everything built on it —
`rank()`, `predict()`, `roc_curve()`, `recovery_rate()`) stays tree-only,
untouched. Only `probability()` — the method this whole investigation is
about — gets the blend. This means the existing contamination-pinned hard
threshold, ranking order, and recovery-rate numbers are unaffected by
construction; only the posterior *value* changes, and only where it needs to.

`fit()` currently builds `Xs`/`Xb` (feature matrices), trains the tree+isotonic
model, and discards the matrices. It will additionally retain `Xs, Xb, ws, wb`
(already-computed, nothing new to build) and the column list on the instance,
since the KDE side needs the same weighted training events at prediction time.

## Mechanics

### Bandwidth

Computed once at `fit()` time, per feature column (`self.keys + self.extra`):

```
h_d = scott_factor * weighted_std_d
scott_factor = n_eff ** (-1 / (d + 4))
```

- `weighted_std_d`: that column's rate-weighted standard deviation, computed
  over whichever training events (pooled signal+background) have it finite.
- `n_eff`: `effective_sample_size` of the pooled training weights.
- `d`: total number of feature columns (`len(self.keys) + len(self.extra)`).
- Standard multivariate KDE bandwidth heuristic (Scott's rule); not a new
  invention. A diagonal (axis-aligned) Gaussian kernel is used — no
  cross-dimension covariance — keeping the kernel simple and each dimension's
  bandwidth independently interpretable (a colour with little natural spread
  gets a narrower kernel than one that's naturally broad).

### Per-point kernel evaluation

For a query point `x` and a class's training events `{x_i}` with rate weights
`{w_i}`, restricted to whichever feature dimensions are finite for `x` (see
"Missing-band handling" below):

```
k_i = exp(-0.5 * sum_d ((x_d - x_i,d) / h_d) ** 2)      # per training event
S(x) = sum_i w_i * k_i      # signal
B(x) = sum_i w_i * k_i      # background (same formula, background events/weights)
```

### Local density ratio

```
P_kde(x) = S(x) / (S(x) + B(x))
```

A direct Nadaraya-Watson-style kernel density ratio. As `x` moves into a region
where background events get vanishingly small kernel weight (nearest background
events many bandwidths away) while at least a few signal events remain within
reach, `B(x) → 0` while `S(x)` stays positive, so `P_kde(x) → 1` — the needed
asymptotic behaviour, a direct consequence of the kernel's tail shape, not a
hardcoded floor.

**Degenerate guard:** if `S(x) + B(x)` underflows below a small floor (`1e-300`),
`P_kde` is undefined (0/0). In that case: treat as the `α=1` fallback below
(pure tree) — a point with no training support from *either* class is exactly
where only the tree's own last-leaf plateau is available anyway, so this guard
cannot make anything worse than today's behaviour.

### Local effective sample size

Reuses `cmsne.importance_sampling.effective_sample_size` unchanged, fed the
kernel-weighted values instead of bin-membership masks:

```
local_ess_sig(x) = effective_sample_size([w_i * k_i for signal events])
local_ess_bg(x)  = effective_sample_size([w_i * k_i for background events])
```

The pointwise generalization of the same ESS concept `diagnose_sparse_cells`
already uses for binned cells, continuous in `x` instead. Shares the same
per-event kernel evaluation as `S(x)`/`B(x)` — no redundant computation.

### Blend weight

```
m(x) = min(local_ess_sig(x), local_ess_bg(x))
alpha(x) = sigmoid((log(m(x) + eps) - log(ess_threshold)) / log_width)
final_p(x) = alpha(x) * p_tree(x) + (1 - alpha(x)) * p_kde(x)
```

- Driven by the **thinner-supported class's** local ESS, not background alone:
  the tree can only have learned a trustworthy boundary near `x` if *both*
  classes had decent local presence there during training. A point where
  signal is locally starved is just as much an extrapolation as one where
  background is.
- `ess_threshold` and `log_width` are `ColourClassifier` constructor parameters
  with defaults chosen empirically during implementation (not guessed a
  priori) — this project's rate weights span many orders of magnitude, so
  local ESS runs low even in the "normal" bulk (`diagnose_sparse_cells` floors
  of order 1–5 were observed on real production data). The default is tuned
  against two concrete targets: in the already-trusted bulk (this project's
  existing headline operating region), `alpha(x)` should land close to 1; in
  the bright+red corner from this investigation, `alpha(x)` should land close
  to 0.
- Blending happens in plain probability space (not log-odds): both `p_tree`
  and `p_kde` are already bounded in `[0, 1]` (the existing code already clips
  `p_tree` to `[1e-12, 1-1e-12]`), and a linear blend is simpler to state, test,
  and check against a concrete acceptance number than a log-odds blend.
- The optional `prior` reweighting already in `probability()` (the likelihood-
  ratio rescaling to a different base rate) applies to `final_p(x)`, unchanged
  in its own logic — it happens *after* the blend, not before.

### Missing-band ("NaN-native") handling and caching

`ColourClassifier` is deliberately NaN-native (a real cadence won't always have
every band). Query and training events can have different subsets of columns
finite.

- Query events are grouped by their **finite-column pattern** (which exact
  subset of `self.keys + self.extra` is non-NaN for that event).
- For each distinct pattern seen in a `probability()`/`score()`-adjacent call:
  filter the stored training events (signal and background separately) down to
  those finite in *that same* subset of columns, then compute `S(x)`, `B(x)`,
  and the two local ESS values using only those dimensions and their
  already-precomputed bandwidths.
- The filtered-training-subset-per-pattern is cached on the `ColourClassifier`
  instance, keyed by the pattern, and reused across calls. `fit()` resets the
  cache.
- `posterior_surface`'s mesh (`{colour_key: c, magnitude_key: m}` dicts — always
  exactly 2 finite columns, one pattern, for up to 40,000 grid points per call)
  is the cheapest possible case: one cached filtered subset, reused for the
  whole grid. This is also the exact caller used throughout this investigation,
  so the main validation path is the cheap path.
- Degenerate case (an event with no finite columns at all): `local_ess_sig` and
  `local_ess_bg` are both 0, which falls through to the same `alpha=1` guard —
  cannot make anything worse than today.

## Backward compatibility

`score()`, `rank()`, `predict()`, `roc_curve()`, `recovery_rate()` are
untouched — they depend only on `score()`, which stays tree-only. Only
`probability()` changes, and only where `alpha(x) < 1`, i.e. only where at
least one class has thin local support.

**Validation, not assumption:** rerun `posterior_surface` on the already-
generated production data from this session (`results/mc_58818280`, boosted,
`results/mc_58819823`, `--no-boost` baseline — both already on BlueBEAR) and
confirm the blended posterior matches the pre-change tree-only posterior within
a tight tolerance (`1e-3`) everywhere **outside** the handful of genuinely
sparse cells identified by `diagnose_sparse_cells`'s `flagged_cells` output for
that class/band-pair.

## Performance

`posterior_surface`'s 200×200 mesh against a ~100-150k-event background
population is, in the worst case, several billion kernel evaluations per band
pair. This must be fully vectorized (numpy broadcasting across query points ×
training events, no Python-level loops over events).

**Safety valve:** a `kde_max_events` constructor parameter on `ColourClassifier`
(default to be chosen during implementation, starting point: low thousands per
class) rate-weighted-subsamples each class's training events before the KDE
side is built from them — the same weighted-subsampling pattern already used
elsewhere in this project (`_draw_resampled_sources` in
`scripts/plot_evolving_boundaries.py`). KDE accuracy saturates well before using
every one of 100k+ points, so this caps cost without a meaningful accuracy loss.
Actual wall-clock will be measured during implementation against real
production data, and the default tuned from that measurement rather than
guessed.

## Testing and validation plan

**Unit tests** (`tests/test_cmsne.py`, following the existing house style —
`check(name, cond, extra='')`, not pytest):

- Local-ESS/kernel helper correctness on a small synthetic case (Kish's ESS
  formula reused correctly; bandwidths positive and scale with weighted std).
- `P_kde → 1` on a synthetic case with signal events clustered near a point and
  zero background events nearby (background events exist, just elsewhere); and
  the symmetric case (background clustered, signal elsewhere) → `P_kde → 0`.
- Dense synthetic region (plenty of both classes nearby a query point):
  `alpha ≈ 1`, blended probability ≈ tree-only probability within tight
  tolerance.
- Sparse synthetic region (very few of one class nearby): `alpha ≈ 0`, blended
  result ≈ `P_kde`, materially different from `p_tree` alone.
- Missing-band/pattern-grouping: a query event with a finite-column pattern
  distinct from others in the same call does not crash and produces a sane
  probability from the correctly-restricted training subset.
- Degenerate guard: a query point with no training data of either class nearby
  (and the no-finite-columns case) falls back to `alpha=1`, no NaN/crash.
- Backward-compatibility regression on production-shaped data: in a
  synthetic "dense bulk" region built the same way Task 5 of the prior plan
  built its production-shaped backward-compat fixture, blended probability
  matches pre-change tree-only probability within tolerance.

**Manual validation** (BlueBEAR-scale, not automatable — run by the project
owner, using data already generated this session):

1. Rerun the exact bright+red corner check (`gr≥2.0, peakmag≤16`, signal =
   `lensed_Ia`, background = the other 9 classes, explicit grid range to reach
   the corner) against the new `ColourClassifier`, using `results/mc_58818280`.
   **Success criterion: P(lensed) exceeds 0.9** in that corner (up from the
   ~0.4576 plateau measured in this session before this fix).
2. Re-check the backward-compatibility tolerance above against the real
   production data, not just the synthetic fixture.
3. Re-generate (or reuse) `scripts/plot_evolving_boundaries.py`'s boundary
   figures on production data and visually confirm the bulk decision boundary
   is unchanged, while the previously-flat extreme corners now show `P → 1`.
4. Timing: record actual wall-clock for a `posterior_surface` call at
   production scale, to confirm the `kde_max_events` default keeps this
   tractable (target: comparable order of magnitude to the existing tree-only
   fit + grid evaluation, not a multi-x slowdown that makes routine use
   impractical).

## Open items carried into the implementation plan

- The exact default values for `ess_threshold`, `log_width`, and
  `kde_max_events` are **not** fixed in this spec — they are tuned during
  implementation against the two concrete empirical targets in "Blend weight"
  and "Performance" above, using the real production data already generated
  this session, and recorded in the implementation plan/PR once chosen.
