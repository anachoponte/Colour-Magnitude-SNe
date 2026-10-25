# Sparse-Corner Importance Sampling Design

## Problem

`cmsne/classifier.py`'s `ColourClassifier` (a `HistGradientBoostingClassifier` with isotonic
`CalibratedClassifierCV`) estimates `P(lensed | x)` for colour-magnitude feature vectors `x`,
used directly by `cmsne/contours.py`'s `posterior_surface` and indirectly by every
`TrajectoryClassifier`/multicolour classifier built the same way. In the very-bright,
very-red corner of colour-magnitude space — a region that is physically near-impossible to
reach as an unlensed event — the fitted posterior does not reliably approach 1 the way the
underlying physics demands. Empirical prototyping on real production data (job 58563115,
`gr` colour + `peakmag`) confirmed this directly:

| Model | Held-out rate-weighted AUC | Behaviour in the deep corner |
|---|---|---|
| Tree + isotonic (current production) | 0.8999 | Erratic: a spurious spike to 0.99 next to frozen values near 0.25 |
| Plain logistic regression | 0.8856 | Smooth, monotonic, but worst bulk accuracy |
| Logistic GAM (splines) | 0.9302 (best) | Collapses to ≈0.5 outside the training box — least reliable of all |
| Monotonic-constrained tree + isotonic | 0.9093 | No more backward artifacts, but plateaus rather than converging to 1 |

No model architecture fixed this, because none of them is the actual problem.

### Reframing

`P(lensed | x) = w_lensed(x) / (w_lensed(x) + w_unlensed(x))` is exactly the quantity a
classifier trained on samples from two simulated populations is proven to converge to
(Cranmer, Pavez & Louppe 2015, "Approximating Likelihood Ratios with Calibrated
Discriminative Classifiers", [arXiv:1506.02169](https://arxiv.org/pdf/1506.02169)) — the
foundational result behind simulation-based/likelihood-free inference. The strategy already
in production is textbook-correct; the formula is not wrong. What fails is **estimation** in
one region: an event being individually rare in the lensed population does not by itself
depress `P(lensed|x)` — only the *ratio* against the unlensed density matters, and if the
unlensed density is rarer still at that point, the ratio should correctly tend to 1 regardless
of how few lensed training events exist there.

That specific failure — a data-driven ratio estimate becoming unreliable where one class's
effective density drops toward zero — is a named, established problem in the propensity-score
/ causal-inference literature: **poor overlap / positivity violation**. The standard diagnosis
there is explicitly *not* "that region is anomalous, downweight it" but "there isn't enough
effective sample size to trust the data-driven estimate there."

### Confirmed on real data

Kish's effective sample size (`ESS = (Σw)² / Σw²`) for lensed Ia vs. all contaminants, job
58563115, `gr` colour + `peakmag`:

| Region | N (lensed) | ESS (lensed) | N (bg) | ESS (bg) |
|---|---|---|---|---|
| Full sample | 2989 | 174.3 | 14626 | 4026.6 |
| g−r ≥ 1.5 | 780 | 80.6 | 752 | 113.9 |
| peakmag ≤ 18 | 981 | 327.3 | 1385 | 250.2 |
| g−r ≥ 1.5 & peakmag ≤ 18 | 54 | 21.4 | 53 | 9.0 |
| g−r ≥ 2.0 & peakmag ≤ 16 | 15 | **7.5** | 19 | **2.4** |

The deep corner is ~2 orders of magnitude thinner than the bulk, on **both** sides — not a
one-sided lensed-data gap. No model, however expressive, can responsibly resolve a ratio from
15–19 raw events of each class; every model tried above produced something closer to sampling
noise than signal in exactly this region.

### No analytic shortcut available

The obvious next idea — evaluate `rate_weight()` directly instead of trusting a classifier in
the sparse corner — does not work. `l_weight(redshift, magnification)` / `ul_weight(redshift)`
(`cmsne/colour_magnitude.py:120-172`) are functions of the *latent* physical parameters, not of
the observed colour/magnitude `x`. The map from `(z, magnification, abs_mag, x1, c)` to an
observed `x` goes through the full forward simulator (SALT3/Nugent template, noise, OpSim
cadence, detection), which has no closed form to invert. This is a genuine
simulation-based-inference setting: there is no tractable likelihood in observed-space to fall
back on.

What *is* available: the weight is known analytically in the latent `(z, magnification)`
space, and the simulator can be driven from there. The fix is therefore not a different
classifier — it's **generating more effective training samples in the latent region that maps
into the sparse corner**, i.e. importance sampling the simulator itself (Hesterberg's
defensive-mixture importance sampling; "Don't Cut Corners",
[arXiv:2608.12470](https://arxiv.org/pdf/2608.12470), studies the same idea — training outside
the natural proposal density — for simulation-based inference generally).

## Current pipeline (confirmed by direct code inspection)

- `MultiColourGenerator.generate_many`/`generate_one` (`cmsne/multicolour.py:147-221`) draw
  latent parameters uniformly: `z ~ Uniform(z_range)`, `magnification ~ Uniform(2, 50)` if
  lensed else 1.0, `abs_mag ~ Normal(abs_mag_avg, abs_mag_sig)`. There is **no stratification
  anywhere** in the repo, and no existing scaffolding for a non-uniform proposal.
  `TrajectoryGenerator`/`DualSchemaGenerator` (`cmsne/trajectory.py`) inherit this unchanged —
  they only override feature extraction (`_event_features`), not sampling.
- Per-event weight is attached **after** `generate_many` returns, in a plain per-event Python
  loop in each of three parallel production drivers: `scripts/run_multicolour.py:49-51`,
  `scripts/run_trajectory.py:51-54`, `scripts/run_dual_schema.py:59-62` — each doing
  `e['weight'] = rate_weight(cls['rate'], e['z'], e['magnification'], source=cls['source'])`.
  This is pure post-hoc importance reweighting of the uniform proposal; `rate_weight()` has no
  role in sampling itself.
- Each of the 10 classes (`lensed_Ia`, `unlensed_Ia`, 4 lensed/unlensed core-collapse pairs) is
  produced by exactly one `generate_many(n, ...)` call, run as one SLURM array task per class
  (`--array=0-9` in the `*_bluebear.sh` scripts), with **no communication between tasks**.
  `z_range` is computed per class from `redshift_limits()` (`cmsne/supernovae.py:54-79`) — it is
  already as wide as the source's spectral coverage allows, so widening it further is not an
  available lever; only the sampling *density* within it can change.

## Goals

- Generate enough effective training samples in sparse/extreme colour-magnitude regions that
  `ColourClassifier`/`TrajectoryClassifier`'s fitted posterior can actually learn the correct
  asymptotic behaviour there, instead of fitting sampling noise.
- Generalize across all five adjacent band pairs (`ug`, `gr`, `ri`, `iz`, `zy` —
  `cmsne/multicolour.py:26-27`) without hand-calibrating a boost direction per pair.
- Leave the bulk (non-sparse) region's statistics unchanged — this is an additive correction,
  not a resampling of the whole population.
- Keep `generate_many`/`generate_one` and the SLURM one-task-per-class array structure
  unmodified, to minimize risk to the validated existing pipeline.

## Non-goals

- Not attempting to fix the "ad-hoc lens geometry" placeholder (`z_l = z/2`, `ang_sep`) flagged
  as unverified physics elsewhere in the project — orthogonal to this change.
- Not changing `ColourClassifier`'s model architecture (tree + isotonic calibration) — the
  investigation showed the architecture isn't the problem; the training data is.
- Not attempting true cross-class-aware diagnosis (boosting only where signal *and*
  background are jointly thin) — see Decision Log.

## Architecture

A two-stage pipeline per class, self-contained within each existing SLURM array task:

1. **Pilot stage** — draw a population from the existing uniform proposal, unchanged
   (`generate_many` exactly as it exists today). No change to `cmsne/multicolour.py`.
2. **Diagnose stage** — from the pilot's own events, for each of the 5 band pairs, bin into a
   coarse 2D grid over `(colour, peakmag)` extending beyond the `robust_range` the production
   figures actually plot, and compute Kish's ESS per cell for this class. Flag cells where ESS
   falls far below this class's own bulk-typical per-cell ESS.
3. **Boost-proposal construction** — for events of this class landing in a flagged cell, fit a
   weighted Gaussian KDE over their stored latent `(z, magnification)` (reusing `gaussian_kde`,
   already used for density contours in `cmsne/colour_magnitude.py`), weighted by each event's
   original `rate_weight`.
4. **Boost stage** — draw additional latent samples from the fitted KDE(s), run each through the
   unchanged `generate_one`.
5. **Reweighting** — merge pilot + boost events. Every event's final weight becomes
   `rate_weight(z, μ) / p_mix(z, μ)`, replacing today's bare `rate_weight(z, μ)`, where
   `p_mix(z, μ) = f_pilot · p_uniform(z, μ) + f_boost · p_boost(z, μ)` is the known mixture
   density of the combined proposal (`f_pilot`/`f_boost` = fraction of total *attempted* draws
   from each stage). **This applies to pilot-origin events too, not only boost-origin ones** —
   once a second proposal source exists, the honest density any event was drawn under includes
   the possibility it came from either source. Reweighting only the boost batch would silently
   bias the bulk.

Where this lives: a new shared module, `cmsne/importance_sampling.py`. `generate_many`/
`generate_one` and the `MultiColourGenerator`/`TrajectoryGenerator`/`DualSchemaGenerator`
classes are not modified — the new module calls them with different latent-parameter draws,
it doesn't change their internals. Only the three driver scripts' orchestration changes.

## Components

### `generate_with_boosting(gen, n, kind, z_range, abs_mag_avg, abs_mag_sig, magnification_range, rate_weight_fn, pilot_frac=0.85)`

The single entry point each driver script calls in place of today's `generate_many` +
bare-`rate_weight()` loop.

- Splits `n` into `n_pilot = n * pilot_frac`, `n_boost = n - n_pilot`.
- Calls `gen.generate_many(n_pilot, kind, z_range, abs_mag_avg, abs_mag_sig, magnification_range=magnification_range)`.
- Attaches provisional `rate_weight_fn(z, μ)` to each pilot event (needed for the ESS
  diagnosis and for the final mixture-corrected weight).
- Runs `_diagnose_sparse_cells` (below) across all 5 band pairs for this class.
- For each flagged cell, runs `_fit_boost_proposal` (below) and allocates a share of
  `n_boost` to it (proportional to how far below the ESS floor the cell was).
- Draws boost latent samples per cell's proposal, runs each through `gen.generate_one`.
- Computes `p_mix(z, μ)` for every event (pilot- and boost-origin alike) and sets
  `event['weight'] = rate_weight_fn(z, μ) / p_mix(z, μ)`.
- Returns `(combined_events, boost_info)`, where `boost_info` records which cells were
  flagged, their ESS before/after, how many boost draws each received, and which fallback
  tier (if any) was used.

### `_diagnose_sparse_cells(events, band_pairs, ess_floor_fraction=0.05)`

For each band pair, bins events into a coarse grid over `(colour, peakmag)` (extending past
`robust_range`'s 1st–99th percentile crop), computes per-cell ESS via Kish's formula, and
flags any cell whose ESS falls below `ess_floor_fraction` of this class's median per-cell ESS
in the bulk. Returns a list of flagged `(band_pair, cell_bounds)` entries.

### `_fit_boost_proposal(events_in_cell)`

Fits a weighted `gaussian_kde` over `(z, magnification)` for the events in a flagged cell,
weighted by their original `rate_weight`. Two fallback tiers (see Error Handling) when there
are too few points to fit reliably.

## Integration

Each driver script's `run_class()` (`run_multicolour.py`, `run_trajectory.py`,
`run_dual_schema.py`) replaces:

```python
pop = gen.generate_many(n, cls['kind'], z_range, cls['am'], cls['sd'])
for e in pop:
    e['weight'] = float(rate_weight(cls['rate'], e['z'], e['magnification'], source=cls['source']))
    e['class'] = cls['name']
```

with:

```python
pop, boost_info = generate_with_boosting(
    gen, n, cls['kind'], z_range, cls['am'], cls['sd'],
    rate_weight_fn=lambda z, mu: rate_weight(cls['rate'], z, mu, source=cls['source']),
)
for e in pop:
    e['class'] = cls['name']
```

`boost_info` is written into the per-class manifest alongside the existing git-commit/host/
`n_per_class` fields, for auditability. A `--no-boost` CLI flag (equivalent to `pilot_frac=1.0`)
reproduces today's exact behaviour, for direct comparison runs. No SLURM script changes are
needed — each array task still produces one class, independently, as today.

## Error Handling

- **Flagged cell has too few points (but not zero) to fit a reliable KDE** (threshold: fewer
  than 5 weighted-effective points): fall back to a padded uniform box over the latent range
  the existing points span, instead of a KDE. Recorded in `boost_info` with `"fallback":
  "box"`.
- **Flagged cell has zero events of this class**: fall back to a rougher,
  physically-plausible latent box (the `(z, magnification)` range consistent with that
  observed cell being reachable at all, not data-derived). Recorded with `"fallback":
  "prior_box"`.
- **`p_mix(z, μ)` underflow**: KDE tails can numerically underflow for an event far from the
  fitted cell. Floor `p_mix` at a small epsilon derived from `p_uniform`'s minimum before
  dividing, and record if the floor was triggered.

## Testing / Validation Plan

1. **Unit test, toy simulator**: a synthetic case with an identity latent→observed map and a
   known analytic target ratio. Verify the pilot+boost reweighted estimate is (a) unbiased in
   the bulk, (b) lower-variance / higher-ESS in the targeted sparse region than pilot-only —
   and specifically that omitting the pilot-reweighting step (the correctness property flagged
   during design) introduces a detectable bias, so the test would catch a regression there.
2. **Backward-compatibility check**: `pilot_frac=1.0` (boosting disabled) must reproduce
   today's bare-`rate_weight()` output exactly on the cached job 58563115 data.
3. **Small live BlueBEAR smoke test**: one class, reduced `n`, boosting enabled. Confirm ESS
   improves in the deep corner (gr ≥ 2.0 & peakmag ≤ 16 and its analogues in the other 4 band
   pairs) while the existing bulk ROC/AUC numbers stay statistically unchanged.
4. **Full production rerun**: all 10 classes × 3 drivers at production `n`, retrain
   `ColourClassifier`/`TrajectoryClassifier`, and confirm `P(lensed)` in the bright+red corner
   now trends toward 1. Cross-check the existing headline colour-first numbers (55.6%
   per-system / 88.8% rate-weighted, [[corrected-run-results]]) are not silently shifted by
   this change — they should be unaffected since boosting only touches the sparse tails this
   analysis targets, but this must be checked, not assumed.

## Decision Log

- **Self-contained per-class diagnosis, not cross-class-aware** (explicit user decision). A
  cross-class-aware design (only boost where signal *and* background are jointly thin) would
  need a new two-phase SLURM structure — a pilot-phase array job, a combination/diagnosis step,
  then a dependent boost-phase array job — roughly doubling wall-clock phases for an
  already-expensive run. Self-contained diagnosis costs at most some wasted compute (boosting
  a class in a cell where its counterpart already has plenty of ESS), never a correctness
  problem, and needs zero SLURM infrastructure changes.
- **General mechanism across all 5 band pairs, not a `gr`-specific point fix** (explicit user
  decision). Diagnosis runs independently per band pair because "bright+red" means a different
  latent region in g−r than in i−z; the adaptive pilot+boost design means no band pair needs
  hand-picked physics reasoning the way the `gr` investigation initially did.
- **Adaptive (pilot + boost) importance sampling chosen over fixed hand-calibrated boost
  strata.** A fixed-strata design (manually identify each band pair's sparse latent region
  once, hardcode it) is simpler per pair but doesn't scale to "general across all band pairs"
  without repeating manual calibration, and can go stale if contaminant mixes or redshift
  limits change. Adaptive sampling costs more implementation (the diagnosis + KDE + mixture
  bookkeeping machinery) but is the literature-standard answer to this generality requirement
  (adaptive/sequential importance sampling, "Population Monte Carlo").
