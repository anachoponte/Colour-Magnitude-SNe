# Detection-Aware Importance-Sampling Proposal — Design

Supersedes `2026-11-20-matched-importance-proposal-design.md`, whose mechanism
(a proposal matched to the rate density alone) was measured to be *worse* than
the uniform proposal it replaced for 8 of 10 classes. That document is retained
with a superseded banner; its Problem, Measured impact and Constraints sections
remain accurate and are carried forward here.

## Problem

Every simulated population in this project is drawn from a **uniform** proposal
over the latent variables and reweighted by the physical rate:

```python
# cmsne/multicolour.py, generate_many
z  = np.random.uniform(*z_range)
mu = np.random.uniform(*magnification_range) if lensed else 1.0
```

The target is nowhere near uniform. For lensed classes
`w(z, mu) = f(z) * dtau_dmu_dz`, with `dtau_dmu_dz = (d_com(z)/31)^3 * mu^-3`.
Over `mu in [2, 50]` the `mu^-3` factor alone caps lensed ESS at `~5%`, and the
`min_detections=5` cut makes it worse: detection favours high `mu`, exactly
where `mu^-3` is smallest.

### Measured impact

`scripts/diag_calib_resolution.py`, BlueBEAR job `62176945`, against
`results/dual_59297152`:

| Pool | Raw events | ESS | ESS/raw | Top 1% of events hold |
|---|---|---|---|---|
| lensed_Ia | 33,147 | 623 | 1.9% | 55.0% of weight |
| all_lensed | 85,381 | 876 | 1.0% | 61.6% |
| lensed_sn2n | 29,770 | 373 | 1.3% | — |
| lensed_sn2l | 7,412 | 214 | 2.9% | — |
| lensed_sn1bc | 6,589 | 73 | 1.1% | — |
| lensed_sn2p | 8,463 | 53 | 0.6% | — |
| background (all unlensed) | 26,383 | 9,282 | 35.2% | 7.0% |

Downstream consequence, measured in `rank_candidates.py` (job `62176559`,
Finding 6): of 120 cumulative operating points, the 38 that are statistically
clean are *all* at the 10% or 1% FPR budget. None at 0.1% or 0.01%, any target,
any epoch.

### Why this is not a boosting failure

`generate_with_boosting(pilot_frac=0.85)` patches sparse *cells* in
(colour, magnitude) space on top of a proposal mismatched *globally*. Its
mixture correction also assumes a uniform pilot (`p_uniform_const`).

## Prior art

This is a solved problem in the statistics literature and a mature methodology
in a structurally identical astronomy application. The references below are for
the eventual paper as much as for the build; each was checked directly, and the
"verified" column states how far.

### The result we are implementing

The textbook variance-minimising proposal is `q_opt ∝ |f| · p` — sample
proportional to the integrand, **no square root**
([Owen, *Monte Carlo theory, methods and examples*, Ch. 9][owen]). Applied
naively here that gives `rate · eps`, and a referee who knows the standard
result will ask why ours has a root in it.

The root appears because **`eps` is never evaluated, only sampled from**:
detection is a random 0/1 outcome of simulating a light curve, not a computable
function. That is the *noisy* importance-sampling regime, which has its own
theory. Llorente & Martino's survey, §6.2, gives the optimal proposal when the
integrand evaluation is itself a random variable:

```
q_opt(theta)  ∝  ||f(theta)||_2 * sqrt( m(theta)^2 + s(theta)^2 )
```

with `m`, `s` the mean and standard deviation of the noisy evaluation. For a
Bernoulli detection indicator `m = eps` and `s^2 = eps(1 - eps)`, so
`m^2 + s^2 = eps^2 + eps - eps^2 = eps`, giving

```
q_opt(z, mu)  ∝  rate(z, mu) * sqrt( eps(z, mu) )
```

**exactly**. Our proposal is the Bernoulli special case of a published general
theorem, not a local derivation. Independently it is also plain **Neyman
allocation** — allocate in proportion to standard deviation rather than to size
— an equivalence the same survey states explicitly.

### References

| Purpose | Reference | Verified |
|---|---|---|
| The `sqrt(eps)` result (§6.2); also that noisy IS **cannot reach zero variance** (§6.1) | Llorente & Martino (2025), *Optimality in importance sampling: a gentle survey*, [arXiv:2502.07396][llm] | §6.1–6.2 read directly |
| Standard `q ∝ \|f\|p`, ESS, failure modes of mismatched proposals | [Owen, *MC theory, methods and examples*, Ch. 9][owen] | Content confirmed |
| Allocation ∝ standard deviation | Neyman (1934), *JRSS* **97**(4), 558–625, [doi:10.2307/2342192][neyman] | Citation confirmed |
| Reweighting one generic injection set to many populations — our exact pattern | Tiwari (2018), *CQG* **35**, 145009, [arXiv:1712.00482][tiwari] | Confirmed |
| How accurately a selection function must be measured; `N_eff = (mu/sigma)^2` | Farr (2019), *RNAAS* **3**, 66, [arXiv:1904.10879][farr] | Confirmed; formulas not read |
| Precision requirements; the `N_eff > 4 N_obs` rule of thumb | Essick & Farr (2022), [arXiv:2204.00461][essick] | Confirmed; derivations not read |
| Defensive mixtures — bounding weights, the basis of this design's safety net | Hesterberg (1995), *Technometrics* **37**, 185; Owen & Zhou (2000), *JASA* **95**, 135, [doi][ownzhou] | Confirmed |
| Kish effective sample size `(sum w)^2 / sum w^2` | Kish (1965), *Survey Sampling* | **Not independently verified** — already used throughout `cmsne/importance_sampling.py`; confirm before citing |

[owen]: https://artowen.su.domains/mc/Ch-var-is.pdf
[llm]: https://arxiv.org/abs/2502.07396
[neyman]: https://doi.org/10.2307/2342192
[tiwari]: https://inspirehep.net/arxiv/1712.00482
[farr]: https://arxiv.org/abs/1904.10879
[essick]: https://arxiv.org/abs/2204.00461
[ownzhou]: https://doi.org/10.1080/01621459.2000.10473909

### Framing for the paper

**The gravitational-wave selection-function literature is the strongest framing
citation.** Tiwari (2018), Farr (2019) and Essick & Farr (2022) address a
structurally identical problem — injections drawn from a reference
distribution, a found/missed detection cut, reweighted to an astrophysical
population, with effective sample size governing how far the result can be
trusted. "The same problem the GW community solved for sensitive-volume
estimation" is a far easier sell than deriving this cold.

**Novelty claim, stated conservatively.** The method is thoroughly established.
What a search did *not* turn up is anyone optimising the *generation*
distribution by `sqrt(eps)` in **supernova survey simulation**. Adjacent
practice exists — a recent SBI paper importance-samples a SNANA bank; DES and
Foundation measure selection efficiency empirically from data — but not this.
The defensible claim is therefore **a known-optimal method applied to lensed-SN
simulation for the first time**, not a new method. Claim it that way.

### Two caveats that belong in the paper

1. **The variance floor is not zero.** Llorente & Martino note the optimal
   noisy-IS estimator cannot reach zero variance while `s` is non-zero. ESS per
   attempt therefore cannot approach 1 even with a perfect proposal; the
   measured gain should not be read as falling short of an unreachable ideal.
2. **A badly estimated `eps_hat` can be worse than uniform.** This is the known
   failure mode of Neyman allocation — with wrong stratum standard deviations
   it can underperform simple random sampling. It is exactly the risk in
   estimating `eps` from a pilot, and it is a documented hazard rather than
   speculation. The defensive mixture below exists to bound it.

## Intent and why

Raise effective sample size per unit compute, so that operating points at
strict FPR budgets become measurable and CC-subtype true-positive statistics
stop resting on ~11 effective events.

This is **variance reduction, not bias correction.** The weights already
correct for the proposal, so existing published numbers are unbiased; they
should land within their current error bars and get much tighter. Any
systematic shift after this change is evidence of a bug — see Validation.

## Goals

- Draw `(z, mu)` from a proposal approximating `p ∝ rate * sqrt(eps)`, with
  weights divided by the proposal density so estimates stay unbiased.
- Estimate `eps` from a pilot stage and regularise it.
- Bound the damage from a bad `eps_hat` so the change cannot do much worse than
  uniform even when the estimate is poor.
- Keep one code path: the new proposal replaces the uniform one.

## Non-goals

- Changing any rate physics. `absolute_rate_weight`, `LensingOpticalDepth` and
  the SFRD model are untouched *as physics*; only the Monte Carlo volume factor
  that assumed uniform sampling changes.
- Changing the classifier, its features, or any analysis script.
- Re-running the full downstream analysis chain. Follow-up work, gated on this
  landing and on a regeneration run.

## Anti-goals

- **A silent bias.** If rate-weighted totals (annual detection rates, TP/yr at
  the 10% budget) shift outside their error bars, the change is wrong.
- **Two divergent sampling paths.** The user chose replacement specifically to
  avoid carrying a second path.
- **Physics duplicated in a second place.** If the proposal hard-codes `mu^-3`
  and the SFRD independently of `absolute_rate_weight`, the two will drift.
- **An unbounded weight.** Any region with `rate > 0` and `p = 0` is an
  unbounded-variance bug, not a tuning problem.

## Constraints

- **`eps` is NOT separable in `(z, mu)`.** Detection depends on apparent
  brightness, which couples them (`mu` brightens, `z` dims). The superseded
  design's "two independent 1-D inverse CDFs" trick is valid for the rate alone
  and **invalid here**. The proposal must be genuinely 2-D for lensed classes.
- `rate` *is* separable, but there is no benefit in exploiting that separately
  once the 2-D machinery exists.
- `z_range` is per-source, from `redshift_limits(source, 'lsstr', 'lsstr')`, so
  proposals are built per class, not globally.
- Weights must stay comparable across classes: they feed
  `cmsne.importance_sampling`'s weighted statistics and the annual-rate
  normalisation used throughout `scripts/`.
- BlueBEAR only for any run (login node is not for compute); `cmsne-env`
  locally for tests.
- The existing baseline `results/dual_59297152` is **valid** for comparison:
  generation code is unchanged since 2026-10-27 and the run is dated
  2026-10-28.

## Approaches

Three were considered for representing the 2-D proposal. (The *statistical*
choice — `rate * sqrt(eps)` — is settled by the theory above and by
`spike_proposal_ceiling.py`, which measured it beating uniform for all 10
classes by 3.2x–60.4x and beating `rate` and `rate*eps` in every case.)

1. **2-D piecewise-constant grid — CHOSEN.** One grid over `(z, mu)` holds
   `eps_hat` and defines the proposal. Sample a cell from a discrete CDF, then
   uniformly within it. The density is piecewise constant and therefore
   *exactly* evaluable — no inversion error entering the weights — and the grid
   that estimates `eps` is the same object that defines the proposal, so the
   two cannot drift. Cheap and easy to test.
2. **Conditional decomposition `p(z) * p(mu | z)`.** Build a marginal in `z`
   and a conditional in `mu`. Exact and continuous, but needs an inverse CDF
   per `z`-slice, and the density evaluation is interpolated, so inversion error
   enters the weights. Rejected: more code, more error, no benefit.
3. **SIR pre-filter.** Keep cheap uniform latent draws, resample proportional
   to `rate * sqrt(eps_hat)` before the expensive `generate_one`. Minimal
   change and spends every expensive simulation on a good latent. Rejected for
   introducing resampling variance and duplicate latents — but it is the
   cheapest fallback if the grid approach proves awkward, so it is recorded
   rather than discarded.

## Mechanics

### New module: `cmsne/proposals.py`

A `DetectionAwareProposal` built per class, holding a 2-D grid over
`(z_range, magnification_range)` (1-D over `z_range` for unlensed classes,
where `mu = 1` deterministically).

- **`rate_grid`** — cell-centre evaluations of the class's own
  `rate_weight_fn`, i.e. the same function the weights use. No physics is
  restated.
- **`eps_hat`** — detection efficiency per cell, estimated in the pilot stage
  as `detected / attempted`, then regularised (below).
- **`weights ∝ rate_grid * sqrt(eps_hat) * cell_area`**, normalised.
- **`sample(n)`** — draw cells from the discrete CDF, then uniform within cell.
- **`density(z, mu)`** — exact piecewise-constant value. Used to divide the
  weight.

Guarantees to preserve, all directly testable: `density` integrates to 1 over
the sampling domain; `sample(n)` reproduces `density`; `density > 0` everywhere
`rate > 0`.

### Regularising `eps_hat`

Three layers, in order. The first two are the builder's to tune; the third is
not optional.

1. **Smoothing.** Raw per-cell `detected/attempted` is noisy in thin cells. A
   small Gaussian or box smoother over the grid, or a shrinkage estimator
   toward the global rate (`(d + a) / (n + a + b)`, a Beta prior), is
   sufficient. Grid resolution and smoother strength are the builder's call,
   subject to the acceptance metric below.
2. **Floor.** `eps_hat = max(eps_hat, eps_min)` so no cell with non-zero rate
   gets zero proposal mass.
3. **Defensive mixture — REQUIRED.**
   `p_final = (1 - alpha) * p_matched + alpha * p_uniform`, with `alpha`
   defaulting to `0.1`. This is Hesterberg (1995) / Owen & Zhou (2000) and it
   is what bounds caveat 2 above: the weight anywhere is at most `1/alpha`
   times its uniform-proposal value, so a badly estimated `eps_hat` can cost at
   most a bounded factor instead of blowing the variance up without limit. It
   must not be removable by a flag.

### `cmsne/multicolour.py` — `generate_many`

Takes the proposal and draws from it instead of uniformly. Each returned event
records the proposal density it was drawn from (`e['proposal_density']`), so
downstream weight correction never re-derives it.

### `cmsne/rates.py` — `absolute_rate_weight`

The `n_attempted` branch currently computes

```python
v_sample = delta_z * delta_mu   # uniform-proposal volume
return (v_sample / n_attempted) * density
```

`delta_z * delta_mu` is exactly the reciprocal of a uniform proposal density.
Generalised, the Monte Carlo weight is `density / (n_attempted * p_proposal)`,
with the uniform case recovered when `p_proposal = 1/(delta_z * delta_mu)`.
The parameter carrying this must be **explicit**, so a caller that forgets it
fails loudly rather than silently reverting to uniform.

### `cmsne/importance_sampling.py` — `generate_with_boosting`

`p_uniform_const` assumes a uniform pilot; it becomes the pilot proposal's
actual density, evaluated per event. The existing mixture correction
(`f_pilot + p_boost_total / p_pilot`) is otherwise structurally unchanged — it
already handles a non-trivial pilot density once that density is supplied per
event rather than as a constant.

The pilot stage gains a second job: it now also produces `eps_hat`. The pilot
itself stays **uniform**, because its purpose is to cover the space and a
detection-aware pilot would bias the very estimate it is meant to produce.

The boost stage keeps working as-is: it targets cells still sparse *after* the
new proposal. Fewer cells should be flagged; that is success, not regression.

### Production scripts

`run_dual_schema.py`, `run_multicolour.py` and `run_trajectory.py` construct
the proposal per class and pass it through. Their `CLASSES` lists are
deliberately identical and must stay so.

## Testing and validation

Unit (`tests/test_cmsne.py`, run via `python tests/test_cmsne.py` in
`cmsne-env`, **not** pytest):

- `density` integrates to 1 over the domain, lensed and unlensed.
- `sample(n)` reproduces `density` within tolerance.
- `density > 0` wherever `rate > 0`, including with `eps_hat` identically zero
  in a region — the defensive mixture must carry it.
- With `eps_hat` constant, the proposal reduces to the rate-matched one; with
  `rate` constant too, it reduces to uniform.
- `absolute_rate_weight` with a uniform proposal density reproduces today's
  value **exactly**.
- `generate_with_boosting(pilot_frac=1.0)` returns weights equal to
  `rate_weight_fn / p_proposal`.

Two existing tests encode the uniform assumption and must be **updated, not
removed** (`tests/test_cmsne.py:2449, 2457`): reframe them as the uniform
special case of the general formula.

Validation (BlueBEAR, sbatch only):

- **Acceptance metric: ESS per generation attempt**, per class, against the
  uniform baseline. Not ESS per kept event, which is misleading — compute is
  spent per attempt. Accept if every class improves and no class regresses.
- **Unbiasedness gate:** a small run compared against `results/dual_59297152`
  on rate-weighted totals (annual detection rate per class, TP/yr and purity at
  the 10% budget). "Agree within error" means: a weighted bootstrap
  (resample events with probability ∝ weight, 1000 draws) of each quantity on
  both runs, with overlapping 90% intervals. A shift outside that **blocks the
  change** — it indicates a bug, since this is variance reduction.
- `scripts/diag_calib_resolution.py` re-run on the new output, against the
  table above as baseline.

## Expected outcome, and what it does not solve

`spike_proposal_ceiling.py` measured ceilings of 3.2x–60.4x in ESS per attempt,
assuming perfect knowledge of `eps`. Realised gains will be **lower**: `eps_hat`
is noisy, smoothing costs sharpness, and the defensive mixture deliberately
gives up some efficiency for safety. The spike's uniform baseline reproduces
measured values well for the Ia classes (`unlensed_Ia` 0.0682 vs 0.0682) but
only within ~2–3x for the CC subtypes, so CC factors are order-of-magnitude.

If the background pool realises even ~5x, calib background ESS moves from
`1,856` toward the `~10,000` range, which would make the **0.1% FPR budget
measurable for the first time** with no extra generation, and bring 0.01%
within a ~2–5x run rather than 54x. That is the outcome to aim at; it is a
projection, not a promise, and Finding 6's table should not be rewritten until
it is measured.

## What is left to the builder

- Grid resolution in `z` and `mu`, subject to the acceptance metric.
- Smoother form and strength; shrinkage prior parameters.
- `eps_min`.
- Whether the proposal caches per `(source, kind)` or is rebuilt per call.
- The exact parameter name and signature carrying proposal density into
  `absolute_rate_weight`, subject to it being explicit and failing loudly.
- Test tolerances.

Settled as my calls, flagged for the user to check:

- ESS **per attempt** as the acceptance metric (compute is per attempt).
- Validation runs use `--no-boost` / `pilot_frac=1.0`, so the proposal is
  measured without the boost stage confounding it.
- The weighted-bootstrap recipe above as the operational meaning of
  "agree within error".
- `alpha = 0.1` as the defensive-mixture default.
- The pilot stays uniform.
