# Rate-Matched Importance-Sampling Proposal — Design

> ## ⚠ STATUS: SUPERSEDED — DO NOT BUILD FROM THIS DOCUMENT
>
> The mechanism below (approach 2: a proposal matched to the **rate density**)
> was invalidated by `scripts/spike_matched_proposal_gain.py` (BlueBEAR job
> `62201604`) **after** this document was written and before any `cmsne/` code
> was touched. Measured in ESS per generation **attempt**, it *loses* for 8 of
> the 10 classes — by `16x` for `lensed_sn2l` — because it concentrates draws
> where the rate density is large and detection efficiency collapses. It loses
> on the background pool too (`unlensed_Ia` `0.63x`), which is the pool gating
> the FPR budgets this work exists to unlock.
>
> `scripts/spike_proposal_ceiling.py` (job `62201753`) identifies the correct
> target: the variance-optimal proposal under a detection cut is
> `p ∝ rate · √ε` by Cauchy-Schwarz, where `ε(z, mu)` is detection efficiency —
> not `rate`, and not `rate · ε`. It wins for all 10 classes by `3.2x`
> (`unlensed_sn2n`) to `60.4x` (`lensed_sn2l`), including `3.2–11.7x` on the
> background pool. Those are ceilings assuming perfect knowledge of `ε`; a real
> implementation must estimate it from the pilot stage and regularise it.
>
> A replacement design is **not yet written and not yet approved**. Everything
> below is retained for the record: the Problem, Measured impact, Intent,
> Goals, Non-goals, Anti-goals and Constraints sections remain accurate and a
> successor should reuse them. **Approaches, Mechanics, Testing and Expected
> outcome describe the dead proposal and must not be implemented.**


## Problem

Every simulated population in this project is drawn from a **uniform** proposal
over the latent variables and then reweighted by the physical rate:

```python
# cmsne/multicolour.py, generate_many
z  = np.random.uniform(*z_range)
mu = np.random.uniform(*magnification_range) if lensed else 1.0
```

The target density is nowhere near uniform. For lensed classes the weight is

```
w(z, mu) = f(z) * dtau_dmu_dz(z, mu),   dtau_dmu_dz = (d_com(z)/31)^3 * mu^-3
```

and for unlensed classes `w(z) = f(z)` where
`f(z) = R(z) * dV/dz * Omega * (1+z)^-1` (`sources_in_redshift_bin`'s integrand).

Sampling uniformly against a `mu^-3` target wastes most of the sample. Over
`mu` in `[2, 50]`:

```
ESS/N = E[w]^2 / E[w^2] = (int mu^-3)^2 / (48 * int mu^-6) ~= 5%
```

so **magnification alone caps lensed ESS at ~5%** before the redshift factor is
considered. The `min_detections=5` cut makes it worse: detection favours high
`mu`, which is exactly where `mu^-3` is smallest, so the kept sample
over-represents the lowest-weight region.

### Measured impact

From `scripts/diag_calib_resolution.py` (BlueBEAR job 62176945, run against
`results/dual_59297152`):

| Pool | Raw events | ESS | ESS/raw | Top 1% of events hold |
|---|---|---|---|---|
| lensed_Ia | 33,147 | 623 | 1.9% | 55.0% of weight |
| all_lensed | 85,381 | 876 | 1.0% | 61.6% |
| lensed_sn2n | 29,770 | 373 | 1.3% | — |
| lensed_sn2l | 7,412 | 214 | 2.9% | — |
| lensed_sn1bc | 6,589 | 73 | 1.1% | — |
| lensed_sn2p | 8,463 | 53 | 0.6% | — |
| background (all unlensed) | 26,383 | 9,282 | 35.2% | 7.0% |

A three-way train/calib/val split divides those further. `lensed_sn2p`'s val
split carries an ESS of **~11**, at `MIN_SAMPLE_FOR_STATS`. Every true-positive
statistic for the CC subtypes currently rests on 11–125 effective events.

### Why this is not a boosting failure

`generate_with_boosting(pilot_frac=0.85)` is already wired into all three
production scripts. It patches sparse *cells* in (colour, magnitude) space on
top of a proposal that is mismatched *globally*. It cannot fix this, and its
mixture correction assumes the pilot is uniform (`p_uniform_const`).

## Intent and why

Raise effective sample size per unit compute so that operating points at strict
FPR budgets become measurable, and so CC-subtype true-positive statistics stop
resting on ~11 effective events.

This is **variance reduction, not bias correction**. The weights already
correct for the proposal, so existing published numbers are unbiased; they
should land within their current error bars and simply get much tighter. Any
systematic shift after this change is evidence of a bug, not an expected
outcome — see Validation.

## Goals

- Sample `z` and `mu` from proposals matched to the rate density, with weights
  divided by the proposal density so estimates stay unbiased.
- Lift lensed ESS/raw from ~1–3% toward the tens of percent.
- Lift background (unlensed) ESS/raw from 35% toward ~100%.
- Keep one code path: the matched proposal replaces the uniform one.

## Non-goals

- Changing any rate physics. `absolute_rate_weight`, `LensingOpticalDepth` and
  the SFRD model are untouched as *physics*; only the Monte Carlo volume factor
  that assumed uniform sampling changes.
- Changing the classifier, its features, or any analysis script.
- Re-running the full downstream analysis chain. That is follow-up work, gated
  on this landing and on a regeneration run.

## Anti-goals

Things that would make this a failure even if ESS improved:

- **A silent bias.** If rate-weighted totals (annual detection rates, TP/yr at
  the 10% budget) shift outside their error bars, the change is wrong.
- **Two divergent sampling paths.** The user chose replacement specifically to
  avoid carrying a second path; reintroducing one defeats it.
- **Physics duplicated in a second place.** If the proposal hard-codes `mu^-3`
  and the SFRD independently of `absolute_rate_weight`, the two will drift.

## Constraints

- `dtau_dmu_dz` is **separable**: `(d_com(z)/31)^3 * mu^-3`, no z–mu coupling.
  Two independent 1-D proposals are therefore sufficient and exact.
- `z_range` is per-source, from `redshift_limits(source, 'lsstr', 'lsstr')`, so
  proposals must be built per class, not globally.
- Weights must remain comparable across classes: they feed
  `cmsne.importance_sampling`'s weighted statistics and the annual-rate
  normalisation used throughout `scripts/`.
- BlueBEAR only for any run (login node is not for compute); `cmsne-env`
  locally for tests.

## Approaches

Three were considered. The user chose a **replacement** of the uniform
proposal (not a flag) in conversation, then picked approach 2.

1. **Analytic `mu`, tabulated `z`.** `mu^-3` has a closed-form inverse CDF;
   build a numeric CDF grid for `z`. Fastest and most explicit. Rejected
   because it hard-codes the rate physics in a second location alongside
   `absolute_rate_weight`, which will drift (an anti-goal).
2. **Generic numeric inverse-CDF — CHOSEN.** Build both 1-D CDFs numerically
   from the same `rate_weight_fn` the weights already use. One mechanism, no
   duplicated physics, automatically correct if the rate model changes. Costs a
   few hundred quadrature evaluations per class at setup, negligible against
   ~20 minutes of light-curve simulation per class.
3. **SIR pre-filter.** Keep cheap uniform latent draws, resample proportional to
   rate weight before the expensive `generate_one`. Minimal change to sampling
   code and spends every expensive simulation on a high-weight latent. Rejected
   for introducing resampling variance and duplicate latents.

## Mechanics

### New module: `cmsne/proposals.py`

A `RateProposal` built once per class, holding two independent 1-D inverse-CDF
samplers derived from the class's own `rate_weight_fn`.

Construction, given `rate_weight_fn(z, mu)`, `z_range`, `magnification_range`
and `lensed`:

- **z marginal.** Evaluate `g(z) = rate_weight_fn(z, mu_ref)` on a grid over
  `z_range` (`mu_ref` is any fixed value; separability means the choice cancels
  in the normalised CDF). Build the normalised CDF by cumulative trapezoid, and
  invert by interpolation.
- **mu marginal** (lensed only). The same procedure over
  `magnification_range` with `z` fixed. For unlensed classes `mu = 1`
  deterministically and the mu factor is absent.
- **Density.** `density(z, mu)` returns the normalised product
  `p_z(z) * p_mu(mu)`, used to divide the weight.

Grid resolution is the builder's call; it must be fine enough that the CDF
inversion error is negligible against Monte Carlo error. A few hundred points
per axis is expected to suffice and should be verified, not assumed.

Guarantee to preserve: `density` integrates to 1 over the sampling domain, and
`sample(n)` reproduces `density` — both directly testable.

### `cmsne/multicolour.py` — `generate_many`

Takes the proposal and draws from it instead of uniformly. Each returned event
records the proposal density it was drawn from (`e['proposal_density']`), so
downstream weight correction never has to re-derive it.

### `cmsne/rates.py` — `absolute_rate_weight`

The `n_attempted` branch currently computes

```python
v_sample = delta_z * delta_mu   # uniform-proposal volume
return (v_sample / n_attempted) * density
```

`delta_z * delta_mu` is exactly the reciprocal of a uniform proposal density.
Generalised, the Monte Carlo weight is `density / (n_attempted * p_proposal)`,
with the uniform case recovered when `p_proposal = 1/(delta_z * delta_mu)`.
The parameter carrying this must be explicit rather than inferred, so a caller
that forgets it fails loudly instead of silently reverting to uniform.

### `cmsne/importance_sampling.py` — `generate_with_boosting`

`p_uniform_const` assumes a uniform pilot. It becomes the pilot proposal's
actual density, evaluated per event. The existing mixture correction
(`f_pilot + p_boost_total / p_pilot`) is otherwise structurally unchanged — it
already handles a non-trivial pilot density correctly once that density is
supplied per event rather than as a constant.

The boost stage keeps working as-is: it targets cells still sparse *after* the
matched proposal, which is the right division of labour. Fewer cells should be
flagged; that is a signal of success, not a regression.

### Production scripts

`run_dual_schema.py`, `run_multicolour.py` and `run_trajectory.py` construct
the proposal per class and pass it through. Their `CLASSES` lists are
deliberately identical and must stay so.

## Testing and validation plan

Unit (in `tests/test_cmsne.py`, which runs via `python tests/test_cmsne.py` in
`cmsne-env`, not pytest):

- `density` integrates to 1 over the domain, lensed and unlensed.
- Sampling reproduces the density: a large draw's histogram matches `density`
  within tolerance; a draw from the `mu` proposal matches `mu^-3`.
- The inverse CDF is monotonic and spans the full range.
- `absolute_rate_weight` with a uniform proposal density reproduces today's
  value exactly — this keeps the two existing tests meaningful rather than
  deleting them.
- `generate_with_boosting(pilot_frac=1.0)` returns weights equal to
  `rate_weight_fn / p_proposal`, the matched analogue of today's
  "matches bare rate_weight" test.

Two existing tests encode the uniform assumption and must be **updated, not
removed** (`tests/test_cmsne.py:2449, 2457`): `absolute_rate_weight MC scales
by delta_z / N` and `MC lensed scales by delta_z * delta_mu / N`. They should
be reframed as the uniform special case of the general formula.

Validation (BlueBEAR, sbatch only):

- A small matched run, compared against `results/dual_59297152` on
  **rate-weighted totals** — annual detection rate per class, TP/yr and purity
  at the 10% budget. These must agree within error. A shift outside error bars
  blocks the change.
- `scripts/diag_calib_resolution.py` re-run on the matched output to measure
  the realised ESS gain per class, against the table above as the baseline.

## Expected outcome, and what it does not solve

Signal ESS/raw should move from 0.6–2.9% toward the tens of percent, which
resolves the CC-subtype thin-sample problem. Background ESS/raw moves from
35.2% toward ~100%, about 2.8x.

**This does not by itself unlock either strict FPR budget.** The FPR floor is
set by calib *background* ESS. At ~100% efficiency the current pool gives calib
ESS ~5,276 and a floor of `10/5276` = 0.19% FPR. Therefore:

- **0.1% budget:** needs this change **plus** roughly 2x more background
  generation (down from 5.4x without it).
- **0.01% budget:** needs roughly 19x more background (down from 54x).

Those generation runs are follow-up work, out of scope here.

## What is left to the builder

- Grid resolution and interpolation scheme for the CDFs, subject to the
  accuracy check above.
- Whether `RateProposal` caches per `(source, kind)` or is rebuilt per call.
- The exact parameter name and signature carrying proposal density into
  `absolute_rate_weight`, subject to it being explicit and failing loudly.
- Tolerances for the sampling-reproduces-density tests.
