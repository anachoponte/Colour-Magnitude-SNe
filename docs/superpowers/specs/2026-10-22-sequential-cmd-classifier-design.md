# Sequential colour-magnitude classifier: design spec

Date: 2026-10-22
Status: approved for implementation planning

## Goal

Build a time-domain companion to the existing static-feature `ColourClassifier`
([[colour-classifier-findings]]): instead of scoring a candidate once from a
fixed two-epoch feature vector (trigger, trigger+15d), maintain an *updating*
P(lensed SN Ia) estimate as new LSST epochs arrive for a candidate, derived
from colour-magnitude distributions that evolve over time and across the ten
ugrizy band pairs.

This is explicitly a **parallel research pipeline**, not a replacement. It
will be benchmarked against the existing classifier on the same metrics
already established in the project, and the result (not an a-priori
architectural decision) determines which approach the project keeps using.

## Scope (decided 2026-10-22)

- **Signal class:** lensed SN Ia only. No other lensed transient types.
- **Background/contaminant class:** SNe only (Ia + core-collapse, the existing
  rate-calibrated composition: `CC_FRACTIONS` × `CC_TO_IA_RATE` in `lsst.py`).
  Non-SN contaminants (AGN, TDEs, novae, variable stars) are an explicit
  non-goal for this design.
- **Colour-only constraint carries over** from the existing project scope: no
  host photo-z/spec-z, no explicit light-curve template fitting.
- Relation to `ColourClassifier`: build alongside it, reusing its machinery
  (NaN-native feature matrix, calibrated GBT, contamination-pinned threshold)
  rather than discarding it.

## Resolved methodology questions

These three questions were open at the start of this design and are resolved
as follows; see conversation log 2026-10-22 for the full reasoning.

### 1. Unknown phase of a newly discovered transient

The time axis for every distribution in this system is **Δt = days since the
candidate's first LSST detection (trigger epoch)** — never days since true
explosion, which is unobservable for a real candidate. This matches the
convention already used by `multicolour.py`'s `evolution_days` feature.

Background and signal CMD distributions are generated on this same
observer-anchored axis from realistic cadence simulation. Because the
simulated population spans the full realistic range of "true phase at
trigger," the unknown-phase nuisance is marginalized *by construction* during
population generation — no explicit phase-marginalization step is needed when
scoring a live candidate.

Rejected alternative: explicit latent-phase marginalization via light-curve
template fitting (à la PSNID/SNANA). More statistically rigorous, but
reconstructs full light-curve fitting, which conflicts with the project's
colour-only, no-fit scope constraint. Not pursued in this design.

### 2. Non-independence of sequential observations

Discrete CMD distributions at fixed Δt-since-trigger steps are a sensible
*training construct* — the existing two-point evolution feature is already
exactly this with two steps. They must **not** be combined by multiplying
independent per-step posteriors/likelihood ratios (naive sequential Bayes),
because colour at nearby epochs is strongly correlated (same object, smoothly
evolving) and naive multiplication double-counts that correlation, producing
systematically overconfident posteriors.

Resolution: generalize the existing cumulative-feature-vector approach. At
each new epoch, append the new (colour, magnitude) snapshot, tagged by its
Δt, to a single NaN-native feature vector, and re-score with **one** jointly
trained classifier — never chain independent per-epoch updates. The outward
behaviour ("probability updates as observations accumulate") is preserved;
the mechanism is re-scoring a longer feature vector, not multiplying
probabilities.

A naive independent-multiplication baseline is still built, but only as a
deliberately mislabeled comparison point to quantify how overconfident that
wrong approach is (see Comparison harness, below) — it is never used to make
a real classification decision.

Stretch goal (explicitly out of scope for v1): an explicit Markov
transition-density model, P(x_t | x_{t-1}, class), would be the
textbook-correct sequential filter and should be reconsidered if the
cumulative-feature-vector approach hits a ceiling. Not pursued now.

### 3. Time-separation constraint for a "colour," and background/candidate consistency

Reuse the existing `window` convention (5 observer-days) from
`multicolour.band_photometry` / `event_multicolour`, unchanged, as a single
shared constant consumed by **both** the training-population generator and
the live candidate scorer. The two must never use independently-chosen
windows. Re-tuning this value is an explicit non-goal for v1; it is flagged
as a future sensitivity check only.

## Architecture

### A. Trajectory generation layer (`cmsne/trajectory.py`, new)

Extends `multicolour.py`'s two-point generator (trigger, trigger+15d) to a
fixed N-point canonical Δt grid. Default grid: **0d / 3d / 14d / 30d / 60d**
since trigger — reusing the exact checkpoints already validated in the
waiting-time study ([[colour-classifier-findings]]) so new results stay
comparable to existing findings, rather than inventing a new grid.

- Same light-curve simulation as `multicolour.py`; same `window=5d` pairing
  logic at every grid step, for every class.
- Output per event: a trajectory of up to 5 `(ugrizy colour vector, peakmag)`
  snapshots, NaN where the cadence didn't deliver a band within the window at
  that step.
- Run once over the SN Ia + CC background (rate-weighted as today), once over
  lensed SN Ia signal — same generator, same rate-weighting
  (`CC_FRACTIONS`/`CC_TO_IA_RATE`), just producing N-point trajectories
  instead of 2-point ones.

### B. CMD contour layer (diagnostic)

At each of the 5 Δt steps and each of the 10 band pairs, compute a continuous
P(lensed | colour, mag) surface, generalizing the existing single-epoch
gradient-boundary study (scratchpad `gradual_boundary.py`,
[[colour-classifier-findings]] "boundary is a gradient, not a line") across
the Δt grid. This is the literal "evolving probability density contour"
artifact from the original goal statement, and doubles as an independent
visualization/sanity-check layer — it does not feed the classifier directly.

### C. Trajectory classifier (`cmsne/trajectory.py` or a new
`cmsne/trajectory_classifier.py`)

Generalizes `ColourClassifier`'s feature matrix from today's 2-point
(trigger, trigger+15d) vector to the full 5-step, NaN-native vector (colour +
peakmag at each of the 5 Δt tags), reusing the same
`HistGradientBoostingClassifier` + isotonic calibration + contamination-pinned
threshold machinery. Trained once on the correlated N-point trajectories.

"Updating as observations arrive" = re-scoring this one trained model as more
of the vector fills in with real observations — resolving question 2 above
by construction (the model learns the true joint/correlated distribution
from training data, rather than assuming independence).

### D. Comparison harness

Benchmarks three approaches side by side on the same held-out trajectories,
using the metrics already established in the project (recovery @ 10% FPR,
colour-first lead time vs 2nd image, calibration/reliability):

1. Today's static 2-point `ColourClassifier` (unchanged, baseline of record).
2. The new 5-step trajectory classifier (Section C).
3. A deliberately labeled naive independent-multiplication baseline (chains
   per-step posteriors via odds multiplication) — included only to quantify
   how much that wrong approach overstates confidence relative to (2).

The harness produces the headline comparison numbers this design exists to
get: does the trajectory classifier actually recover more lensed Ia / trigger
earlier / stay better calibrated than the static classifier, and how badly
wrong is naive sequential multiplication by comparison.

## Non-goals (explicit, for this design)

- Non-SN contaminant classes (AGN, TDE, nova, variable stars).
- Lensed transient types other than SN Ia.
- Explicit latent-phase marginalization / light-curve template fitting.
- Markov transition-density sequential filtering (stretch goal only).
- Re-tuning the 5-day colour-pairing window.
- Replacing or deprecating the existing static `ColourClassifier` — that
  decision is deferred to the comparison harness's results.

## Testing approach

Follow the project's existing self-contained-synthetic-sample test philosophy
(`tests/test_cmsne.py`): extend with synthetic trajectory fixtures (small N,
deterministic) covering trajectory generation (correct Δt tagging, NaN
handling when cadence misses a step), the trajectory classifier (fit/score/
probability/rank, matching `ColourClassifier`'s existing test pattern), and
the naive-multiplication baseline (a pure function, easy to unit test against
hand-computed odds).

## Next step

Hand this spec to the writing-plans skill to produce an implementation plan.
