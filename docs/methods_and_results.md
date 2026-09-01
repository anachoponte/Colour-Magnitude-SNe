# Colour as an early trigger for strongly lensed Type Ia supernovae in LSST

**Methods and results.** This document describes the method implemented in the `cmsne`
package and the results of the production runs on the Rubin/LSST OpSim baselines. The
guiding question is deliberately narrow:

> Can **colour alone** — with no host photometric redshift and no spectroscopy — act as
> an **early trigger** that flags a strongly lensed SN Ia from its first-detection
> photometry, before the system reveals itself as multiply imaged?

Everything below is built around that constraint. Redshift-based features (comparing an
observed magnitude to the expected unlensed magnitude at a known redshift) are powerful
but **out of scope by construction**: using them would defeat the purpose of a
photometry-only trigger.

---

## 1. Motivation

Strongly lensed SNe Ia are rare (LSST is expected to find of order ~10 per year;
Wojtak et al. 2019; Goldstein et al. 2019) but scientifically valuable: the multiple
images and their time delays measure H₀ and probe the lens mass distribution. Their
value is highest if the *trailing* images can be caught, which requires flagging the
system early — ideally from the first image, before the delayed image appears. A
lensed SN Ia is magnified but otherwise a normal SN Ia, so magnification alone
(achromatic) leaves no colour signature. What colour *can* do is **type** the transient
— separate a SN Ia from the core-collapse background — cheaply and early, from whatever
bands the survey happens to have measured.

## 2. Methods

### 2.1 Transient populations and the lensing model

Five source classes are simulated: SN Ia (SALT3) and four core-collapse templates
(`nugent-sn1bc`, `nugent-sn2l`, `nugent-sn2n`, `nugent-sn2p`), each in an *unlensed*
and a *strongly lensed* variant — ten populations in total. Peak absolute magnitudes
and scatters for the CC templates are set from `cmsne.lsst.contaminant_info`.

Strong lensing is applied with a four-component magnification model
(`cmsne.colour_magnitude.time_delay`): galaxy-, group-, and cluster-scale lenses use a
`(μ/4)^-3` magnification-dependent time-delay prefactor, and a fourth lens population
uses `(μ/4)^-1`. The lensing optical depth scales as `(d/31 Gpc)³ · μ^-2`, so
high-magnification, signal-like configurations are strongly suppressed. Frame
conventions are handled explicitly: observer-frame observation times are divided by
`(1+z)` to rest frame where the model requires it, while magnitudes are never
redshift-divided.

### 2.2 Volumetric rates and cross-class weighting

Every event enters all rate-weighted statistics with a weight equal to the integral of
its class's volumetric rate over the sampled redshift (and, for lensed classes,
magnification) range. The SN Ia rate follows a Madau star-formation history convolved
with a delay-time distribution; the CC rate tracks the SFH directly
(`cmsne.sn_rates`).

The Ia and CC rate grids carry independent normalisations, so a single constant sets
the **cross-class balance**. This is anchored to the local literature:

- SN Ia volumetric rate ~2.4 × 10⁻⁵ yr⁻¹ Mpc⁻³ (Frohmaier et al. 2019);
- core-collapse rate ~7 × 10⁻⁵ yr⁻¹ Mpc⁻³ (Li et al. 2011; Taylor et al. 2014),

i.e. **CC:Ia ≈ 3:1** locally. The code's own z≈0.05 weight ratio is ~0.86, so
`CC_TO_IA_RATE = 3.5` (≈ 3 / 0.86) scales it to the literature value; its redshift
evolution then follows the SFH (CC) and DTD (Ia) shapes. The four CC sub-types are
further weighted by realistic volumetric fractions (`CC_FRACTIONS`: Ib/c 0.36,
II-P 0.55, II-n 0.06, II-L 0.03; Li et al. 2011; Shivvers et al. 2017) rather than
treated as one rate shape — so the luminous but rare SN IIn does not carry the same
weight as the common SN IIP.

### 2.3 Survey simulation and detection

Light curves are realised through a real LSST cadence using OpSim visit histories
(via `opsimsummaryv2`), with per-band single-visit limiting magnitudes as the detection
threshold (`cmsne.lsst.mask`). The baseline used for the production results is
**v5.3.2**; **v5.3.5** is used as an independent robustness check (§3.7). An event is
"detected" when it clears the threshold in at least one visit; multi-band photometry is
recorded at the **trigger (first-detection) epoch** and at a short evolution baseline
thereafter (`cmsne.multicolour`).

### 2.4 Colour features

The feature vector is the five adjacent-band colours **u−g, g−r, r−i, i−z, z−y**,
optionally augmented by their short-baseline evolution and by the apparent peak
magnitude. Crucially, at the first-detection epoch a real cadence rarely delivers a
*specific* colour: a given pair such as g−r is present only ~9% of the time, and ~48%
of detected events have just one band (no colour at all). The features are therefore
treated as **missing-data-native** — the method uses whatever bands are present rather
than committing to one fixed colour.

### 2.5 Classifier and decision boundary

`cmsne.classifier.ColourClassifier` wraps a `HistGradientBoostingClassifier`
(NaN-native, so missing colours need no imputation) on the colour vector, rate-weighted,
with the signal up-weighted to a balanced training prior. Scores are calibrated with
isotonic `CalibratedClassifierCV`, and a decision threshold is pinned to a target
rate-weighted contamination (default 10%). The classifier exposes a calibrated
`probability(events, prior=…)` (which folds in the true base rate via a likelihood-ratio
update) and a `rank()` interface, reflecting the finding (§3.1) that the lensed/unlensed
transition is a **gradient**, not a sharp line — candidates should be ranked, not hard-cut.

### 2.6 Evaluation metrics

The primary metric is **recovery at fixed contamination**: the rate-weighted fraction of
lensed SNe Ia retained when only 1 survivor in 10 is a contaminant (recovery @10%
false-positive rate). We also report ROC-AUC, the **colour-first fraction** (does the
colour trigger fire before the second image becomes detectable, i.e. before one lensing
time delay), the median lead time, and — at the true base rate — the **purity** and
**follow-up cost** (candidates to vet per genuine lensed Ia).

## 3. Results

### 3.1 The decision boundary is a gradient, not a line

Comparing six boundary families against the rate-weighted background (signal = lensed
SN Ia; background = unlensed Ia + all CC, lensed and unlensed), the current straight
line recovers ~63% of lensed SNe Ia at 10% contamination, while a KDE density-ratio,
a decision tree, and even a simple per-magnitude step or sloped step all reach ~88–89%.
Any flexible boundary beats the line; the elaborate 2-D tree is barely ahead of a simple
step once the rates are realistic. Diagnostics of the underlying posterior
P(lensed | colour, magnitude) show a smooth ridge: the transition is broad (the
ambiguous 0.2 < P < 0.8 band holds ~11–14% of rate-weighted events for blue pairs but
40% for i−z and 48% for z−y). The recommendation implemented in the package is to
report a calibrated probability and rank candidates, not apply a hard cut.

### 3.2 The detected background is SN-Ia-dominated

Although core-collapse SNe outnumber SNe Ia ~3:1 intrinsically, the **detected**
background is the opposite. Flux-limited detection favours the brighter SNe Ia (seen to
larger volume), and the realistic sub-type mix down-weights the luminous-but-rare
SN IIn. Rate-weighted and restricted to detected events, the contaminant background is

> **~70% unlensed SN Ia, ~30% core-collapse, and lensed core-collapse utterly
> negligible (~10⁻³ %).**

The operational consequence is important: the dominant contaminant is the
**colour-identical unlensed SN Ia**, which magnification cannot distinguish by colour.
Multi-colour therefore earns its keep in two ways — cleanly rejecting the ~30%
core-collapse, and exploiting the mild reddening and extra brightness of the
higher-redshift lensed Ia — while the lensing verdict itself is deferred to a later
stage (light-curve modelling, imaging for multiple images, spectroscopy).

### 3.3 Under a real cadence, multi-colour is essential

Idealised (every band at peak), colour-only recovery of lensed SNe Ia rises from ~0.40
(one colour) to ~0.98 (all five ugrizy colours); five colours alone match
colour+magnitude. Under a **real cadence at first detection**, where a specific colour
is usually missing, the picture is more sobering but the ranking is the same:

| feature set                | recovery @10% contamination |
|----------------------------|:---------------------------:|
| 1 fixed colour (g−r)       | 0.11 |
| all 5 colours              | 0.32 |
| all 5 + colour evolution   | 0.39 |
| all 5 + peak magnitude     | 0.41 |
| all 5 + evolution + mag    | **0.48** |

The value of multi-colour in practice is **band redundancy**: a single fixed colour is
missing nine times out of ten at first light, so the trigger must use the bands it
happens to have. A first-detection trigger sees only ~2 bands, which caps recovery
near 0.5.

### 3.4 Waiting a few weeks lifts recovery, then plateaus

Waiting after first detection to accumulate more bands roughly doubles recovery and
then flattens: colour-only recovery goes 0.28 (wait 3 d) → 0.47 (30 d) → 0.49 (60 d),
and with peak magnitude reaches ~0.56 by 60 d. The yield of lensed SNe Ia with at least
one colour rises from 31% (3 d) to 62% (30 d) and plateaus near 65% — about a third of
lensed SNe Ia are only ever seen in a single band, an irreducible cadence limit. The
practical sweet spot is **~2–4 weeks** after first light.

### 3.5 Colour-first: the trigger beats the second image

Packaging the features into the actual identifier and asking the science question — does
the colour trigger fire before the second image becomes detectable (~one time delay
later)? — the multi-colour identifier **flags 56% of lensed SNe Ia**, fires a median of
~3 days after first detection, and **when it flags a system it beats the second image
85% of the time, by a median of 88 days.** That large lead is the argument for a
colour-only trigger: it flags the majority of lensed SNe Ia, early, from photometry
alone.

### 3.6 The base rate is brutal — colour is a pre-filter, not a verdict

At the literature prior (~9 lensed SNe Ia/yr against ~10⁵ detected SNe/yr, i.e.
~1 in 10⁴), even the multi-colour identifier gives sub-0.1% purity at useful recovery:
flagging half the lensed SNe Ia yields a sample only ~0.03% real. What colour *does* buy
is a drastic cut in the follow-up list — from ~10,000 detected SNe per genuine lensed Ia
(no filter) to a few hundred at tight recovery (~40×). Colour is a strong **prioritiser**
feeding a second stage, and should be scored by purity / follow-up cost at the true
prior, not by balanced accuracy.

### 3.7 Robust across OpSim baselines

Re-running the entire calibrated pipeline on an independent OpSim baseline (**v5.3.5**,
~201k detected events vs ~199k for v5.3.2) reproduces every headline within noise, if
anything marginally better on v5.3.5:

| metric                          | v5.3.2 | v5.3.5 |
|---------------------------------|:------:|:------:|
| detected background (uIa / CC)  | 70 / 30 | 70 / 30 |
| recovery @10% (5 colours)       | 0.319  | 0.348  |
| recovery @10% (best, +evo+mag)  | 0.481  | 0.518  |
| colour-first flagged            | 56%    | 58%    |
| beats 2nd image                 | 85%    | 86%    |
| median lead                     | 88 d   | 92 d   |

The colour-only trigger does not hinge on one particular cadence realisation.

### 3.8 Which brightness feature? Cross-band brightest beats a fitted peak

The colour vector is optionally augmented by a single brightness feature. We recorded
three candidates for every event (`cmsne.multicolour.event_multicolour`) and compared
them as the magnitude input, calibrated throughout:

- **first-detected** — the magnitude of the earliest detection (available soonest, but
  the faintest and noisiest; median lensed-Ia magnitude 21.76);
- **fitted peak** — a parabola fitted to the best-sampled band, interpolating the true
  peak between visits (`cmsne.lightcurve.fitted_peak_magnitude`; median 20.77);
- **brightest observed** — the brightest sampled point across *all* bands (median 20.42).

| magnitude feature       | recovery, 5 colours | recovery, 5 colours + evolution |
|-------------------------|:-------------------:|:-------------------------------:|
| *(none)*                | 0.319               | —     |
| first-detected          | 0.365               | 0.440 |
| fitted peak             | 0.384               | 0.462 |
| **brightest observed**  | **0.406**           | **0.481** |

Any brightness proxy helps (recovery rises from 0.319), and the fitted peak clearly
beats the first-detected magnitude — but it does **not** beat the brightest-observed
sample, which is the best of the three. The single-band parabola fit trades away
cross-band information: the brightest-observed feature captures the brightest point in
*whichever* band the cadence happened to sample, whereas the fit is confined to one band
and so lands ~0.35 mag fainter on average. The production pipeline therefore keeps
**brightest observed** as the default magnitude feature (`peakmag`); `peakmag_fit` and
`peakmag_firstdet` are recorded alongside it for this comparison.

## 4. Limitations and future work

- **The magnitude feature is settled (not an open item).** An earlier draft flagged the
  brightness input as a first-detected placeholder to be replaced by a fitted peak. On
  test (§3.8) the pipeline was already using the *brightest-observed* magnitude, and that
  beats both the first-detected value and a single-band fitted peak, so no change is
  warranted. A multi-band SED-aware peak fit could be revisited but is unlikely to help.
- **Train/eval realism.** The boundary-shape study (§3.1) is fit on idealised model light
  curves; the multi-colour results (§3.2–3.7) put every class through a real cadence.
- **Achromatic degeneracy is fundamental.** No colour feature can separate a lensed from
  an unlensed SN Ia; the ~70% unlensed-Ia background is irreducible by colour and sets
  the ceiling. The lensing decision must come from brightness, multiplicity, or the time
  delay at a later stage.

## 5. Summary

Framed correctly — lensed SN Ia against the full, rate-weighted, *detected* background —
colour alone is a fast, cadence-robust **early trigger**: it flags ~56% of lensed SNe Ia
a median of ~3 days after first detection and beats the trailing image ~85% of the time
by ~88 days. It is a strong pre-filter (cutting the follow-up list ~40×) but not a
standalone identifier: at the true rarity of lensed SNe the flagged sample is sub-percent
pure, so the colour trigger's role is to hand a short, early candidate list to a second,
lensing-aware stage.

---

### Reproducibility

- Rate calibration: `cmsne/lsst.py` (`CC_TO_IA_RATE`, `CC_FRACTIONS`).
- Classifier and boundary: `cmsne/classifier.py`, `cmsne/colour_magnitude.py`.
- Multi-colour cadence photometry: `cmsne/multicolour.py`; drivers in `scripts/run_multicolour*`.
- Production data: BlueBEAR array 52264909 (boundary study) and multicolour arrays
  52585403 / 52639521 on OpSim v5.3.2; robustness array 52781668 on v5.3.5.

*Rate references: Frohmaier et al. 2019 (SN Ia rate); Li et al. 2011, Taylor et al. 2014
(CC rate); Li et al. 2011, Shivvers et al. 2017 (CC sub-type fractions). Lensed SN Ia
yield: Wojtak et al. 2019, Goldstein et al. 2019.*
