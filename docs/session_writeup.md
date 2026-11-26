# Session Writeup: Rate-Weighted Diagnostics, Purity Trade-offs, Visual Encodings, and Full-Trajectory Classifier Upgrades

**Dataset:** `results/dual_59297152` on BlueBEAR (`1,000,000` simulated LSST supernovae across 10 classes: `100,000` per class for `lensed_Ia`, `lensed_sn1bc`, `lensed_sn2l`, `lensed_sn2n`, `lensed_sn2p` and their 5 `unlensed_*` counterparts, evaluated on a held-out 70/30 train/test split with physical annual LSST rate weights). **Findings 7-8 use a different dataset:** `results/brute15_62368493` (`1.5M` attempts per class, OpSim `baseline_v5.3.5_10yrs.db`, three-way train/calib/val split, annual rates derived from the simulation). Their absolute rates are not comparable to Findings 1-6.

---

## 1. Executive Summary

Across this session, we audited, visualised, and substantially upgraded the LSST gravitationally lensed supernova classification pipeline:

1. **Removed Cross-Subtype Lensed Contamination:** Fixed both `scripts/plot_roc.py` and `scripts/plot_evolving_boundaries.py` so that when evaluating an individual lensed subtype (such as `lensed_Ia`), other lensed supernova subtypes are **never** treated as false-positive background contaminants (`background = unlensed_*` strictly).
2. **Quantified the LSST Base-Rate Trap & Zero-False-Positive Sweet Spot:** Demonstrated how the `1 : 10,000` (`10/yr` lensed Ia vs `100,000/yr` unlensed -- the literature figure already cited elsewhere in this project, Oguri et al. 2010; Wojtak et al. 2019; Goldstein et al. 2019, not the uncited `50/yr` and `1,000,000/yr` an earlier draft of this document used) and `~1 : 2,200` (`45/yr` all lensed vs `100,000/yr` unlensed -- an order-of-magnitude placeholder pending [issue #43](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/43), not an independently literature-validated figure) cosmic rate ratios affect real telescope follow-up purity, and mapped the `TP / FP` ratio and `FPR = 0` "Zero-False-Positive Sweet Spot".
3. **Resolved the Scatter-vs-Heatmap Visual Paradox (`size + opacity` + High-Contrast Palette + `1σ` Error Bars):** Identified why unweighted scatter plots previously appeared inconsistent with rate-weighted probability surfaces, compared 4 visual rate-weight encodings, adopted **`size + opacity` (`size_alpha`)** with a high-contrast **`PuBu` + vivid marker** palette and **`1σ` photometric error bars**, and regenerated 208 standard boundary figures across all filter pairs (`ug, gr, ri, iz, zy`) and subtypes.
4. **Benchmarked Improvements & Non-Tree Replacements for the Full-Trajectory Classifier (`~4.1x`/`>7x` claims RETIRED -- see point 6):** Built a 6-model benchmark testing smooth Sigmoid calibration, physics-enriched trajectory features (snapshot magnitudes, time-dilated fade from peak, broad-baseline colours, photometric uncertainties), causal prefix training for early `+0d .. +14d` alerts, Smooth Spline GAMs, Mask-Aware Neural MLP ensembles, and a **Hybrid (Physics-Enriched GBDT + Mask-Aware Smooth MLP)** ensemble. This benchmark originally reported the Hybrid ensemble's zero-false-positive yield as **2.3 lensed Ia/yr (23.3% recall)**, a claimed `~4.1x`/`>7x` improvement over two different baselines. **Once issue #50's calibration discount was applied to this benchmark's own models too (point 6), every one of the 6 architectures' zero-FP yield collapsed to `0.0/yr` for both targets** -- the original `0.2-2.3/yr` figures were themselves an uncorrected-overconfidence artifact, the same pathology issue #50 found in the real deployed classifier. There is no longer a meaningful multiplier to quote; see Finding 4's revised table. The AUC comparison is still informative but also shifted: the early-epoch advantage of the physics-enriched features survives the correction (`lensed_Ia +0d` AUC `0.764 -> 0.817`), but their apparent advantage *by* `+60d` does not (`0.905 -> 0.894`) -- see Finding 4's dimensionality caveat before reading that reversal as a clean result.
5. **Found and Fixed a Look-Ahead ("Clairvoyance") Bug in the Purity/Follow-Up Analysis:** Discovered that `investigate_purity_tradeoff.py` scored every test event on its full, fully-realised light curve -- however many of the 5 trajectory epochs (`+0d, +3d, +14d, +30d, +60d`) it happened to have been detected at over the *entire* 3-year survey -- rather than at a single, causally-consistent decision epoch. This meant every previously-reported "detection rate" implicitly assumed a broker could wait up to 60 days after a trigger before deciding whether to allocate spectroscopic follow-up, which defeats the purpose of early alerting on fast, highly-magnified transients. **Fixed `investigate_purity_tradeoff.py` itself** to use the codebase's own pre-existing (but previously unapplied here) causal machinery, `cmsne.trajectory.reveal_up_to`/`TrajectoryClassifier.roc_curve_by_step` (already used correctly in `scripts/plot_roc.py`/`scripts/compare_trajectory_models.py`'s AUC-by-epoch columns) -- it now fits one classifier and reports purity/recovery separately at each of the 5 decision epochs, never mixing them into one curve -- see Finding 2's revised table. Filed and fixed as [issue #51](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/51).
6. **Found and Fixed `score()`'s Missing Calibration Discount, Which Was Masking Widespread Overconfidence ([issue #50](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/50)):** The existing local-density ("KDE blend") calibration discount -- meant to down-weight scores wherever an event's missing-data pattern has thin local training support -- was wired into `probability()` only; every operational consumer (`predict()`, `roc_curve()`, `recovery_rate()`, and `fit()`'s own threshold search) calls `score()`, which never applied it. Folding the blend into `score()` itself (see Finding 2's revised table, now rerun with this fix on top of #51) shows the effect was much larger than "a few sparse events slipping through": the zero-FP "pure" tier -- previously `0.56`/`yr` (`5.6%`) for `lensed_Ia` at `+60d` -- collapses to `0.03/yr` (`0.31%`), and even the standard `10%`-FPR operating point's annual yield, previously rising with epoch count (`3.9 -> 7.8/yr` lensed Ia from `+0d` to `+60d`), is now roughly flat at `4.0-4.8/yr` across all 5 decision epochs. In other words, a large share of the project's previously-reported detection rate -- at *every* decision epoch, not only the early/sparse ones -- was the classifier being confidently wrong about events whose specific missing-data pattern (plausibly compounded by [issue #49](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/49)'s `lensed_Ia`-specific `NaN` dropout) was never well-populated in training. Verified the fix is *selective*, not a blanket penalty: a genuinely well-supported sparse event (`z=1.33`, `mu=48.1`, only 1 of 5 epochs observed) still scores almost unchanged (`0.9455` vs `0.9451` pre-fix). Getting this fix to run at production scale (thousands of distinct missing-data patterns) surfaced two further bugs: an unbounded per-pattern KDE cache that OOM'd at 8GB (fixed with an LRU cache), and a latent `effective_sample_size` overflow (`inf/inf -> NaN`) only ever exercised once `score()`'s new full-scale usage hit an extreme weight-rescaling edge case in `_weighted_subsample`. **Applying the same discount to Finding 4's own 6-model benchmark (a model-agnostic, vectorized port, since that benchmark doesn't use the real classifier at all) collapsed every one of its models' zero-FP yield to `0.0/yr` too -- confirming this wasn't a quirk of the real classifier specifically: none of the 7 architectures tested in this project show genuine zero-false-positive discrimination at this survey scale once thin-local-support overconfidence is corrected for. See Finding 4's revised table and its dimensionality caveat.**
7. **Reframed the Classifier Around Ranked Follow-Up at a Tunable FPR Budget, and Characterized What's Actually Contaminating Each Budget (`investigate_contaminant_properties.py`):** With zero-false-positive discrimination retired across all 7 tested architectures (points 4, 6), this classifier's real value is as a **ranked candidate list at a chosen FPR budget** -- `rank()`/`probability()`, already this codebase's own recommended API over a hard `predict()` cut -- not a zero-contamination filter. A real, non-zero operating point still exists and is worth stating plainly: e.g. `lensed_Ia` at the 10%-FPR budget yields `4.0-4.8/yr` (Finding 2). This session investigated what's actually leaking through at each budget and whether a cheap secondary cut can clean it up further. "Host galaxy type" was the originally proposed secondary cut, but it does not exist anywhere in this project's simulated dataset, and using it in the trigger itself would contradict this project's explicit colour-only, no-host-redshift design (`docs/methods_and_results.md`). Redshift `z` and magnification `mu` **are** simulated for every event but deliberately excluded from the classifier's own features for that same reason, making them a legitimate downstream triage cut -- after the photometric trigger has already fired, not part of it. Measured results (rate-weighted -- an earlier, unweighted version of this analysis materially overstated some of these numbers and is corrected in Finding 5): false-positive composition is dominated by `unlensed_Ia` at realistic budgets (`66.6%` of FPs at `lensed_Ia`'s `10%`-FPR/`+60d` operating point, rising to `71.4%`/`99.1%` at tighter budgets) -- the expected colour-twin confusion, since a lensed Ia is an unlensed Ia's colour profile at higher apparent brightness -- and redshift cleanly separates true from false positives even though it plays no role in the classifier's own features (magnification lets true positives reach higher `z` than an unmagnified false positive could). A secondary redshift-window cut, calibrated on each operating point's own true-positive `z`-range, delivers measured (not projected) purity gains at near-zero recall cost: e.g. `lensed_Ia` at `1%`-FPR/`+3d` removes `64.7%` of residual false-positive weight for a `4.9%` true-positive cost (purity `0.213% -> 0.570%`). Extending the investigation to all 4 individual lensed CC subtypes (not just the `all_lensed` aggregate) shows this `unlensed_Ia` colour-twin confusion is not specific to the `lensed_Ia` target: it is the single largest contaminant for `lensed_sn1bc`, `lensed_sn2l`, and `lensed_sn2p` too, and a close second for `lensed_sn2n` and `all_lensed`. Whether false positives are photometrically noisier than true positives turned out NOT to be a consistent pattern once properly weighted (retracted, not merely revised -- see Finding 5). The cut's gain also largely survives an independent out-of-sample check (`validate_zcut_outofsample.py`: a proper train/calib/val split, plus a stress test against realistic host-photo-z measurement noise, since that -- not the true redshift -- is what would actually be available in deployment). See Finding 5 for the full results and caveats.

8. **Built the Ranked Candidate List the Previous Point Recommends, and Found Its Real Limit Is Simulation Sampling, Not the Classifier (`rank_candidates.py`):** Point 7 argued the classifier's value is a ranked list at a tunable FPR budget; `scripts/rank_candidates.py` builds it, on a proper three-way `train`/`calib`/`val` split, for all 6 targets (BlueBEAR job `62176559`). Two things came out of it. First, the **lift curve, not the FPR-budget table, is the deliverable**: a follow-up programme has a number of nights, not an FPR budget, and "if I review `N` candidates a year, how many lensed SNe do I get?" needs only the ordering -- no threshold calibration at all. Rate-weighted, `lensed_Ia` reviewing `10,000` candidates/yr (`10%` of the alert stream) recovers `~4/yr` of the `10/yr` that exist, and the ranking concentrates `~63x` more lensed Ia into the top `100/yr` than random selection would. `all_lensed` is far weaker at the sharp end (`1.6-9.8x` vs `38-83x`), because aggregating the four CC subtypes dilutes the top of the list with the ones the classifier separates worst -- for a small follow-up budget, targeting `lensed_Ia` specifically is the better configuration. Second, **of the 120 cumulative operating points (6 targets x 5 epochs x 4 budgets), the 38 that are statistically clean are *all* at the `10%` or `1%` budget; not one clean operating point exists at `0.1%` or `0.01%`, for any target at any epoch.** The script now reports this honestly in two distinct forms rather than hiding it -- `NOT RESOLVABLE` where the `calib` background ESS (`~1,856`, against a `0.54%` true FPR floor) cannot place the threshold at all, and `THIN SAMPLE` where it can but the yield past it rests on too few effective events (flagged if **any** of `n_tp`, `n_fp`, `tp_ess`, `fp_ess` falls below `MIN_SAMPLE_FOR_STATS = 10`). An earlier version snapped to the nearest achievable ROC point via `np.argmin(|fpr - target|)` and reported it as though the budget had been met; that was a reporting bug, fixed. The root cause is **importance-sampling proposal mismatch, not a classifier limit** -- every population is drawn uniformly in `(z, mu)` against a `mu^-3` target, giving `0.6-2.9%` ESS/raw on signal pools with the top `1%` of events carrying `55-62%` of all weight ([issue #37](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/37)). This is **variance, not bias**: every number in Findings 2, 5 and 6 is unbiased, just noisier than its raw event count suggests. Two throwaway spikes measured the fix: a proposal matched to the **rate density** -- the intuitive answer -- *loses* for 8 of 10 classes including the background pool that gates the FPR budgets, because it concentrates draws where events are too faint to detect; the variance-optimal `p ∝ rate * sqrt(eps)` (Cauchy-Schwarz, `eps` = detection efficiency) wins for all 10 by `3.2-60.4x`. The spikes killed an already-written design for the losing proposal before any production code was touched. **Nothing in `cmsne/` has been changed and no number in this document depends on that work landing** -- see Findings 6 and 6a.

9. **The Strict `0.1%` Budget Is Now Measured, Signal ESS Is the Binding Side, and Rates Are Derived From the Simulation (Finding 7):** Point 8 attributed the missing `0.1%`/`0.01%` operating points to importance-sampling proposal mismatch. That was right about the cause and wrong about the cure. The measurable-FPR floor is simply `MIN_SAMPLE_FOR_STATS / ESS_background` (`10/1,856 = 0.54%` on the baseline), and ESS is linear in `N` at fixed proposal. The `p ∝ rate*sqrt(eps)` design that point 8 recommended was **measured and rejected**: its gain at the `0.1%` background tail is `0.75x`, worse than uniform (`scripts/spike_tail_transfer.py`, job `62298093`). Brute force at `15x` (job `62368493`, `1.5M` attempts/class) was analysed once a transient `/rds` `ENOSPC` fault cleared. **`0.1%` is now resolvable for every target at every epoch, and 18 of 25 operating points there are clean, against 0 of 30 in Finding 6.** Background ESS clears the bar everywhere (`>= 10.6`). The 7 thin points are all thin on the *signal* side, in `lensed_sn2n`, `lensed_sn2p` and `lensed_sn2l`. Annual rates are now **derived from the simulation's own LTT weights rather than hardcoded** (`a11d2e6`, [issue #63](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/63)): `lensed_Ia` `148.8/yr`, `all_lensed` `449.3/yr`, background `804,889/yr`. These replace the `10/yr`/`45/yr`/`100,000/yr` placeholders of points 2 and 8 for all 15x numbers. The old table took volumetric unlensed subtype fractions as the observed lensed composition, which got the subtype *ordering* wrong. Corrected `lensed_Ia` purity at `0.1%` FPR is `0.49 / 0.65 / 1.98 / 0.76 / 0.36%` across `+0/+3/+14/+30/+60d`, peaking at `8.3` true vs `413` false candidates/yr at `+14d`. `lensed_sn2l` (`0.40/yr`) is a null result at every budget and is reported as such. An ESS-targeted follow-up run with per-class budgets (`max(1, 13/worst_ESS)`) is in progress. Its first submission mis-mapped eight of ten budgets onto the wrong class (fixed in `5d3f600`, corrective jobs running), and **no number here depends on it yet**. `0.01%` is out of scope: it would need a `~55M`-attempt run. The earlier `54x` plan is superseded.

10. **The Most Common Lensed Supernova Is the One the Method Cannot Find (Finding 8):** With derived rates, `lensed_sn2n` is the **dominant lensed class**: `272.2/yr`, `1.83x` `lensed_Ia` and `61%` of all lensed SNe. It is also the class the colour method separates worst. TPR at `10%` FPR is `12-23%` (`21%` at `+30d`, against `lensed_Ia`'s `56%`). Its enrichment over random selection **never exceeds `2.3x` at any budget or epoch**, and is consistent with chance at strict budgets up to `+14d`. `lensed_Ia` reaches `19-107x` at `0.1%`. `0.1%` purity is `4.7-48x` lower than `lensed_Ia`'s depending on epoch (`0.020-0.076%` vs `0.36-1.98%`). This changes the paper's claim from "lensed SNe" to **lensed SNe Ia**, and means `all_lensed` (pending) is dominated by its weakest member. The `272.2/yr` rate itself is model-dependent (IIn luminosity function and template) and should be checked before it is quoted as an LSST prediction. The separation result is not rate-dependent.

---

## 2. Summary of All Code Edits & New Scripts

> **Rate-weighting audit across `scripts/`** (same class of bug as [issue #50](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/50)'s contaminant-analysis fix): after finding and fixing the raw-count-vs-rate-weighted bug in `investigate_contaminant_properties.py` (Finding 5), audited the rest of `scripts/` for the same pattern -- a plain `np.median`/`np.mean` applied to an importance-sampled population instead of a weighted statistic. Found and fixed two genuine instances, both in already-committed scripts predating this session: `scripts/run_multicolour.py`'s per-class `median_n_bands` summary (now `cmsne.importance_sampling.weighted_median`; this value has no downstream consumer today, so no published number needed correcting), and two stray `np.median(...)` calls in `scripts/build_notebooks.py`'s "second-image epoch" demonstration cell that had fallen back to an unweighted statistic despite `weighted_quantile`/`weighted_fraction` already being imported and used correctly elsewhere in the very same cell. Everything else checked (`plot_evolving_boundaries.py`'s weight-percentile visual-encoding helper, `compare_trajectory_models.py`'s ensemble-prediction averaging, `build_notebooks.py`'s unweighted toy-demo cell, which has no `weight` field by design) was confirmed not to need weighting.

1. **`scripts/plot_roc.py`**
   - **Change:** Updated `load_population`, `plot_rate_by_subtype`, and `make_rate_gif` to define background strictly as `UNLENSED_CLASSES` (`unlensed_Ia`, `unlensed_sn1bc`, `unlensed_sn2l`, `unlensed_sn2n`, `unlensed_sn2p`), preventing non-target lensed SNe from inflating the false-positive rate in `lensed_Ia` and per-subtype evaluations.
   - **Regenerated Outputs:** All 19 ROC and subtype-rate PNGs and animated GIFs in `docs/figures/` (`roc_overall*.png`, `roc_by_step*.png/gif`, `roc_by_subtype_alllensed.png`, `rate_by_subtype_alllensed.png/gif`).

2. **`scripts/plot_evolving_boundaries.py`**
   - **Subtype target support & clean background:** Added `--target {lensed_ia, all_lensed, lensed_sn1bc, lensed_sn2l, lensed_sn2n, lensed_sn2p}`, always comparing the chosen lensed signal against strictly `unlensed_*` background.
   - **Snapshot vs Matched magnitude modes:** Added `--mag-mode {matched, snapshot}` to support both cumulative-best `peakmag_{b}` and contemporaneous snapshot `{b}_d{t}` y-axes.
   - **Rate-weight visual encoding (`--weight-encoding`):** Added `{none, size, alpha, colour, size_alpha}` (setting `size_alpha` as the new default), mapping `log10(weight)` within the active 99.5%-mass dynamic range to marker area and opacity so high-rate physical populations stand out while ultra-low-weight Monte Carlo tail draws recede proportionally.
   - **High-contrast visual palette:** Replaced the `viridis` heatmap + `tab10` scatter colours (which collided across blue/teal/green/yellow) with a cool-slate `PuBu` probability gradient (`white` at `P = 0` to `deep slate-navy` at `P = 1`, dark dashed `P = 0.5` decision contour) and `HIGH_CONTRAST_CLASS_COLORS` (vivid crimson `#e6194B` stars vs warm amber-gold `#f59e0b` circles in binary mode; distinct saturated warm stars vs dark-outlined circles in 10-class mode).
   - **Photometric `1σ` error bars:** Added `--show-errors` and `--also-plot-errors` (`_draw_error_bars` via `LineCollection`), rendering `{mag}_err` (vertical) and `{colour}_err` (horizontal) crosses scaled by event rate weight (`*_errorbars.png`).

### B. New Analysis, Diagnostic & SLURM Scripts

3. **`scripts/compare_mag_modes.py`** (SLURM wrapper: `scripts/run_compare_mag_modes_bluebear.sh`)
   - Quantitatively benchmarks 2D (`colour + magnitude`) classification in `snapshot` vs `matched` (`peakmag`) mode across `g-r`, `r-i`, and `i-z` at `+0d, +3d, +14d, +30d, +60d`.
4. **`scripts/investigate_purity_tradeoff.py`** (SLURM wrapper: `scripts/run_investigate_purity_bluebear.sh`) -- **fixed for [issue #51](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/51)**
   - Fits one rate-weighted `TrajectoryClassifier` per target (`lensed_ia`: `10/yr`, Oguri 2010; Wojtak 2019; Goldstein 2019; `all_lensed`: `45/yr`, order-of-magnitude placeholder pending issue #43, both against `100,000/yr` unlensed SNe), then re-evaluates annual `TP/yr`, `FP/yr`, sample purity `TP / (TP + FP)`, and the empirical `FPR = 0` pure sweet spot **separately at each of the 5 decision epochs** (`+0d, +3d, +14d, +30d, +60d`) via `cmsne.trajectory.reveal_up_to`, instead of scoring every test event on its full, fully-realised light curve as the pre-fix version did.
5. **`scripts/plot_weight_encodings.py`** (SLURM wrapper: `scripts/run_weight_encodings_bluebear.sh`)
   - Generates 4-panel (`1x4`) and 5-row (`5x4`) comparison figures contrasting `unweighted`, `size`, `opacity`, `colour`, and `size + opacity` encodings on identical data slices.
6. **`scripts/compare_trajectory_models.py`** (SLURM wrapper: `scripts/run_model_comparison_bluebear.sh`)
   - Benchmarks 6 full-trajectory classification architectures across full (`+60d`) and causal partial (`+0d, +3d, +14d, +30d, +60d`) trajectories, outputting 3-panel comparison figures and JSON/CSV metric tables.
   - Added `_VectorizedDensityBlender` (fixed for [issue #50](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/50)): applies the same local-density calibration discount `cmsne.classifier.ColourClassifier.score()` uses, as a model-agnostic, vectorized post-hoc correction to all 6 of this benchmark's own models (none of which use the real classifier) -- see Finding 4's revised table.
7. **Parallel BlueBEAR Plot-Suite Scripts:**
   - `scripts/run_plots_parallel_bluebear.sh`, `scripts/run_standard_size_alpha_bluebear.sh`, `scripts/run_plots_ia_rem_bluebear.sh`.
8. **`scripts/investigate_contaminant_properties.py`** (SLURM wrapper: `scripts/run_contaminant_properties_bluebear.sh`)
   - Companion to `investigate_purity_tradeoff.py`: fits the same `TrajectoryClassifier` per target (now all 6: `lensed_ia`, `all_lensed`, `lensed_sn1bc`, `lensed_sn2l`, `lensed_sn2n`, `lensed_sn2p` -- the last 4 added to break per-subtype contamination out of `all_lensed`'s aggregate), then at each FPR budget and decision epoch tags every surviving false positive with its true unlensed subtype, redshift, survey epoch of first detection, and final photometric uncertainty, and tests a secondary redshift-window cut calibrated on that operating point's own true-positive `z`-range -- see Finding 5.
9. **`scripts/validate_zcut_outofsample.py`** (SLURM wrapper: `scripts/run_zcut_validation_bluebear.sh`)
   - Out-of-sample validation of Finding 5's secondary `z`-window cut: a proper three-way train/calib/val split (instead of calibrating and evaluating the cut on the same split), plus a photo-z-noise stress test (the window is calibrated on clean `z`, as a historical confirmed sample would give, and applied to `val`'s `z` after adding realistic Gaussian photo-z-like noise, as a new unconfirmed candidate's host photo-z would give) -- see Finding 5's "Out-of-sample validation" subsection.
10. **`scripts/rank_candidates.py`** (SLURM wrapper: `scripts/run_rank_candidates_bluebear.sh`)
   - Operationalises Executive Summary point 7 / Finding 5's recommendation: fits the deployed `TrajectoryClassifier` on the same three-way `split3` as `validate_zcut_outofsample.py`, calibrates FPR thresholds on `calib`, and emits held-out `val` candidates in rank order tagged with the strictest budget each clears. Emits a review-budget lift curve (the primary output -- it needs only the ordering, no threshold calibration), exclusive tier stats, cumulative operating points comparable to Finding 2, and per-event CSVs. All statistics rate-weighted via `cmsne.importance_sampling` with class weights renormalised to annual on-sky rates (`annual_shares()`); `assign_tier()` skips budgets flagged unresolvable so their candidates fall through rather than vanishing from the table. See Finding 6.
11. **`scripts/diag_calib_resolution.py`** (SLURM wrapper: `scripts/run_calib_resolution_bluebear.sh`)
   - Scoping diagnostic for Finding 6's FPR floor: reads the result pickles only (no classifier fit) and reports per-pool ESS, ESS/raw, weight concentration, and the generation multiple each FPR budget would require. See Finding 6a.
12. **Throwaway proposal spikes** -- `scripts/spike_matched_proposal_gain.py`, `scripts/spike_proposal_ceiling.py` (wrappers `scripts/run_spike_gain_bluebear.sh`, `scripts/run_spike_ceiling_bluebear.sh`)
   - Explicitly labelled throwaway. Measure **ESS per generation attempt** (the compute-relevant metric) under uniform, `rate`, `rate*sqrt(eps)` and `rate*eps` proposals, recovering `eps` from the existing uniform baseline's bin counts. Their result killed an already-written design before any `cmsne/` code was touched. See Finding 6a. Not part of any production path.

13. **`scripts/spike_tail_transfer.py`** (SLURM wrapper: `scripts/run_spike_tail_bluebear.sh`) -- throwaway
   - The decisive spike, and the one that killed the *second* proposal design. Asks whether the `(z, mu)` ESS gain reaches the background **score** tail that actually sets a strict FPR threshold, by noting that restricting the estimator to a tail multiplies each bin's contribution by `f_b` (the fraction of that bin's detections in the tail) -- so `ESS_tail` is the global formula with `eps -> eps*f`, and the tail-optimal proposal is `rate*sqrt(eps*f)`. Reports `gain_global`, `gain_tail` and the tail-aware ceiling per class and pooled, with the reading rule printed by the script itself so it could not be rationalised after the fact. Result: `0.75x` at the `0.1%` tail. See Finding 7. Not part of any production path.

14. **`cmsne/proposals.py`** (new module) + the `generate_with_boosting` weight fix in `cmsne/importance_sampling.py`
   - `DetectionAwareProposal`: a 2-D piecewise-constant grid over `(z, mu)` with log-spaced `mu` edges, Beta-shrunk per-cell efficiency estimates, count-weighted smoothing, and a defensive uniform mixture (Hesterberg 1995; Owen & Zhou 2000) bounding weights at `1/alpha`. **Committed, tested, and wired into nothing** -- the objective it was built to serve was rejected (Finding 7). Retained as a mechanism, not a decision.
   - Separately, a **real 15% weight deficit** was found and fixed in `generate_with_boosting`: when no sparse cells were flagged it returned after only `n_pilot` attempts, while `rate_weight_fn` had already been normalised by `n_attempted = n`. Every weight was therefore low by `n_pilot/n` (`15%` at the production `pilot_frac=0.85`). The error is **asymmetric across classes** -- in `results/dual_59297152` only `lensed_Ia` and `unlensed_Ia` flagged cells and were weighted correctly; the other eight took the early return -- so it distorts relative rates and hence composition statistics. Finding 5's `unlensed_Ia`-at-`66.6%` figure moves under this fix and needs recomputing. Closing this also exposed a hole in the test harness, where a mid-file `sys.exit(1)` meant failures in the tail block never failed the suite.

15. **`scripts/run_brute15_bluebear.sh`** -- the 15x production run
   - `1,500,000` attempts/class across all 10 classes as a 10-way array, targeting `25` effective background events at a `0.1%` FPR cut (`20%` Poisson error). Carries the floor derivation, the precision table, and the literature precedent (Metcalf et al. 2019's `TPR_10`; HOLISMOKES XXI Part II's `O(10^-4)`) as a header comment, so the choice of `15x` is auditable from the script itself. Job `62368493`: completed, `7h45m`, `~55` CPU-hours, `9.5 GB`. Also records that `baseline_v5.3.2_10yrs.db` no longer exists on disk, so this run's absolute rates are not comparable to the baseline's.

16. **`scripts/diag_baseline_uniformity.py`** (SLURM wrapper: `scripts/run_baseline_uniformity_bluebear.sh`) and **`scripts/run_noboost_ia_bluebear.sh`**
   - The first reads pickles only and reports `n_pilot`, `n_boost` and `total_attempted` per class -- it is what established that only `lensed_Ia` and `unlensed_Ia` took the boosted path, and therefore which spike results were contaminated. The second regenerates those two classes with `--no-boost` (`pilot_frac=1.0`) for a clean uniform-proposal measurement (job `62299092`, `32` minutes).

---

## 3. Detailed Scientific & Methodological Findings

### Finding 1: Snapshot Magnitude vs Cumulative `peakmag` Mode (`compare_mag_modes.py`)
- At **early epochs (`+0d` and `+3d`)**, snapshot magnitude `{b}_d{t}` and cumulative `peakmag_{b}_d{t}` perform identically (e.g. `g-r` `+0d` `AUC = 0.843`, `r-i` `+0d` `AUC = 0.864`).
- At **intermediate post-peak epochs (`+14d` to `+30d`)**, **contemporaneous snapshot magnitudes** outperform cumulative `peakmag` in single-epoch 2D slices:
  - `g - r` at `+14d`: `AUC = 0.763` (snapshot) vs `0.626` (matched `peakmag`).
  - `g - r` at `+30d`: `AUC = 0.945` (snapshot, `TPR@10% = 92.3%`) vs `0.821` (matched, `TPR@10% = 41.9%`).
  - `r - i` at `+14d`: `AUC = 0.906` (snapshot) vs `0.865` (matched).
  - *Physical reason:* `(1 + z)` cosmological time dilation stretches the observed light-curve timescale for *any* SN at redshift `z`, lensed or not -- it is not an intrinsic lensing effect. The lensed-vs-unlensed difference here is a **selection** effect: unlensed detections are flux-limited to comparatively low redshift (little dilation, fast observed fading), while a *detectable* lensed source needs enough lensing optical depth to favour higher redshift (more dilation, slower observed fading). So nearby unlensed SNe evolve rapidly and dim out of the bright region by `+14d .. +30d`, while the (typically higher-z) lensed population remains bright longer. Locking the y-axis to historical `peakmag` freezes faded unlensed SNe at their old peak brightness.
- By **`+60d`**, keeping *both* current snapshot brightness and historical peak brightness—specifically their difference `fade_b(t) = mag_b(t) - peakmag_b(t)`—gives the classifier direct access to the light-curve decline rate.

### Finding 2: The LSST Base-Rate Paradox & Telescope Follow-Up Operating Points, by Decision Epoch (`investigate_purity_tradeoff.py`)
Even when a classifier achieves `ROC-AUC > 0.91`, multiplying `TPR` and `FPR` by true annual LSST rates (`N_sig = 10/yr` for lensed Ia -- Oguri et al. 2010; Wojtak et al. 2019; Goldstein et al. 2019, the same figure already cited in `docs/methods_and_results.md` -- or `45/yr` for all lensed, an order-of-magnitude placeholder pending [issue #43](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/43), vs `N_bg = 100,000/yr` unlensed) exposes why moderate false-positive rates fail for spectroscopic follow-up.

**Why "by decision epoch" and not one curve:** an earlier version of this script scored every test event with `clf.score(event)` on whatever the light curve looked like once the *entire* 3-year survey simulation had finished -- not what would actually be known at the moment a real alert broker needs to decide whether to trigger follow-up. An event credited as a "detection" there might only have been distinguishable from background once its `+60d` photometry existed, 60 days after decision time, which overstates what the classifier can do in deployment -- precisely for the short-lived, highly-magnified transients this project cares about most (see [issue #51](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/51), fixed). `investigate_purity_tradeoff.py` now fits one `TrajectoryClassifier` (isotonic calibration, uncertainty-aware noise injection, `target_fpr=0.10`) and re-evaluates ROC, the standard operating points, and the zero-FP sweet spot **separately at each of the 5 decision epochs**, using `cmsne.trajectory.reveal_up_to` to mask out anything not yet observable -- the same causal machinery `scripts/plot_roc.py`/`scripts/compare_trajectory_models.py` already use correctly for their per-epoch AUC columns.

**Table below also includes the [issue #50](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/50) fix** (`score()` now applies the classifier's local-density calibration discount, which previously only ran inside `probability()` and so never reached any of these operating points):

| Decision Epoch | Lensed Ia TP/yr @ 10% FPR | Lensed Ia Purity | Lensed Ia Zero-FP Pure/yr (recall) | All Lensed TP/yr @ 10% FPR | All Lensed Purity | All Lensed Zero-FP Pure/yr (recall) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+0d (at trigger, immediate)** | **4.0 / yr** | 0.04% | **0.00 / yr (0.0%)** | **9.2 / yr** | 0.09% | **0.00 / yr (0.0%)** |
| +3d | 4.6 / yr | 0.05% | 0.00 / yr (0.0%) | 9.8 / yr | 0.10% | 0.00 / yr (0.0%) |
| +14d | 4.7 / yr | 0.05% | 0.00 / yr (0.02%) | 10.1 / yr | 0.10% | 0.00 / yr (0.0%) |
| +30d | 4.8 / yr | 0.05% | 0.02 / yr (0.21%) | 10.0 / yr | 0.10% | 0.00 / yr (0.0%) |
| +60d (full light curve -- what every pre-#51-fix table on this project implicitly assumed) | 4.0 / yr | 0.04% | 0.03 / yr (0.31%) | 10.3 / yr | 0.10% | 0.00 / yr (0.0%) |

*Conclusion:* with the #50 fix applied, the picture changes in two ways from the #51-only table this project previously reported. First, at the standard `10%`-FPR operating point, yield is now roughly **flat across all 5 decision epochs** (`4.0-4.8/yr` lensed Ia, `9.2-10.3/yr` all lensed) rather than climbing steadily from `+0d` to `+60d` as it previously appeared to -- that climb was partly an artifact of `score()` growing more confident as more epochs accumulated, independent of whether the local training support actually justified it. Second, the zero-FP "pure" tier -- previously the headline `0.56/yr` (`5.6%`) `lensed_Ia` / `1.02/yr` (`2.3%`) all-lensed figures at `+60d` -- is now close to zero at every epoch (`<=0.03/yr` lensed Ia, `0.00/yr` all lensed): most of what previously cleared zero false positives was doing so on thin local support, not genuine separation from background. Real spectroscopic target selection still lives entirely in the `FPR <= 0.01%` to `FPR = 0%` tail at every epoch, so this tail is also the part of the ROC curve most sensitive to exactly this kind of calibration error -- which is why issue #50 mattered more here than its "1-2 epoch events" framing suggested. **The pre-#50-fix numbers this project reported until now (`3.9 -> 7.8/yr` lensed Ia TP/yr across epochs, `0.56/yr` zero-FP) are superseded by this table** and should not be cited going forward; Finding 4's `~4.1x` improvement claim was computed against the old `0.56/yr` baseline and has since been retired entirely, not merely rebased -- see Finding 4's revised table and caveats.

### Finding 3: Resolving the Heatmap vs Scatter-Plot Mismatch
We investigated why the 2D probability heatmaps previously appeared misaligned with the overlaid scatter points in certain regions (e.g. high-probability zones containing many visible unlensed dots, or low-probability zones with isolated lensed stars):
1. **Unweighted Markers vs Rate-Weighted Classifier:** Each of the 10 classes was simulated with `100,000` Monte Carlo events so rare tails were well sampled, but each event carries a physical `weight` spanning several orders of magnitude. Visually plotting unweighted dots made rare, near-zero-weight Monte Carlo tail draws look like massive background clusters, whereas the classifier rightly weighted them according to their tiny physical sky rate.
2. **Balanced Training Prior (`P_balanced`) on 2D Heatmaps:** The 2D probability surface shows balanced class likelihood (`50/50` total signal vs background weight) so the `P = 0.5` contour marks where normalized signal density equals normalized background density (`TPR` vs `FPR` crossover), rather than the `1 : 2,200` raw sky prior (under which `P_sky` would be `< 0.01` almost everywhere on a single 2D colour slice).
3. **Other Lensed Subtypes Previously Counted as Background in `lensed_Ia` Plots:** Previously, `lensed_sn1bc`, `lensed_sn2l`, `lensed_sn2n`, and `lensed_sn2p` sitting in the bright/red lensed region were labelled `y = 0` ("background") during `lensed_Ia` 2D fits, carving artificial low-probability holes right through the lensed locus. Excluding non-target lensed SNe from the contaminant pool eliminated this artifact.
4. **Axis-Aligned GBDT Extrapolation & Isotonic Step Plateaus in Sparse Corners:** Standard decision trees extrapolate flat rectangular boxes to the plot borders where data density vanishes, and Isotonic calibration snaps continuous tree scores onto discrete step plateaus.
5. **Photometric Scatter Near the LSST Detection Limit:** Adding `1σ` error bars (`*_errorbars.png`) confirmed that while bright events (`r < 21.5`) have tight uncertainties (`< 0.04 mag`), faint unlensed SNe near `r ~ 23.5 - 24.5` have `±0.20 - 0.35 mag` colour uncertainties that scatter a thin halo horizontally into redder colours.

### Finding 4: Full-Trajectory Architecture Benchmark (`compare_trajectory_models.py`)
We tested 6 full-trajectory models on the exact same 70/30 held-out test split (132 enriched physical columns -- 102 beyond the 30 baseline columns, recomputed directly from the feature-building code; an earlier draft of this document claimed 165/100 in error):

1. **Baseline GBDT (30 features, Isotonic calibration, trained on `+60d` only)**
2. **Smooth-Calibrated GBDT (30 features, Platt Sigmoid calibration + leaf regularisation `min_samples_leaf=80, l2=5.0`)**
3. **Physics-Enriched + Causal-Prefix GBDT (Snapshot magnitudes + per-band fade from peak `mag(t) - peakmag(t)` + broad colours `g-i, r-z, g-z` + colour evolution + `r/i/z` photometric errors + causal partial-trajectory training prefixes + Sigmoid calibration)**
4. **Smooth Spline Logistic GAM (Cubic B-Splines + explicit observation masks + Ridge Logistic Regression — non-tree additive model)**
5. **Mask-Aware Neural MLP Ensemble (4 smooth `128-64-32` ReLU MLPs taking `[X_standardized, Observation_Mask]` on rate-weighted balanced importance resamples — non-tree neural replacement)**
6. **Hybrid Ensemble (`0.65 * Enriched GBDT + 0.35 * Mask-Aware Neural MLP` blended in log-odds space)**

> **Caveat on "Model 1: Baseline GBDT" and the word "baseline":** this benchmark's Model 1 fits a bare `HistGradientBoostingClassifier` directly and does **not** include the production default's uncertainty-aware noise-injection replication (`n_uncertainty_replicates=8`, see `cmsne/classifier.py`) that the actually-deployed `cmsne.trajectory.TrajectoryClassifier` always applies. So "Model 1" here isolates calibration/feature/architecture choices *from* noise injection; it is not the currently-deployed system's own operating point, and this table should not be read as using the same "baseline" as Finding 2.
>
> **Causal caveat (added after Finding 2b / [issue #51](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/51)):** the `AUC (+0d)`/`AUC (+14d)`/`AUC (+60d)` columns below *are* already evaluated causally per epoch (this benchmark's own "causal prefix training" already handles that correctly). The `Recall @ X% FPR`, `Zero-FP Recall`, and `Pure Yield / yr` columns are still a single number per model, not broken out by decision epoch -- they are the full-light-curve (`+60d`) figures, not a real-time decision point.
>
> **Local-density discount now applied here too (added after [issue #50](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/50); table below is the rerun with it applied):** every one of this benchmark's 6 models originally had no correction at all for thin-local-training-support overconfidence -- the exact pathology issue #50 found and fixed in the real deployed classifier's `score()`. Applying the same mechanism here (a vectorized, model-agnostic port -- see `scripts/compare_trajectory_models.py`'s `_VectorizedDensityBlender`) changes the picture completely: **every one of the 6 models' zero-FP "pure" yield collapses to 0.0/yr, for both targets.** The previously reported `0.2-2.3/yr` (lensed Ia) and `0.1-1.6/yr` (all lensed) zero-FP figures -- including the Hybrid ensemble's headline `2.3/yr` -- were themselves an artifact of the same uncorrected overconfidence Finding 2 found in the real classifier, not a genuine architectural capability. This is consistent with Finding 2's own corrected numbers (the real deployed classifier's zero-FP yield is also ~0 at every decision epoch) -- **none of the 7 architectures tested in this project (the 6 here + the real deployed classifier) demonstrate genuine zero-false-positive discrimination at this survey scale once this discount is applied.** The `>7x`/`~4.1x` pure-yield improvement claims previously made in this section and the Executive Summary are retired, not merely stale: there is no longer a meaningful ratio to quote between two populations of zero.
>
> **A caveat on this correction's own fairness, in turn:** Models 1-2 use 30 raw features; Models 3-6 use 132. The local-density discount fires more often (and more strongly) for a query row whose missing-data *pattern* has thin local support -- and a 132-dimensional feature space has combinatorially more distinct possible patterns than a 30-dimensional one, so the enriched models may be penalized somewhat more mechanically by dimensionality alone, not purely because they are more overconfident per se. The AUC reversal below (baseline models now beating enriched ones at `+60d` for `all_lensed`, and even for `lensed_Ia`) should be read with this in mind -- it is evidence the previous enriched-feature advantage was at least partly inflated by uncorrected overconfidence, not yet a clean, confound-free demonstration that the enriched features are actively worse.

#### Summary Benchmark — Lensed Type Ia SNe (`10 / yr` vs `100,000 / yr` Unlensed)

| Model | AUC (+0d) | AUC (+14d) | AUC (+60d) | Recall @ 10% FPR | Recall @ 1% FPR | Recall @ 0.1% FPR | Zero-FP Recall (FPR = 0%) | Pure Yield / yr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline GBDT** | 0.764 | 0.862 | **0.905** | 73.0% | **33.2%** | 7.1% | 0.00% | **0.0 / yr** |
| **2. Smooth-Calibrated GBDT** | 0.758 | 0.854 | 0.905 | 71.9% | 32.2% | 6.6% | 0.00% | **0.0 / yr** |
| **3. Physics-Enriched + Prefix GBDT** | 0.817 | 0.887 | 0.894 | **76.4%** | 22.8% | 7.2% | 0.00% | **0.0 / yr** |
| **4. Smooth Spline GAM** | **0.818** | **0.889** | 0.896 | 75.3% | 23.0% | 9.5% | 0.00% | **0.0 / yr** |
| **5. Mask-Aware Neural MLP** | 0.775 | 0.885 | 0.892 | 75.5% | 23.4% | **9.8%** | 0.00% | **0.0 / yr** |
| **6. Hybrid (Enriched GBDT + MLP)** | 0.807 | 0.888 | 0.894 | 74.9% | 23.0% | 8.4% | 0.00% | **0.0 / yr** |

#### Summary Benchmark — All Lensed SNe (`45 / yr` vs `100,000 / yr` Unlensed)

| Model | AUC (+0d) | AUC (+14d) | AUC (+60d) | Recall @ 10% FPR | Recall @ 1% FPR | Recall @ 0.1% FPR | Zero-FP Recall (FPR = 0%) | Pure Yield / yr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline GBDT** | 0.587 | **0.695** | 0.729 | 36.3% | 5.7% | **2.8%** | 0.00% | **0.0 / yr** |
| **2. Smooth-Calibrated GBDT** | 0.592 | 0.689 | **0.730** | **37.0%** | 5.6% | 2.4% | 0.00% | **0.0 / yr** |
| **3. Physics-Enriched + Prefix GBDT** | 0.645 | 0.678 | 0.681 | 25.2% | 8.6% | **3.5%** | 0.00% | **0.0 / yr** |
| **4. Smooth Spline GAM** | 0.614 | 0.676 | 0.677 | 24.6% | 7.7% | 2.3% | 0.00% | **0.0 / yr** |
| **5. Mask-Aware Neural MLP** | 0.633 | 0.676 | 0.684 | 25.4% | 8.2% | 1.5% | 0.00% | **0.0 / yr** |
| **6. Hybrid (Enriched GBDT + MLP)** | **0.649** | 0.681 | 0.684 | 25.2% | **8.7%** | 2.9% | 0.00% | **0.0 / yr** |

(AUC values above are rounded to 3 d.p.; see `docs/figures/model_comparison/trajectory_model_comparison{,_alllensed}.json` for full precision and the `+3d`/`+30d` columns omitted here for space.)

#### Why Each Upgrade Worked (revised after the issue #50 rerun):
- **Sigmoid vs Isotonic Calibration:** A small, consistent AUC edge survives the correction (`lensed_Ia +60d`: `0.905` both, `all_lensed +60d`: `0.729 -> 0.730`) -- a genuine, if now much smaller, effect than the pre-correction table suggested.
- **Physics Enrichment (Time-Dilated Fade + Broad Colours + Errors):** Still wins clearly at `+0d`/`+14d` for `lensed_Ia` (`0.764/0.862 -> 0.817/0.887`) and at `+0d` for `all_lensed` (`0.587 -> 0.645`), i.e. the early-warning benefit holds up. It no longer wins by `+60d` for either target (`lensed_Ia`: `0.905 -> 0.894`; `all_lensed`: `0.729 -> 0.681`) -- a reversal from the pre-correction table, though see the dimensionality caveat above before reading this as "enrichment hurts."
- **Causal Prefix Augmentation:** The `+0d` out-of-distribution fix still holds directionally (`lensed_Ia +0d` AUC `0.764 -> 0.817` baseline-to-enriched), smaller than the pre-correction `0.708 -> 0.812` gap but not eliminated.
- **Hybrid GBDT + Mask-Aware Smooth MLP:** No longer the best zero-FP performer -- all 6 models tie at `0.0/yr` once genuinely unsupported confidence is discounted (see the discount caveat above). Its AUC remains competitive (within the enriched-model cluster at every epoch) but is not a clear outright winner the way the pre-correction table suggested.

---

### Finding 5: Ranked Follow-Up, Not a Hard Cut -- Characterizing False-Positive Contaminants & a Measured Secondary Redshift Cut (`investigate_contaminant_properties.py`)

With zero-false-positive discrimination retired across all 7 architectures (Findings 2, 4), the classifier's real value is as a **ranked candidate list at a chosen FPR budget** -- `rank()`/`probability()`, already this codebase's own recommended API over a hard `predict()` cut (see their docstrings: *"the prioritised follow-up list: work down it until the budget runs out"*; *"a hard boundary is the wrong picture for a broad, gradual transition"*) -- not a zero-contamination filter. `docs/figures/purity_investigation/ranked_followup_summary{,_alllensed}.png` restates Finding 2's operating points this way: annual yield and purity at the `10%/1%/0.1%/0.01%` FPR budgets, one line per decision epoch.

This raises the natural follow-up question: at a realistic budget, what is actually leaking through as a false positive, and can a cheap secondary cut remove more of it before committing scarce spectroscopic time? **"Host galaxy type" was the originally proposed secondary cut, but it does not exist anywhere in this project's simulated dataset**, and using it *in the trigger itself* would contradict this project's own explicit design constraint (`docs/methods_and_results.md`: *"colour alone -- no host photometric redshift, no spectroscopy"*). Redshift `z` and magnification `mu` **are** simulated for every event (signal and background alike) but are deliberately excluded from the classifier's own feature set for that same reason. Using them as a **downstream** triage cut -- after the photometric trigger has already fired, when deciding which of the ranked candidates get follow-up time -- does not touch that constraint, since it plays no role in the trigger decision itself.

> **Correction (rate-weighted stats, not raw test-set counts):** the first version of this Finding computed subtype composition, every reported median, and the `z`-cut's own window/removed/retained percentages from raw, unweighted test-set counts. This dataset is produced by pilot+boost importance sampling (`cmsne.importance_sampling`), which deliberately over-samples rare, extreme `(z, magnification)` regions for Monte Carlo statistics and corrects each such event's own `weight` down accordingly -- so an unweighted characterization silently re-introduces exactly the sampling-density bias those weights exist to correct for. This surfaced when a "surprising high-z non-lensed contaminant" turned out to be partly a real, expected effect (an unmagnified Ia, `abs_mag ~ -19.23 +/- 0.10`, is genuinely LSST-detectable out to `z ~ 1+` on its own) and partly this exact bug. All numbers below are now rate-weighted (`cmsne.importance_sampling.weighted_composition`/`weighted_quantile`, added and unit-tested alongside this fix), and the "thin sample" flag now also requires adequate *effective* sample size (`effective_sample_size`, issue #50's own tool), not just a raw event count. Finding 2/4's own purity/yield numbers were never affected -- those were always computed from the weighted ROC curve, not from raw counts.

**Method:** fits the same rate-weighted `TrajectoryClassifier` as Finding 2 (one classifier per target, scored per decision epoch via `reveal_up_to`), for all **6 targets**: `lensed_ia`, `all_lensed`, and -- extending the investigation to the individual CC subtypes previously only visible lumped inside `all_lensed` -- `lensed_sn1bc`, `lensed_sn2l`, `lensed_sn2n`, `lensed_sn2p`, each against the same full `unlensed_*` background pool used throughout this project. The 4 new per-subtype annual rates are not invented: they split the already-used lensed-CC total (`all_lensed - lensed_ia = 35/yr`) by `cmsne.lsst.CC_FRACTIONS` (`sn1bc=0.36, sn2p=0.55, sn2n=0.06, sn2l=0.03`; Li et al. 2011, Shivvers et al. 2017) -- the same literature-sourced relative CC demographics already baked into every event's own rate `weight` -- giving `12.6/yr, 1.05/yr, 2.10/yr, 19.25/yr` respectively. These inherit (and compound) `all_lensed`'s own not-independently-validated caveat (issue #43): treat them as more provisional still. At each of the 4 standard FPR budgets and all 5 decision epochs, every surviving background false positive is tagged with its true unlensed subtype, redshift, survey epoch of first detection (`trigger_day`), and final photometric uncertainty (`peakmag_d60_err`, an SNR proxy) -- the last two simulated per event but never used as classifier features, same rationale as `z`/`mu`, reported here diagnostically only (no cut proposed on them this round). A secondary cut is tested on `z` alone, restricting to the central `95%` (`2.5-97.5` weighted percentile) of that operating point's own true-positive redshift range, and measuring exactly how much rate-weighted contamination it removes vs. how much true-positive weight it costs.

**Results** (`docs/figures/purity_investigation/contaminant_properties{,_alllensed,_snibc,_sn2l,_sn2n,_sn2p}.json`, full per-epoch tables):

1. **`unlensed_Ia` dominates false-positive composition across nearly every lensed subtype, and becomes MORE dominant (not less) as the budget tightens.** At the standard `10%`-FPR/`+60d` operating point, `unlensed_Ia` is the single largest contaminant for `lensed_Ia` (`66.6%` of FPs, rate-weighted), `lensed_sn1bc` (`66.7%`), `lensed_sn2l` (`59.3%`), and `lensed_sn2p` (`57.2%`) -- the expected colour-twin confusion, since a lensed Ia is simply an unlensed Ia's colour profile at higher apparent brightness, which magnification (achromatic) cannot by itself fix. For `lensed_Ia` specifically, this share *rises* as the budget tightens (`66.6%` at `10%` FPR -> `71.4%` at `1%` FPR -> `99.1%` at `0.1%` FPR) before the strictest `0.01%`-FPR tier (`n = 9` test events, flagged thin) shows a different, unreliable mix -- the opposite of this Finding's original (pre-correction) claim that a rare CC subtype's share grew at tighter budgets, which was itself an artifact of the same unweighted-count bug.
2. **`lensed_sn2n` is the one target where `unlensed_Ia` is not the top contaminant**, and `all_lensed` is close to a tie. At `10%`-FPR/`+60d`, `lensed_sn2n`'s false positives are `43.0%` `unlensed_sn2n` (a magnified core-collapse event most often confused with an unmagnified example of *its own* subtype) vs `35.0%` `unlensed_Ia`; `all_lensed` is `42.2%` `unlensed_Ia` vs `37.4%` `unlensed_sn2n`. `unlensed_sn2l` -- the rarest contaminant subtype (`cmsne.lsst.CC_FRACTIONS`: `3%` of core-collapse) -- drops to a consistently negligible `0.2-0.4%` share everywhere once rate-weighted, versus `7-12%` in the pre-correction raw-count version: it was being deliberately over-sampled for Monte Carlo statistics and almost all of that raw count carried negligible real weight.

   **Why `sn2n` specifically:** `cmsne.lsst.contaminant_info` gives `sn2n` a peak absolute magnitude of `-19.05`, within `0.2` mag of SN~Ia's `-19.23` and far brighter than `sn1bc`/`sn2l`/`sn2p` (`-17.51`/`-17.46`/`-16.90`). Combined with the same Malmquist-style, flux-limited detection bias already documented for the project's colour-only classifier (`docs/paper/paper.tex` Section 4.2: the *detected* background is SN~Ia-dominated despite core-collapse outnumbering Ia `~3:1` volumetrically, because flux-limited detection favours the intrinsically brighter class), `unlensed_sn2n`'s *detected* share of the background (`~25%` of the overall unlensed test-set mix, per `fp_subtype_composition_sn2n.png`'s reference bar) sits roughly `4x` above its `6%` volumetric `CC_FRACTIONS` weight -- the same inflation mechanism, just acting on one CC sub-type instead of on Ia overall. That inflated, nearly-Ia-bright detected population, combined with being an exact colour-twin of itself by construction (magnification cannot change it), is what lets self-type confusion win out over the usual `unlensed_Ia` colour-twin problem for this one target. The self-confusion share also grows with decision epoch (`30.2%` at `+0d` to `43.0%` at `+60d`, while `unlensed_Ia`'s share falls `43.5%` to `35.0%` over the same epochs), consistent with `sn2n`'s own colour evolution diverging from Ia's more than from its own unmagnified self as more of the light curve is observed.
3. **Redshift cleanly separates true from false positives, despite playing no role in the classifier's own features.** `docs/figures/purity_investigation/contaminant_redshift_10pct{,_alllensed,_snibc,_sn2l,_sn2n,_sn2p}.png` shows true positives (magnification-boosted) concentrated at higher `z` than false positives (unmagnified unlensed SNe, which can only be bright enough to trigger at lower `z`) -- the expected magnification-bias signature, recovered here purely as a byproduct of characterizing the classifier's mistakes, not something it was told to look for. This qualitative picture held before and after the weighting correction; only the quantitative composition/cut numbers changed.
4. **A secondary `z`-window cut delivers measured, non-uniform gains at a near-fixed true-positive cost (~5%, by construction of the 95% window) -- and the correction shows the cut is more effective than originally reported.** `lensed_Ia` at `1%`-FPR/`+3d` removes `64.7%` of residual false-positive weight for a `4.9%` true-positive cost (purity `0.213% -> 0.570%`, a ~2.7x gain). At the `10%`-FPR/`+60d` reference point across all 6 targets, the gain ranges from `19.4%` of false-positive weight removed (`lensed_sn1bc`) to `61.1%` (`lensed_sn2n`), tracking how cleanly magnification separates that specific subtype's redshift range from its contaminants' rather than being a fixed property of the cut. (The pre-correction version of this bullet also cited an `all_lensed` `0.01%`-FPR/`+60d` example; under the corrected, effective-sample-size-aware thin-sample check that operating point is now flagged too thin to report a cut on at all -- it was already borderline by raw count alone.)
5. **Whether false positives are photometrically noisier than true positives is NOT a consistent pattern once rate-weighted -- this bullet is retracted, not merely revised.** The pre-correction version of this Finding reported false positives as noisier than true positives for 5 of 6 targets. Rate-weighted, the direction is mixed: `lensed_Ia`, `lensed_sn1bc`, and `lensed_sn2l` still show false positives noisier (median `peakmag_d60_err` `0.119` vs `0.094`, `0.126` vs `0.098`, `0.125` vs `0.122` mag respectively), but `all_lensed` and `lensed_sn2n` now show the *opposite* (true positives noisier: `0.131` vs `0.105`, `0.129` vs `0.111`), and `lensed_sn2p` is an even tie (`0.125` vs `0.127`). `trigger_day` differences remain small and inconsistent in direction, as before, and are still reported as measured without being read as a meaningful pattern.

**Caveats:**
> **In-sample cut calibration (now independently checked -- see below):** the `z`-window above is calibrated and evaluated on the *same* held-out test split, not an independent third split, so the numbers above are an optimistic estimate of what an operationally deployed cut would achieve on genuinely unseen data. `scripts/validate_zcut_outofsample.py` tests this directly with a proper three-way split and finds the gain largely survives -- see "Out-of-sample validation" below.
>
> **Thin samples at strict budgets:** the `0.1%`/`0.01%`-FPR tiers have as few as `5-24` false-positive test events (and, now, correspondingly low effective sample sizes); their subtype-composition and (where present) `z`-cut percentages are correspondingly unreliable and are marked `*`/`[THIN SAMPLE]` in the figures and raw output rather than silently reported as precise.
>
> **Per-subtype annual rates are a further split of an already-provisional total:** the 4 new CC-subtype rates divide `all_lensed`'s own not-independently-validated `45/yr` (issue #43), so their absolute annual-TP/FP numbers should be read as even more provisional than `all_lensed`'s; the rate-weighted composition/`z`/`trigger_day`/SNR findings above do not depend on this particular rate split (they use each event's own simulated `weight`, not the annualization) and are not affected by this caveat.
>
> **Not a replacement for Finding 2's purity numbers:** this investigation characterizes *what* is contaminating each budget and tests one candidate secondary filter: it does not change Finding 2's annual yield/purity table, which remains the authoritative per-epoch operating-point reference and was unaffected by the weighting bug above (it was always computed from the weighted ROC curve).

**Out-of-sample validation (`scripts/validate_zcut_outofsample.py`):** a practical question about how this cut would actually be applied surfaced an operational detail worth stating plainly: the `z` used throughout this Finding is each event's true redshift, which is not directly observable. The realistic path is a resolved host-galaxy photometric redshift (host photo-z never touches the SN's own trigger photometry, so it doesn't reopen the "colour alone" constraint this project is built around) -- the alternative, spectroscopy, defeats the purpose of the triage this cut exists to support. Host photo-z carries real measurement uncertainty, typically `sigma_z/(1+z) ~ 0.02-0.05` for broad-band photometry.

This script re-does the cut with a proper three-way split -- `train` (classifier fit), `calib` (chooses the FPR threshold and the `z`-window bounds, exactly as the single test split did before), and `val` (held out from both, where the already-chosen threshold/window are applied fresh) -- and reports the gain both with clean `z` and with Gaussian photo-z-like noise (`sigma = 0.03 * (1+z)`) added to `val`'s query `z` only, mimicking a new unconfirmed candidate's host photo-z while the window itself stays calibrated on clean `z` (the realistic scenario for a historical, already spectroscopically-confirmed calibration sample).

**Results** (`docs/figures/purity_investigation/zcut_outofsample_validation{,_alllensed,_snibc,_sn2l,_sn2n,_sn2p}.json`): the gain largely survives out-of-sample, and is fairly robust to realistic photo-z noise. At `10%`-FPR/`+60d`: `lensed_Ia` removes `63.5%` of false-positive weight out-of-sample (clean `z`) for a `10.0%` true-positive cost (`64.6%`/`11.6%` with photo-z noise) -- comparable to or better than the in-sample `50.1%`/unspecified-cost figure above; `all_lensed` removes `49.1%`/`47.6%` (clean/noisy) for `5.6%`/`6.9%` cost, closely matching its in-sample `50.6%`; `lensed_sn2n` removes `58.4%`/`56.5%` for only `1.4%` cost either way, close to its in-sample `61.1%`. The two smaller-magnitude-gain targets (`lensed_sn1bc`: `22.2%`/`23.7%`; `lensed_sn2l`: `14.9%`/`14.5%`, both for `2-5%` cost) also land close to their in-sample figures (`19.4%`/`20.6%`). The previously-highlighted `lensed_Ia` `1%`-FPR/`+3d` example holds up particularly well out-of-sample: `69.3%` of false-positive weight removed (clean `z`) for a `6.1%` true-positive cost (purity `0.236% -> 0.717%`, a `~3.0x` gain, if anything better than the in-sample `64.7%`/`0.213% -> 0.570%`), and noisy photo-z removes even more (`82.3%`) at a similar cost (`5.7%`) -- in this particular case the added noise happens to push more false positives outside the window without pushing many true positives out too, not a general rule. `docs/figures/purity_investigation/zcut_outofsample_validation.png` shows the gain holding up at every decision epoch, not just the ones highlighted here.

> **Caveat on this validation itself:** splitting into three parts (rather than two) shrinks each operating point's effective sample size further -- `val_tp_ess` runs as low as `13-55` at the `10%`-FPR/`+60d` reference point across targets (above the `MIN_SAMPLE_FOR_STATS = 10` thin-sample threshold, but modest). The out-of-sample numbers above should be read as "the gain survives a genuine train/calib/val split, with the usual sampling noise that implies," not as a precision claim to the same number of significant figures as the in-sample table.

---

### Finding 6: The Ranked Candidate List Itself (`rank_candidates.py`) -- A Review-Budget Curve, Not an FPR Cut

Findings 2 and 5 established *that* the classifier's value is a ranked list at a tunable FPR budget. `scripts/rank_candidates.py` operationalises it: it fits the deployed `TrajectoryClassifier` on a three-way `train`/`calib`/`val` split (the same `split3` convention as `validate_zcut_outofsample.py`), calibrates thresholds on `calib`, and emits the held-out `val` candidates in rank order, each tagged with the strictest FPR budget it clears. BlueBEAR job `62176559`, all 6 targets against `results/dual_59297152`, `00:37:21`.

**The primary output is the lift curve, not the budget table.** An FPR budget is an awkward way to state what a follow-up programme actually has, which is a finite number of nights. The operationally meaningful question is "if I can review `N` candidates a year, how many lensed SNe do I get?", and that question needs only the *ordering* -- no threshold calibration and no background-quantile estimate. Rate-weighted, against the `10/yr` lensed Ia vs `100,000/yr` unlensed base rate (parenthesised figure is lift over reviewing a random `N`):

| Target | Decision epoch | Review 100/yr | Review 1,000/yr | Review 10,000/yr |
|---|---|---|---|---|
| `lensed_Ia` (10/yr available) | `+0d` | `0.63` (`63x`) | `1.42` (`14x`) | `3.99` (`4.0x`) |
| | `+3d` | `0.83` (`83x`) | `1.92` (`19x`) | `4.24` (`4.2x`) |
| | `+14d` | `0.38` (`38x`) | `1.58` (`16x`) | `4.76` (`4.8x`) |
| | `+30d` | `0.40` (`40x`) | `1.12` (`11x`) | `3.98` (`4.0x`) |
| | `+60d` | `0.43` (`43x`) | `1.08` (`11x`) | `3.54` (`3.5x`) |
| `all_lensed` (45/yr available) | `+0d` | `0.44` (`9.8x`) | `0.83` (`1.9x`) | `8.30` (`1.8x`) |
| | `+3d` | `0.07` (`1.6x`) | `2.37` (`5.3x`) | `10.55` (`2.3x`) |
| | `+60d` | `0.24` (`5.3x`) | `0.76` (`1.7x`) | `9.64` (`2.1x`) |

Read plainly: reviewing `10,000` candidates a year -- `10%` of the alert stream -- recovers roughly `40%` of all lensed Ia that exist (`~4/yr` of `10/yr`), and the ranking is doing real work at the sharp end, concentrating `~63x` more lensed Ia into the top `100/yr` than random selection would. Notably `all_lensed` is *much* weaker at the sharp end (`1.6-9.8x` at `100/yr`) than `lensed_Ia` alone (`38-83x`): aggregating the four CC subtypes into one target dilutes the top of the list with the subtypes the classifier separates worst, which is consistent with Finding 4's per-subtype AUC spread. For a programme with a small follow-up budget, targeting `lensed_Ia` specifically is the better configuration.

**The FPR-budget table is reported too, but two of the four budgets are not measurable on this pool.** Across all 120 cumulative operating points (6 targets x 5 decision epochs x 4 budgets), **38 are clean, and every one of those 38 is either the `10%` or the `1%` budget. Not a single clean operating point exists at `0.1%` or `0.01%`, for any target at any epoch.** `lensed_Ia` at `+0d`:

| Budget | TP/yr | FP/yr | Purity | TP ESS | |
|---|---|---|---|---|---|
| `10%` FPR | `3.99` | `10,646` | `0.037%` | `43.5` | |
| `1%` FPR | `1.55` | `1,073` | `0.144%` | `13.3` | |
| `0.1%` FPR | `0.63` | `222` | `0.282%` | `3.5` | **THIN SAMPLE** |
| `0.01%` FPR | `0.28` | `46` | `0.600%` | `3.1` | **THIN SAMPLE** |

This is a *sampling* limit, not a classifier limit, and the script now reports it as such rather than hiding it. Two mechanisms are separated:

- **`NOT RESOLVABLE`.** Placing a cut at `0.1%` FPR requires the `99.9th` percentile of the `calib` background score distribution. With `calib_n_bg = 5276` raw background events the unweighted floor is `1/5276 = 0.019%`, but the binding constraint is effective sample size, and the rate-weighted `calib` background ESS is only `~1,856`. Where the nearest achievable ROC point misses the target by more than `FPR_MATCH_TOLERANCE = 0.5` (50%), the budget is reported as unresolvable and **no operating point is emitted** -- e.g. `lensed_Ia` at `+3d`: *`0.1%` FPR: NOT RESOLVABLE on this calib split (target `0.1000%`, nearest achievable `0.0484%`)*. An earlier version of this script used `np.argmin(|fpr - target|)`, which silently snapped to the nearest achievable point and reported it as though the budget had been met; that was a reporting bug, now fixed. Candidates in an unresolvable tier are not dropped from the exclusive-tier table -- they fall through to the next budget that *is* resolvable.
- **`THIN SAMPLE`.** Even at a resolvable threshold, the yield past it may rest on too few effective events. Matching the `MIN_SAMPLE_FOR_STATS = 10` convention already used by `investigate_contaminant_properties.py`, a tier is flagged if **any** of `n_tp`, `n_fp`, `tp_ess`, `fp_ess` falls below it. The `OR` matters: a tier with `tp_ess = 1.4` and `fp_ess = 20` is not a usable measurement, and an earlier `tp_ess + fp_ess < 10` form would have passed it. The worst case in this run is `lensed_sn2p` at `+30d`/`0.01%`, which reads a superficially impressive `0.63%` purity on `tp_ess = 4.7` / `fp_ess = 1.9`.

A `THIN SAMPLE` yield is roughly `+/- 1/sqrt(ESS)` at best -- `~15%` on the `10%`-budget row above (`ESS 43.5`) versus `~53%` on the `0.1%` row (`ESS 3.5`), and that is optimistic, since Kish ESS understates the error for weights as skewed as these (Finding 6a below: the top `1%` of `lensed_Ia` events carry `55%` of the total weight).

**What this does and does not block.** It does not block the deliverable: the ordering needs no calibration pool, and the `10%`/`1%` rows are solid. What it blocks is attaching a *trustworthy yield figure* to the sharp end of the list -- precisely where a spectroscopic programme operates. The `63x` lift at `100/yr` above is the same statement as the `0.1%`-FPR row and carries the same `~50%` uncertainty. Fixing that is a simulation-sampling problem, not an analysis problem, and is tracked separately (see Finding 6a and [issue #37](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/37)).

**Rate weighting.** Every purity, composition and yield figure in this Finding is rate-weighted via `cmsne.importance_sampling`. An earlier draft of this script computed tier purity from raw simulated counts; on smoke data that read `~65%` purity at the `10%` budget where the rate-weighted answer is `0.09%` -- a ~700x overstatement, the same class of bug as [PR #57](https://github.com/anachoponte/Colour-Magnitude-SNe/pull/57). Class weights are renormalised to annual on-sky rates (`annual_shares()`) before any statistic is taken, so `TP/yr` and `FP/yr` are directly comparable to Findings 2 and 5. The per-event CSVs carry both the raw counts and the annual-rate columns, explicitly suffixed `_display_only` where the cumulative figure is a plotting convenience rather than an independent estimate.

> **Caveat:** these numbers are conditioned on `results/dual_59297152`, whose `calib` background ESS sets the `0.54%` true FPR floor quoted above. Any regeneration of the simulation pool -- in particular the importance-sampling proposal work in Finding 6a -- will move every figure in this Finding, and the two strict budgets specifically are expected to become measurable for the first time. The lift-curve numbers are the more durable half of this Finding; the budget table is the half with a known expiry.

### Finding 6a: Why the Strict Budgets Are Unmeasurable -- Importance-Sampling Proposal Mismatch (`diag_calib_resolution.py`, spikes)

Finding 6's floor is not a quirk of one split. `scripts/diag_calib_resolution.py` (BlueBEAR job `62176945`) measures where the effective sample actually goes:

| Pool | Raw events | ESS | ESS/raw | Top `1%` of events hold |
|---|---|---|---|---|
| `lensed_Ia` | `33,147` | `623` | `1.9%` | `55.0%` of weight |
| `all_lensed` | `85,381` | `876` | `1.0%` | `61.6%` |
| `lensed_sn2n` | `29,770` | `373` | `1.3%` | -- |
| `lensed_sn2l` | `7,412` | `214` | `2.9%` | -- |
| `lensed_sn1bc` | `6,589` | `73` | `1.1%` | -- |
| `lensed_sn2p` | `8,463` | `53` | `0.6%` | -- |
| background (all unlensed) | `26,383` | `9,282` | `35.2%` | `7.0%` |

The cause is a proposal mismatch. Every population in this project is drawn **uniformly** in `(z, mu)` and reweighted by the physical rate `w(z, mu) = f(z) * dtau_dmu_dz`, with `dtau_dmu_dz = (d_com(z)/31)^3 * mu^-3`. Over `mu in [2, 50]`, the `mu^-3` factor alone caps lensed ESS at `~5%` before redshift is considered, and the `min_detections = 5` cut makes it worse, since detection favours high `mu` -- exactly where `mu^-3` is smallest. This is **variance, not bias**: the weights do correct for the proposal, so every number in Findings 2, 5 and 6 is unbiased. They are simply far noisier than their raw event counts suggest. Boosting (`generate_with_boosting(pilot_frac=0.85)`) cannot fix it: it patches sparse *cells* in colour-magnitude space on top of a proposal that is mismatched *globally*, and its mixture correction assumes a uniform pilot (`p_uniform_const`).

Brute force is expensive: reaching the `0.1%` budget needs `~5.4x` more background generation, and `0.01%` needs `~54x`. Two throwaway spikes (`scripts/spike_matched_proposal_gain.py`, `scripts/spike_proposal_ceiling.py`; jobs `62201604`, `62201753`) tested whether a better proposal is the cheaper route, by measuring **ESS per generation attempt** -- the compute-relevant metric, not ESS per kept event:

- A proposal matched to the **rate density** -- the intuitive fix -- **loses for 8 of the 10 classes**, and `16x` for `lensed_sn2l`. It concentrates draws where the rate density is large, which is where events are too faint to pass the detection cut. Critically it loses on the *background* classes (`unlensed_Ia` `0.63x`), which are the pool that gates the FPR budgets. This result killed an already-written design for exactly that proposal before any production code was touched.
- The variance-optimal proposal under a detection cut is `p ∝ rate * sqrt(eps)` by Cauchy-Schwarz, where `eps(z, mu)` is detection efficiency -- not `rate`, and not `rate * eps`. It **wins for all 10 classes**, by `3.2x` (`unlensed_sn2n`) to `60.4x` (`lensed_sn2l`), including `3.2-11.7x` on the background pool.

These are *ceilings*, assuming perfect knowledge of `eps`; a real implementation must estimate it from the pilot stage and regularise it, and will realise less. Sanity check: the spike's uniform baseline reproduces the measured values well for the Ia classes (`unlensed_Ia` `0.0682` vs `0.0682`) but only within `~2-3x` for the CC subtypes, so the CC factors are order-of-magnitude. Design work is tracked in `docs/superpowers/specs/2026-11-20-matched-importance-proposal-design.md` and [issue #37](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/37); **nothing in `cmsne/` has been changed**, and no number anywhere in this document depends on that work landing.

---

### Finding 7: The `0.1%` FPR Budget Is a Sample-Size Problem With a Known Price -- Why the Clever Proposal Failed and Brute Force Won (`spike_tail_transfer.py`, `run_brute15_bluebear.sh`)

Finding 6 established that no operating point below the `1%` FPR budget is
statistically clean, and attributed it to importance-sampling proposal
mismatch ([issue #37](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/37)).
That attribution was correct about the *cause* and wrong about the *cure*.

**The floor is arithmetic, not mystery.** The finest measurable FPR is

```
FPR_floor = MIN_SAMPLE_FOR_STATS / ESS_background
```

The baseline's calibration-split background ESS of `1,856` gives
`10 / 1,856 = 0.54%` -- exactly where Finding 6's clean operating points stop.
Because ESS is linear in `N` at fixed proposal, the required generation
multiple follows directly:

| Effective events at threshold | Poisson error | ESS needed for `0.1%` | x baseline |
| :--- | :---: | ---: | ---: |
| `10` | `32%` | `10,000` | `5.4x` |
| `25` | `20%` | `25,000` | `13.5x` |
| `100` | `10%` | `100,000` | `54x` |

This table prices the **background** side only, which was the binding side on
the baseline. At 15x the binding side turned out to be **signal** for three CC
subtypes (see "What 15x delivered" below). The `54x` row is arithmetic, not a
plan.

**Why the detection-aware proposal was abandoned.** The variance-optimal
proposal under noisy importance sampling is `p ∝ rate * sqrt(eps)` (see
`docs/superpowers/specs/2026-11-20-detection-aware-importance-proposal-design.md`
for the derivation and verified bibliography). It minimises the variance of a
class's **total** annual detection rate. But a strict FPR threshold is set by
the extreme **tail** of the background *score* distribution, and the two are
anti-correlated: `rate*sqrt(eps)` concentrates attempts on bright, common,
easily-detected bins, while the unlensed mimics of a lensed Ia occupy a rare,
faint corner. `scripts/spike_tail_transfer.py` (job `62298093`, target
`lensed_ia`, `+60d`, val split) measured ESS per attempt relative to uniform:

| Background score tail | `gain_global` | `gain_tail` | Tail-aware ceiling | Raw events in tail |
| :--- | :---: | :---: | :---: | ---: |
| top `10%` | `1.65x` | `1.70x` | `2.89x` | `438` |
| top `1%` | `1.65x` | `1.23x` | `11.61x` | `80` |
| top `0.1%` | `1.65x` | **`0.75x`** | `21.63x` | `45` |

At the `0.1%` tail the optimised proposal is **worse than uniform**. Two
corrections to Finding 6's own account follow: the background gain is `1.65x`
pooled, **not** the `3.4x` previously quoted (that came from `unlensed_Ia`, one
of the two classes whose weights carried the `n_attempted` deficit fixed in
`0489253`; clean per-class values are `1.42x-1.53x`), and `1.65x` would move the
floor only from `0.54%` to `~0.33%` -- it does **not** unlock `0.1%`. The spike
cost 83 seconds and killed a design that was already written.

**Why not the tail-aware proposal.** The same algebra gives the tail-optimal
proposal as `p ∝ rate * sqrt(eps * f)`, where `f_b` is the fraction of bin `b`'s
detections landing in the tail (restricting the estimator to a tail multiplies
each bin by `f_b`, i.e. the global formula with `eps -> eps*f`). Its ceiling is
`21.63x`. It was not pursued because `f` depends on a **fitted classifier**,
closing the loop `proposal -> sample -> classifier -> f -> proposal`. This is
categorically harder than the `eps` loop: `eps` is a property of the universe and
the cadence, fixed and unchanged by the proposal, whereas `f` is a property of a
model trained on the proposal's own output. The loop can be broken by freezing a
stage-1 classifier fitted on a uniform pilot and never refitting on stage-2 data
-- at the cost of reporting with the classifier trained on the smaller sample.
Compounding this, the `21.63x` is fitted to **45 raw events** across 5 classes
and 24 bins each; `unlensed_sn2n` reading `0.13x` at the `1%` tail is
instability, not signal. No cross-fit estimate was made.

**What precision to target, and what the literature does.** Two standards exist,
and they differ by an order of magnitude in cost.

For an FPR used as a **selection threshold** -- this project's case -- the field
operates at roughly ten events and does not quote an error bar. The Strong
Gravitational Lens Finding Challenge ([Metcalf et al. 2019, A&A 625,
A119](https://www.aanda.org/articles/aa/full_html/2019/05/aa32797-18/aa32797-18.html))
uses `TPR_0` (highest TPR before a *single* false positive in `100,000`) and
`TPR_10` (TPR at fewer than *ten* false positives) as its headline metrics --
`TPR_10` is `FPR = 1e-4` at ten events, which is precisely this project's
`MIN_SAMPLE_FOR_STATS = 10`. The closest methodological precedent,
[HOLISMOKES XXI Part II](https://arxiv.org/html/2605.05318) (lensed SNe Ia from
LSST-like multi-band time series), quotes its TPR at an FPR of `O(10^-4)` --
order of magnitude only -- on a test set of `~88,600` with `~80,000` background,
i.e. `~8` surviving events, with **no stated statistical uncertainty**.

For a contamination fraction entering a **cosmology error budget**, the standard
is far tighter: [Jones et al. 2017](https://arxiv.org/html/1611.07042.pdf) find
`2%` core-collapse contamination shifts mean distances by `0.02-0.04` mag with
the contamination systematic at `~29%` of the statistical error, and LSST/DES
forecasts quote contamination to `~10%` relative on a `~1%` baseline
([Carreres et al. 2025](https://arxiv.org/pdf/2507.00157)). That standard does
not apply here: this project's FPR sets a follow-up triage cut, not a `w` error
budget.

**The run.** `scripts/run_brute15_bluebear.sh` targets `25` effective events at a
`0.1%` cut (`20%` Poisson error) -- clearing the field precedent rather than
matching it, since we would report an error bar where the comparable papers
report none. BlueBEAR job `62368493`, `1,500,000` attempts/class across all 10
classes as a 10-way array: **completed**, `7h45m` wall clock, `~55` CPU-hours,
`9.5 GB` in `results/brute15_62368493`. Detection fractions track the baseline
closely (`lensed_Ia` `32.5%` vs `34.2%`), confirming nothing pathological at the
larger `N`. Boosting was deliberately left at the baseline `pilot_frac=0.85`:
holding the proposal fixed is what makes the ESS-linear-in-`N` scaling valid.
Generation peak RSS was `6-22 GB` against a `120 GB` request; the fixed OpSim
overhead dominates.

**The analysis ran once the filesystem recovered.** Two first attempts (jobs
`62461638`, `62633153`) were killed at the five-minute mark because the BlueBEAR
`/rds` GPFS filesystem was returning `ENOSPC` on all writes, including to a home
directory at `4.8%` of quota and including `rm`. Reads were unaffected and the
generated data was intact. The fault cleared on its own; no support ticket was
needed. `rank_candidates.py` then ran per target as jobs `62886634`-`62886638`
(`scripts/run_rank_one_bluebear.sh`). Memory is dominated by the shared
`~299k`-event background pool, not the signal class: every single-class job
peaked at `61.2 GB`, so the script needs `96 GB` per target and `240 GB` for
`all_lensed`. A `64 GB` allocation OOM-kills every target. Outputs are in
`docs/figures/purity_investigation/brute15/`.

**Annual rates are now derived from the simulation, not hardcoded** (`a11d2e6`,
[issue #63](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/63)).
The earlier `ANNUAL_RATES` table was built from `CC_FRACTIONS` (Li et al. 2011;
Shivvers et al. 2017). Those are volumetric, *unlensed* subtype demographics:
correct as an input to the LTT event weight, wrong as the *observed lensed*
composition. The simulation turns one into the other through detectability and
lensing optical depth. The table skipped that step and got the subtype
**ordering** wrong, not just the scale. `annual_rates_from_data()` now sums
`annual_detection_rate` from each class's `summary.json`. On
`results/brute15_62368493`, per year:

| Target | Derived rate | Base purity (rate / (rate + background)) |
| :--- | ---: | ---: |
| `all_lensed` | `449.3` | `0.0558%` |
| `lensed_sn2n` | `272.2` | `0.0338%` |
| `lensed_Ia` | `148.8` | `0.0185%` |
| `lensed_sn2p` | `17.8` | `0.0022%` |
| `lensed_sn1bc` | `10.1` | `0.0013%` |
| `lensed_sn2l` | `0.40` | `0.00005%` |
| background (all `unlensed_*`) | `804,889` | -- |

These replace the `10/yr` vs `100,000/yr` and `45/yr` placeholders used in
Findings 2-6 (which remain as measured on `results/dual_59297152`). The
independent ESS-targeted run below reproduces them to `0.3-3%` (`lensed_sn2n`
`272.9`, `lensed_sn2p` `18.2`, `lensed_sn1bc` `10.4`). That is a useful
consistency check across two generations on the same cadence.

The rate correction is **exactly analytic** for purity. `annual_shares` is linear,
and thresholds, ESS, TPR and FPR are all calibrated on weights that never see
the rates, so `purity *= (N_sig_new/N_sig_old) / (N_bg_new/N_bg_old)`. The
re-run matched the predicted `lensed_Ia` purities to three figures. It was a
check, not a rescue.

**What 15x delivered: background now clears `0.1%` everywhere, and signal
became the binding side.** The ESS table above counts background only, because
Finding 6's floor was set by background. Measured on the `val` split
(cumulative operating points, rate-weighted), the `0.1%` budget is now
resolvable for every target at every epoch. Effective events at the threshold,
signal / background:

| Target | `+0d` | `+3d` | `+14d` | `+30d` | `+60d` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `lensed_Ia` | `73.0` / `17.9` | `76.6` / `33.6` | `130.6` / `31.0` | `124.1` / `33.3` | `80.2` / `32.0` |
| `lensed_sn2n` | **`3.6`** / `57.7` | `15.0` / `89.0` | `13.5` / `67.5` | **`7.0`** / `29.5` | `13.7` / `52.9` |
| `lensed_sn2p` | `13.0` / `18.1` | `27.5` / `30.7` | **`2.6`** / `50.7` | **`2.5`** / `37.1` | `26.4` / `30.5` |
| `lensed_sn1bc` | `20.5` / `29.1` | `10.4` / `12.9` | `14.3` / `12.9` | `20.5` / `12.7` | `21.1` / `10.6` |
| `lensed_sn2l` | **`4.8`** / `20.3` | `20.5` / `27.3` | `12.7` / `23.3` | **`3.4`** / `17.0` | **`6.0`** / `18.1` |
| `all_lensed` | *pending (job `62886633`)* | | | | |

Bold marks `THIN SAMPLE` (any ESS or raw count `< MIN_SAMPLE_FOR_STATS = 10`).
**18 of 25 operating points at `0.1%` are now clean, against 0 of 30 in Finding
6.** Every background ESS is `>= 10.6`. All seven thin points are thin on the
**signal** side, in three CC subtypes. All 50 operating points at the `10%` and
`1%` budgets are clean. `0.01%` remains thin for most targets and epochs and
stays out of scope (see below).

Two things this table says about the method, not just the sampling:

- **ESS at a tail cut is lumpy, not smooth in epoch.** `lensed_sn2p` goes
  `13.0 -> 27.5 -> 2.6` across consecutive epochs. A tail threshold's ESS is set
  by whether a few heavy-weight events fall just above or just below the cut, so
  "ESS is linear in `N`" holds in expectation, with large epoch-to-epoch scatter
  around it. Budgets sized from the *worst* epoch are a target, not a guarantee.
- **The purities these ESS support** (`0.1%` budget, cumulative; `*` = thin):

| Target | `+0d` | `+3d` | `+14d` | `+30d` | `+60d` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `lensed_Ia` | `0.49%` | `0.65%` | `1.98%` | `0.76%` | `0.36%` |
| `lensed_sn2n` | `0.021%`* | `0.020%` | `0.041%` | `0.066%`* | `0.076%` |
| `lensed_sn2p` | `0.004%` | `0.006%` | `0.094%`* | `0.058%`* | `0.024%` |
| `lensed_sn1bc` | `0.006%` | `0.021%` | `0.040%` | `0.033%` | `0.044%` |
| `lensed_sn2l` | `0.000%`* | `0.000%` | `0.000%` | `0.000%`* | `0.000%`* |

  `lensed_Ia` is the only target with purity above `0.1%` at this budget. It
  peaks at `+14d`, with `8.3` true candidates/yr among `413` false. The `+14d`
  peak is a feature of the decision epoch: purity *falls* again by `+60d`.
  `lensed_sn2l` is a null result, not an omission: at `0.40/yr` against
  `805k/yr` of background it cannot produce a candidate list at any budget, and
  it is reported for completeness. `lensed_sn2n` is the substantive surprise and
  gets its own Finding (8).

**The ESS-targeted run** (`scripts/run_ess_targeted_bluebear.sh`, in progress).
A flat multiple buys `lensed_Ia` signal it already has (worst ESS `73`) while
leaving three CC subtypes short. The follow-up therefore scales each class by
`max(1, 13/worst_ESS)`. It targets `13` rather than `10` so the worst epoch
clears the bar instead of landing on it.

| Class | 15x attempts | Worst `0.1%` ESS (15x) | Budget |
| :--- | ---: | :---: | ---: |
| `lensed_Ia` | `1.5M` | `73.0` | `1.5M` (`1x`) |
| `lensed_sn1bc` | `1.5M` | `10.4` | `1.88M` |
| `lensed_sn2n` | `1.5M` | `3.6` | `5.42M` |
| `lensed_sn2p` | `1.5M` | `2.5` | `7.8M` |
| `lensed_sn2l` | `1.5M` | `3.4` | `5.74M` |
| each `unlensed_*` | `1.5M` | `10.6` (background, `sn1bc` target) | `1.84M` |

Boosting stays at `pilot_frac=0.85`, because the linear-in-`N` scaling holds only
at fixed proposal. **The first submission (array `62886623`) mis-mapped the
budgets.** `run_dual_schema.py`'s `CLASSES` order interleaves unlensed/lensed
per subtype after the first two entries, and the budget table assumed
all-lensed-then-all-unlensed. Eight of ten budgets landed on the wrong class:
`lensed_sn2n` and `lensed_sn2p`, the two binding classes, got `1.84M` each, while
`lensed_sn1bc`, `lensed_sn2l` and `unlensed_sn2l` were over-provisioned. Fixed
in `5d3f600`. Corrective job `63065791` regenerates `lensed_sn2n` (`5.42M`) and
`lensed_sn2p` (`7.8M`). Job `63073971` regenerates `lensed_sn2l` at its
designed `5.74M` with a 72h limit, as insurance for the mis-mapped `7.8M` task
(`62886623_5`), which risks overrunning its 48h limit. **No number in this
document yet depends on the ESS-targeted run.** Whether it gets every target
above `13` at `0.1%` will be reported per target once it lands.

**Out of scope: `0.01%`.** Reaching it would need background `x4.3` and
`lensed_sn2l` `x6.7` (`10M` attempts alone, for a class with `~4` events per
decade), taking the run to `~55M` attempts. Reporting `0.1%` with honest ESS
and marking `0.01%` unresolved already exceeds the precedent above. This
supersedes the `54x` row of the first table as a plan: that row remains correct
arithmetic but is not being pursued.

**Caveats carried forward.** `results/dual_59297152` was generated with
`baseline_v5.3.2_10yrs.db`, which no longer exists on BlueBEAR; only `v5.3.5` is
present. The 15x and ESS-targeted runs therefore use a different survey cadence,
and their **absolute** rates are not comparable to Findings 2-6. Each run is
internally consistent. Finding 5's false-positive composition (`unlensed_Ia` at
`66.6%`) also shifts under the `0489253` weight fix and has not yet been
recomputed.

---

### Finding 8: The Most Common Lensed Supernova Is the One the Colour Method Cannot Find -- Lensed SNe IIn (`rank_candidates.py`, 15x run)

With rates derived from the simulation (Finding 7), `lensed_sn2n` is the
**dominant lensed class by number**: `272.2/yr`, `1.83x` `lensed_Ia`'s `148.8/yr`
and `61%` of all lensed SNe (`449.3/yr`). The hardcoded table had it as a minor
subtype. It is also the class the colour method separates **worst**. This
changes what the method can claim, not just its numbers.

**TPR at the `10%` FPR budget** (cumulative, `val` split, rate-weighted; all
points clean):

| Target | `+0d` | `+3d` | `+14d` | `+30d` | `+60d` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `lensed_Ia` | `36.0%` | `42.2%` | `48.9%` | `55.7%` | `50.7%` |
| `lensed_sn2p` | `26.0%` | `39.5%` | `49.7%` | `44.9%` | `37.9%` |
| `lensed_sn1bc` | `36.9%` | `41.3%` | `47.4%` | `45.1%` | `41.2%` |
| `lensed_sn2l` | `28.2%` | `40.5%` | `34.0%` | `22.5%` | `21.2%` |
| **`lensed_sn2n`** | **`11.6%`** | **`12.9%`** | **`19.0%`** | **`21.1%`** | **`22.6%`** |

`lensed_sn2n` has the lowest TPR at four of five epochs. At `+60d`,
`lensed_sn2l` is marginally lower. At `+30d` it is `21%` against `lensed_Ia`'s
`56%`.

**The sharper statement is in enrichment** (purity divided by the base purity
the target would have under random selection). For `lensed_Ia` and
`lensed_sn1bc`, tightening the budget concentrates signal: enrichment rises from
`~4-5x` at `10%` to `~5-107x` at `0.1%`, mostly `>20x`. `lensed_sn2p` does the
same from `+14d` (`11-42x`), but not at `+0d`/`+3d` (`1.8-2.5x`). The other type
II with a strict-budget problem, `lensed_sn2l`, peaks at `9.5x` (`1%`, `+3d`)
and stays `<= 3.2x` at `0.1%`. `lensed_sn2n` is the extreme case. Its
enrichment barely rises at all and never exceeds `2.3x` anywhere:

| `lensed_sn2n` enrichment | `+0d` | `+3d` | `+14d` | `+30d` | `+60d` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `10%` FPR | `1.14x` | `1.30x` | `1.91x` | `2.12x` | `2.29x` |
| `1%` FPR | `0.93x` | `0.90x` | `0.94x` | `1.76x` | `2.13x` |
| `0.1%` FPR | `0.63x`* | `0.58x` | `1.22x` | `1.96x`* | `2.24x` |
| *`lensed_Ia` at `0.1%`, for comparison* | *`26.6x`* | *`35.1x`* | *`106.9x`* | *`41.0x`* | *`19.4x`* |

(`*` = thin sample.) At every budget and epoch, the ranking puts `lensed_sn2n`
at the top of the list **no more than `~2.3x` more often than chance**. At
`+0d`-`+14d` and strict budgets it is consistent with chance. The `+3d`/`0.1%`
value (`0.58x`, signal ESS `15.0`, i.e. `~+/-26%` relative) sits nominally
`~2.8 sigma` below unity. Kish ESS understates the error for weights this
skewed (Finding 6), and this is one of 15 values in the table, so it is
**flagged, not claimed**, as anti-selection. What is established is the absence
of selection.
At `0.1%`, `lensed_sn2n` purity is `4.7x`-`48x` lower than `lensed_Ia`'s,
depending on epoch (`0.020-0.076%` vs `0.36-1.98%`). Because `lensed_sn2n` is the
more common class, the gap in *base* purity runs the other way (`0.034%` vs
`0.019%`), so the whole difference comes from separation.

**Absolute yield tells the same story with a twist.** At the loose `10%` budget,
volume partly compensates: `lensed_sn2n` contributes `32-61` true candidates/yr,
comparable to `lensed_Ia`'s `54-83`. At `0.1%` it contributes `0.15-0.73/yr`
against `lensed_Ia`'s `3.0-8.3/yr`. The colour method recovers lensed IIn only in
a broad, highly contaminated alert stream, and loses them in a spectroscopic
follow-up list.

**What this means for the paper's claims.**
- The method's headline is **lensed SNe Ia**, not "lensed SNe". Of the classes
  with a non-negligible rate, `lensed_Ia` is the only one with `0.1%` purity
  above `0.1%`. `lensed_sn1bc` (from `+3d`) and `lensed_sn2p` (from `+14d`)
  separate well in *enrichment* (`~11-42x` at `0.1%`) but are too rare
  (`10.1/yr`, `17.8/yr`) to matter in yield.
- **`all_lensed` will be dominated by its weakest member.** `61%` of its rate is
  `lensed_sn2n`, which quantitatively explains Finding 6's observation that
  `all_lensed` is far weaker at the sharp end than `lensed_Ia` alone. The
  `all_lensed` 15x result (job `62886633`, pending) should be read against this.
- A colour-only trigger aimed at "lensed supernovae" in general would miss most
  of them by number. That is a scope statement the paper should make
  explicitly, not leave for a referee.

**Why, and what is not yet established.** Two hypotheses are plausible and
**untested** here. (1) SNe IIn have a broad intrinsic luminosity function, so
the `0.75-4.2 mag` brightening from `mu in [2, 50]` is partly degenerate with
intrinsic scatter, removing the "too bright for its colour" signature the method
relies on for Ia. (2) IIn colours are blue and slowly evolving, and so less
distinctive than the Ia colour trajectory. Separately, the `272.2/yr` rate is
**model-dependent**. It inherits the simulation's IIn template, luminosity
function and `CC_FRACTIONS` input, and IIn are luminous enough to be detected to
high `z`, where lensing optical depth is largest. The ordering
`lensed_sn2n > lensed_Ia` should be checked against the IIn luminosity-function
assumption before it is stated as a prediction for LSST. Both the TPR and the
enrichment results above are robust to the rate: they are calibrated on weights
that never see it (Finding 7).

---

## 4. Index of Generated Figures & Data Tables

- **`docs/figures/` (Root):**
  - Updated clean-background ROC curves & animated GIFs (`roc_overall*.png`, `roc_by_step*.png/gif`, `roc_by_subtype_alllensed.png`, `rate_by_subtype_alllensed.png/gif`).
  - Snapshot vs `peakmag` benchmark (`comparison_recovery_snapshot_vs_peakmag.png`, `comparison_snapshot_vs_peakmag.json`).
- **`docs/figures/purity_investigation/`** (Finding 2, `investigate_purity_tradeoff.py`, fixed for [issue #51](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/51) and regenerated with [issue #50](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/50)'s fix):
  - Causal, per-decision-epoch ROC curves and the "cost of deciding early" plots (`roc_by_decision_epoch{,_alllensed}.png`, `cost_of_early_decision{,_alllensed}.png`, `purity_investigation{,_alllensed}.json`).
  - **Finding 5** (`investigate_contaminant_properties.py`, all 6 targets -- suffixes `{,_alllensed,_snibc,_sn2l,_sn2n,_sn2p}`): ranked-follow-up yield/purity vs. FPR budget (`ranked_followup_summary*.png`), false-positive subtype composition by budget (`fp_subtype_composition*.png`), redshift distribution of true vs. false positives at the 10%-FPR budget with the secondary `z`-window cut marked (`contaminant_redshift_10pct*.png`), the same at the 10%-FPR budget for survey epoch of first detection (`contaminant_trigday_10pct*.png`, diagnostic only) and final photometric uncertainty (`contaminant_snr_10pct*.png`, diagnostic only), and the full per-epoch/per-budget tables (`contaminant_properties*.json`).
  - **Finding 5, out-of-sample validation** (`validate_zcut_outofsample.py`, all 6 targets, same suffixes): clean-`z` vs. photo-z-noise-stressed `z`-cut gain and before/after purity, both out-of-sample, by decision epoch (`zcut_outofsample_validation*.png`), and the full per-epoch/per-budget tables including both the clean and noisy results (`zcut_outofsample_validation*.json`).
  - **Finding 6** (`rank_candidates.py`, all 6 targets -- suffixes `{,_alllensed,_snibc,_sn2l,_sn2n,_sn2p}`): the review-budget lift curve, cumulative annual candidates reviewed vs. cumulative annual true positives recovered, log-log (`rank_lift*.png`); the rate-weighted score distribution of true vs. false positives with each budget's threshold marked (`rank_distribution*.png`); and the full per-epoch tables -- thresholds, exclusive tier stats, cumulative operating points, per-budget `resolvable` flags, `calib_n_bg`, `finest_resolvable_fpr`, and a 202-point sampled lift curve (`rank_candidates*.json`).
    - **`brute15/`** (Findings 7-8): the same `rank_candidates.py` outputs (`rank_lift*.png`, `rank_distribution*.png`, `rank_candidates*.json`) on `results/brute15_62368493` with derived annual rates, jobs `62886634`-`62886638`. Five targets so far (unsuffixed = `lensed_ia`); `all_lensed` (`_alllensed`) is pending job `62886633`. Its per-event CSVs (`~117 MB`) stay on BlueBEAR.
    - The per-event ranked lists themselves (`ranked_candidates*.csv`, 6 files, 9.4 MB) are **deliberately not committed**, matching this directory's existing convention of tracking figures and JSON tables only. They regenerate from `scripts/run_rank_candidates_bluebear.sh` in ~37 minutes. Each carries both raw simulated counts and annual-rate columns, with the cumulative annual columns explicitly suffixed `_display_only`.

- **`docs/figures/weight_encodings/`:**
  - Side-by-side visual comparisons of `unweighted`, `size`, `alpha`, `colour`, and `size_alpha` (`encoding_comparison_gr_day0*.png`, `encoding_comparison_gr_day14*.png`, `boundary_gr_corner_*`).
- **`docs/figures/snapshot_size_alpha/` (80 PNGs):**
  - Full `ug, gr, ri, iz, zy` suite in **snapshot magnitude mode** with `size_alpha`, `PuBu` + high-contrast markers, binary & 10-class views, and `1σ` error-bar versions (`*_errorbars.png`).
- **`docs/figures/standard_size_alpha/` (80 PNGs):**
  - Full `ug, gr, ri, iz, zy` suite in **matched cumulative-best `peakmag` mode** with `size_alpha`, `PuBu` + high-contrast markers, and `1σ` error-bar versions (`*_errorbars.png`).
- **`docs/figures/subtypes_size_alpha/` (48 PNGs):**
  - Individual core-collapse lensed subtypes (`lensed_sn1bc`, `lensed_sn2l`, `lensed_sn2n`, `lensed_sn2p`) in `g-r` and `r-i` against strictly unlensed background, with `size_alpha`, high-contrast colours, and `*_errorbars.png` counterparts.
- **`docs/figures/model_comparison/`:**
  - 6-architecture benchmark plots (`trajectory_model_comparison_lensed_ia.png`, `trajectory_model_comparison_alllensed.png`) and raw JSON metrics (`*.json`), regenerated with [issue #50](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/50)'s fix applied (see Finding 4).
  - The `slide_table_*.csv` files are a separate, manually-maintained presentation artifact, **not regenerated by this script**. Manually refreshed against `trajectory_model_comparison_{lensed_ia,alllensed}.json` (post-#50-fix): every `ROC-AUC`/`TPR @ X% FPR` cell matches Finding 4's table exactly, `Zero-FP Recall`/`Pure Yield` now read `0.0%`/`0.0` for all 6 models (consistent with Finding 4's zero-FP collapse), and `Fit Time` reflects the corrected rerun's timings. They are not auto-regenerated by any script, so a future benchmark rerun will need the same manual refresh.
