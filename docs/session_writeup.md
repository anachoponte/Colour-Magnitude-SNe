# Session Writeup: Rate-Weighted Diagnostics, Purity Trade-offs, Visual Encodings, and Full-Trajectory Classifier Upgrades

**Dataset:** `results/dual_59297152` on BlueBEAR (`1,000,000` simulated LSST supernovae across 10 classes: `100,000` per class for `lensed_Ia`, `lensed_sn1bc`, `lensed_sn2l`, `lensed_sn2n`, `lensed_sn2p` and their 5 `unlensed_*` counterparts, evaluated on a held-out 70/30 train/test split with physical annual LSST rate weights).

---

## 1. Executive Summary

Across this session, we audited, visualised, and substantially upgraded the LSST gravitationally lensed supernova classification pipeline:

1. **Removed Cross-Subtype Lensed Contamination:** Fixed both `scripts/plot_roc.py` and `scripts/plot_evolving_boundaries.py` so that when evaluating an individual lensed subtype (such as `lensed_Ia`), other lensed supernova subtypes are **never** treated as false-positive background contaminants (`background = unlensed_*` strictly).
2. **Quantified the LSST Base-Rate Trap & Zero-False-Positive Sweet Spot:** Demonstrated how the `1 : 10,000` (`10/yr` lensed Ia vs `100,000/yr` unlensed -- the literature figure already cited elsewhere in this project, Oguri et al. 2010; Wojtak et al. 2019; Goldstein et al. 2019, not the uncited `50/yr` and `1,000,000/yr` an earlier draft of this document used) and `~1 : 2,200` (`45/yr` all lensed vs `100,000/yr` unlensed -- an order-of-magnitude placeholder pending [issue #43](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/43), not an independently literature-validated figure) cosmic rate ratios affect real telescope follow-up purity, and mapped the `TP / FP` ratio and `FPR = 0` "Zero-False-Positive Sweet Spot".
3. **Resolved the Scatter-vs-Heatmap Visual Paradox (`size + opacity` + High-Contrast Palette + `1σ` Error Bars):** Identified why unweighted scatter plots previously appeared inconsistent with rate-weighted probability surfaces, compared 4 visual rate-weight encodings, adopted **`size + opacity` (`size_alpha`)** with a high-contrast **`PuBu` + vivid marker** palette and **`1σ` photometric error bars**, and regenerated 208 standard boundary figures across all filter pairs (`ug, gr, ri, iz, zy`) and subtypes.
4. **Benchmarked Improvements & Non-Tree Replacements for the Full-Trajectory Classifier (`~4.1x` Increase in Zero-FP Pure Discoveries vs the Deployed Baseline):** Built a 6-model benchmark testing smooth Sigmoid calibration, physics-enriched trajectory features (snapshot magnitudes, time-dilated fade from peak, broad-baseline colours, photometric uncertainties), causal prefix training for early `+0d .. +14d` alerts, Smooth Spline GAMs, Mask-Aware Neural MLP ensembles, and a **Hybrid (Physics-Enriched GBDT + Mask-Aware Smooth MLP)** ensemble. Measured against the *real, currently-deployed* classifier's own zero-false-positive operating point (Finding 2: **0.6 lensed Ia/yr**, 5.6% recall, from running the actual production `TrajectoryClassifier`), the Hybrid ensemble's pure zero-false-positive yield of **2.3 lensed Ia/yr (23.3% recall)** is a **~4.1x improvement** -- smaller than the `>7x` an earlier draft of this document quoted, which compared against this benchmark's own internal, non-deployed "Model 1" row (0.3/yr) instead of the deployed baseline; see Finding 4's caveat for why those two "baselines" differ (Model 1 omits the production default's noise-injection replication). The AUC comparison is unaffected by this: `+14d` early-warning `ROC-AUC` jumped from **0.846 to 0.924** within the benchmark, surpassing the weakest model's `+60d` AUC (0.918) within 2 weeks of trigger.

---

## 2. Summary of All Code Edits & New Scripts

### A. Modified Existing Scripts

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
4. **`scripts/investigate_purity_tradeoff.py`** (SLURM wrapper: `scripts/run_investigate_purity_bluebear.sh`)
   - Computes annual LSST `TP/yr`, `FP/yr`, `TP : FP` ratios, sample purity `TP / (TP + FP)`, and the empirical `FPR = 0` pure sweet spot across the full ROC curve for both `lensed_ia` (`10/yr`, Oguri 2010; Wojtak 2019; Goldstein 2019) and `all_lensed` (`45/yr`, order-of-magnitude placeholder pending issue #43) against `100,000/yr` unlensed SNe.
5. **`scripts/plot_weight_encodings.py`** (SLURM wrapper: `scripts/run_weight_encodings_bluebear.sh`)
   - Generates 4-panel (`1x4`) and 5-row (`5x4`) comparison figures contrasting `unweighted`, `size`, `opacity`, `colour`, and `size + opacity` encodings on identical data slices.
6. **`scripts/compare_trajectory_models.py`** (SLURM wrapper: `scripts/run_model_comparison_bluebear.sh`)
   - Benchmarks 6 full-trajectory classification architectures across full (`+60d`) and causal partial (`+0d, +3d, +14d, +30d, +60d`) trajectories, outputting 3-panel comparison figures and JSON/CSV metric tables.
7. **Parallel BlueBEAR Plot-Suite Scripts:**
   - `scripts/run_plots_parallel_bluebear.sh`, `scripts/run_standard_size_alpha_bluebear.sh`, `scripts/run_plots_ia_rem_bluebear.sh`.

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

### Finding 2: The LSST Base-Rate Paradox & Telescope Follow-Up Operating Points (`investigate_purity_tradeoff.py`)
Even when a classifier achieves `ROC-AUC > 0.91`, multiplying `TPR` and `FPR` by true annual LSST rates (`N_sig = 10/yr` for lensed Ia -- Oguri et al. 2010; Wojtak et al. 2019; Goldstein et al. 2019, the same figure already cited in `docs/methods_and_results.md` -- or `45/yr` for all lensed, an order-of-magnitude placeholder pending [issue #43](https://github.com/anachoponte/Colour-Magnitude-SNe/issues/43), vs `N_bg = 100,000/yr` unlensed) exposes why moderate false-positive rates fail for spectroscopic follow-up. These are the real, currently-deployed `TrajectoryClassifier`'s own numbers (isotonic calibration, uncertainty-aware noise injection, `target_fpr=0.10`):

| Operating Regime | Target FPR | Lensed Ia TP / yr | Unlensed FP / yr | Lensed Ia Purity | All Lensed TP / yr | All Lensed Purity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard Alert Broker | 10% | 7.8 / yr | ~10,290 / yr | 0.08% | 19.1 / yr | 0.19% |
| Low Contamination | 1% | 4.4 / yr | ~1,000 / yr | 0.44% | 8.4 / yr | 0.75% |
| Strict Filter | 0.1% | 1.9 / yr | ~133 / yr | 1.41% | 3.9 / yr | 3.97% |
| Spectroscopic Candidate | 0.01% | 1.5 / yr | ~7 / yr | 17.1% | 2.6 / yr | 20.7% |
| **Zero-FP Sweet Spot (Deployed Baseline)** | **0%** | **0.6 / yr (5.6%)** | **0 / yr** | **100%** | **1.0 / yr (2.3%)** | **100%** |

*Conclusion:* Real spectroscopic target selection lives entirely in the `FPR <= 0.01%` to `FPR = 0%` tail—making extreme-tail probability resolution and false-outlier rejection far more important than bulk `FPR > 5%` ROC area. (An earlier draft of this document used uncited 50/250/1,000,000-per-year assumptions here, giving larger but unsupported absolute `/yr` figures; the recall percentages in parentheses are unaffected by that fix and were always correct -- only the annual counts rescale.)

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

> **Caveat on "Model 1: Baseline GBDT" and the word "baseline":** this benchmark's Model 1 fits a bare `HistGradientBoostingClassifier` directly and does **not** include the production default's uncertainty-aware noise-injection replication (`n_uncertainty_replicates=8`, see `cmsne/classifier.py`) that the actually-deployed `cmsne.trajectory.TrajectoryClassifier` always applies. Finding 2's numbers, by contrast, come from running the *real* deployed classifier (via `investigate_purity_tradeoff.py`) on the same data and split, and its zero-FP yield is noticeably higher than this table's Model 1 row -- it lands close to Model 2 instead. So "Model 1" here isolates calibration/feature/architecture choices *from* noise injection; it is not the currently-deployed system's own operating point, and the two tables below should not be read as using the same "baseline". The improvement multiplier quoted in the Executive Summary is computed against Finding 2's real deployed baseline, not against this table's Model 1.

#### Summary Benchmark — Lensed Type Ia SNe (`10 / yr` vs `100,000 / yr` Unlensed)

| Model | AUC (+0d) | AUC (+14d) | AUC (+60d) | Recall @ 10% FPR | Recall @ 1% FPR | Recall @ 0.1% FPR | Zero-FP Recall (FPR = 0%) | Pure Yield / yr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline GBDT** | 0.708 | 0.846 | 0.918 | 71.8% | 40.6% | 21.8% | 3.3% | **0.3 / yr** |
| **2. Smooth-Calibrated GBDT** | 0.708 | 0.844 | 0.924 | 73.8% | 44.3% | 19.2% | 5.6% | **0.6 / yr** |
| **3. Physics-Enriched + Prefix GBDT** | **0.812** | 0.923 | **0.946** | 79.5% | 48.0% | 32.6% | 12.3% | **1.2 / yr** |
| **4. Smooth Spline GAM** | 0.806 | 0.903 | 0.931 | 80.0% | 37.0% | 15.0% | 2.1% | **0.2 / yr** |
| **5. Mask-Aware Neural MLP** | 0.780 | 0.911 | 0.930 | 77.6% | 48.2% | 35.1% | 16.7% | **1.7 / yr** |
| **6. Hybrid (Enriched GBDT + MLP)** | 0.804 | **0.924** | **0.946** | **81.4%** | **52.9%** | **36.8%** | **23.3%** | **2.3 / yr** |

#### Summary Benchmark — All Lensed SNe (`45 / yr` vs `100,000 / yr` Unlensed)

| Model | AUC (+0d) | AUC (+14d) | AUC (+60d) | Recall @ 10% FPR | Recall @ 1% FPR | Recall @ 0.1% FPR | Zero-FP Recall (FPR = 0%) | Pure Yield / yr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline GBDT** | 0.578 | 0.678 | 0.748 | 38.2% | 16.9% | 9.3% | 1.8% | **0.8 / yr** |
| **2. Smooth-Calibrated GBDT** | 0.579 | 0.673 | 0.755 | 40.8% | 18.2% | 8.2% | 2.5% | **1.1 / yr** |
| **3. Physics-Enriched + Prefix GBDT** | 0.645 | **0.733** | **0.783** | 47.2% | 18.8% | **12.1%** | 3.0% | **1.4 / yr** |
| **4. Smooth Spline GAM** | 0.607 | 0.715 | 0.753 | 37.5% | 14.7% | 3.7% | 0.2% | **0.1 / yr** |
| **5. Mask-Aware Neural MLP** | 0.641 | 0.700 | 0.748 | 44.4% | 16.4% | 5.8% | **3.4%** | **1.6 / yr** |
| **6. Hybrid (Enriched GBDT + MLP)** | **0.655** | 0.732 | 0.775 | **49.8%** | **20.2%** | 9.7% | 3.3% | **1.5 / yr** |

#### Why Each Upgrade Worked:
- **Sigmoid vs Isotonic Calibration:** Eliminated top-plateau score ties so a single outlier background event no longer blocks dozens of higher-log-odds lensed SNe (`0.3 -> 0.6/yr`).
- **Physics Enrichment (Time-Dilated Fade + Broad Colours + Errors):** Explicit `mag(t) - peakmag(t)` captures slow time-dilated decline; `g-i, r-z, g-z` rescue epochs where intermediate bands weren't visited; `{band}_err` downweights noisy faint detections.
- **Causal Prefix Augmentation:** Exposing partial `+0d .. +30d` light curves during training fixed Day-0 out-of-distribution `NaN` patterns (`+0d` Ia AUC `0.708 -> 0.812`; `+14d` Ia AUC `0.846 -> 0.924`).
- **Hybrid GBDT + Mask-Aware Smooth MLP:** Pure additive GAMs miss non-linear `redness x brightness` interactions in the extreme tail, while pure GBDTs occasionally assign high scores to boxy corner outliers. Log-odds blending requires **both** the missing-data tree interactions and smooth diagonal neural boundaries to agree—boosting `FPR = 0` pure lensed Ia recovery by **>7x relative to this benchmark's own internal Model 1 row** (`0.3 -> 2.3 / yr`), or **~4.1x relative to the real deployed baseline** (`0.6 -> 2.3 / yr`, see Finding 2 and the Executive Summary).

---

## 4. Index of Generated Figures & Data Tables

- **`docs/figures/` (Root):**
  - Updated clean-background ROC curves & animated GIFs (`roc_overall*.png`, `roc_by_step*.png/gif`, `roc_by_subtype_alllensed.png`, `rate_by_subtype_alllensed.png/gif`).
  - Snapshot vs `peakmag` benchmark (`comparison_recovery_snapshot_vs_peakmag.png`, `comparison_snapshot_vs_peakmag.json`).
- **`docs/figures/purity_investigation/`:**
  - ROC with `TP / FP` ratio & follow-up purity curves (`roc_purity_ratio*.png`, `telescope_followup_yield*.png`, `purity_investigation*.json`).
- **`docs/figures/weight_encodings/`:**
  - Side-by-side visual comparisons of `unweighted`, `size`, `alpha`, `colour`, and `size_alpha` (`encoding_comparison_gr_day0*.png`, `encoding_comparison_gr_day14*.png`, `boundary_gr_corner_*`).
- **`docs/figures/snapshot_size_alpha/` (80 PNGs):**
  - Full `ug, gr, ri, iz, zy` suite in **snapshot magnitude mode** with `size_alpha`, `PuBu` + high-contrast markers, binary & 10-class views, and `1σ` error-bar versions (`*_errorbars.png`).
- **`docs/figures/standard_size_alpha/` (80 PNGs):**
  - Full `ug, gr, ri, iz, zy` suite in **matched cumulative-best `peakmag` mode** with `size_alpha`, `PuBu` + high-contrast markers, and `1σ` error-bar versions (`*_errorbars.png`).
- **`docs/figures/subtypes_size_alpha/` (48 PNGs):**
  - Individual core-collapse lensed subtypes (`lensed_sn1bc`, `lensed_sn2l`, `lensed_sn2n`, `lensed_sn2p`) in `g-r` and `r-i` against strictly unlensed background, with `size_alpha`, high-contrast colours, and `*_errorbars.png` counterparts.
- **`docs/figures/model_comparison/`:**
  - 6-architecture benchmark plots (`trajectory_model_comparison_lensed_ia.png`, `trajectory_model_comparison_alllensed.png`), raw JSON metrics (`*.json`), and slide-ready CSV tables (`slide_table_*.csv`).
