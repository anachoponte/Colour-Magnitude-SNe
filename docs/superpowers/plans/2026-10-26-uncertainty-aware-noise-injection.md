# Uncertainty-Aware Noise Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Smear each event's measurement-uncertain feature values across a cloud of noise-injection replicas, instead of treating every colour/magnitude as an exact point estimate, so a single near-detection-limit measurement can no longer single-handedly dominate a local density estimate or unduly influence the tree.

**Architecture:** New `*_err` fields propagate real per-observation photometric uncertainty (already computed, currently discarded) from `cmsne/lightcurve.py` through `cmsne/multicolour.py`'s feature extraction into `cmsne/classifier.py`. A new primitive, `_replicate_with_uncertainty`, expands each event into `K` noisy copies at `weight/K`; the tree trains on the full replicated population once, while the KDE replicates lazily, per missing-data pattern, only after the existing `kde_max_events` cap. A new `_grouped_effective_sample_size` helper fixes a real correctness wrinkle: naive replication would inflate Kish's ESS by a factor of `K`.

**Tech Stack:** NumPy, scikit-learn (`HistGradientBoostingClassifier`, `CalibratedClassifierCV`), existing `cmsne` package conventions.

**Spec:** `docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md`

## Global Constraints

- **Backward compatibility is automatic, not a flag.** `_replicate_with_uncertainty` must auto-skip (return inputs unchanged, zero extra compute) whenever no finite, positive sigma exists over the relevant dimensions. Every existing test fixture and every already-generated population lacks `*_err` fields, so the ENTIRE pre-existing test suite (PR #35, #36, the subsampling fix, commit `1fe0eb3`) MUST continue to pass with byte-identical results, with no special-casing in the tests themselves.
- **Grouped ESS, not plain ESS, everywhere a possibly-replicated array feeds `effective_sample_size`.** This includes `local_ess_sig`/`local_ess_bg` in `_local_density_ratio`, both ESS checks inside `_widened_kernel_values` (the initial check AND the `ceiling` check), and `_local_density_ratio`'s own ceiling-gate fallback (`ess_ceiling_sig`/`ess_ceiling_bg` when `None`). The ONE exception, which MUST stay a plain, non-grouped computation: the ceiling value cached by `_kde_subset_for_pattern` into `_KDEPatternSubset.ess_ceiling_sig`/`ess_ceiling_bg` is computed from the CAPPED-BUT-PRE-REPLICATION weights (`ws_cap`/`wb_cap`), never from the post-replication `ws_rep`/`wb_rep` -- this is already correct by construction (computed before replication happens), not something that needs "fixing" with grouping.
- **Two different replication injection points, never conflated.** The tree replicates the FULL balanced population once, in `fit()`, before training. The KDE replicates ONLY the already-capped per-pattern subset, inside `_kde_subset_for_pattern`, AFTER its existing finite-filter+cap step (commit `1fe0eb3`'s fix). Replicating the KDE's raw population before capping would reintroduce that exact bug.
- **`_store_kde_training_data` keeps storing the ORIGINAL, un-replicated population.** Replication for the KDE side happens lazily per pattern, not globally at storage time.
- **`peakmag_fit_err` is explicitly out of scope.** Do not attempt to propagate uncertainty for `peakmag_fit` (a parabola-fit quantity, not a single observation). A classifier using `magnitude_feature='peakmag_fit'` simply has nothing to perturb that one feature with, which is the correct, intentional fallback (not a bug to fix).
- **Tree-side threshold-setting and `train_prior` stay on the ORIGINAL, un-replicated arrays** (`self.clf.predict_proba(Xb)` for `self.threshold`, `ws.sum()`/`wb.sum()` for `train_prior`) -- only the actual tree-training call (`CalibratedClassifierCV(...).fit(...)` / `base.fit(...)`) uses the replicated expansion.

**User decisions (already made):**
- Monte Carlo noise-injection replication (Approach A), not exact analytic KDE-bandwidth convolution or query-time-only averaging -- "go with A".
- Apply to BOTH the tree and the KDE, not the KDE alone -- "Also explore tree-side uncertainty handling".
- Real-data validation (BlueBEAR population regeneration + re-running Manual Validation) is IN SCOPE for this plan, not deferred to a separate follow-up -- "Treat regeneration as part of this same plan".

---

## Task 1: Propagate photometric uncertainty into colour/magnitude features

**Goal:** `event_multicolour` emits new `*_err` fields (colour and magnitude measurement uncertainty), reusing the per-observation magnitude errors `cmsne/lightcurve.py` already computes but currently discards, without changing `band_photometry`'s existing return signature or behaviour (it is also called directly by `cmsne/trajectory.py`).

**Files:**
- Modify: `cmsne/multicolour.py:30-133` (`band_photometry`, `_colours`, `event_multicolour`)
- Test: `tests/test_cmsne.py` (the `=== multicolour: all-band photometry extraction ===` section, ~line 609-638; extend the `_LCm` fixture)

**Acceptance Criteria:**
- [ ] `band_photometry`'s signature, return value, and behaviour are UNCHANGED (verified by the existing tests passing unchanged) -- `cmsne/trajectory.py`'s direct call site is not touched.
- [ ] A new function, `band_photometry_errors(lightcurve, epoch_obs, window=5.0)`, returns each band's magnitude error at the SAME nearest-detection index `band_photometry` would pick for the same arguments.
- [ ] `event_multicolour` gains `{colour}_err = sqrt(err_a**2 + err_b**2)` for every entry `_colours` produces (every adjacent-band colour and its `_d` evolved counterpart), `peakmag_err` (error of whichever observation achieved the brightest-observed magnitude), and `peakmag_firstdet_err` (error of the first-detection observation). No `peakmag_fit_err` field is added.
- [ ] `python tests/test_cmsne.py` -- all tests pass, 0 failures.

**Verify:** `python tests/test_cmsne.py` → all pass, 0 failures.

**Steps:**

- [ ] **Step 1: Extend the `_LCm` test fixture to carry per-observation magnitude errors**

In `tests/test_cmsne.py`, find:
```python
print('\n=== multicolour: all-band photometry extraction ===')
from cmsne import multicolour as mcol
class _LCm:
    def __init__(s, d, f, m):
        s.obs_days = np.array(d, float); s.obs_filters = np.array(f); s.obs_mag = np.array(m, float)
# g@2, r@3 detected; r@4 fainter; i@20 below depth (inf). Trigger = first detection (day 2).
_lc = _LCm([2, 3, 20, 4], ['g', 'r', 'i', 'r'], [22.0, 21.5, np.inf, 21.7])
```
Replace with:
```python
print('\n=== multicolour: all-band photometry extraction ===')
from cmsne import multicolour as mcol
class _LCm:
    def __init__(s, d, f, m, e=None):
        s.obs_days = np.array(d, float); s.obs_filters = np.array(f); s.obs_mag = np.array(m, float)
        s.obs_mag_error = np.full(len(m), 0.1) if e is None else np.array(e, float)
# g@2, r@3 detected; r@4 fainter; i@20 below depth (inf). Trigger = first detection (day 2).
# Errors: g@2=0.05, r@3=0.08, i@20=inf (undetected anyway), r@4=0.12.
_lc = _LCm([2, 3, 20, 4], ['g', 'r', 'i', 'r'], [22.0, 21.5, np.inf, 21.7], [0.05, 0.08, np.inf, 0.12])
```
This keeps every OTHER `_LCm(...)` call site in the file (which omits `e`) working unchanged -- they fall back to a harmless uniform `0.1` error that no existing check reads.

- [ ] **Step 2: Write the failing tests**

Immediately after the existing line:
```python
check('band_photometry NaN for undetected band', np.isnan(_bp['lssti']) and np.isnan(_bp['lsstu']))
```
add:
```python
_bpe = mcol.band_photometry_errors(_lc, 2.0, window=5.0)
check('band_photometry_errors matches the SAME nearest-detection index as band_photometry',
      np.isclose(_bpe['lsstg'], 0.05) and np.isclose(_bpe['lsstr'], 0.08), (_bpe['lsstg'], _bpe['lsstr']))
check('band_photometry_errors NaN for undetected band', np.isnan(_bpe['lssti']) and np.isnan(_bpe['lsstu']))
```
And immediately after the existing line:
```python
check('n_bands counts bands at trigger', _f['n_bands'] == 2, _f['n_bands'])
```
add:
```python
check('event_multicolour: gr_err is the quadrature sum of the two bands\' own errors',
      np.isclose(_f['gr_err'], np.sqrt(0.05 ** 2 + 0.08 ** 2)), _f['gr_err'])
check('event_multicolour: colour error is NaN exactly where the colour itself is NaN',
      np.isnan(_f['ri']) and np.isnan(_f['ri_err']))
check('event_multicolour: peakmag_err is the error of whichever observation achieved peakmag',
      np.isclose(_f['peakmag_err'], 0.08), _f['peakmag_err'])
check('event_multicolour: peakmag_firstdet_err is the error of the first-detection observation',
      np.isclose(_f['peakmag_firstdet_err'], 0.05), _f['peakmag_firstdet_err'])
check('event_multicolour: no peakmag_fit_err field is added (explicitly out of scope)',
      'peakmag_fit_err' not in _f)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py 2>&1 | grep -A2 "band_photometry_errors\|gr_err\|peakmag_err\|peakmag_firstdet_err\|peakmag_fit_err"`
Expected: `AttributeError`/`KeyError`-driven failures -- `band_photometry_errors` doesn't exist yet, `gr_err`/`peakmag_err`/etc. aren't in `_f`.

- [ ] **Step 4: Implement `band_photometry_errors`, keeping `band_photometry` unchanged**

In `cmsne/multicolour.py`, replace:
```python
def band_photometry(lightcurve, epoch_obs, window=5.0):
    """Nearest detected magnitude in each LSST band within ``window`` observer-days
    of ``epoch_obs``.

    ``lightcurve.obs_days`` are observer-frame days since this image's t0, and
    ``obs_mag`` is inf below the single-visit depth / nan outside the model's
    wavelength range, so ``finite`` == detected.

    :return: dict ``{band: magnitude or nan}`` over all six LSST bands.
    """
    days = np.asarray(lightcurve.obs_days, dtype=float)
    filt = np.asarray(lightcurve.obs_filters)
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    det = np.isfinite(mag)

    out = {}
    for b in LSST_BANDS:
        m = det & (filt == b[4:]) & (np.abs(days - epoch_obs) <= window)
        if not np.any(m):
            out[b] = np.nan
            continue
        idx = np.where(m)[0]
        nearest = idx[np.argmin(np.abs(days[idx] - epoch_obs))]
        out[b] = float(mag[nearest])
    return out
```
with:
```python
def _nearest_detection_indices(lightcurve, epoch_obs, window=5.0):
    """Index of the nearest detected observation in each LSST band within
    ``window`` observer-days of ``epoch_obs``, or ``None`` where no
    detection falls in that window. Shared by :func:`band_photometry` and
    :func:`band_photometry_errors` so both pick exactly the same
    observation per band."""
    days = np.asarray(lightcurve.obs_days, dtype=float)
    filt = np.asarray(lightcurve.obs_filters)
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    det = np.isfinite(mag)

    idx = {}
    for b in LSST_BANDS:
        m = det & (filt == b[4:]) & (np.abs(days - epoch_obs) <= window)
        if not np.any(m):
            idx[b] = None
            continue
        candidates = np.where(m)[0]
        idx[b] = candidates[np.argmin(np.abs(days[candidates] - epoch_obs))]
    return idx


def band_photometry(lightcurve, epoch_obs, window=5.0):
    """Nearest detected magnitude in each LSST band within ``window`` observer-days
    of ``epoch_obs``.

    ``lightcurve.obs_days`` are observer-frame days since this image's t0, and
    ``obs_mag`` is inf below the single-visit depth / nan outside the model's
    wavelength range, so ``finite`` == detected.

    :return: dict ``{band: magnitude or nan}`` over all six LSST bands.
    """
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    idx = _nearest_detection_indices(lightcurve, epoch_obs, window)
    return {b: (float(mag[i]) if i is not None else np.nan) for b, i in idx.items()}


def band_photometry_errors(lightcurve, epoch_obs, window=5.0):
    """Magnitude error at the SAME nearest-detection index
    :func:`band_photometry` would pick for the same arguments -- used to
    propagate measurement uncertainty into colour features (see
    docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md).

    :return: dict ``{band: magnitude error or nan}`` over all six LSST bands.
    """
    err = np.asarray(lightcurve.obs_mag_error, dtype=float)
    idx = _nearest_detection_indices(lightcurve, epoch_obs, window)
    return {b: (float(err[i]) if i is not None else np.nan) for b, i in idx.items()}
```

- [ ] **Step 5: Add `_colours_errors`, mirroring `_colours`**

Replace:
```python
def _colours(mags, suffix=''):
    return {f'{a[-1]}{b[-1]}{suffix}': (mags[a] - mags[b]) for a, b in ADJ_COLOURS}
```
with:
```python
def _colours(mags, suffix=''):
    return {f'{a[-1]}{b[-1]}{suffix}': (mags[a] - mags[b]) for a, b in ADJ_COLOURS}


def _colours_errors(errs, suffix=''):
    """Quadrature-sum colour error for every pair :func:`_colours` forms,
    matching its key naming exactly (e.g. ``gr`` -> ``gr_err``,
    ``gr_d`` -> ``gr_d_err``)."""
    return {f'{a[-1]}{b[-1]}{suffix}_err': np.sqrt(errs[a] ** 2 + errs[b] ** 2) for a, b in ADJ_COLOURS}
```

- [ ] **Step 6: Wire the new fields into `event_multicolour`**

Replace:
```python
    trigger = float(days[det].min())
    m0 = band_photometry(lightcurve, trigger, window)
    m1 = band_photometry(lightcurve, trigger + evolution_days, window)

    feat = {**_colours(m0), **_colours(m1, '_d')}
    # Three brightness features, so the classifier's magnitude input can be chosen
    # and compared (see docs/methods_and_results.md):
    #   peakmag          -- brightest observed sample across all bands (nanmin);
    #   peakmag_firstdet -- magnitude of the earliest detection (the trigger point),
    #                       the noisiest/faintest-biased option;
    #   peakmag_fit      -- parabola-fitted light-curve peak in the best-sampled
    #                       band, which interpolates the true peak between visits.
    det_idx = np.where(det)[0]
    first_idx = det_idx[np.argmin(days[det_idx])]
    feat['peakmag'] = float(np.nanmin(mag[det]))
    feat['peakmag_firstdet'] = float(mag[first_idx])
    filt = np.asarray(lightcurve.obs_filters)
    det_filters = filt[det]
    vals, counts = np.unique(det_filters, return_counts=True)
    best_band = 'lsst' + str(vals[int(np.argmax(counts))])
    feat['peakmag_fit'] = fitted_peak_magnitude(lightcurve, band=best_band)
    feat['trigger_day'] = trigger
    feat['n_bands'] = int(sum(np.isfinite(v) for v in m0.values()))
```
with:
```python
    trigger = float(days[det].min())
    m0 = band_photometry(lightcurve, trigger, window)
    m1 = band_photometry(lightcurve, trigger + evolution_days, window)
    err0 = band_photometry_errors(lightcurve, trigger, window)
    err1 = band_photometry_errors(lightcurve, trigger + evolution_days, window)

    feat = {**_colours(m0), **_colours(m1, '_d'), **_colours_errors(err0), **_colours_errors(err1, '_d')}
    # Three brightness features, so the classifier's magnitude input can be chosen
    # and compared (see docs/methods_and_results.md):
    #   peakmag          -- brightest observed sample across all bands (nanmin);
    #   peakmag_firstdet -- magnitude of the earliest detection (the trigger point),
    #                       the noisiest/faintest-biased option;
    #   peakmag_fit      -- parabola-fitted light-curve peak in the best-sampled
    #                       band, which interpolates the true peak between visits.
    # peakmag_fit has NO uncertainty companion -- it's a parabola-fit quantity, not
    # a single observation, and proper error propagation there is out of scope (see
    # docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md).
    mag_err = np.asarray(lightcurve.obs_mag_error, dtype=float)
    det_idx = np.where(det)[0]
    first_idx = det_idx[np.argmin(days[det_idx])]
    peak_idx = det_idx[np.argmin(mag[det_idx])]
    feat['peakmag'] = float(mag[peak_idx])
    feat['peakmag_err'] = float(mag_err[peak_idx])
    feat['peakmag_firstdet'] = float(mag[first_idx])
    feat['peakmag_firstdet_err'] = float(mag_err[first_idx])
    filt = np.asarray(lightcurve.obs_filters)
    det_filters = filt[det]
    vals, counts = np.unique(det_filters, return_counts=True)
    best_band = 'lsst' + str(vals[int(np.argmax(counts))])
    feat['peakmag_fit'] = fitted_peak_magnitude(lightcurve, band=best_band)
    feat['trigger_day'] = trigger
    feat['n_bands'] = int(sum(np.isfinite(v) for v in m0.values()))
```
(`feat['peakmag'] = float(mag[peak_idx])` is numerically identical to the previous `float(np.nanmin(mag[det]))` -- `mag[det]` contains no NaN by construction (`det = np.isfinite(mag)`), so its `nanmin` is just its `min`, and `peak_idx` is exactly the position of that minimum. This refactor only exists to get the index needed for `peakmag_err`.)

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python tests/test_cmsne.py 2>&1 | tail -40`
Expected: all pass, including the 6 new checks from Step 2, and every pre-existing check in this section (bands, colours, `peakmag`/`peakmag_firstdet`/`peakmag_fit`, `n_bands`) unchanged.

- [ ] **Step 8: Commit**

```bash
git add cmsne/multicolour.py tests/test_cmsne.py
git commit -m "feat: propagate photometric uncertainty into colour/magnitude features"
```

---

## Task 2: Uncertainty-injection primitives

**Goal:** Three new, pure, standalone functions in `cmsne/classifier.py` -- `colour_feature_sigma_matrix` (mirrors `colour_feature_matrix` for `*_err` fields), `_replicate_with_uncertainty` (the noise-injection primitive, with its auto-skip no-op), and `_grouped_effective_sample_size` (the fix for the ESS-inflation problem) -- fully testable in isolation, with no `ColourClassifier` wiring yet.

**Files:**
- Modify: `cmsne/classifier.py:48-58` (add `colour_feature_sigma_matrix` after `colour_feature_matrix`), `cmsne/classifier.py:38-45` (new constant), `cmsne/classifier.py:102-125` (add the two new functions after `_kernel_weighted_values`)
- Test: `tests/test_cmsne.py` (new section, after the existing `_kernel_weighted_values` tests, before `=== local density-ratio blend: density ratio and blend weight ===`)

**Acceptance Criteria:**
- [ ] `DEFAULT_N_UNCERTAINTY_REPLICATES = 8` added near `DEFAULT_KDE_MIN_ESS`.
- [ ] `colour_feature_sigma_matrix(events, keys, extra)` returns an `(n, d)` matrix in the SAME column order as `colour_feature_matrix`, reading `f'{c}_err'` from each event dict, `NaN` where absent.
- [ ] `_replicate_with_uncertainty(X, sigma, weights, dims, n_replicates, seed=0)` is a strict no-op (identical `X`/`weights`, `event_id = np.arange(len(X))`, zero RNG draws) when `sigma[:, dims]` is entirely non-finite or non-positive, or `n_replicates <= 1`, or `dims` is empty.
- [ ] Otherwise: expands each row into `n_replicates` copies; perturbs each dimension in `dims` independently by that row's own `sigma` (a non-finite/non-positive entry leaves that one dimension unperturbed for every replica of that row); each replica's weight is the original weight divided by `n_replicates`, so `np.bincount(event_id, weights=w_rep)` exactly recovers the original `weights` array.
- [ ] `_grouped_effective_sample_size(g, event_id, n_original)` sums each original event's values back together (`np.bincount`) before calling `effective_sample_size`, and is proven (not just argued) to report ESS ~1, not ~K, for K replicas of one real point.
- [ ] `python tests/test_cmsne.py` -- all tests pass, 0 failures.

**Verify:** `python tests/test_cmsne.py` → all pass, 0 failures.

**Steps:**

- [ ] **Step 1: Write the failing tests**

After the existing block ending in:
```python
check('_kernel_weighted_values returns zeros for zero training rows',
      np.all(_kernel_weighted_values(np.array([0.0]), np.zeros((0, 1)), np.zeros(0), _kw_h, _kw_dims) == 0.0))
```
insert:
```python
print('\n=== uncertainty-aware noise injection: primitives ===')
from cmsne.classifier import (colour_feature_sigma_matrix, _replicate_with_uncertainty,
                              _grouped_effective_sample_size, DEFAULT_N_UNCERTAINTY_REPLICATES)
from cmsne.importance_sampling import effective_sample_size as _ess_ref2

check('DEFAULT_N_UNCERTAINTY_REPLICATES is a sane positive default', DEFAULT_N_UNCERTAINTY_REPLICATES >= 2,
      DEFAULT_N_UNCERTAINTY_REPLICATES)

_sig_events = [{'gr': 1.0, 'peakmag': 20.0, 'gr_err': 0.1},
               {'gr': 2.0, 'peakmag': 21.0}]  # second event has NO gr_err/peakmag_err at all
_sig_matrix = colour_feature_sigma_matrix(_sig_events, keys=['gr'], extra=('peakmag',))
check('colour_feature_sigma_matrix reads the *_err field when present',
      np.isclose(_sig_matrix[0, 0], 0.1), _sig_matrix[0, 0])
check('colour_feature_sigma_matrix is NaN when no *_err field exists',
      np.isnan(_sig_matrix[0, 1]) and np.isnan(_sig_matrix[1, 0]) and np.isnan(_sig_matrix[1, 1]))

# No-op: sigma entirely non-finite over dims -> unchanged, no RNG draws.
_rwu_X = np.array([[1.0, 2.0], [3.0, 4.0]])
_rwu_w = np.array([0.5, 0.5])
_rwu_sigma_none = np.full_like(_rwu_X, np.nan)
_rwu_Xr, _rwu_wr, _rwu_id = _replicate_with_uncertainty(_rwu_X, _rwu_sigma_none, _rwu_w, np.array([0, 1]), 8)
check('_replicate_with_uncertainty is a no-op when sigma is entirely non-finite',
      np.array_equal(_rwu_Xr, _rwu_X) and np.array_equal(_rwu_wr, _rwu_w) and
      np.array_equal(_rwu_id, np.arange(2)))

# Weight preservation: each original event's replica weights sum back exactly.
_rwu_sigma = np.array([[0.2, np.nan], [np.nan, 0.5]])
_rwu_Xr2, _rwu_wr2, _rwu_id2 = _replicate_with_uncertainty(_rwu_X, _rwu_sigma, _rwu_w, np.array([0, 1]), 8)
check('_replicate_with_uncertainty expands each row into n_replicates copies',
      len(_rwu_Xr2) == 2 * 8, len(_rwu_Xr2))
check('_replicate_with_uncertainty preserves each original event\'s total weight exactly',
      np.allclose(np.bincount(_rwu_id2, weights=_rwu_wr2, minlength=2), _rwu_w))
check('_replicate_with_uncertainty leaves a dimension unperturbed when that row\'s sigma is non-finite',
      np.all(_rwu_Xr2[_rwu_id2 == 0, 1] == 2.0) and np.all(_rwu_Xr2[_rwu_id2 == 1, 0] == 3.0))

# Perturbation: with a large, finite sigma, replicas measurably scatter.
_rwu_sigma_big = np.array([[5.0, np.nan]])
_rwu_Xr3, _, _ = _replicate_with_uncertainty(np.array([[10.0, 0.0]]), _rwu_sigma_big, np.array([1.0]),
                                              np.array([0]), 2000, seed=1)
check('_replicate_with_uncertainty replicas scatter with roughly the given sigma',
      abs(np.std(_rwu_Xr3[:, 0]) - 5.0) < 0.5, np.std(_rwu_Xr3[:, 0]))

# The ESS-inflation bug, and the fix: 5 replicas of ONE real point must not look
# like 5 independent points.
_grp_event_id = np.array([0, 0, 0, 0, 0])
_grp_g = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
check('naive (ungrouped) ESS is fooled into thinking 5 replicas are 5 independent samples',
      np.isclose(_ess_ref2(_grp_g), 5.0), _ess_ref2(_grp_g))
check('_grouped_effective_sample_size correctly reports ESS~1 for 5 replicas of ONE real point',
      np.isclose(_grouped_effective_sample_size(_grp_g, _grp_event_id, 1), 1.0),
      _grouped_effective_sample_size(_grp_g, _grp_event_id, 1))

_grp_event_id2 = np.array([0, 0, 0, 1, 1, 1])
_grp_g2 = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
_grp_expected = _ess_ref2(np.array([0.3, 0.15]))
check('_grouped_effective_sample_size matches a direct group-sum-then-ESS calculation',
      np.isclose(_grouped_effective_sample_size(_grp_g2, _grp_event_id2, 2), _grp_expected),
      (_grouped_effective_sample_size(_grp_g2, _grp_event_id2, 2), _grp_expected))
check('_grouped_effective_sample_size reduces to plain ESS for a trivial (non-replicated) event_id',
      np.isclose(_grouped_effective_sample_size(_grp_g2, np.arange(6), 6), _ess_ref2(_grp_g2)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py 2>&1 | tail -30`
Expected: `ImportError` -- none of `colour_feature_sigma_matrix`/`_replicate_with_uncertainty`/`_grouped_effective_sample_size`/`DEFAULT_N_UNCERTAINTY_REPLICATES` exist yet.

- [ ] **Step 3: Add the new constant**

In `cmsne/classifier.py`, replace:
```python
DEFAULT_ESS_THRESHOLD = 10.0
DEFAULT_LOG_WIDTH = 0.3
DEFAULT_KDE_MAX_EVENTS = 4000
DEFAULT_KDE_MIN_ESS = 10.0
```
with:
```python
DEFAULT_ESS_THRESHOLD = 10.0
DEFAULT_LOG_WIDTH = 0.3
DEFAULT_KDE_MAX_EVENTS = 4000
DEFAULT_KDE_MIN_ESS = 10.0
# Uncertainty-aware noise injection (see docs/superpowers/specs/
# 2026-10-26-uncertainty-aware-noise-injection-design.md). Provisional, like
# the other blend constants above -- tuned against real data in this plan's
# own Manual Validation step.
DEFAULT_N_UNCERTAINTY_REPLICATES = 8
```

- [ ] **Step 4: Add `colour_feature_sigma_matrix`**

Replace:
```python
def colour_feature_matrix(events, keys=COLOUR_KEYS, extra=(DEFAULT_MAGNITUDE_FEATURE,)):
    """Stack ``events`` (dicts) into a feature matrix over ``keys`` (+ ``extra``).

    Missing entries become NaN, which the classifier treats as "band not observed".

    :return: ``(X, columns)`` where ``columns`` is the ordered feature name list.
    """
    columns = list(keys) + [k for k in extra if k]
    X = np.array([[float(e.get(c, np.nan)) if e.get(c, None) is not None else np.nan
                   for c in columns] for e in events], dtype=float)
    return X, columns
```
with:
```python
def colour_feature_matrix(events, keys=COLOUR_KEYS, extra=(DEFAULT_MAGNITUDE_FEATURE,)):
    """Stack ``events`` (dicts) into a feature matrix over ``keys`` (+ ``extra``).

    Missing entries become NaN, which the classifier treats as "band not observed".

    :return: ``(X, columns)`` where ``columns`` is the ordered feature name list.
    """
    columns = list(keys) + [k for k in extra if k]
    X = np.array([[float(e.get(c, np.nan)) if e.get(c, None) is not None else np.nan
                   for c in columns] for e in events], dtype=float)
    return X, columns


def colour_feature_sigma_matrix(events, keys=COLOUR_KEYS, extra=(DEFAULT_MAGNITUDE_FEATURE,)):
    """Stack ``events``' per-feature measurement uncertainty (``f'{c}_err'``)
    into a matrix matching :func:`colour_feature_matrix`'s column order.

    Missing/absent ``_err`` entries become NaN -- :func:`_replicate_with_uncertainty`
    treats this as "nothing to perturb with", not an error. ``peakmag_fit`` in
    particular has no ``_err`` companion anywhere in this codebase (see
    docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md)
    and will always read NaN here.

    :return: ``(n, d)`` sigma matrix, same column order as :func:`colour_feature_matrix`.
    """
    columns = list(keys) + [k for k in extra if k]
    sigma = np.array([[float(e.get(f'{c}_err', np.nan)) if e.get(f'{c}_err', None) is not None else np.nan
                        for c in columns] for e in events], dtype=float)
    return sigma
```

- [ ] **Step 5: Add `_replicate_with_uncertainty` and `_grouped_effective_sample_size`**

In `cmsne/classifier.py`, after `_kernel_weighted_values` (ends with `return np.asarray(weights, dtype=float) * k`) and before `_widened_kernel_values`, insert:
```python
def _replicate_with_uncertainty(X, sigma, weights, dims, n_replicates, seed=0):
    """Expand each row of ``X`` into ``n_replicates`` noisy copies, perturbed
    per dimension in ``dims`` by that row's own ``sigma`` (non-finite or
    non-positive sigma leaves that dimension exactly as-is for every
    replica of that row -- nothing to perturb with is not an error). Each
    replica carries ``weights / n_replicates``, so summed weight mass per
    original row is preserved exactly.

    :return: ``(X_rep, w_rep, event_id)``. ``event_id[k]`` is the index into
        the ORIGINAL ``X``/``weights`` that ``X_rep[k]`` came from -- needed
        to group replicas back together (e.g. for
        :func:`_grouped_effective_sample_size`).

    If every entry of ``sigma[:, dims]`` is non-finite or non-positive (no
    real uncertainty data available), or ``n_replicates <= 1``, or ``dims``
    is empty, returns ``(X, weights, np.arange(len(X)))`` unchanged -- zero
    extra compute, exact backward compatibility with any population/fixture
    lacking uncertainty fields.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    dims = np.asarray(dims, dtype=int)
    if n_replicates <= 1 or len(dims) == 0 or not np.any(np.isfinite(sigma[:, dims]) & (sigma[:, dims] > 0)):
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


def _grouped_effective_sample_size(g, event_id, n_original):
    """Kish's effective sample size computed AFTER summing each original
    event's noise-injection replicas' values back together, so
    ``n_replicates`` correlated copies of one real observation contribute
    ESS ~1, not ~n_replicates (see docs/superpowers/specs/
    2026-10-26-uncertainty-aware-noise-injection-design.md). Reduces to a
    plain :func:`~cmsne.importance_sampling.effective_sample_size` call
    when ``event_id`` is trivial (``np.arange(n_original)``, i.e. no
    replication occurred).
    """
    # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
    from .importance_sampling import effective_sample_size

    g_grouped = np.bincount(event_id, weights=g, minlength=n_original)
    return effective_sample_size(g_grouped)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python tests/test_cmsne.py 2>&1 | tail -40`
Expected: all pass, including every new check from Step 1.

- [ ] **Step 7: Commit**

```bash
git add cmsne/classifier.py tests/test_cmsne.py
git commit -m "feat: add uncertainty-aware noise-injection primitives"
```

---

## Task 3: Tree-side wiring (train and predict)

**Goal:** `fit()` trains the tree on a noise-injection-replicated expansion of the full population; `score()` evaluates a query event by replicating it from its own uncertainty and averaging the tree's predictions back together. Both reduce to exactly today's behaviour when no `*_err` data exists.

**Files:**
- Modify: `cmsne/classifier.py:300-324` (`__init__`), `cmsne/classifier.py:325-360` (`fit`), `cmsne/classifier.py:391-398` (`score`)
- Test: `tests/test_cmsne.py` (new section after Task 2's)

**Acceptance Criteria:**
- [ ] `ColourClassifier(..., n_uncertainty_replicates=DEFAULT_N_UNCERTAINTY_REPLICATES)` stores `self.n_uncertainty_replicates`.
- [ ] `fit()` builds `sigma_s`/`sigma_b` via `colour_feature_sigma_matrix`, balances `ws`/`wb` exactly as before, THEN replicates the balanced `Xs`/`ws`/`Xb`/`wb` via `_replicate_with_uncertainty` before calling `CalibratedClassifierCV(...).fit(...)`/`base.fit(...)`.
- [ ] `self.threshold` (via `self.clf.predict_proba(Xb)`) and `self.train_prior` (via `ws.sum()`/`wb.sum()`) use the ORIGINAL, un-replicated `Xb`/`ws`/`wb` -- unchanged from before this task.
- [ ] `score()` replicates each query event from ITS OWN `*_err` fields and averages `self.clf.predict_proba` over its replicas; with no `*_err` data, this is numerically identical to today's `self.clf.predict_proba(X)[:, 1]`.
- [ ] A fitted classifier on a population WITH real, large, synthetic `gr_err` values trains and predicts without error, producing a finite, calibrated-looking output (sanity check, not a deep ML-quality test).
- [ ] The full pre-existing test suite passes unchanged (no `*_err` fields anywhere in existing fixtures -> every `fit()`/`score()` call is a byte-identical no-op under replication).
- [ ] `python tests/test_cmsne.py` -- all tests pass, 0 failures.

**Verify:** `python tests/test_cmsne.py` → all pass, 0 failures.

**Steps:**

- [ ] **Step 1: Write the failing tests**

This section reuses `_dense_sig`/`_dense_bg`/`_dense_sw`/`_dense_bw`, which are
defined in the `=== local density-ratio blend: wired into ColourClassifier ===`
section -- NOT immediately after Task 2's section (which sits one section
*before* that, and does not yet have those fixtures in scope). Insert
immediately after that section's own checks finish, right before
`=== local density-ratio blend: backward compatibility and kde_max_events ===`:
```python
print('\n=== uncertainty-aware noise injection: tree-side wiring ===')
from cmsne.classifier import ColourClassifier

check('ColourClassifier stores n_uncertainty_replicates',
      ColourClassifier(n_uncertainty_replicates=3).n_uncertainty_replicates == 3)

# A population with real, large gr_err values trains and predicts without error.
_rng_unc = np.random.default_rng(11)
_n_unc = 300
_unc_sig = [{'gr': float(_rng_unc.normal(1.0, 0.3)), 'peakmag': float(_rng_unc.normal(22.0, 1.0)),
             'gr_err': float(abs(_rng_unc.normal(0.3, 0.1))), 'peakmag_err': 0.1}
            for _ in range(_n_unc)]
_unc_bg = [{'gr': float(_rng_unc.normal(1.3, 0.3)), 'peakmag': float(_rng_unc.normal(22.0, 1.0)),
            'gr_err': float(abs(_rng_unc.normal(0.3, 0.1))), 'peakmag_err': 0.1}
           for _ in range(_n_unc)]
_unc_w = np.full(_n_unc, 1e-5)
_clf_unc = ColourClassifier(keys=['gr'], include_peakmag=True, n_uncertainty_replicates=4)
_clf_unc.fit(_unc_sig, _unc_bg, _unc_w, _unc_w)
_unc_probe = _clf_unc.score([{'gr': 1.1, 'peakmag': 22.0, 'gr_err': 0.2}])
check('a classifier trained with real uncertainty data produces a finite, calibrated-range score',
      np.isfinite(_unc_probe[0]) and 0.0 <= _unc_probe[0] <= 1.0, _unc_probe[0])

# With NO uncertainty data anywhere (training OR query -- reusing the
# error-free _dense_sig/_dense_bg/_dense_sw/_dense_bw fixtures from the
# "wired into ColourClassifier" section above), score() is numerically
# identical to predict_proba directly: replication is a true no-op start to
# finish, not just at the query.
_clf_noerr = ColourClassifier(keys=['gr'], include_peakmag=True)
_clf_noerr.fit(_dense_sig, _dense_bg, _dense_sw, _dense_bw)
_X_direct, _ = colour_feature_matrix([{'gr': 1.1, 'peakmag': 22.0}], ['gr'], ('peakmag',))
check('score() with no *_err data anywhere matches predict_proba directly '
      '(replication is a true no-op, training and query alike)',
      np.isclose(_clf_noerr.score([{'gr': 1.1, 'peakmag': 22.0}])[0],
                 _clf_noerr.clf.predict_proba(_X_direct)[0, 1]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py 2>&1 | tail -20`
Expected: `TypeError` on `n_uncertainty_replicates=3` -- `__init__` doesn't accept it yet.

- [ ] **Step 3: Update `__init__`**

Replace:
```python
    def __init__(self, keys=COLOUR_KEYS, include_peakmag=True,
                 magnitude_feature=DEFAULT_MAGNITUDE_FEATURE, target_fpr=0.10,
                 calibrate=True, ess_threshold=DEFAULT_ESS_THRESHOLD,
                 log_width=DEFAULT_LOG_WIDTH, kde_max_events=DEFAULT_KDE_MAX_EVENTS,
                 kde_min_ess=DEFAULT_KDE_MIN_ESS, **hgb):
        self.keys = list(keys)
        if include_peakmag and magnitude_feature:
            if magnitude_feature not in MAGNITUDE_FEATURES:
                raise ValueError("magnitude_feature must be one of %r, got %r"
                                 % (MAGNITUDE_FEATURES, magnitude_feature))
            self.magnitude_feature = magnitude_feature
            self.extra = (magnitude_feature,)
        else:
            self.magnitude_feature = None
            self.extra = ()
        self.target_fpr = target_fpr
        self.calibrate = calibrate
        self.params = dict(max_depth=4, learning_rate=0.1, max_iter=300,
                           l2_regularization=1.0, random_state=0)
        self.params.update(hgb)
        self.ess_threshold = ess_threshold
        self.log_width = log_width
        self.kde_max_events = kde_max_events
        self.kde_min_ess = kde_min_ess
```
with:
```python
    def __init__(self, keys=COLOUR_KEYS, include_peakmag=True,
                 magnitude_feature=DEFAULT_MAGNITUDE_FEATURE, target_fpr=0.10,
                 calibrate=True, ess_threshold=DEFAULT_ESS_THRESHOLD,
                 log_width=DEFAULT_LOG_WIDTH, kde_max_events=DEFAULT_KDE_MAX_EVENTS,
                 kde_min_ess=DEFAULT_KDE_MIN_ESS,
                 n_uncertainty_replicates=DEFAULT_N_UNCERTAINTY_REPLICATES, **hgb):
        self.keys = list(keys)
        if include_peakmag and magnitude_feature:
            if magnitude_feature not in MAGNITUDE_FEATURES:
                raise ValueError("magnitude_feature must be one of %r, got %r"
                                 % (MAGNITUDE_FEATURES, magnitude_feature))
            self.magnitude_feature = magnitude_feature
            self.extra = (magnitude_feature,)
        else:
            self.magnitude_feature = None
            self.extra = ()
        self.target_fpr = target_fpr
        self.calibrate = calibrate
        self.params = dict(max_depth=4, learning_rate=0.1, max_iter=300,
                           l2_regularization=1.0, random_state=0)
        self.params.update(hgb)
        self.ess_threshold = ess_threshold
        self.log_width = log_width
        self.kde_max_events = kde_max_events
        self.kde_min_ess = kde_min_ess
        self.n_uncertainty_replicates = n_uncertainty_replicates
```

- [ ] **Step 4: Update `fit()`**

Replace:
```python
        Xs, cols = colour_feature_matrix(signal_events, self.keys, self.extra)
        Xb, _ = colour_feature_matrix(background_events, self.keys, self.extra)
        self.columns = cols

        ws = np.ones(len(Xs)) if signal_weights is None else np.asarray(signal_weights, float)
        wb = np.ones(len(Xb)) if background_weights is None else np.asarray(background_weights, float)
        if ws.sum() > 0:
            ws = ws * (wb.sum() / ws.sum())            # balance totals

        X = np.vstack([Xs, Xb])
        y = np.r_[np.ones(len(Xs)), np.zeros(len(Xb))]
        w = np.r_[ws, wb]
        # Signal and background totals are balanced, so the trained posterior is at a
        # 0.5 prior; probability(prior=...) rescales it to any other prior.
        self.train_prior = float(ws.sum() / (ws.sum() + wb.sum()))

        base = HistGradientBoostingClassifier(**self.params)
        if self.calibrate:
            self.clf = CalibratedClassifierCV(base, method='isotonic', cv=3).fit(X, y, sample_weight=w)
        else:
            self.clf = base.fit(X, y, sample_weight=w)

        sb = self.clf.predict_proba(Xb)[:, 1]
        self.threshold = float(weighted_quantile(sb, wb, q=1.0 - self.target_fpr))
        self._store_kde_training_data(Xs, ws, Xb, wb)
        return self
```
with:
```python
        Xs, cols = colour_feature_matrix(signal_events, self.keys, self.extra)
        Xb, _ = colour_feature_matrix(background_events, self.keys, self.extra)
        self.columns = cols
        sigma_s = colour_feature_sigma_matrix(signal_events, self.keys, self.extra)
        sigma_b = colour_feature_sigma_matrix(background_events, self.keys, self.extra)

        ws = np.ones(len(Xs)) if signal_weights is None else np.asarray(signal_weights, float)
        wb = np.ones(len(Xb)) if background_weights is None else np.asarray(background_weights, float)
        if ws.sum() > 0:
            ws = ws * (wb.sum() / ws.sum())            # balance totals

        # Noise-injection replication for the TREE only, on the full balanced
        # population -- see Global Constraints. _store_kde_training_data below
        # keeps the ORIGINAL, un-replicated arrays; the KDE replicates lazily,
        # per pattern, after its own kde_max_events cap (Task 4).
        dims_all = np.arange(Xs.shape[1])
        Xs_rep, ws_rep, _ = _replicate_with_uncertainty(
            Xs, sigma_s, ws, dims_all, self.n_uncertainty_replicates)
        Xb_rep, wb_rep, _ = _replicate_with_uncertainty(
            Xb, sigma_b, wb, dims_all, self.n_uncertainty_replicates)

        X = np.vstack([Xs_rep, Xb_rep])
        y = np.r_[np.ones(len(Xs_rep)), np.zeros(len(Xb_rep))]
        w = np.r_[ws_rep, wb_rep]
        # Signal and background totals are balanced, so the trained posterior is at a
        # 0.5 prior; probability(prior=...) rescales it to any other prior. Uses the
        # ORIGINAL ws/wb -- a class total is unaffected by replication either way
        # (weights sum back exactly), but there is no reason to route this through
        # the replicated arrays.
        self.train_prior = float(ws.sum() / (ws.sum() + wb.sum()))

        base = HistGradientBoostingClassifier(**self.params)
        if self.calibrate:
            self.clf = CalibratedClassifierCV(base, method='isotonic', cv=3).fit(X, y, sample_weight=w)
        else:
            self.clf = base.fit(X, y, sample_weight=w)

        # Threshold-setting stays on the REAL, un-replicated background -- this is
        # calibration against the actual rate-weighted population, not a training
        # input, so it should not see synthetic noise copies.
        sb = self.clf.predict_proba(Xb)[:, 1]
        self.threshold = float(weighted_quantile(sb, wb, q=1.0 - self.target_fpr))
        self._store_kde_training_data(Xs, ws, Xb, wb, sigma_s, sigma_b)
        return self
```

- [ ] **Step 5: Update `score()`**

Replace:
```python
    def score(self, events):
        """Raw ranking score (calibrated posterior at the balanced training prior).

        Monotonic with :meth:`probability`, so it gives the same ordering; use it
        when you only need to rank.
        """
        X, _ = colour_feature_matrix(events, self.keys, self.extra)
        return self.clf.predict_proba(X)[:, 1]
```
with:
```python
    def score(self, events):
        """Raw ranking score (calibrated posterior at the balanced training prior).

        Monotonic with :meth:`probability`, so it gives the same ordering; use it
        when you only need to rank.

        Each query event is noise-injection-replicated from its OWN ``*_err``
        fields (if present) and the tree's predictions are averaged back to one
        value per event -- symmetric treatment with how training data is
        handled (see docs/superpowers/specs/
        2026-10-26-uncertainty-aware-noise-injection-design.md). With no
        ``*_err`` data, this is numerically identical to a direct
        ``predict_proba`` call.
        """
        X, _ = colour_feature_matrix(events, self.keys, self.extra)
        sigma = colour_feature_sigma_matrix(events, self.keys, self.extra)
        dims_all = np.arange(X.shape[1])
        X_rep, _, event_id = _replicate_with_uncertainty(
            X, sigma, np.ones(len(X)), dims_all, self.n_uncertainty_replicates)
        p_rep = self.clf.predict_proba(X_rep)[:, 1]
        counts = np.bincount(event_id, minlength=len(X))
        sums = np.bincount(event_id, weights=p_rep, minlength=len(X))
        return sums / counts
```

- [ ] **Step 6: `_store_kde_training_data` must accept the new sigma arguments (full implementation lands in Task 4) -- for now, accept and ignore**

This task does not yet use `sigma_s`/`sigma_b` inside `_store_kde_training_data` (that wiring is Task 4's), but `fit()` (Step 4 above) now calls it with 6 arguments. Replace:
```python
    def _store_kde_training_data(self, Xs, ws, Xb, wb):
```
with:
```python
    def _store_kde_training_data(self, Xs, ws, Xb, wb, sigma_s, sigma_b):
```
and, inside the method body, after the existing `pos_b = wb > 0` line, add:
```python
        self._kde_sigma_s_unfiltered, self._kde_sigma_b_unfiltered = sigma_s, sigma_b
```
(Task 4 replaces this placeholder-storage line with the real filtered `self._kde_sigma_s`/`self._kde_sigma_b` used by `_kde_subset_for_pattern`; naming it `_unfiltered` here makes it unambiguous this is an intermediate step, not the final shape, and nothing else reads it yet in this task.)

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python tests/test_cmsne.py 2>&1 | tail -40`
Expected: all pass, including every new check from Step 1, and the full pre-existing suite unchanged.

- [ ] **Step 8: Commit**

```bash
git add cmsne/classifier.py tests/test_cmsne.py
git commit -m "feat: wire noise-injection replication into tree training and prediction"
```

---

## Task 4: KDE-side wiring (train and predict)

**Goal:** `_kde_subset_for_pattern` replicates its own per-pattern, post-cap subset using each surviving event's own uncertainty; `_local_density_ratio` and `_widened_kernel_values` use grouped ESS wherever a possibly-replicated array feeds `effective_sample_size`; `_blend_with_local_density` is updated for both the new cache shape and query-side replication-and-averaging. This is the task that actually fixes the issue #37 mechanism.

**Files:**
- Modify: `cmsne/classifier.py:251-277` (`_weighted_subsample`), `cmsne/classifier.py` (new `_KDEPatternSubset` NamedTuple, placed just before `class ColourClassifier`), `cmsne/classifier.py:128-165` (`_widened_kernel_values`), `cmsne/classifier.py:168-232` (`_local_density_ratio`), `cmsne/classifier.py` (`_store_kde_training_data`, finishing Task 3's placeholder), `cmsne/classifier.py` (`_kde_subset_for_pattern`), `cmsne/classifier.py` (`_blend_with_local_density`)
- Test: `tests/test_cmsne.py` -- rewrite the THREE existing test blocks that unpack `_kde_subset_for_pattern`'s return positionally (the `kde_max_events` tests, the "budget is not wasted" tests, and the `ess_ceiling` cache tests), and add new tests for the grouped-ESS behaviour integrated end-to-end and the issue #37 smoking-gun reproduction.

**Acceptance Criteria:**
- [ ] `_weighted_subsample` returns `(X_sub, weights_sub, chosen)`, where `chosen` indexes into the INPUT arrays (needed to align sigma with the kept subsample). Behaviour and values of the first two return elements are UNCHANGED.
- [ ] `_KDEPatternSubset` is a `NamedTuple` with fields `Xs, ws, Xb, wb, ess_ceiling_sig, ess_ceiling_bg, event_id_s, event_id_b, n_orig_s, n_orig_b`, in that order (the first 6 positions match the OLD 6-tuple's order exactly, so old positional reads of indices 0-5 still work).
- [ ] `_kde_subset_for_pattern` computes `ess_ceiling_sig`/`ess_ceiling_bg` from the CAPPED, PRE-replication weights (never the replicated ones), then replicates using each surviving event's own stored sigma.
- [ ] `_widened_kernel_values` gains `event_id=None, n_original=None` keyword parameters (defaulting to a trivial, non-grouping arange when omitted -- fully backward compatible with every existing direct call) and uses `_grouped_effective_sample_size` for BOTH its initial ESS check and its `ceiling` stopping condition.
- [ ] `_local_density_ratio` gains `event_id_s=None, event_id_b=None, n_orig_s=None, n_orig_b=None` keyword parameters (same trivial-default backward compatibility) and uses `_grouped_effective_sample_size` for `local_ess_sig`/`local_ess_bg` and for its `ess_ceiling_sig`/`ess_ceiling_bg` fallback-computation branch.
- [ ] `_blend_with_local_density` replicates each query event from its own uncertainty, evaluates the blended probability on each replica, and averages the result back to one value per event; with no query-side uncertainty, this reduces to exactly one evaluation (today's behaviour).
- [ ] A direct, small-scale reproduction of the issue #37 finding: one high-sigma background point whose fixed-bandwidth kernel contribution currently dominates a query point WITHOUT noise injection, and is shown to smear out and shift the ratio WITH it.
- [ ] The full pre-existing test suite (including the three rewritten blocks) passes, with VALUES unchanged where no uncertainty data is present (only the unpacking syntax changes, not the numbers).
- [ ] `python tests/test_cmsne.py` -- all tests pass, 0 failures.

**Verify:** `python tests/test_cmsne.py` → all pass, 0 failures.

**Steps:**

- [ ] **Step 1: Update `_weighted_subsample` to expose the chosen indices**

Replace:
```python
def _weighted_subsample(X, weights, max_events, seed=0):
    """Rate-weighted subsample of up to ``max_events`` positive-weight rows
    of ``X``/``weights``, without replacement, rescaled to preserve total
    weight mass.

    Zero-weight rows are dropped unconditionally -- they contribute nothing
    to the kernel-weighted sums this feeds and carry no rate mass to
    preserve. Matches the weighted-subsampling convention already used
    elsewhere in this project (``scripts/plot_evolving_boundaries.py``'s
    ``_draw_resampled_sources``), adapted to sample *without* replacement
    since this becomes the KDE's own reference set, where duplicate points
    would distort local density estimates.

    :return: ``(X_sub, weights_sub)``. If there are ``max_events`` or fewer
        positive-weight rows, returns all of them (no subsampling), with
        weights unchanged.
    """
    weights = np.asarray(weights, dtype=float)
    positive = np.where(weights > 0)[0]
    if len(positive) <= max_events:
        return X[positive], weights[positive]
    wsum = weights[positive].sum()
    p = weights[positive] / wsum
    rng = np.random.default_rng(seed)
    chosen = rng.choice(positive, size=max_events, replace=False, p=p)
    kept_w = weights[chosen] * (wsum / weights[chosen].sum())
    return X[chosen], kept_w
```
with:
```python
def _weighted_subsample(X, weights, max_events, seed=0):
    """Rate-weighted subsample of up to ``max_events`` positive-weight rows
    of ``X``/``weights``, without replacement, rescaled to preserve total
    weight mass.

    Zero-weight rows are dropped unconditionally -- they contribute nothing
    to the kernel-weighted sums this feeds and carry no rate mass to
    preserve. Matches the weighted-subsampling convention already used
    elsewhere in this project (``scripts/plot_evolving_boundaries.py``'s
    ``_draw_resampled_sources``), adapted to sample *without* replacement
    since this becomes the KDE's own reference set, where duplicate points
    would distort local density estimates.

    :return: ``(X_sub, weights_sub, chosen)``. ``chosen`` indexes into the
        INPUT ``X``/``weights`` -- needed by callers that must align another
        per-row array (e.g. per-event measurement-uncertainty sigma, see
        docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md)
        with the same kept subsample. If there are ``max_events`` or fewer
        positive-weight rows, returns all of them (no subsampling), with
        weights unchanged.
    """
    weights = np.asarray(weights, dtype=float)
    positive = np.where(weights > 0)[0]
    if len(positive) <= max_events:
        return X[positive], weights[positive], positive
    wsum = weights[positive].sum()
    p = weights[positive] / wsum
    rng = np.random.default_rng(seed)
    chosen = rng.choice(positive, size=max_events, replace=False, p=p)
    kept_w = weights[chosen] * (wsum / weights[chosen].sum())
    return X[chosen], kept_w, chosen
```

- [ ] **Step 2: Add `_KDEPatternSubset`**

In `cmsne/classifier.py`, add right before `class ColourClassifier:`:
```python
from typing import NamedTuple


class _KDEPatternSubset(NamedTuple):
    """Per-pattern KDE reference set, cached by
    :meth:`ColourClassifier._kde_subset_for_pattern`. ``Xs``/``ws``/``Xb``/``wb``
    are the noise-injection-REPLICATED subset (see
    docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md);
    ``event_id_s``/``event_id_b`` map each replica back to its original
    (capped, pre-replication) row, and ``n_orig_s``/``n_orig_b`` are how many
    original rows that was -- both needed by
    :func:`_grouped_effective_sample_size`. ``ess_ceiling_sig``/``ess_ceiling_bg``
    are computed from the CAPPED, PRE-replication weights, not ``ws``/``wb``.
    """
    Xs: np.ndarray
    ws: np.ndarray
    Xb: np.ndarray
    wb: np.ndarray
    ess_ceiling_sig: float
    ess_ceiling_bg: float
    event_id_s: np.ndarray
    event_id_b: np.ndarray
    n_orig_s: int
    n_orig_b: int
```
Move the `import numpy as np` / existing imports at the top of the file are untouched; add `from typing import NamedTuple` alongside them instead of inline before the class, i.e. place it with the other imports at the top of `cmsne/classifier.py` (after `from .colour_magnitude import weighted_quantile`), and define `_KDEPatternSubset` where indicated (right before `class ColourClassifier:`).

- [ ] **Step 3: Update `_widened_kernel_values` for grouped ESS**

Replace:
```python
def _widened_kernel_values(x, X, weights, bandwidths, dims, min_ess, growth=2.0, max_iter=20):
    """Kernel-weighted values at ``x``, with the kernel's bandwidth
    isotropically widened (by ``growth`` per iteration, up to ``max_iter``
    times) until the resulting local ESS reaches ``min_ess``, or until
    widening stops helping (plateaus at this class's own overall ESS
    ceiling) -- whichever comes first.

    This exists ONLY to make the KDE side's own ``S``/``B`` sums
    statistically meaningful in genuinely data-starved regions, where a
    fixed bandwidth's local ESS can be dominated by a single surviving
    training point (see docs/superpowers/specs/
    2026-10-26-adaptive-kde-bandwidth-design.md). It is NOT used for the
    tree-vs-KDE blend-weight decision, which stays on the fixed-bandwidth
    ESS computed separately in :func:`_local_density_ratio`.

    :return: ``(g, ess)`` -- the kernel-weighted values and the local ESS
        at whichever scale the widening stopped at.
    """
    # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
    from .importance_sampling import effective_sample_size

    scale = 1.0
    g = _kernel_weighted_values(x, X, weights, bandwidths, dims)
    ess = effective_sample_size(g)
    if ess >= min_ess:
        return g, ess
    ceiling = effective_sample_size(weights)
    for _ in range(max_iter):
        if ess >= min_ess or ess >= ceiling - 1e-9:
            break
        scale *= growth
        g_new = _kernel_weighted_values(x, X, weights, bandwidths * scale, dims)
        ess_new = effective_sample_size(g_new)
        if ess_new <= ess:
            ess = ess_new
            break
        g, ess = g_new, ess_new
    return g, ess
```
with:
```python
def _widened_kernel_values(x, X, weights, bandwidths, dims, min_ess, growth=2.0, max_iter=20,
                            event_id=None, n_original=None):
    """Kernel-weighted values at ``x``, with the kernel's bandwidth
    isotropically widened (by ``growth`` per iteration, up to ``max_iter``
    times) until the resulting local ESS reaches ``min_ess``, or until
    widening stops helping (plateaus at this class's own overall ESS
    ceiling) -- whichever comes first.

    This exists ONLY to make the KDE side's own ``S``/``B`` sums
    statistically meaningful in genuinely data-starved regions, where a
    fixed bandwidth's local ESS can be dominated by a single surviving
    training point (see docs/superpowers/specs/
    2026-10-26-adaptive-kde-bandwidth-design.md). It is NOT used for the
    tree-vs-KDE blend-weight decision, which stays on the fixed-bandwidth
    ESS computed separately in :func:`_local_density_ratio`.

    :param event_id, n_original: if ``X``/``weights`` hold noise-injection
        replicas of the same original training events (see
        docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md),
        pass the replica-to-original mapping here so ESS is computed on
        each original event's GROUPED contribution, not per replica --
        otherwise ``n_replicates`` copies of one real point would look
        like ``n_replicates`` independent points. ``None`` (the default,
        used by every pre-existing direct caller) means no replication is
        in play: a trivial, non-grouping mapping is used.
    :return: ``(g, ess)`` -- the kernel-weighted values and the local ESS
        at whichever scale the widening stopped at.
    """
    if event_id is None:
        event_id = np.arange(len(weights))
    if n_original is None:
        n_original = len(weights)

    scale = 1.0
    g = _kernel_weighted_values(x, X, weights, bandwidths, dims)
    ess = _grouped_effective_sample_size(g, event_id, n_original)
    if ess >= min_ess:
        return g, ess
    ceiling = _grouped_effective_sample_size(weights, event_id, n_original)
    for _ in range(max_iter):
        if ess >= min_ess or ess >= ceiling - 1e-9:
            break
        scale *= growth
        g_new = _kernel_weighted_values(x, X, weights, bandwidths * scale, dims)
        ess_new = _grouped_effective_sample_size(g_new, event_id, n_original)
        if ess_new <= ess:
            ess = ess_new
            break
        g, ess = g_new, ess_new
    return g, ess
```

- [ ] **Step 4: Update `_local_density_ratio` for grouped ESS**

Replace:
```python
def _local_density_ratio(x, Xs, ws, Xb, wb, bandwidths, dims, min_kde_ess=DEFAULT_KDE_MIN_ESS,
                          ess_ceiling_sig=None, ess_ceiling_bg=None):
```
through (the whole function body, ending at):
```python
    S, B = float(np.sum(gs)), float(np.sum(gb))
    if S + B < 1e-300:
        return None, local_ess_sig, local_ess_bg
    return S / (S + B), local_ess_sig, local_ess_bg
```
with:
```python
def _local_density_ratio(x, Xs, ws, Xb, wb, bandwidths, dims, min_kde_ess=DEFAULT_KDE_MIN_ESS,
                          ess_ceiling_sig=None, ess_ceiling_bg=None,
                          event_id_s=None, event_id_b=None, n_orig_s=None, n_orig_b=None):
    """Local weighted kernel density ratio and local ESS at point ``x``.

    :param Xs, ws: signal training feature matrix and rate weights -- may
        hold noise-injection replicas of fewer original events (see
        docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md).
    :param Xb, wb: background training feature matrix and rate weights, same caveat.
    :param min_kde_ess: minimum effective sample size each class's kernel is
        adaptively widened to reach before computing the density ratio (see
        :func:`_widened_kernel_values`). This does NOT affect
        ``local_ess_sig``/``local_ess_bg`` below, which stay at the fixed
        bandwidth and drive the tree-vs-KDE blend-weight decision. A class
        is skipped from widening entirely (its fixed-bandwidth value is used
        for ``p_kde`` unchanged) when that class's own overall ESS ceiling
        -- ``effective_sample_size`` of its raw training weights, ignoring
        distance -- is already below ``min_kde_ess``: widening such a class
        would only swallow its whole (small) population regardless of true
        distance from ``x``, destroying locality rather than recovering it.
    :param ess_ceiling_sig, ess_ceiling_bg: precomputed
        ``effective_sample_size(ws)``/``effective_sample_size(wb)``, for
        callers that query the same ``(Xs, ws, Xb, wb)`` repeatedly (e.g.
        :meth:`ColourClassifier._blend_with_local_density`, once per event)
        and already have these cached -- they depend only on ``ws``/``wb``,
        never on ``x``, so recomputing them on every call is wasted work at
        this project's training-set scale. ``None`` (the default, used by
        one-off/direct callers) computes them here instead.
    :param event_id_s, event_id_b, n_orig_s, n_orig_b: replica-to-original
        mapping for ``Xs``/``Xb`` respectively (see
        :func:`_widened_kernel_values`). ``None`` (the default, used by
        every pre-existing direct caller) means no replication is in play.
    :return: ``(p_kde, local_ess_sig, local_ess_bg)``. ``p_kde`` is ``None``
        when both classes' FIXED-bandwidth kernel-weighted mass underflows
        to (numerically) zero at ``x`` -- the degenerate case callers fall
        back to the tree-only posterior for. This guard is checked before
        any widening is attempted, so a point with no real local
        information from either class is never "rescued" into a
        near-arbitrary ratio by widening far enough to reach unrelated data.
    """
    if event_id_s is None:
        event_id_s = np.arange(len(ws))
    if n_orig_s is None:
        n_orig_s = len(ws)
    if event_id_b is None:
        event_id_b = np.arange(len(wb))
    if n_orig_b is None:
        n_orig_b = len(wb)

    gs_fixed = _kernel_weighted_values(x, Xs, ws, bandwidths, dims)
    gb_fixed = _kernel_weighted_values(x, Xb, wb, bandwidths, dims)
    local_ess_sig = _grouped_effective_sample_size(gs_fixed, event_id_s, n_orig_s)
    local_ess_bg = _grouped_effective_sample_size(gb_fixed, event_id_b, n_orig_b)
    if float(np.sum(gs_fixed)) + float(np.sum(gb_fixed)) < 1e-300:
        return None, local_ess_sig, local_ess_bg

    # Only widen a class if doing so can plausibly reach min_kde_ess -- if a
    # class's own overall ESS ceiling is already below that target, widening
    # toward it only swallows the class's entire (small) population
    # regardless of true distance, destroying locality rather than
    # recovering it. Use the fixed-bandwidth value unchanged instead.
    if ess_ceiling_sig is None:
        ess_ceiling_sig = _grouped_effective_sample_size(ws, event_id_s, n_orig_s)
    if ess_ceiling_bg is None:
        ess_ceiling_bg = _grouped_effective_sample_size(wb, event_id_b, n_orig_b)
    if ess_ceiling_sig >= min_kde_ess:
        gs, _ = _widened_kernel_values(x, Xs, ws, bandwidths, dims, min_kde_ess,
                                        event_id=event_id_s, n_original=n_orig_s)
    else:
        gs = gs_fixed
    if ess_ceiling_bg >= min_kde_ess:
        gb, _ = _widened_kernel_values(x, Xb, wb, bandwidths, dims, min_kde_ess,
                                        event_id=event_id_b, n_original=n_orig_b)
    else:
        gb = gb_fixed

    S, B = float(np.sum(gs)), float(np.sum(gb))
    if S + B < 1e-300:
        return None, local_ess_sig, local_ess_bg
    return S / (S + B), local_ess_sig, local_ess_bg
```

- [ ] **Step 5: Finish `_store_kde_training_data` (replacing Task 3's placeholder)**

Replace:
```python
    def _store_kde_training_data(self, Xs, ws, Xb, wb, sigma_s, sigma_b):
```
through the end of that method (which currently ends with `self._kde_pattern_cache = {}`, plus Task 3's placeholder line `self._kde_sigma_s_unfiltered, self._kde_sigma_b_unfiltered = sigma_s, sigma_b`) with:
```python
    def _store_kde_training_data(self, Xs, ws, Xb, wb, sigma_s, sigma_b):
        """Retain the full (positive-weight) training events/weights/sigma +
        bandwidths for the local density-ratio blend in :meth:`probability`,
        and reset the per-pattern cache. ``ws``/``wb`` must already be the
        *balanced* weights computed earlier in :meth:`fit`, matching
        ``p_tree``'s own balanced-training-prior convention. ``sigma_s``/
        ``sigma_b`` are per-event measurement-uncertainty matrices, same row
        order and column order as ``Xs``/``Xb`` (see
        :func:`colour_feature_sigma_matrix`).

        ``kde_max_events`` subsampling does NOT happen here -- it happens
        lazily, per missing-data pattern, in :meth:`_kde_subset_for_pattern`.
        Real survey populations can have most events missing one or more
        features (different bands get a detection for different events);
        subsampling the full, NaN-mixed population *before* any pattern-
        specific finite-row filter wastes most of the budget on rows a
        given pattern discards anyway -- e.g. on BlueBEAR production data,
        capping at 4000 events before filtering left only ~65 rows (1.6%)
        usable for the all-features-finite pattern, versus ~2600 actually
        available. Filtering first, then capping per pattern, spends the
        budget only on rows that pattern can use. See docs/superpowers/
        specs/2026-10-26-kde-pattern-aware-subsampling-design.md.

        Noise-injection replication for the KDE also does NOT happen here --
        it happens lazily, per pattern, in :meth:`_kde_subset_for_pattern`,
        AFTER that pattern's own kde_max_events cap, for the same reason: a
        cap applied before replication would be spent on redundant copies of
        the same few original events instead of genuinely diverse ones. See
        docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md.
        """
        pos_s = ws > 0
        pos_b = wb > 0
        self._kde_Xs, self._kde_ws = Xs[pos_s], ws[pos_s]
        self._kde_sigma_s = sigma_s[pos_s]
        self._kde_Xb, self._kde_wb = Xb[pos_b], wb[pos_b]
        self._kde_sigma_b = sigma_b[pos_b]
        pooled_X = np.vstack([self._kde_Xs, self._kde_Xb])
        pooled_w = np.r_[self._kde_ws, self._kde_wb]
        self._kde_bandwidths = _scott_bandwidths(pooled_X, pooled_w)
        self._kde_pattern_cache = {}
```

- [ ] **Step 6: Update `_kde_subset_for_pattern`**

Replace:
```python
    def _kde_subset_for_pattern(self, pattern, dims):
        """Training rows finite over ``dims``, weighted-subsampled to at
        most ``kde_max_events`` (independently per class), plus each
        resulting subset's ESS ceiling -- cached per ``pattern`` on this
        instance (reset by :meth:`fit`).

        Subsampling happens HERE, after the finite-row filter, not in
        :meth:`_store_kde_training_data` -- so the ``kde_max_events`` budget
        is spent only on rows this specific pattern can actually use. Each
        distinct pattern gets its own independent cap, rather than all
        patterns competing for one global cap drawn before anyone knows
        which rows which pattern needs (see docs/superpowers/specs/
        2026-10-26-kde-pattern-aware-subsampling-design.md for why this
        matters on real data). The ceilings depend only on the resulting
        subset's weights, never on the query point, so caching them here
        (rather than recomputing in :func:`_local_density_ratio` on every
        event that shares this pattern) avoids an O(n) pass per event.
        """
        if pattern not in self._kde_pattern_cache:
            # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
            from .importance_sampling import effective_sample_size

            finite_s = np.all(np.isfinite(self._kde_Xs[:, dims]), axis=1)
            finite_b = np.all(np.isfinite(self._kde_Xb[:, dims]), axis=1)
            Xs_p, ws_p = _weighted_subsample(
                self._kde_Xs[finite_s], self._kde_ws[finite_s], self.kde_max_events)
            Xb_p, wb_p = _weighted_subsample(
                self._kde_Xb[finite_b], self._kde_wb[finite_b], self.kde_max_events)
            self._kde_pattern_cache[pattern] = (
                Xs_p, ws_p, Xb_p, wb_p,
                effective_sample_size(ws_p), effective_sample_size(wb_p))
        return self._kde_pattern_cache[pattern]
```
with:
```python
    def _kde_subset_for_pattern(self, pattern, dims):
        """Training rows finite over ``dims``, weighted-subsampled to at
        most ``kde_max_events`` (independently per class), then
        noise-injection-replicated from each surviving event's own
        measurement uncertainty -- cached per ``pattern`` on this instance
        (reset by :meth:`fit`) as a :class:`_KDEPatternSubset`.

        Subsampling happens HERE, after the finite-row filter, not in
        :meth:`_store_kde_training_data` -- so the ``kde_max_events`` budget
        is spent only on rows this specific pattern can actually use (see
        docs/superpowers/specs/2026-10-26-kde-pattern-aware-subsampling-design.md).
        Replication ALSO happens here, AFTER that cap -- replicating before
        capping would spend the budget on redundant copies of a few original
        events instead of genuinely diverse ones (see docs/superpowers/specs/
        2026-10-26-uncertainty-aware-noise-injection-design.md). The ESS
        ceilings are computed from the CAPPED, PRE-replication weights --
        replicating a point into K pieces must not make its own class look
        like it has K times more statistical support.
        """
        if pattern not in self._kde_pattern_cache:
            # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
            from .importance_sampling import effective_sample_size

            finite_s = np.all(np.isfinite(self._kde_Xs[:, dims]), axis=1)
            finite_b = np.all(np.isfinite(self._kde_Xb[:, dims]), axis=1)
            finite_s_idx = np.where(finite_s)[0]
            finite_b_idx = np.where(finite_b)[0]
            Xs_cap, ws_cap, chosen_s = _weighted_subsample(
                self._kde_Xs[finite_s], self._kde_ws[finite_s], self.kde_max_events)
            Xb_cap, wb_cap, chosen_b = _weighted_subsample(
                self._kde_Xb[finite_b], self._kde_wb[finite_b], self.kde_max_events)
            sigma_s_cap = self._kde_sigma_s[finite_s_idx[chosen_s]]
            sigma_b_cap = self._kde_sigma_b[finite_b_idx[chosen_b]]

            Xs_rep, ws_rep, event_id_s = _replicate_with_uncertainty(
                Xs_cap, sigma_s_cap, ws_cap, dims, self.n_uncertainty_replicates)
            Xb_rep, wb_rep, event_id_b = _replicate_with_uncertainty(
                Xb_cap, sigma_b_cap, wb_cap, dims, self.n_uncertainty_replicates)

            self._kde_pattern_cache[pattern] = _KDEPatternSubset(
                Xs_rep, ws_rep, Xb_rep, wb_rep,
                effective_sample_size(ws_cap), effective_sample_size(wb_cap),
                event_id_s, event_id_b, len(Xs_cap), len(Xb_cap))
        return self._kde_pattern_cache[pattern]
```

- [ ] **Step 7: Update `_blend_with_local_density`**

Replace:
```python
    def _blend_with_local_density(self, events, p_tree):
        """Blend ``p_tree`` with the local KDE density ratio, per event,
        weighted by local training support (see
        docs/superpowers/specs/2026-10-25-local-density-posterior-blend-design.md).
        Falls back to ``p_tree`` unchanged wherever the KDE side is
        degenerate (no nearby training data of either class, or no finite
        feature columns for that event).
        """
        X, _ = colour_feature_matrix(events, self.keys, self.extra)
        out = np.array(p_tree, dtype=float, copy=True)
        for i in range(len(events)):
            x = X[i]
            pattern = _finite_pattern(x)
            dims = np.where(np.asarray(pattern))[0]
            if len(dims) == 0:
                continue
            Xs_p, ws_p, Xb_p, wb_p, ceil_sig, ceil_bg = self._kde_subset_for_pattern(pattern, dims)
            p_kde, ess_sig, ess_bg = _local_density_ratio(
                x, Xs_p, ws_p, Xb_p, wb_p, self._kde_bandwidths, dims, self.kde_min_ess,
                ess_ceiling_sig=ceil_sig, ess_ceiling_bg=ceil_bg)
            if p_kde is None:
                continue
            alpha = _blend_weight(ess_sig, ess_bg, self.ess_threshold, self.log_width)
            out[i] = alpha * p_tree[i] + (1 - alpha) * p_kde
        return out
```
with:
```python
    def _blend_with_local_density(self, events, p_tree):
        """Blend ``p_tree`` with the local KDE density ratio, per event,
        weighted by local training support (see
        docs/superpowers/specs/2026-10-25-local-density-posterior-blend-design.md).
        Falls back to ``p_tree`` unchanged wherever the KDE side is
        degenerate (no nearby training data of either class, or no finite
        feature columns for that event).

        Each query event is ALSO noise-injection-replicated from its own
        ``*_err`` fields (if present), and the blended result is averaged
        back over those replicas -- symmetric with how training data is
        handled (see docs/superpowers/specs/
        2026-10-26-uncertainty-aware-noise-injection-design.md). ``p_tree``
        arrives already averaged over its own replicas (see
        :meth:`score`); only the KDE-side evaluation is repeated here.
        """
        X, _ = colour_feature_matrix(events, self.keys, self.extra)
        sigma = colour_feature_sigma_matrix(events, self.keys, self.extra)
        out = np.array(p_tree, dtype=float, copy=True)
        for i in range(len(events)):
            x = X[i]
            pattern = _finite_pattern(x)
            dims = np.where(np.asarray(pattern))[0]
            if len(dims) == 0:
                continue
            subset = self._kde_subset_for_pattern(pattern, dims)
            x_rep, _, _ = _replicate_with_uncertainty(
                x[np.newaxis, :], sigma[i][np.newaxis, :], np.array([1.0]),
                dims, self.n_uncertainty_replicates)
            blended_vals = []
            for xr in x_rep:
                p_kde, ess_sig, ess_bg = _local_density_ratio(
                    xr, subset.Xs, subset.ws, subset.Xb, subset.wb, self._kde_bandwidths, dims,
                    self.kde_min_ess, ess_ceiling_sig=subset.ess_ceiling_sig,
                    ess_ceiling_bg=subset.ess_ceiling_bg, event_id_s=subset.event_id_s,
                    event_id_b=subset.event_id_b, n_orig_s=subset.n_orig_s, n_orig_b=subset.n_orig_b)
                if p_kde is None:
                    continue
                alpha = _blend_weight(ess_sig, ess_bg, self.ess_threshold, self.log_width)
                blended_vals.append(alpha * p_tree[i] + (1 - alpha) * p_kde)
            if blended_vals:
                out[i] = float(np.mean(blended_vals))
        return out
```

- [ ] **Step 8: Rewrite the three existing test blocks that unpack `_kde_subset_for_pattern`'s return positionally**

In `tests/test_cmsne.py`:

(a) The `kde_max_events` section -- replace:
```python
_Xs_capped, _ws_capped, _, _, _, _ = _clf_cap._kde_subset_for_pattern((True,), np.array([0]))
check('kde_max_events caps the PER-PATTERN subsample (via _kde_subset_for_pattern) to at most the cap',
      _Xs_capped.shape[0] <= 50, _Xs_capped.shape[0])
check('kde_max_events preserves total rate mass after per-pattern subsampling (within 1%)',
      abs(_ws_capped.sum() - _cap_weights.sum()) / _cap_weights.sum() < 0.01,
      (_ws_capped.sum(), _cap_weights.sum()))

# Below the cap: no subsampling, nothing changes at the pattern level either.
_clf_nocap = ColourClassifier(keys=['gr'], include_peakmag=False, kde_max_events=10000)
_clf_nocap.fit(_cap_events, _cap_events, _cap_weights, _cap_weights)
_Xs_nocap, _, _, _, _, _ = _clf_nocap._kde_subset_for_pattern((True,), np.array([0]))
check('kde_max_events above the actual event count leaves the per-pattern subset untouched',
      _Xs_nocap.shape[0] == np.sum(_cap_weights > 0), _Xs_nocap.shape[0])
```
with:
```python
_subset_capped = _clf_cap._kde_subset_for_pattern((True,), np.array([0]))
check('kde_max_events caps the PER-PATTERN subsample (via _kde_subset_for_pattern) to at most the cap',
      _subset_capped.Xs.shape[0] <= 50, _subset_capped.Xs.shape[0])
check('kde_max_events preserves total rate mass after per-pattern subsampling (within 1%)',
      abs(_subset_capped.ws.sum() - _cap_weights.sum()) / _cap_weights.sum() < 0.01,
      (_subset_capped.ws.sum(), _cap_weights.sum()))

# Below the cap: no subsampling, nothing changes at the pattern level either.
_clf_nocap = ColourClassifier(keys=['gr'], include_peakmag=False, kde_max_events=10000)
_clf_nocap.fit(_cap_events, _cap_events, _cap_weights, _cap_weights)
_subset_nocap = _clf_nocap._kde_subset_for_pattern((True,), np.array([0]))
check('kde_max_events above the actual event count leaves the per-pattern subset untouched',
      _subset_nocap.Xs.shape[0] == np.sum(_cap_weights > 0), _subset_nocap.Xs.shape[0])
```

(b) The "budget is not wasted" section -- replace:
```python
_clf_waste = ColourClassifier(keys=['gr'], include_peakmag=True, kde_max_events=10)
_clf_waste.columns = ['gr', 'peakmag']
_clf_waste._store_kde_training_data(_Xs_waste, _ws_waste, _Xs_waste, _ws_waste)
check('_store_kde_training_data retains the FULL population (no global subsampling)',
      _clf_waste._kde_Xs.shape[0] == _n_both + _n_col0_only, _clf_waste._kde_Xs.shape[0])
_Xs_both, _, _, _, _, _ = _clf_waste._kde_subset_for_pattern((True, True), np.array([0, 1]))
check('the all-finite pattern reaches its FULL kde_max_events cap, drawn only from rows it can use '
      '(not starved by 950 unrelated-pattern rows competing for a shared budget)',
      _Xs_both.shape[0] == 10, _Xs_both.shape[0])
_Xs_col0, _, _, _, _, _ = _clf_waste._kde_subset_for_pattern((True, False), np.array([0]))
check('a second, disjoint pattern independently reaches its own full cap too (patterns do not starve each other)',
      _Xs_col0.shape[0] == 10, _Xs_col0.shape[0])
```
with:
```python
_clf_waste = ColourClassifier(keys=['gr'], include_peakmag=True, kde_max_events=10)
_clf_waste.columns = ['gr', 'peakmag']
_sigma_waste = np.full_like(_Xs_waste, np.nan)  # no uncertainty data -- isolates the subsampling behaviour
_clf_waste._store_kde_training_data(_Xs_waste, _ws_waste, _Xs_waste, _ws_waste, _sigma_waste, _sigma_waste)
check('_store_kde_training_data retains the FULL population (no global subsampling)',
      _clf_waste._kde_Xs.shape[0] == _n_both + _n_col0_only, _clf_waste._kde_Xs.shape[0])
_subset_both = _clf_waste._kde_subset_for_pattern((True, True), np.array([0, 1]))
check('the all-finite pattern reaches its FULL kde_max_events cap, drawn only from rows it can use '
      '(not starved by 950 unrelated-pattern rows competing for a shared budget)',
      _subset_both.Xs.shape[0] == 10, _subset_both.Xs.shape[0])
_subset_col0 = _clf_waste._kde_subset_for_pattern((True, False), np.array([0]))
check('a second, disjoint pattern independently reaches its own full cap too (patterns do not starve each other)',
      _subset_col0.Xs.shape[0] == 10, _subset_col0.Xs.shape[0])
```

(c) The `ess_ceiling` cache section -- replace:
```python
_clf_cache = ColourClassifier(keys=['gr'], include_peakmag=False)
_clf_cache.fit(_dense_sig, _dense_bg, _dense_sw, _dense_bw)
_pattern_dense = (True,)
_dims_dense = np.array([0])
_subset = _clf_cache._kde_subset_for_pattern(_pattern_dense, _dims_dense)
check('_kde_subset_for_pattern now returns a 6-tuple including the two cached ESS ceilings',
      len(_subset) == 6, len(_subset))
_expected_ceil_sig = _ess_ref(_subset[1])
_expected_ceil_bg = _ess_ref(_subset[3])
check('_kde_subset_for_pattern caches effective_sample_size(ws_p)/effective_sample_size(wb_p) correctly',
      np.isclose(_subset[4], _expected_ceil_sig) and np.isclose(_subset[5], _expected_ceil_bg),
      (_subset[4], _expected_ceil_sig, _subset[5], _expected_ceil_bg))
check('_kde_subset_for_pattern caches the same tuple object on a repeated call (no recomputation)',
      _clf_cache._kde_subset_for_pattern(_pattern_dense, _dims_dense) is _subset)
```
with:
```python
_clf_cache = ColourClassifier(keys=['gr'], include_peakmag=False)
_clf_cache.fit(_dense_sig, _dense_bg, _dense_sw, _dense_bw)
_pattern_dense = (True,)
_dims_dense = np.array([0])
_subset = _clf_cache._kde_subset_for_pattern(_pattern_dense, _dims_dense)
check('_kde_subset_for_pattern now returns a _KDEPatternSubset with 10 fields '
      '(Xs, ws, Xb, wb, ess_ceiling_sig, ess_ceiling_bg, event_id_s, event_id_b, n_orig_s, n_orig_b)',
      len(_subset) == 10, len(_subset))
_expected_ceil_sig = _ess_ref(_subset.ws)
_expected_ceil_bg = _ess_ref(_subset.wb)
check('_kde_subset_for_pattern caches effective_sample_size(ws)/effective_sample_size(wb) correctly '
      '(no uncertainty data here, so ws/wb are unreplicated and this matches directly)',
      np.isclose(_subset.ess_ceiling_sig, _expected_ceil_sig) and
      np.isclose(_subset.ess_ceiling_bg, _expected_ceil_bg),
      (_subset.ess_ceiling_sig, _expected_ceil_sig, _subset.ess_ceiling_bg, _expected_ceil_bg))
check('_kde_subset_for_pattern caches the same object on a repeated call (no recomputation)',
      _clf_cache._kde_subset_for_pattern(_pattern_dense, _dims_dense) is _subset)
```

- [ ] **Step 9: Write the new grouped-ESS-integration and issue #37 smoking-gun tests**

Add a new section after the rewritten blocks above:
```python
print('\n=== uncertainty-aware noise injection: KDE-side wiring ===')
# Grouped-ESS integration: _widened_kernel_values and _local_density_ratio must
# not be fooled by replicas of one real point, end to end (not just via the
# standalone _grouped_effective_sample_size helper tested in Task 2).
_gi_h = np.array([1.0])
_gi_dims = np.array([0])
_gi_X_one_point_x5 = np.array([[3.0], [3.0], [3.0], [3.0], [3.0]])  # 5 "replicas" of ONE real point
_gi_w_one_point_x5 = np.full(5, 1e-5 / 5)                           # each at weight/5
_gi_event_id = np.array([0, 0, 0, 0, 0])
_gi_g_ungrouped = _kernel_weighted_values(np.array([0.0]), _gi_X_one_point_x5, _gi_w_one_point_x5, _gi_h, _gi_dims)
check('WITHOUT the event_id/n_original fix, 5 replicas of one point look like ESS~5',
      _ess_ref(_gi_g_ungrouped) > 4.0, _ess_ref(_gi_g_ungrouped))
_, _gi_ess_grouped = _widened_kernel_values(
    np.array([0.0]), _gi_X_one_point_x5, _gi_w_one_point_x5, _gi_h, _gi_dims, min_ess=10.0,
    event_id=_gi_event_id, n_original=1)
check('_widened_kernel_values with event_id/n_original correctly reports ESS~1 for the same 5 replicas',
      _gi_ess_grouped < 1.5, _gi_ess_grouped)

# Issue #37 smoking-gun reproduction, at the _local_density_ratio level (the
# same style as PR #36's own bug-reproduction test, _bug_Xs/_bug_ws above):
# one background point sitting RIGHT AT the query with a large, unmodelled
# uncertainty (mirrors the real gr=6.16 event -- a near-detection-limit
# colour) single-handedly dominates the density estimate WITHOUT noise
# injection, exactly as observed on real data, and is correctly smeared back
# down WITH it. Signal sits far enough away (3 bandwidths) that its 20 points
# don't outweigh background's single exact-match point until that point's own
# uncertainty is taken into account.
_sg_h = np.array([1.0])
_sg_dims = np.array([0])
_sg_Xs = np.array([[3.0]] * 20)       # signal: 20 points, each 3 bandwidths from the query
_sg_ws = np.full(20, 1e-5)
_sg_Xb = np.array([[0.0]])            # background: ONE point exactly at the query
_sg_wb = np.array([1e-5])
_sg_sigma_b = np.array([[5.0]])       # ...with a large, unmodelled measurement uncertainty

_p_sg_noinj, _, _ = _local_density_ratio(np.array([0.0]), _sg_Xs, _sg_ws, _sg_Xb, _sg_wb, _sg_h, _sg_dims)
check('WITHOUT noise injection, the single exact-match high-uncertainty background point '
      'dominates the query, as observed in the real issue #37 finding',
      _p_sg_noinj is not None and _p_sg_noinj < 0.3, _p_sg_noinj)

_sg_Xb_rep, _sg_wb_rep, _sg_event_id_b = _replicate_with_uncertainty(
    _sg_Xb, _sg_sigma_b, _sg_wb, _sg_dims, n_replicates=500, seed=3)
_p_sg_inj, _, _ = _local_density_ratio(
    np.array([0.0]), _sg_Xs, _sg_ws, _sg_Xb_rep, _sg_wb_rep, _sg_h, _sg_dims,
    event_id_b=_sg_event_id_b, n_orig_b=1)
check('WITH noise injection, that point\'s own large uncertainty smears its contribution and '
      'the probability moves back toward the confident signal population',
      _p_sg_inj is not None and _p_sg_inj > _p_sg_noinj, (_p_sg_noinj, _p_sg_inj))
```
(Hand-verified: fixed bandwidth `h=1.0`; signal's 20 points at distance 3 give kernel `exp(-4.5)≈0.0111` each, `S_fixed≈2.22e-6`; background's single exact-match point gives `B_fixed=1e-5` (kernel=1) → `p_noinj≈0.182`. After replicating that one point with `σ=5.0`, its expected post-injection kernel value is `h/sqrt(h²+σ²)=1/sqrt(26)≈0.196` → `B_after≈1.96e-6`, giving `p_inj≈0.531` -- both comfortably clear of the asserted thresholds. Signal's own ceiling is 20 (≥ the default `min_kde_ess=10.0`), so `_local_density_ratio` attempts widening for it too, but since all 20 signal points share an identical kernel value, `_widened_kernel_values`'s very first ESS check already reads 20 and returns immediately unwidened -- `gs` ends up identical to `gs_fixed` either way, so this doesn't change the numbers above.)

- [ ] **Step 10: Run the tests to verify they fail, then pass**

Run (before implementation, Steps 1-7): `python tests/test_cmsne.py 2>&1 | tail -40`
Expected: failures from the old positional-tuple tests (now updated to expect attribute access, which doesn't exist until Steps 1-7 land) and from the new grouped-ESS/smoking-gun tests (missing `event_id`/`n_original` params, `_KDEPatternSubset` not defined).

Run (after Steps 1-7): `python tests/test_cmsne.py 2>&1 | tail -60`
Expected: all pass, including every rewritten and newly-added check.

- [ ] **Step 11: Commit**

```bash
git add cmsne/classifier.py tests/test_cmsne.py
git commit -m "fix: wire noise-injection replication and grouped ESS into the KDE blend"
```

---

## Task 5: Integration verification

**Goal:** Confirm the assembled mechanism holds together end to end: `rank()` and `TrajectoryClassifier` inherit noise injection with no new code, the full pre-existing test suite remains green, and the issue #37 reproduction from Task 4 also holds when driven through the PUBLIC `probability()` API exactly as a real caller would use it (Task 4's own test already does this, but this task adds the inheritance checks and a final whole-suite confirmation as its own, separately-committable unit).

**Files:**
- Test: `tests/test_cmsne.py` (new section after Task 4's)

**Acceptance Criteria:**
- [ ] `rank()` on a population with real uncertainty data produces the same ordering as sorting `score()`'s own (already-averaged) output directly -- proving `rank()` needs no separate change.
- [ ] `TrajectoryClassifier` (PR #14/#15), which subclasses `ColourClassifier`, trains and predicts correctly on a population carrying real uncertainty data, with no code changes of its own.
- [ ] The complete pre-existing test suite (PR #35, #36, the subsampling fix, this plan's own Tasks 1-4) passes together, in one run, with 0 failures.

**Verify:** `python tests/test_cmsne.py` → all pass, 0 failures.

**Steps:**

- [ ] **Step 1: Write the failing tests**

After Task 4's new section, insert:
```python
print('\n=== uncertainty-aware noise injection: integration ===')
from cmsne.trajectory import TrajectoryClassifier

_rank_sig = [{'gr': float(_rng_unc.normal(1.0, 0.3)), 'peakmag': float(_rng_unc.normal(22.0, 1.0)),
              'gr_err': 0.2} for _ in range(50)]
_rank_bg = [{'gr': float(_rng_unc.normal(1.5, 0.3)), 'peakmag': float(_rng_unc.normal(22.0, 1.0)),
             'gr_err': 0.2} for _ in range(50)]
_rank_w = np.full(50, 1e-5)
_clf_rank = ColourClassifier(keys=['gr'], include_peakmag=True, n_uncertainty_replicates=4)
_clf_rank.fit(_rank_sig, _rank_bg, _rank_w, _rank_w)
_rank_query = [{'gr': float(_rng_unc.normal(1.2, 0.3)), 'peakmag': 22.0, 'gr_err': 0.2} for _ in range(10)]
check('rank() orders by score() with no separate change needed',
      np.array_equal(_clf_rank.rank(_rank_query), np.argsort(_clf_rank.score(_rank_query))[::-1]))


# TrajectoryClassifier hardcodes include_peakmag=False internally and sets its
# own self.extra -- follow the SAME construction the existing "inherits the
# sparse-corner blend" test above already uses (_tc_event/TRAJECTORY_COLOUR_KEYS/
# TRAJECTORY_MAGNITUDE_KEYS/_tc_key/_rng_sparse), only adding an *_err field;
# passing keys=.../include_peakmag=... here would collide with its __init__.
_tc_unc_sig = [_tc_event(float(_rng_sparse.normal(1.0, 0.3)), float(_rng_sparse.normal(22.0, 1.0)))
              for _ in range(50)]
_tc_unc_bg = [_tc_event(float(_rng_sparse.normal(1.3, 0.3)), float(_rng_sparse.normal(22.0, 1.0)))
             for _ in range(50)]
for _e in _tc_unc_sig + _tc_unc_bg:
    _e[f'{_tc_key}_err'] = 0.2
_tc_unc_w = np.full(50, 1e-5)
_traj_clf_unc = TrajectoryClassifier(n_uncertainty_replicates=4)
_traj_clf_unc.fit(_tc_unc_sig, _tc_unc_bg, _tc_unc_w, _tc_unc_w)
_tc_unc_query = [_tc_event(1.1, 22.0)]
_tc_unc_query[0][f'{_tc_key}_err'] = 0.2
_traj_probe = _traj_clf_unc.probability(_tc_unc_query)
check('TrajectoryClassifier inherits noise injection with no extra code',
      np.all(np.isfinite(_traj_probe)) and np.all((_traj_probe >= 0) & (_traj_probe <= 1)), _traj_probe)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py 2>&1 | tail -20`
Expected: failures only if the inheritance assumptions don't hold (they should, by construction -- this step mainly confirms that, catching any surprise rather than expecting a red bar).

- [ ] **Step 3: Run the full suite**

Run: `python tests/test_cmsne.py 2>&1 | tail -3`
Expected: `===== N passed, 0 failed =====` for the complete suite, including every test from Tasks 1-5 and every pre-existing test from PR #35, #36, and the subsampling fix.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cmsne.py
git commit -m "test: confirm rank()/TrajectoryClassifier inherit noise injection, full suite green"
```

---

## Task 6: Real-data validation

**Goal:** Regenerate the Monte Carlo population with real `*_err` fields, then re-run the full Manual Validation sequence with `n_uncertainty_replicates` as a new, fourth tuning axis -- confirming (or revising) the issue #37 conclusions with real uncertainty data in hand, on BlueBEAR.

**This is not a code-TDD task** -- it is the "Real-data validation" step this plan's spec scoped in explicitly (unlike every prior spec in this investigation, which deferred it). It runs real BlueBEAR jobs and produces a report, the same way every manual-validation pass earlier in this investigation was actually executed (never via bare `ssh` compute -- always `sbatch`, per project memory).

**Files:**
- None (no code changes) -- the report lands as an amendment to `docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md`'s own "Validation results" section (mirroring how `2026-10-26-kde-pattern-aware-subsampling-design.md`'s was appended), and/or a comment on issue #37, whichever the result calls for.

**Acceptance Criteria:**
- [ ] The regenerated population's event dicts contain finite `gr_err`/`peakmag_err`/etc. values wherever the corresponding colour/magnitude itself is finite (spot-checked directly, not assumed).
- [ ] The `ess_threshold`/`log_width`/`kde_min_ess`/`n_uncertainty_replicates` four-axis sweep is re-run against the regenerated population.
- [ ] The bright+red corner re-check, bulk backward-compatibility re-check, boundary figure regeneration, and timing re-measurement are all re-run and reported, exactly as in every prior Manual Validation pass this investigation.
- [ ] Whatever the real-data result turns out to be (corner criterion met, still unmet for a now-understood reason, or something new) is reported plainly -- this task's job is to produce that evidence, not to presuppose a particular outcome.

**Verify:** A written report (this plan's final status update, or a follow-up GitHub issue/comment if the result reopens a design question) covering every item above, backed by actual BlueBEAR job output -- not inferred or estimated.

**Steps:**

- [ ] **Step 1: Sync BlueBEAR to the commit including Tasks 1-5**

```bash
ssh bluebear "cd /rds/projects/s/smithgp-lensed-transients/andres/Colour_Magnitude/ && git fetch origin --quiet && git pull --quiet && git log --oneline -3"
```
Expected: the three commits from Tasks 1, 4, and 5 (or however Tasks 1-5 ended up committed/merged) are present.

- [ ] **Step 2: Regenerate the Monte Carlo population**

Use the EXISTING generation pipeline (`scripts/run_multicolour.py` + `scripts/run_multicolour_bluebear.sh`) unchanged -- `MultiColourGenerator`/`rate_weight` call `event_multicolour` internally, so the new `*_err` fields flow through automatically once Task 1 has landed; no changes to this script are needed.

```bash
ssh bluebear "cd /rds/projects/s/smithgp-lensed-transients/andres/Colour_Magnitude/ && sbatch --array=0-9 scripts/run_multicolour_bluebear.sh --n 100000"
```
This matches the scale (10 classes, 100k attempts each) used to produce `results/mc_58818280`, for direct before/after comparability. Each array task has a 6-hour budget (per the script's own `#SBATCH --time=06:00:00`); the full job may take multiple hours across all 10 array tasks -- this is the expensive step flagged in the spec, not a quick script. Monitor via `squeue -u $USER` and the per-task logs in `slurm_logs/mc-<job>_<task>.out`; do not proceed until all 10 array tasks report completion.

- [ ] **Step 3: Spot-check the regenerated population has real uncertainty fields**

Write a small diagnostic script (following this investigation's established pattern: write locally, `scp` to BlueBEAR, submit via `sbatch`, never run compute on the login node) that loads a few regenerated class pkls and checks, for a sample of events with a finite `gr`, that `gr_err` is also finite and positive; same for `peakmag`/`peakmag_err`. Report the fraction of finite-colour events that now also have a finite error (expected: all of them, since Task 1 computes the error alongside the colour unconditionally).

- [ ] **Step 4: Extend the Manual Validation script with the fourth tuning axis**

Build on this investigation's existing `manual_validation_v2.py`-style script (the one used for the subsampling-fix validation): add `n_uncertainty_replicates` as a sweep axis (e.g. `[1, 4, 8, 16, 32]`, with `1` as the explicit "no injection" baseline) alongside the existing `ess_threshold`/`log_width`/`kde_min_ess` axes, pointed at the NEWLY regenerated population directory (`results/mc_<new job ID>`), not the old `mc_58818280`. Re-run: the bright+red corner check (same success criterion, P(lensed) > 0.9), the bulk backward-compatibility check (grid points reaching `alpha > 0.99`), the boundary figure, and the timing comparison (now three stacked costs: density blend + adaptive widening + noise injection, relative to the tree-only baseline).

- [ ] **Step 5: Submit and run via `sbatch`, exactly as every prior Manual Validation pass this investigation**

```bash
ssh bluebear "cd /rds/projects/s/smithgp-lensed-transients/andres/Colour_Magnitude/ && sbatch scripts/_manual_validation_scratch.sh"
```
(Reuse this investigation's established script-naming convention for scratch validation scripts.) Wait for completion, then read the output log.

- [ ] **Step 6: Report**

Document the outcome in a short follow-up -- either an amendment to this plan's spec (mirroring how `docs/superpowers/specs/2026-10-26-kde-pattern-aware-subsampling-design.md`'s own "Validation results" section was appended after its real-data run) or a comment on issue #37, whichever the result calls for:
- If the corner's `P(lensed) > 0.9` criterion is now met: report at what `n_uncertainty_replicates` and why, and whether the issue #37 "background genuinely has more density there" conclusion changes once replicated with real uncertainty.
- If it is still not met: report whether the specific smoking-gun event (or its regenerated-run analogue) now contributes a visibly smeared, rather than sharp, local density -- even if the headline corner number doesn't cross 0.9, confirming the mechanism itself works as designed is still real signal.
- Either way: report the combined timing overhead and whether it remains acceptable for real use (`rank()`-ing a candidate list), and close or update issue #37 accordingly.
