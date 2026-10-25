# Local Density-Ratio Posterior Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ColourClassifier.probability()` (and everything that inherits or
calls it — `TrajectoryClassifier`, `cmsne.contours.posterior_surface`) approach
`P(lensed) ≈ 1` in physically-unambiguous bright+red regions, instead of
plateauing around 0.46, without changing its behaviour anywhere the model is
already well-supported by training data.

**Architecture:** Blend the existing tree+isotonic posterior with a local
weighted-KDE density ratio computed directly from the same training events,
with the blend weight driven by a pointwise, kernel-weighted generalization of
Kish's effective sample size (reusing `cmsne.importance_sampling.effective_sample_size`
unchanged). In the dense bulk the blend trusts the tree; where either class has
thin local support, it shifts toward the KDE ratio, which naturally rises
toward 1 as background's local kernel-weighted density vanishes.

**Tech Stack:** numpy (vectorized kernel evaluation), existing
`cmsne.importance_sampling.effective_sample_size`, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-25-local-density-posterior-blend-design.md`

## Global Constraints

- All new code lives in `cmsne/classifier.py` — no new module. The file is
  currently 212 lines; this addition (~180 lines of new functions/methods)
  does not make it unwieldy by this project's own standard (`importance_sampling.py`
  is already 300+ lines).
- `score()`, `rank()`, `predict()`, `roc_curve()`, `recovery_rate()` stay
  tree-only — their behaviour and numeric output must not change. Only
  `probability()` changes.
- Blend formula exactly as specified: `alpha(x)` is a sigmoid of
  `(log(min(local_ess_sig, local_ess_bg) + eps) - log(ess_threshold)) / log_width`;
  `final_p(x) = alpha*p_tree + (1-alpha)*p_kde`, in plain probability space (not
  log-odds).
- Starting defaults — `DEFAULT_ESS_THRESHOLD = 10.0`, `DEFAULT_LOG_WIDTH = 1.0`,
  `DEFAULT_KDE_MAX_EVENTS = 4000` — are provisional engineering choices, not
  empirically final. The spec explicitly defers final tuning to the Manual
  Validation step, run against real BlueBEAR production data
  (`results/mc_58818280`) already generated this session; nothing in this plan's
  automated tasks should claim these numbers as "tuned."
- Tests go in `tests/test_cmsne.py`, following the existing house style: a
  module-level `check(name, cond, extra='')` helper appending to `PASS`/`FAIL`
  lists, `ok`/`FAIL` printed per check — NOT pytest.
- After every task, `python tests/test_cmsne.py` must report no failures
  beyond the one known pre-existing, unrelated failure
  (`threshold pins contamination near target`) — confirm this by name, not
  just by count, since a new failure could coincidentally keep the count the
  same if an old one is simultaneously fixed.

**User decisions (already made):**
- Approach: local weighted-KDE density-ratio blend (not calibration-only, not
  a full KDE replacement of the tree) — user chose this explicitly after
  seeing all three options.
- Generality: the fix must work for any query point, not only pre-registered
  boxes like "bright+red" — user chose this explicitly over the cheaper,
  simpler pre-registered-boxes alternative.
- Spec approved with no changes.

---

## Task 1: Kernel and bandwidth helper functions

**Goal:** Add three pure, standalone helper functions to `cmsne/classifier.py`
— rate-weighted standard deviation, Scott's-rule bandwidths, and per-point
kernel-weighted values — with unit tests. No `ColourClassifier` changes yet.

**Files:**
- Modify: `cmsne/classifier.py` (add module-level functions before the
  `ColourClassifier` class, i.e. before the current line 52)
- Test: `tests/test_cmsne.py` (new section appended near the end, before the
  final summary block)

**Acceptance Criteria:**
- [ ] `_weighted_std(values, weights)` returns the rate-weighted standard
      deviation over finite, non-negative-weight entries, and `1.0` (a safe,
      finite fallback) when fewer than 2 such entries exist or the weighted
      variance is non-positive.
- [ ] `_scott_bandwidths(X, weights)` returns a `(d,)` array of positive,
      finite bandwidths via Scott's rule (`n_eff ** (-1/(d+4))` times each
      column's weighted std), using `cmsne.importance_sampling.effective_sample_size`
      for `n_eff`.
- [ ] `_kernel_weighted_values(x, X, weights, bandwidths, dims)` returns a
      `(n,)` array of `weights[i] * exp(-0.5 * sum_d ((x_d - X[i,d])/h_d)**2)`,
      evaluated only over the columns named in `dims`; returns an all-zero
      array when `dims` is empty or `X` has zero rows.
- [ ] `python tests/test_cmsne.py` reports no new failures (only the known
      pre-existing `threshold pins contamination near target`).

**Verify:** `python tests/test_cmsne.py` → `threshold pins contamination near target` is the only failure.

**Steps:**

- [ ] **Step 1: Add the import**

In `cmsne/classifier.py`, after the existing imports (after line 23,
`from .colour_magnitude import weighted_quantile`), add:

```python
from .importance_sampling import effective_sample_size
```

(No circular import: `importance_sampling.py` imports only from `.config`,
`.multicolour`, `.opsim`, none of which import `.classifier`.)

- [ ] **Step 2: Write the three helper functions**

Insert before the `class ColourClassifier:` line (currently line 52), after
the `colour_feature_matrix` function:

```python
def _weighted_std(values, weights):
    """Rate-weighted standard deviation of ``values``, over finite,
    non-negative-weight entries only.

    Returns ``1.0`` (a safe, finite fallback) if fewer than 2 such entries
    exist or the weighted variance is non-positive -- this keeps downstream
    bandwidths finite and positive rather than zero or NaN.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if np.sum(finite) < 2 or np.sum(weights[finite]) <= 0:
        return 1.0
    v, w = values[finite], weights[finite]
    mean = np.sum(v * w) / np.sum(w)
    var = np.sum(w * (v - mean) ** 2) / np.sum(w)
    return float(np.sqrt(var)) if var > 0 else 1.0


def _scott_bandwidths(X, weights):
    """Per-column Gaussian-kernel bandwidth via Scott's rule, over pooled
    ``X``/``weights``.

    :param X: ``(n, d)`` feature matrix, NaN allowed (ignored per-column by
        :func:`_weighted_std`).
    :param weights: ``(n,)`` rate weights, pooled across every class the
        kernel is shared between.
    :return: ``(d,)`` array of positive, finite bandwidths.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_eff = effective_sample_size(weights)
    d = X.shape[1]
    scott_factor = n_eff ** (-1.0 / (d + 4)) if n_eff > 0 else 1.0
    stds = np.array([_weighted_std(X[:, j], weights) for j in range(d)])
    return scott_factor * stds


def _kernel_weighted_values(x, X, weights, bandwidths, dims):
    """Per-training-row kernel-weighted value ``w_i * k_i``, over columns
    ``dims`` only.

    :param x: ``(d,)`` query feature vector (full column set; only ``dims``
        entries are read).
    :param X: ``(n, d)`` training feature matrix, already filtered to rows
        that are finite over ``dims`` -- this function does not filter rows
        itself.
    :param weights: ``(n,)`` training rate weights.
    :param bandwidths: ``(d,)`` per-column bandwidths (full column set; only
        ``dims`` entries are read).
    :param dims: 1D int array, which columns to evaluate the kernel over.
    :return: ``(n,)`` array, ``weights[i] * exp(-0.5 * sum_d ((x_d - X[i,d])/h_d)**2)``.
    """
    n = X.shape[0]
    if len(dims) == 0 or n == 0:
        return np.zeros(n)
    xd = np.asarray(x, dtype=float)[dims]
    Xd = X[:, dims]
    hd = np.asarray(bandwidths, dtype=float)[dims]
    z = (Xd - xd) / hd
    k = np.exp(-0.5 * np.sum(z ** 2, axis=1))
    return np.asarray(weights, dtype=float) * k
```

- [ ] **Step 3: Write the tests**

Append to `tests/test_cmsne.py`, before the final summary block:

```python
print('\n=== local density-ratio blend: kernel/bandwidth helpers ===')
from cmsne.classifier import _weighted_std, _scott_bandwidths, _kernel_weighted_values

_ws_vals = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
_ws_wts = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
check('_weighted_std ignores NaN and matches plain std of the finite subset',
      np.isclose(_weighted_std(_ws_vals, _ws_wts), np.std([1.0, 2.0, 3.0, 4.0]), atol=1e-9),
      _weighted_std(_ws_vals, _ws_wts))
check('_weighted_std returns the safe fallback 1.0 for a degenerate (all-equal) column',
      _weighted_std(np.array([5.0, 5.0, 5.0]), np.array([1.0, 1.0, 1.0])) == 1.0)
check('_weighted_std returns the safe fallback 1.0 for fewer than 2 finite entries',
      _weighted_std(np.array([5.0, np.nan]), np.array([1.0, 1.0])) == 1.0)

_bw_X = np.column_stack([np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                         np.array([10.0, 10.0, 10.0, 10.0, 10.0])])
_bw_w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
_bw = _scott_bandwidths(_bw_X, _bw_w)
check('_scott_bandwidths returns one positive, finite bandwidth per column',
      _bw.shape == (2,) and np.all(_bw > 0) and np.all(np.isfinite(_bw)), _bw)
check('_scott_bandwidths gives the degenerate (constant) column the smaller bandwidth',
      _bw[1] < _bw[0], _bw)

_kw_X = np.array([[0.0], [1.0], [10.0]])
_kw_w = np.array([1.0, 1.0, 1.0])
_kw_h = np.array([1.0])
_kw_dims = np.array([0])
_kw_vals = _kernel_weighted_values(np.array([0.0]), _kw_X, _kw_w, _kw_h, _kw_dims)
check('_kernel_weighted_values peaks at the exact-match training row',
      _kw_vals[0] > _kw_vals[1] > _kw_vals[2], _kw_vals)
check('_kernel_weighted_values matches weight*1.0 exactly at zero distance',
      np.isclose(_kw_vals[0], 1.0), _kw_vals[0])
check('_kernel_weighted_values returns zeros for empty dims',
      np.all(_kernel_weighted_values(np.array([0.0]), _kw_X, _kw_w, _kw_h, np.array([], dtype=int)) == 0.0))
check('_kernel_weighted_values returns zeros for zero training rows',
      np.all(_kernel_weighted_values(np.array([0.0]), np.zeros((0, 1)), np.zeros(0), _kw_h, _kw_dims) == 0.0))
```

- [ ] **Step 4: Run the test suite**

Run: `python tests/test_cmsne.py`
Expected: every new check above prints `ok`; the only failure anywhere in the
suite is the pre-existing `threshold pins contamination near target`.

- [ ] **Step 5: Commit**

```bash
git add cmsne/classifier.py tests/test_cmsne.py
git commit -m "feat: add kernel/bandwidth helpers for local density-ratio blend"
```

```json:metadata
{"files": ["cmsne/classifier.py", "tests/test_cmsne.py"], "verifyCommand": "python tests/test_cmsne.py", "acceptanceCriteria": ["_weighted_std computes rate-weighted std over finite entries, 1.0 fallback for degenerate/too-few cases", "_scott_bandwidths returns positive finite per-column bandwidths via Scott's rule using effective_sample_size", "_kernel_weighted_values returns weight*kernel per training row restricted to dims, zeros for empty dims/rows", "python tests/test_cmsne.py has no new failures beyond the known pre-existing one"], "modelTier": "mechanical"}
```

---

## Task 2: Local density ratio and blend weight

**Goal:** Add `_local_density_ratio` and `_blend_weight` — pure functions
building on Task 1's kernel helper — with tests proving the asymptotic
behaviour (density ratio → 1 when background has no local presence, blend
weight → 1 in dense regions) directly, as pure math, independent of
`ColourClassifier` wiring.

**Files:**
- Modify: `cmsne/classifier.py` (add two more module-level functions, after
  the ones from Task 1)
- Test: `tests/test_cmsne.py` (same new section, extended)

**Acceptance Criteria:**
- [ ] `_local_density_ratio(x, Xs, ws, Xb, wb, bandwidths, dims)` returns
      `(p_kde, local_ess_sig, local_ess_bg)`; `p_kde` is `S/(S+B)` where `S`,
      `B` are the summed kernel-weighted values for signal/background; `p_kde`
      is `None` when `S+B` underflows below `1e-300` (both classes locally
      absent).
- [ ] `local_ess_sig`/`local_ess_bg` are computed via
      `effective_sample_size` fed the per-event kernel-weighted values —
      not a new ESS formula.
- [ ] A synthetic case with signal events clustered at a point and zero
      background events within many bandwidths of it gives `p_kde` within
      `1e-6` of `1.0`; the symmetric case gives `p_kde` within `1e-6` of `0.0`.
- [ ] `_blend_weight(local_ess_sig, local_ess_bg, ess_threshold, log_width)`
      returns a value in `(0, 1)`; is driven by `min(local_ess_sig,
      local_ess_bg)`, not either alone; returns a value close to `1.0` when
      both local ESS values are far above `ess_threshold`, and close to `0.0`
      when either is far below it.
- [ ] `python tests/test_cmsne.py` reports no new failures.

**Verify:** `python tests/test_cmsne.py` → `threshold pins contamination near target` is the only failure.

**Steps:**

- [ ] **Step 1: Write the two functions**

Insert into `cmsne/classifier.py`, directly after `_kernel_weighted_values`
(from Task 1):

```python
def _local_density_ratio(x, Xs, ws, Xb, wb, bandwidths, dims):
    """Local weighted kernel density ratio and local ESS at point ``x``.

    :param Xs, ws: signal training feature matrix and rate weights.
    :param Xb, wb: background training feature matrix and rate weights.
    :return: ``(p_kde, local_ess_sig, local_ess_bg)``. ``p_kde`` is ``None``
        when both classes' kernel-weighted mass underflows to (numerically)
        zero at ``x`` -- the degenerate case callers fall back to the
        tree-only posterior for.
    """
    gs = _kernel_weighted_values(x, Xs, ws, bandwidths, dims)
    gb = _kernel_weighted_values(x, Xb, wb, bandwidths, dims)
    S, B = float(np.sum(gs)), float(np.sum(gb))
    local_ess_sig = effective_sample_size(gs)
    local_ess_bg = effective_sample_size(gb)
    if S + B < 1e-300:
        return None, local_ess_sig, local_ess_bg
    return S / (S + B), local_ess_sig, local_ess_bg


def _blend_weight(local_ess_sig, local_ess_bg, ess_threshold, log_width):
    """Sigmoid blend weight favouring the tree posterior when both classes
    have ample local support, and the KDE density ratio when either is thin.

    :return: ``alpha`` in ``(0, 1)``; 1.0 means "fully trust the tree".
    """
    m = min(local_ess_sig, local_ess_bg)
    z = (np.log(m + 1e-12) - np.log(ess_threshold)) / log_width
    return float(1.0 / (1.0 + np.exp(-z)))
```

- [ ] **Step 2: Write the tests**

Append to the same test section in `tests/test_cmsne.py`:

```python
print('\n=== local density-ratio blend: density ratio and blend weight ===')
from cmsne.classifier import _local_density_ratio, _blend_weight

_ldr_h = np.array([1.0])
_ldr_dims = np.array([0])
# Signal clustered at x=0; background far away (20 bandwidths), contributes
# essentially nothing to the kernel sum there.
_ldr_Xs = np.array([[0.0], [0.1], [-0.1]])
_ldr_ws = np.array([1e-5, 1e-5, 1e-5])
_ldr_Xb = np.array([[20.0], [21.0], [19.0]])
_ldr_wb = np.array([1e-5, 1e-5, 1e-5])
_p_kde, _ess_s, _ess_b = _local_density_ratio(
    np.array([0.0]), _ldr_Xs, _ldr_ws, _ldr_Xb, _ldr_wb, _ldr_h, _ldr_dims)
check('_local_density_ratio -> 1 when signal is local and background is many bandwidths away',
      _p_kde is not None and np.isclose(_p_kde, 1.0, atol=1e-6), _p_kde)
check('_local_density_ratio reports near-zero local ESS for the far-away class',
      _ess_b < 1e-6, _ess_b)

# Symmetric case: swap roles.
_p_kde2, _, _ = _local_density_ratio(
    np.array([0.0]), _ldr_Xb, _ldr_wb, _ldr_Xs, _ldr_ws, _ldr_h, _ldr_dims)
check('_local_density_ratio -> 0 when background is local and signal is far away',
      _p_kde2 is not None and np.isclose(_p_kde2, 0.0, atol=1e-6), _p_kde2)

# Degenerate case: both classes far from x.
_p_kde3, _, _ = _local_density_ratio(
    np.array([1000.0]), _ldr_Xs, _ldr_ws, _ldr_Xb, _ldr_wb, _ldr_h, _ldr_dims)
check('_local_density_ratio returns None when both classes are far from x',
      _p_kde3 is None, _p_kde3)

check('_blend_weight is near 1.0 when both local ESS values are far above threshold',
      _blend_weight(1000.0, 1000.0, ess_threshold=10.0, log_width=1.0) > 0.999)
check('_blend_weight is near 0.0 when either local ESS value is far below threshold',
      _blend_weight(1000.0, 0.01, ess_threshold=10.0, log_width=1.0) < 0.001)
check('_blend_weight is driven by the MINIMUM of the two local ESS values, not their average',
      _blend_weight(1000.0, 0.01, ess_threshold=10.0, log_width=1.0) ==
      _blend_weight(0.01, 0.01, ess_threshold=10.0, log_width=1.0))
```

- [ ] **Step 3: Run the test suite**

Run: `python tests/test_cmsne.py`
Expected: all new checks pass; only pre-existing failure remains.

- [ ] **Step 4: Commit**

```bash
git add cmsne/classifier.py tests/test_cmsne.py
git commit -m "feat: add local density-ratio and blend-weight functions"
```

```json:metadata
{"files": ["cmsne/classifier.py", "tests/test_cmsne.py"], "verifyCommand": "python tests/test_cmsne.py", "acceptanceCriteria": ["_local_density_ratio returns S/(S+B) from kernel-weighted sums, None when both underflow", "local ESS values computed via effective_sample_size on kernel-weighted values", "p_kde approaches 1 when signal is local and background is many bandwidths away, and 0 in the symmetric case", "_blend_weight is a sigmoid of log(min(ess_sig,ess_bg)) relative to ess_threshold/log_width, near 1 when both ample, near 0 when either thin", "python tests/test_cmsne.py has no new failures beyond the known pre-existing one"], "modelTier": "mechanical"}
```

---

## Task 3: Wire the blend into `ColourClassifier`

**Goal:** Make `ColourClassifier.fit()` store the (balanced) training events,
weights, and bandwidths needed for the blend, and make `probability()` use
them — grouping query events by which features are finite ("pattern"),
filtering/caching the matching training subset per pattern, computing the
blend per event, and falling back cleanly to the tree-only posterior wherever
the KDE side is degenerate. `score()`/`rank()`/`predict()`/`roc_curve()`
remain untouched. Because `TrajectoryClassifier` subclasses `ColourClassifier`
without overriding `fit`/`probability`, it picks up the blend automatically —
this task's tests confirm that directly.

**Files:**
- Modify: `cmsne/classifier.py:72-125` (`__init__`, `fit`), `:136-157`
  (`probability`) — exact line numbers from the pre-Task-1 file; after Tasks 1
  and 2 insert new content above the class, these method line numbers will
  have shifted down by the inserted lines, but the methods themselves are
  unchanged in content until this task.
- Test: `tests/test_cmsne.py` (same section, extended)

**Acceptance Criteria:**
- [ ] `ColourClassifier.__init__` accepts `ess_threshold`, `log_width`,
      `kde_max_events` constructor parameters with defaults
      `DEFAULT_ESS_THRESHOLD`, `DEFAULT_LOG_WIDTH`, `DEFAULT_KDE_MAX_EVENTS`.
- [ ] `fit()` stores the (balanced) signal/background training feature
      matrices and weights, pooled-data bandwidths, and resets a per-pattern
      cache — using the *balanced* `ws` (already rescaled to match `wb`'s
      total earlier in `fit()`), not the raw `signal_weights` argument, so the
      KDE side shares the same balanced-prior convention as the tree.
- [ ] `probability()` groups query events by finite-column pattern, filters
      the stored training data to that pattern once per distinct pattern per
      call (cached on the instance, reused across calls, reset by `fit()`),
      computes `p_kde` and the blend weight per event, and returns
      `alpha*p_tree + (1-alpha)*p_kde`.
- [ ] An event with no finite feature columns falls back to `p_tree`
      unchanged (no crash, no NaN).
- [ ] A dense synthetic region (plenty of both classes nearby) gives blended
      `probability()` within `1e-3` of `score()`-derived `p_tree` at the same
      points.
- [ ] A synthetic sparse-corner region, mimicking the real bug (one class has
      a few points in a corner the other class never reaches), gives blended
      `probability()` materially higher (or lower, for the symmetric case)
      than `p_tree` alone at that corner.
- [ ] A batch containing two distinct finite-column patterns in the same
      `probability()` call does not crash and produces finite, sane
      probabilities for both groups.
- [ ] A `TrajectoryClassifier` instance exhibits the same sparse-corner blend
      behaviour as `ColourClassifier`, with no `TrajectoryClassifier`-specific
      code added — confirming the fix propagates via inheritance.
- [ ] `score()`, `rank()`, `predict()`, `roc_curve()`, `recovery_rate()`
      outputs are bit-for-bit unchanged (they don't call the new blend code at
      all).
- [ ] `python tests/test_cmsne.py` reports no new failures.

**Verify:** `python tests/test_cmsne.py` → `threshold pins contamination near target` is the only failure.

**Steps:**

- [ ] **Step 1: Add module-level constants**

In `cmsne/classifier.py`, near `COLOUR_KEYS`/`MAGNITUDE_FEATURES` (around the
current lines 25-36), add:

```python
# Local density-ratio blend defaults (see docs/superpowers/specs/
# 2026-10-25-local-density-posterior-blend-design.md). These are provisional
# starting points, not empirically final -- real tuning against production
# data is a manual-validation step, not an automated test.
DEFAULT_ESS_THRESHOLD = 10.0
DEFAULT_LOG_WIDTH = 1.0
DEFAULT_KDE_MAX_EVENTS = 4000
```

- [ ] **Step 2: Add the pattern-grouping and subsampling helpers**

Insert into `cmsne/classifier.py`, directly after `_blend_weight` (from Task 2):

```python
def _finite_pattern(x):
    """Boolean tuple: which entries of feature vector ``x`` are finite."""
    return tuple(bool(v) for v in np.isfinite(np.asarray(x, dtype=float)))


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

- [ ] **Step 3: Update `__init__`**

In `cmsne/classifier.py`, the current `__init__` (lines 72-89) is:

```python
    def __init__(self, keys=COLOUR_KEYS, include_peakmag=True,
                 magnitude_feature=DEFAULT_MAGNITUDE_FEATURE, target_fpr=0.10,
                 calibrate=True, **hgb):
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
```

Change the signature and add three lines at the end:

```python
    def __init__(self, keys=COLOUR_KEYS, include_peakmag=True,
                 magnitude_feature=DEFAULT_MAGNITUDE_FEATURE, target_fpr=0.10,
                 calibrate=True, ess_threshold=DEFAULT_ESS_THRESHOLD,
                 log_width=DEFAULT_LOG_WIDTH, kde_max_events=DEFAULT_KDE_MAX_EVENTS,
                 **hgb):
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
```

- [ ] **Step 4: Update `fit()`**

The current `fit()` ends with:

```python
        sb = self.clf.predict_proba(Xb)[:, 1]
        self.threshold = float(weighted_quantile(sb, wb, q=1.0 - self.target_fpr))
        return self
```

Change to:

```python
        sb = self.clf.predict_proba(Xb)[:, 1]
        self.threshold = float(weighted_quantile(sb, wb, q=1.0 - self.target_fpr))
        self._store_kde_training_data(Xs, ws, Xb, wb)
        return self

    def _store_kde_training_data(self, Xs, ws, Xb, wb):
        """Retain (optionally subsampled) training events/weights + bandwidths
        for the local density-ratio blend in :meth:`probability`, and reset
        the per-pattern cache. ``ws``/``wb`` must already be the *balanced*
        weights computed earlier in :meth:`fit`, matching ``p_tree``'s own
        balanced-training-prior convention.
        """
        self._kde_Xs, self._kde_ws = _weighted_subsample(Xs, ws, self.kde_max_events)
        self._kde_Xb, self._kde_wb = _weighted_subsample(Xb, wb, self.kde_max_events)
        pooled_X = np.vstack([self._kde_Xs, self._kde_Xb])
        pooled_w = np.r_[self._kde_ws, self._kde_wb]
        self._kde_bandwidths = _scott_bandwidths(pooled_X, pooled_w)
        self._kde_pattern_cache = {}
```

- [ ] **Step 5: Update `probability()`**

The current `probability()` (lines 136-157) is:

```python
    def probability(self, events, prior=None):
        """Posterior probability P(lensed SN Ia) for each event.
        ...
        """
        p = np.clip(self.score(events), 1e-12, 1 - 1e-12)
        if prior is None:
            return p
        p0 = self.train_prior
        lr = (p / (1 - p)) * ((1 - p0) / p0)
        odds = lr * (prior / (1 - prior))
        return odds / (1 + odds)
```

Change to:

```python
    def probability(self, events, prior=None):
        """Posterior probability P(lensed SN Ia) for each event.
        ...
        """
        p_tree = np.clip(self.score(events), 1e-12, 1 - 1e-12)
        p = self._blend_with_local_density(events, p_tree)
        if prior is None:
            return p
        p0 = self.train_prior
        lr = (p / (1 - p)) * ((1 - p0) / p0)
        odds = lr * (prior / (1 - prior))
        return odds / (1 + odds)

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
            Xs_p, ws_p, Xb_p, wb_p = self._kde_subset_for_pattern(pattern, dims)
            p_kde, ess_sig, ess_bg = _local_density_ratio(
                x, Xs_p, ws_p, Xb_p, wb_p, self._kde_bandwidths, dims)
            if p_kde is None:
                continue
            alpha = _blend_weight(ess_sig, ess_bg, self.ess_threshold, self.log_width)
            out[i] = alpha * p_tree[i] + (1 - alpha) * p_kde
        return out

    def _kde_subset_for_pattern(self, pattern, dims):
        """Training rows finite over ``dims``, cached per ``pattern`` on this
        instance (reset by :meth:`fit`)."""
        if pattern not in self._kde_pattern_cache:
            finite_s = np.all(np.isfinite(self._kde_Xs[:, dims]), axis=1)
            finite_b = np.all(np.isfinite(self._kde_Xb[:, dims]), axis=1)
            self._kde_pattern_cache[pattern] = (
                self._kde_Xs[finite_s], self._kde_ws[finite_s],
                self._kde_Xb[finite_b], self._kde_wb[finite_b])
        return self._kde_pattern_cache[pattern]
```

(Keep the existing docstring body of `probability()` — the `...` above stands
for the unchanged docstring text already in the file; only the code body
changes.)

- [ ] **Step 6: Write the tests**

Append to the same test section in `tests/test_cmsne.py`:

```python
print('\n=== local density-ratio blend: wired into ColourClassifier ===')
from cmsne.classifier import ColourClassifier
from cmsne.trajectory import TrajectoryClassifier

# Dense bulk: plenty of both classes everywhere in a shared box -> alpha
# should be close to 1, blended probability() close to tree-only score().
_rng_dense = np.random.default_rng(3)
_n_dense = 1500
_dense_sig = [{'gr': float(_rng_dense.normal(1.0, 0.3)), 'peakmag': float(_rng_dense.normal(22.0, 1.0))}
             for _ in range(_n_dense)]
_dense_bg = [{'gr': float(_rng_dense.normal(1.2, 0.3)), 'peakmag': float(_rng_dense.normal(22.0, 1.0))}
            for _ in range(_n_dense)]
_dense_sw = np.full(_n_dense, 1e-5)
_dense_bw = np.full(_n_dense, 1e-5)
_clf_dense = ColourClassifier(keys=['gr'], include_peakmag=True)
_clf_dense.fit(_dense_sig, _dense_bg, _dense_sw, _dense_bw)
_dense_pts = [{'gr': 1.0, 'peakmag': 22.0}, {'gr': 1.1, 'peakmag': 21.7}]
_dense_blended = _clf_dense.probability(_dense_pts)
_dense_tree_only = np.clip(_clf_dense.score(_dense_pts), 1e-12, 1 - 1e-12)
check('dense region: blended probability() matches tree-only score() closely',
      np.allclose(_dense_blended, _dense_tree_only, atol=1e-3),
      (_dense_blended, _dense_tree_only))

# Sparse corner: background NEVER appears past gr=3.0, signal has a few rare
# points out there -- mimics the real bright+red bug at small scale.
_rng_sparse = np.random.default_rng(4)
_n_sparse = 1500
_sparse_sig_bulk = [{'gr': float(_rng_sparse.normal(1.0, 0.3)), 'peakmag': float(_rng_sparse.normal(22.0, 1.0))}
                    for _ in range(_n_sparse)]
_sparse_sig_corner = [{'gr': float(_rng_sparse.normal(4.0, 0.1)), 'peakmag': float(_rng_sparse.normal(16.0, 0.1))}
                      for _ in range(3)]
_sparse_sig = _sparse_sig_bulk + _sparse_sig_corner
_sparse_bg = [{'gr': float(_rng_sparse.normal(1.2, 0.3)), 'peakmag': float(_rng_sparse.normal(22.0, 1.0))}
             for _ in range(_n_sparse)]
_sparse_sw = np.full(len(_sparse_sig), 1e-5)
_sparse_bw = np.full(_n_sparse, 1e-5)
_clf_sparse = ColourClassifier(keys=['gr'], include_peakmag=True)
_clf_sparse.fit(_sparse_sig, _sparse_bg, _sparse_sw, _sparse_bw)
_corner_pt = [{'gr': 4.0, 'peakmag': 16.0}]
_corner_blended = _clf_sparse.probability(_corner_pt)[0]
_corner_tree_only = np.clip(_clf_sparse.score(_corner_pt), 1e-12, 1 - 1e-12)[0]
check('sparse corner: blended probability() is materially higher than tree-only score()',
      _corner_blended > _corner_tree_only + 0.05,
      (_corner_blended, _corner_tree_only))
check('sparse corner: blended probability() is clearly above the balanced-prior midpoint',
      _corner_blended > 0.7, _corner_blended)

# No finite feature columns at all -> falls back to p_tree unchanged.
_nan_pt = [{'gr': np.nan, 'peakmag': np.nan}]
_nan_blended = _clf_dense.probability(_nan_pt)
_nan_tree_only = np.clip(_clf_dense.score(_nan_pt), 1e-12, 1 - 1e-12)
check('event with no finite columns falls back to tree-only probability unchanged',
      np.allclose(_nan_blended, _nan_tree_only))

# Two distinct finite-column patterns in the same call.
_clf_multi = ColourClassifier(keys=['gr', 'ri'], include_peakmag=True)
_multi_sig = [{'gr': float(_rng_dense.normal(1.0, 0.3)), 'ri': float(_rng_dense.normal(0.5, 0.2)),
              'peakmag': float(_rng_dense.normal(22.0, 1.0))} for _ in range(500)]
_multi_bg = [{'gr': float(_rng_dense.normal(1.2, 0.3)), 'ri': float(_rng_dense.normal(0.6, 0.2)),
             'peakmag': float(_rng_dense.normal(22.0, 1.0))} for _ in range(500)]
_clf_multi.fit(_multi_sig, _multi_bg, np.full(500, 1e-5), np.full(500, 1e-5))
_mixed_batch = [{'gr': 1.0, 'ri': 0.5, 'peakmag': 22.0}, {'gr': 1.0, 'peakmag': 22.0}]
_mixed_result = _clf_multi.probability(_mixed_batch)
check('a batch with two distinct finite-column patterns does not crash and is finite',
      len(_mixed_result) == 2 and np.all(np.isfinite(_mixed_result)), _mixed_result)

# TrajectoryClassifier inherits the blend via inheritance, no extra code.
from cmsne.trajectory import TRAJECTORY_COLOUR_KEYS, TRAJECTORY_MAGNITUDE_KEYS, _step_suffix, TRAJECTORY_GRID
_tc_suffix0 = _step_suffix(TRAJECTORY_GRID[0])
_tc_key = TRAJECTORY_COLOUR_KEYS[0]
_tc_mag_key = TRAJECTORY_MAGNITUDE_KEYS[0]
_tc_sig_bulk = [{_tc_key: float(_rng_sparse.normal(1.0, 0.3)), _tc_mag_key: float(_rng_sparse.normal(22.0, 1.0))}
               for _ in range(_n_sparse)]
_tc_sig_corner = [{_tc_key: float(_rng_sparse.normal(4.0, 0.1)), _tc_mag_key: float(_rng_sparse.normal(16.0, 0.1))}
                  for _ in range(3)]
_tc_sig = _tc_sig_bulk + _tc_sig_corner
_tc_bg = [{_tc_key: float(_rng_sparse.normal(1.2, 0.3)), _tc_mag_key: float(_rng_sparse.normal(22.0, 1.0))}
         for _ in range(_n_sparse)]
_tc = TrajectoryClassifier()
_tc.fit(_tc_sig, _tc_bg, np.full(len(_tc_sig), 1e-5), np.full(_n_sparse, 1e-5))
_tc_corner_pt = [{_tc_key: 4.0, _tc_mag_key: 16.0}]
_tc_corner_blended = _tc.probability(_tc_corner_pt)[0]
_tc_corner_tree_only = np.clip(_tc.score(_tc_corner_pt), 1e-12, 1 - 1e-12)[0]
check('TrajectoryClassifier inherits the sparse-corner blend with no extra code',
      _tc_corner_blended > _tc_corner_tree_only + 0.05,
      (_tc_corner_blended, _tc_corner_tree_only))
```

- [ ] **Step 7: Run the full test suite**

Run: `python tests/test_cmsne.py`
Expected: all new checks pass; only the pre-existing `threshold pins
contamination near target` failure remains.

- [ ] **Step 8: Commit**

```bash
git add cmsne/classifier.py tests/test_cmsne.py
git commit -m "feat: wire local density-ratio blend into ColourClassifier.probability"
```

```json:metadata
{"files": ["cmsne/classifier.py", "tests/test_cmsne.py"], "verifyCommand": "python tests/test_cmsne.py", "acceptanceCriteria": ["__init__ gains ess_threshold/log_width/kde_max_events params with the documented defaults", "fit() stores the BALANCED ws/wb (not raw signal_weights) plus bandwidths, resets pattern cache", "probability() groups by finite-column pattern, caches filtered training subsets, blends per event, falls back to p_tree on degenerate cases", "dense synthetic region: blended probability matches tree-only score within 1e-3", "sparse synthetic corner mimicking the real bug: blended probability is materially higher than tree-only score and clearly above 0.7", "two distinct finite-column patterns in one call do not crash", "TrajectoryClassifier exhibits the same blend with zero TrajectoryClassifier-specific code", "score/rank/predict/roc_curve/recovery_rate unaffected", "python tests/test_cmsne.py has no new failures beyond the known pre-existing one"], "modelTier": "standard"}
```

---

## Task 4: Backward-compatibility regression and `kde_max_events` validation

**Goal:** Prove, on a production-shaped synthetic population (not just the
small fixtures from Task 3), that the blend leaves `probability()` unchanged
in a dense, well-sampled region, and that `kde_max_events` caps the stored KDE
training set without crashing or silently distorting the rate-weighted total
mass used in the density ratio.

**Files:**
- Test: `tests/test_cmsne.py` (same section, extended)

**Acceptance Criteria:**
- [ ] A production-shaped dense population (wide dynamic range of rate
      weights, no sparse corners) gives blended `probability()` within `1e-3`
      of tree-only `score()` at representative interior points.
- [ ] `kde_max_events` caps the number of rows stored for the KDE side to at
      most the cap, for a class with more positive-weight events than the cap.
- [ ] After capping, the stored KDE weights' total sum is within 1% of the
      original (pre-cap) positive-weight total — subsampling must not shrink
      one class's total rate mass relative to the other's.
- [ ] `python tests/test_cmsne.py` reports no new failures.

**Verify:** `python tests/test_cmsne.py` → `threshold pins contamination near target` is the only failure.

**Steps:**

- [ ] **Step 1: Write the backward-compatibility regression test**

Append to the same test section in `tests/test_cmsne.py`:

```python
print('\n=== local density-ratio blend: backward compatibility and kde_max_events ===')

# Production-shaped: wide dynamic range of rate weights (spanning several
# orders of magnitude, like real rate_weight() output), no sparse corners --
# the blend must leave probability() unchanged here.
_rng_bc = np.random.default_rng(5)
_n_bc = 2000
_bc_sig = [{'gr': float(_rng_bc.normal(1.0, 0.3)), 'peakmag': float(_rng_bc.normal(22.0, 1.0))}
          for _ in range(_n_bc)]
_bc_bg = [{'gr': float(_rng_bc.normal(1.2, 0.3)), 'peakmag': float(_rng_bc.normal(22.0, 1.0))}
         for _ in range(_n_bc)]
# Realistic order-of-magnitude spread, same distribution for both classes so
# neither is artificially starved -- this is specifically testing "dense and
# well-sampled", not sparsity.
_bc_sw = 10.0 ** _rng_bc.uniform(-7, -5, _n_bc)
_bc_bw = 10.0 ** _rng_bc.uniform(-7, -5, _n_bc)
_clf_bc = ColourClassifier(keys=['gr'], include_peakmag=True)
_clf_bc.fit(_bc_sig, _bc_bg, _bc_sw, _bc_bw)
_bc_pts = [{'gr': 1.0, 'peakmag': 22.0}, {'gr': 1.1, 'peakmag': 21.5}, {'gr': 0.9, 'peakmag': 22.5}]
_bc_blended = _clf_bc.probability(_bc_pts)
_bc_tree_only = np.clip(_clf_bc.score(_bc_pts), 1e-12, 1 - 1e-12)
check('production-shaped dense region: blended probability() unchanged from tree-only score()',
      np.allclose(_bc_blended, _bc_tree_only, atol=1e-3), (_bc_blended, _bc_tree_only))

# kde_max_events caps the stored KDE training set.
_rng_cap = np.random.default_rng(6)
_n_cap = 500
_cap_events = [{'gr': float(_rng_cap.normal(0, 1))} for _ in range(_n_cap)]
_cap_weights = np.abs(_rng_cap.normal(1e-6, 1e-7, _n_cap))
_clf_cap = ColourClassifier(keys=['gr'], include_peakmag=False, kde_max_events=50)
_clf_cap.fit(_cap_events, _cap_events, _cap_weights, _cap_weights)
check('kde_max_events caps the stored KDE signal set to at most the cap',
      _clf_cap._kde_Xs.shape[0] <= 50, _clf_cap._kde_Xs.shape[0])
check('kde_max_events preserves total rate mass after subsampling (within 1%)',
      abs(_clf_cap._kde_ws.sum() - _cap_weights.sum()) / _cap_weights.sum() < 0.01,
      (_clf_cap._kde_ws.sum(), _cap_weights.sum()))

# Below the cap: no subsampling, nothing changes.
_clf_nocap = ColourClassifier(keys=['gr'], include_peakmag=False, kde_max_events=10000)
_clf_nocap.fit(_cap_events, _cap_events, _cap_weights, _cap_weights)
check('kde_max_events above the actual event count leaves the stored set untouched',
      _clf_nocap._kde_Xs.shape[0] == np.sum(_cap_weights > 0), _clf_nocap._kde_Xs.shape[0])
```

- [ ] **Step 2: Run the full test suite**

Run: `python tests/test_cmsne.py`
Expected: all new checks pass; only the pre-existing `threshold pins
contamination near target` failure remains.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cmsne.py
git commit -m "test: backward-compat regression and kde_max_events for density-ratio blend"
```

```json:metadata
{"files": ["tests/test_cmsne.py"], "verifyCommand": "python tests/test_cmsne.py", "acceptanceCriteria": ["production-shaped dense population: blended probability matches tree-only score within 1e-3", "kde_max_events caps the stored KDE set to at most the cap for an over-cap population", "subsampling preserves total rate mass within 1%", "a population below the cap is left untouched", "python tests/test_cmsne.py has no new failures beyond the known pre-existing one"], "modelTier": "standard"}
```

---

## Manual Validation (after Task 4 lands — not automatable, run by the project owner)

These are the spec's deferred empirical-tuning and real-data checks. They
need real BlueBEAR access and the production data already generated this
session, so they are **not** implementation tasks a coding agent can run
unsupervised.

1. **Tune `ess_threshold`/`log_width` against real data.** Using
   `results/mc_58818280` (the boosted production run already on BlueBEAR),
   refit `ColourClassifier` (signal=`lensed_Ia`, background = the other 9
   classes) and check `alpha(x)` in the already-trusted bulk region lands
   close to 1, and in the bright+red corner (`gr≥2.0, peakmag≤16`) lands close
   to 0. Adjust `DEFAULT_ESS_THRESHOLD`/`DEFAULT_LOG_WIDTH` if it doesn't, and
   record the chosen values and rationale in a follow-up commit.
2. **Re-run the exact bright+red corner check from this investigation**
   (`gr≥2.0, peakmag≤16`, explicit grid range `(-1.5, 3.5) × (13, 25)` to
   reach it) against the new `ColourClassifier`. **Success criterion:
   P(lensed) exceeds 0.9** in that corner (up from the ~0.4576 plateau
   measured before this fix).
3. **Re-check backward compatibility against real production data**, not just
   the synthetic fixtures in Task 4 — confirm the blend leaves the bulk
   posterior surface unchanged outside `diagnose_sparse_cells`' flagged cells.
4. **Re-generate `scripts/plot_evolving_boundaries.py`'s boundary figures** on
   production data and visually confirm the bulk decision boundary is
   unchanged, while previously-flat extreme corners now show `P → 1`.
5. **Timing.** Record actual wall-clock for a `posterior_surface` call at
   production scale (200×200 grid, ~100-150k background events after
   `kde_max_events` subsampling). If it's a multi-x slowdown over the
   tree-only fit + grid evaluation, consider vectorizing
   `_blend_with_local_density` across query points too (chunked, to bound
   memory) rather than the current per-query-point Python loop — this plan
   deliberately deferred that optimization to be driven by a real
   measurement, not a guess.
