# Adaptive KDE Bandwidth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the real-data failure found in PR #35's local density-ratio
blend — `P(lensed)` collapsing to ~0.0002 instead of >0.9 at the bright+red
corner — by widening each class's KDE kernel, independently and per query
point, until its local effective sample size is large enough for the
resulting density ratio to be statistically meaningful, instead of a
single-point coin flip.

**Architecture:** One new helper, `_widened_kernel_values`, performs
isotropic geometric bandwidth widening with two independent stopping
conditions (reaching a minimum ESS, or plateauing at the class's own overall
ESS ceiling). `_local_density_ratio` uses it only to compute `p_kde`'s `S`/`B`
sums; the blend-weight decision (`local_ess_sig`/`local_ess_bg`, consumed by
`_blend_weight`) stays on the existing fixed-bandwidth ESS, unchanged.

**Tech Stack:** numpy, `cmsne.importance_sampling.effective_sample_size`
(already used, no new dependency).

**Spec:** `docs/superpowers/specs/2026-10-26-adaptive-kde-bandwidth-design.md`
(builds on `docs/superpowers/specs/2026-10-25-local-density-posterior-blend-design.md`)

## Global Constraints

- All new code lives in `cmsne/classifier.py` — no new module.
- `local_ess_sig`/`local_ess_bg` (the values `_blend_weight` consumes) MUST
  remain computed at the fixed (unwidened) Scott's-rule bandwidth —
  byte-for-byte the same as before this plan. Widening affects only the
  `p_kde` value itself.
- The existing degenerate guard (`S+B < 1e-300` → `p_kde=None`, pure tree
  fallback) MUST be evaluated against the FIXED-bandwidth kernel sums,
  BEFORE any widening is attempted — not against the widened sums. This
  preserves the exact existing contract for genuinely out-of-range query
  points (e.g. the existing "both classes far from x" test), where widening
  enough would eventually "rescue" the sums from underflow and return a
  near-arbitrary global-prior-like ratio instead of correctly signalling "no
  local information, use the tree." Only points that already have *some*
  non-negligible fixed-bandwidth presence from at least one class proceed to
  widening.
- `ColourClassifier(..., kde_min_ess=DEFAULT_KDE_MIN_ESS)` — new constructor
  parameter, same pattern as `ess_threshold`/`log_width`/`kde_max_events`.
  `DEFAULT_KDE_MIN_ESS = 10.0` is provisional (documented as such in the
  code), re-tuned against real BlueBEAR data in Manual Validation, not
  guessed as final here.
- Tests in `tests/test_cmsne.py`, house style (`check()` helper, not
  pytest). After every task, `python tests/test_cmsne.py` must report no
  failures beyond the one known pre-existing, unrelated failure
  (`threshold pins contamination near target`) — and note that this failure
  has been observed to be itself environment-sensitive (see memory
  `test-env-cmsne-env-not-bare-python`); verify in `cmsne-env` specifically
  before merging, not just whatever `python` resolves to locally.

**User decisions (already made):**
- Approach: adaptive ("balloon") per-class, per-query-point bandwidth
  widening on the KDE side only — user chose this explicitly after seeing
  the real-data diagnostic and three alternatives (targeted deep-boosting,
  an explicit physical-feasibility floor).
- Spec approved with no changes.

---

## Task 1: `_widened_kernel_values` helper

**Goal:** Add a pure, standalone function to `cmsne/classifier.py` that
isotropically widens a kernel's bandwidth until a class's local kernel-sum
ESS reaches a target, or plateaus at that class's own overall ESS ceiling —
with unit tests covering the no-op (already-dense), successful-widening, and
ceiling-plateau cases.

**Files:**
- Modify: `cmsne/classifier.py:42-44` (add `DEFAULT_KDE_MIN_ESS` constant
  after the existing three); insert the new function after
  `_kernel_weighted_values` (currently ending at line 124, directly before
  `_local_density_ratio` at line 127)
- Test: `tests/test_cmsne.py` (new section appended to the existing
  "local density-ratio blend" area, before the final summary block)

**Acceptance Criteria:**
- [ ] `DEFAULT_KDE_MIN_ESS = 10.0` added near `DEFAULT_ESS_THRESHOLD`/
      `DEFAULT_LOG_WIDTH`/`DEFAULT_KDE_MAX_EVENTS`.
- [ ] `_widened_kernel_values(x, X, weights, bandwidths, dims, min_ess, growth=2.0, max_iter=20)`
      returns `(g, ess)`: a no-op (identical to a direct
      `_kernel_weighted_values` call) when `min_ess` is already satisfied at
      `scale=1.0`; otherwise geometrically widens (`scale *= growth`) up to
      `max_iter` times, stopping as soon as `ess >= min_ess` OR `ess` stops
      improving (plateau) OR `ess` reaches the class's own overall ESS
      ceiling (`effective_sample_size(weights)`).
- [ ] A synthetic sparse scenario (one near point dominating, several
      farther points that only contribute once the kernel widens) shows
      `_widened_kernel_values` raising ESS above the unwidened value and
      reaching the requested `min_ess`.
- [ ] A synthetic ceiling scenario (only 2 positive-weight points, both far
      from the query, `min_ess` set unreachably high) shows the function
      stopping at that class's ceiling rather than iterating uselessly to
      `max_iter`.
- [ ] `python tests/test_cmsne.py` reports no new failures beyond the known
      pre-existing one.

**Verify:** `python tests/test_cmsne.py` → `threshold pins contamination near target` is the only failure.

**Steps:**

- [ ] **Step 1: Add the constant**

In `cmsne/classifier.py`, directly after the existing:
```python
DEFAULT_ESS_THRESHOLD = 10.0
DEFAULT_LOG_WIDTH = 0.3
DEFAULT_KDE_MAX_EVENTS = 4000
```
add:
```python
DEFAULT_KDE_MIN_ESS = 10.0
```

- [ ] **Step 2: Write `_widened_kernel_values`**

Insert directly after `_kernel_weighted_values` (before `def _local_density_ratio`):

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

- [ ] **Step 3: Write the tests**

Append to `tests/test_cmsne.py`, before the final summary block:

```python
print('\n=== adaptive KDE bandwidth: _widened_kernel_values ===')
from cmsne.classifier import _widened_kernel_values, DEFAULT_KDE_MIN_ESS
from cmsne.importance_sampling import effective_sample_size as _ess_ref

# Dense case: ess already >= min_ess at scale=1 -> no-op.
_wk_h = np.array([1.0])
_wk_dims = np.array([0])
_wk_X_dense = np.array([[0.0], [0.1], [-0.1], [0.05], [-0.05],
                        [0.2], [-0.2], [0.15], [-0.15], [0.02]])
_wk_w_dense = np.full(10, 1e-5)
_g_unwidened = _kernel_weighted_values(np.array([0.0]), _wk_X_dense, _wk_w_dense, _wk_h, _wk_dims)
_g_widened, _ess_widened = _widened_kernel_values(
    np.array([0.0]), _wk_X_dense, _wk_w_dense, _wk_h, _wk_dims, min_ess=5.0)
check('_widened_kernel_values is a no-op when min_ess is already satisfied at scale=1',
      np.allclose(_g_widened, _g_unwidened), (_g_widened, _g_unwidened))

# Sparse case: one near point dominates (ESS~1 at scale=1); several farther
# points only start contributing once the kernel widens.
_wk_X_sparse = np.array([[2.0], [10.0], [10.0], [10.0]])
_wk_w_sparse = np.full(4, 1e-5)
_g_fixed = _kernel_weighted_values(np.array([0.0]), _wk_X_sparse, _wk_w_sparse, _wk_h, _wk_dims)
_ess_fixed = _ess_ref(_g_fixed)
_g_w, _ess_w = _widened_kernel_values(
    np.array([0.0]), _wk_X_sparse, _wk_w_sparse, _wk_h, _wk_dims, min_ess=2.0)
check('_widened_kernel_values raises local ESS above the unwidened (single-point-dominated) value',
      _ess_w > _ess_fixed and _ess_fixed < 1.5, (_ess_w, _ess_fixed))
check('_widened_kernel_values reaches the requested min_ess in this scenario',
      _ess_w >= 2.0 - 1e-6, _ess_w)

# Ceiling/plateau case: only 2 points have positive weight -- no amount of
# widening can push ESS above effective_sample_size(weights) itself.
_wk_X_ceiling = np.array([[50.0], [50.1]])
_wk_w_ceiling = np.array([1e-5, 1e-5])
_ceiling = _ess_ref(_wk_w_ceiling)
_g_c, _ess_c = _widened_kernel_values(
    np.array([0.0]), _wk_X_ceiling, _wk_w_ceiling, _wk_h, _wk_dims, min_ess=100.0, max_iter=20)
check('_widened_kernel_values stops at the class ceiling rather than looping uselessly to max_iter',
      np.isclose(_ess_c, _ceiling, atol=1e-3), (_ess_c, _ceiling))
```

If the sparse-case numeric margins (`min_ess=2.0`, the `< 1.5` fixed-ESS
check) don't hold exactly as specified when actually run, that means the
hand-derived fixture doesn't match the real floating-point computation
closely enough — adjust the fixture's distances/weights or the margins
while preserving the qualitative claims (`_ess_w > _ess_fixed`,
`_ess_fixed` close to 1, widening reaches a real improvement), and say so in
your report. Do not weaken a check to make it pass without preserving what
it's actually testing.

- [ ] **Step 4: Run the test suite**

Run: `python tests/test_cmsne.py`
Expected: every new check prints `ok`; the only failure anywhere in the
suite is the pre-existing `threshold pins contamination near target`.

- [ ] **Step 5: Commit**

```bash
git add cmsne/classifier.py tests/test_cmsne.py
git commit -m "feat: add adaptive balloon-bandwidth widening for the KDE side"
```

```json:metadata
{"files": ["cmsne/classifier.py", "tests/test_cmsne.py"], "verifyCommand": "python tests/test_cmsne.py", "acceptanceCriteria": ["DEFAULT_KDE_MIN_ESS constant added", "_widened_kernel_values is a no-op when min_ess already satisfied at scale=1", "widening raises ESS above the unwidened single-point-dominated value and reaches min_ess in a sparse scenario", "widening stops at the class's own ESS ceiling rather than looping to max_iter when min_ess is unreachable", "python tests/test_cmsne.py has no new failures beyond the known pre-existing one"], "modelTier": "mechanical"}
```

---

## Task 2: Wire adaptive widening into `_local_density_ratio`, reproduce and fix the real bug

**Goal:** Use `_widened_kernel_values` inside `_local_density_ratio` to
compute `p_kde`'s `S`/`B` sums, while keeping `local_ess_sig`/`local_ess_bg`
(the blend-weight inputs) on the fixed bandwidth and the degenerate guard
evaluated against the fixed-bandwidth sums before any widening is
attempted. Add `kde_min_ess` to `ColourClassifier.__init__` and thread it
through `_blend_with_local_density`'s call site. Reproduce the exact
real-world failure mode at small scale and confirm this fixes it.

**Files:**
- Modify: `cmsne/classifier.py:127-147` (`_local_density_ratio`),
  `:215-237` (`__init__`), `:323-346` (`_blend_with_local_density`) — line
  numbers are from the pre-Task-1 file; Task 1's insertion shifts
  `_local_density_ratio` and everything after it down by the inserted
  lines, but the three blocks' own content (before this task's edits) is
  otherwise as shown below.
- Test: `tests/test_cmsne.py` (same section, extended)

**Acceptance Criteria:**
- [ ] `_local_density_ratio` gains a `min_kde_ess=DEFAULT_KDE_MIN_ESS`
      parameter; computes `local_ess_sig`/`local_ess_bg` from the
      fixed-bandwidth kernel sums exactly as before (unchanged values);
      checks the degenerate guard (`S+B < 1e-300`) against those SAME
      fixed-bandwidth sums BEFORE calling `_widened_kernel_values`; only
      then computes the actual `S`/`B` used for `p_kde` via
      `_widened_kernel_values` for each class independently.
- [ ] `ColourClassifier.__init__` gains `kde_min_ess=DEFAULT_KDE_MIN_ESS`;
      `_blend_with_local_density`'s call to `_local_density_ratio` passes
      `self.kde_min_ess`.
- [ ] A synthetic scenario that reproduces the real bug's signature (one
      near-dominant point per class at the fixed bandwidth, i.e.
      `local_ess_sig`/`local_ess_bg` near 1, where the unwidened ratio
      favours the wrong class) shows the ratio flipping to favour the
      correct class once adaptive widening is applied.
- [ ] `local_ess_sig`/`local_ess_bg` returned by `_local_density_ratio` are
      IDENTICAL regardless of `min_kde_ess` (direct regression test, not
      inferred from reading the code).
- [ ] All existing `_local_density_ratio`/`_blend_weight`/
      `_blend_with_local_density` tests (from the original PR #35 plan)
      still pass unchanged — including the "both classes far from x"
      degenerate case, which must still return `p_kde=None` (confirming the
      guard-before-widening ordering is correct).
- [ ] `python tests/test_cmsne.py` reports no new failures.

**Verify:** `python tests/test_cmsne.py` → `threshold pins contamination near target` is the only failure.

**Steps:**

- [ ] **Step 1: Update `_local_density_ratio`**

The current function (after Task 1's insertion, this is the next function
down) is:

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
    # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
    from .importance_sampling import effective_sample_size

    gs = _kernel_weighted_values(x, Xs, ws, bandwidths, dims)
    gb = _kernel_weighted_values(x, Xb, wb, bandwidths, dims)
    S, B = float(np.sum(gs)), float(np.sum(gb))
    local_ess_sig = effective_sample_size(gs)
    local_ess_bg = effective_sample_size(gb)
    if S + B < 1e-300:
        return None, local_ess_sig, local_ess_bg
    return S / (S + B), local_ess_sig, local_ess_bg
```

Replace with:

```python
def _local_density_ratio(x, Xs, ws, Xb, wb, bandwidths, dims, min_kde_ess=DEFAULT_KDE_MIN_ESS):
    """Local weighted kernel density ratio and local ESS at point ``x``.

    :param Xs, ws: signal training feature matrix and rate weights.
    :param Xb, wb: background training feature matrix and rate weights.
    :param min_kde_ess: minimum effective sample size each class's kernel is
        adaptively widened to reach before computing the density ratio (see
        :func:`_widened_kernel_values`). This does NOT affect
        ``local_ess_sig``/``local_ess_bg`` below, which stay at the fixed
        bandwidth and drive the tree-vs-KDE blend-weight decision.
    :return: ``(p_kde, local_ess_sig, local_ess_bg)``. ``p_kde`` is ``None``
        when both classes' FIXED-bandwidth kernel-weighted mass underflows
        to (numerically) zero at ``x`` -- the degenerate case callers fall
        back to the tree-only posterior for. This guard is checked before
        any widening is attempted, so a point with no real local
        information from either class is never "rescued" into a
        near-arbitrary ratio by widening far enough to reach unrelated data.
    """
    # Local import: importance_sampling -> contours -> classifier would be a cycle at module load.
    from .importance_sampling import effective_sample_size

    gs_fixed = _kernel_weighted_values(x, Xs, ws, bandwidths, dims)
    gb_fixed = _kernel_weighted_values(x, Xb, wb, bandwidths, dims)
    local_ess_sig = effective_sample_size(gs_fixed)
    local_ess_bg = effective_sample_size(gb_fixed)
    if float(np.sum(gs_fixed)) + float(np.sum(gb_fixed)) < 1e-300:
        return None, local_ess_sig, local_ess_bg

    gs, _ = _widened_kernel_values(x, Xs, ws, bandwidths, dims, min_kde_ess)
    gb, _ = _widened_kernel_values(x, Xb, wb, bandwidths, dims, min_kde_ess)
    S, B = float(np.sum(gs)), float(np.sum(gb))
    if S + B < 1e-300:
        return None, local_ess_sig, local_ess_bg
    return S / (S + B), local_ess_sig, local_ess_bg
```

- [ ] **Step 2: Update `__init__`**

The current `__init__` ends with:

```python
        self.ess_threshold = ess_threshold
        self.log_width = log_width
        self.kde_max_events = kde_max_events
```

and its signature is:

```python
    def __init__(self, keys=COLOUR_KEYS, include_peakmag=True,
                 magnitude_feature=DEFAULT_MAGNITUDE_FEATURE, target_fpr=0.10,
                 calibrate=True, ess_threshold=DEFAULT_ESS_THRESHOLD,
                 log_width=DEFAULT_LOG_WIDTH, kde_max_events=DEFAULT_KDE_MAX_EVENTS,
                 **hgb):
```

Change the signature to add `kde_min_ess`:

```python
    def __init__(self, keys=COLOUR_KEYS, include_peakmag=True,
                 magnitude_feature=DEFAULT_MAGNITUDE_FEATURE, target_fpr=0.10,
                 calibrate=True, ess_threshold=DEFAULT_ESS_THRESHOLD,
                 log_width=DEFAULT_LOG_WIDTH, kde_max_events=DEFAULT_KDE_MAX_EVENTS,
                 kde_min_ess=DEFAULT_KDE_MIN_ESS, **hgb):
```

And the body's ending to:

```python
        self.ess_threshold = ess_threshold
        self.log_width = log_width
        self.kde_max_events = kde_max_events
        self.kde_min_ess = kde_min_ess
```

- [ ] **Step 3: Update `_blend_with_local_density`'s call site**

The current call is:

```python
            Xs_p, ws_p, Xb_p, wb_p = self._kde_subset_for_pattern(pattern, dims)
            p_kde, ess_sig, ess_bg = _local_density_ratio(
                x, Xs_p, ws_p, Xb_p, wb_p, self._kde_bandwidths, dims)
```

Change to:

```python
            Xs_p, ws_p, Xb_p, wb_p = self._kde_subset_for_pattern(pattern, dims)
            p_kde, ess_sig, ess_bg = _local_density_ratio(
                x, Xs_p, ws_p, Xb_p, wb_p, self._kde_bandwidths, dims, self.kde_min_ess)
```

- [ ] **Step 4: Write the bug-reproduction and regression tests**

Append to the same test section in `tests/test_cmsne.py`:

```python
print('\n=== adaptive KDE bandwidth: reproduces and fixes the real corner bug ===')
from cmsne.classifier import _local_density_ratio

# Mimics the real bright+red corner failure: each class has one dominant
# near point at the fixed bandwidth (local ESS~1 for both), but background's
# near point carries more weight at a slightly closer distance -- so the
# UNWIDENED ratio wrongly favours background. Each class also has a larger,
# farther-out population that should correctly swing the ratio toward
# signal once the kernel widens enough to actually see it.
_bug_h = np.array([1.0])
_bug_dims = np.array([0])
_bug_Xs = np.array([[3.0]] + [[6.0]] * 30)
_bug_ws = np.full(31, 1e-5)
_bug_Xb = np.array([[2.9]] + [[30.0]] * 30)
_bug_wb = np.array([5e-5] + [1e-5] * 30)

_p_unwidened, _, _ = _local_density_ratio(
    np.array([0.0]), _bug_Xs, _bug_ws, _bug_Xb, _bug_wb, _bug_h, _bug_dims, min_kde_ess=1.0)
check('without meaningful widening, the corner-mimicking scenario wrongly favours background',
      _p_unwidened is not None and _p_unwidened < 0.3, _p_unwidened)

_p_widened, _ess_sig_check, _ess_bg_check = _local_density_ratio(
    np.array([0.0]), _bug_Xs, _bug_ws, _bug_Xb, _bug_wb, _bug_h, _bug_dims, min_kde_ess=10.0)
check('with adaptive widening (min_kde_ess=10), the same scenario correctly favours signal',
      _p_widened is not None and _p_widened > 0.5, _p_widened)

_fixed_ess_sig_expected = effective_sample_size(_kernel_weighted_values(
    np.array([0.0]), _bug_Xs, _bug_ws, _bug_h, _bug_dims))
_fixed_ess_bg_expected = effective_sample_size(_kernel_weighted_values(
    np.array([0.0]), _bug_Xb, _bug_wb, _bug_h, _bug_dims))
check('local_ess_sig/local_ess_bg (the blend-weight inputs) are identical regardless of min_kde_ess',
      np.isclose(_ess_sig_check, _fixed_ess_sig_expected) and
      np.isclose(_ess_bg_check, _fixed_ess_bg_expected),
      (_ess_sig_check, _fixed_ess_sig_expected, _ess_bg_check, _fixed_ess_bg_expected))
```

If the exact margins (`< 0.3`, `> 0.5`) don't hold when actually run, that
means the hand-derived fixture's distances/weights need adjusting to
reproduce the qualitative effect more clearly — adjust them and say so in
your report, while preserving the core claim being tested: the unwidened
ratio favours the wrong class, and widening fixes it. Do not weaken a check
to make it pass without preserving what it's actually testing.

Also run the full existing test suite (unchanged calls to
`_local_density_ratio` from the original PR #35 work, e.g. the "both classes
far from x" degenerate case and the dense/sparse `ColourClassifier`-level
tests) to confirm nothing there is affected — these calls omit `min_kde_ess`
and should behave exactly as before, since it now has a default value and
the degenerate guard still runs against the fixed-bandwidth sums first.

- [ ] **Step 5: Run the full test suite**

Run: `python tests/test_cmsne.py`
Expected: all new checks pass, all pre-existing checks from the PR #35 work
still pass unchanged; only the known pre-existing `threshold pins
contamination near target` failure remains.

- [ ] **Step 6: Commit**

```bash
git add cmsne/classifier.py tests/test_cmsne.py
git commit -m "fix: adaptively widen KDE bandwidth to fix real-data corner collapse"
```

```json:metadata
{"files": ["cmsne/classifier.py", "tests/test_cmsne.py"], "verifyCommand": "python tests/test_cmsne.py", "acceptanceCriteria": ["_local_density_ratio computes local_ess_sig/local_ess_bg from the fixed bandwidth, unchanged by min_kde_ess", "degenerate guard checked against fixed-bandwidth sums before any widening", "ColourClassifier gains kde_min_ess constructor param threaded through to _local_density_ratio's call site", "a synthetic reproduction of the real bug (near-dominant single point per class, unwidened ratio favours the wrong class) is fixed by widening", "local_ess_sig/local_ess_bg are identical regardless of min_kde_ess, verified directly not inferred", "all pre-existing PR #35 tests, including the degenerate both-far-from-x case, still pass unchanged", "python tests/test_cmsne.py has no new failures beyond the known pre-existing one"], "modelTier": "standard"}
```

---

## Manual Validation (after Task 2 lands — not automatable, run by the project owner)

Re-run the exact `manual_validation.py`/`diagnose_bandwidth.py` scripts
already used for PR #35's manual validation (against
`results/mc_58818280` on BlueBEAR), now against the fixed classifier:

1. **Re-run the bright+red corner check.** Success criterion unchanged:
   **P(lensed) exceeds 0.9** in the corner. This is the check that
   previously returned ~0.0002 — confirm it now gives the right answer.
2. **Re-run the `ess_threshold`/`log_width`/`kde_min_ess` tuning sweep**
   against real data, now including `kde_min_ess` as a third axis (in
   addition to the two the first manual-validation pass already swept).
3. **Re-check backward compatibility** in the bulk — should be unaffected
   per this plan's Global Constraints (widening is a no-op in the dense
   bulk), but verify directly against real data, not just the synthetic
   fixtures above.
4. **Regenerate the boundary figure** (same scoped-down direct heatmap
   approach as before, not the full `plot_evolving_boundaries.py` dual-
   schema pipeline) and visually confirm the bulk is unchanged while the
   corner now shows `P → 1` instead of `P → 0`.
5. **Re-measure timing**, now including widening's added iteration cost.
   If it's a problematic slowdown, the per-query-point Python loop in
   `_blend_with_local_density` (already flagged as a deferred optimization
   in the original PR #35 plan) is the first place to look.
