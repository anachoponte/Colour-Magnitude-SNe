# Sequential CMD Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the time-domain companion to the existing static `ColourClassifier` described in the design spec: N-point colour-magnitude trajectories, a classifier that re-scores a growing feature vector as new epochs arrive, an evolving posterior-surface diagnostic, and a benchmarking harness against both the existing classifier and a deliberately-wrong naive baseline.

**Architecture:** Extend `cmsne.multicolour`'s two-point (trigger, trigger+15d) generator to a fixed 5-point Δt-since-trigger grid via a new `cmsne.trajectory` module; train a NaN-native gradient-boosted classifier on that full feature vector by subclassing the existing `ColourClassifier`; add a diagnostic posterior-surface module (`cmsne.contours`) and a comparison module (`cmsne.comparison`) that benchmarks the static classifier, the new trajectory classifier, and a naive odds-multiplication baseline on the same data.

**Tech Stack:** Python, numpy, scikit-learn (`HistGradientBoostingClassifier` + `CalibratedClassifierCV`, already a dependency), sncosmo/OpSim (existing `cmsne` simulation stack). Tests follow the repo's existing flat-script convention (`tests/test_cmsne.py`, `check(name, cond, extra)`), not pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-sequential-cmd-classifier-design.md`

## Global Constraints

- The time axis for every new feature is **Δt = days since the candidate's first LSST detection (trigger epoch)** — never days since explosion. No task introduces an explosion-phase-anchored feature.
- The canonical Δt grid is `TRAJECTORY_GRID = (0.0, 3.0, 14.0, 30.0, 60.0)` days, defined once in `cmsne/trajectory.py` and reused by every other module (`cmsne/contours.py`, `cmsne/comparison.py`, `scripts/run_trajectory.py`) — no task hardcodes its own grid or a different set of checkpoints.
- The colour-pairing tolerance is `window=5.0` days, reusing `cmsne.multicolour.band_photometry`'s existing default unchanged — no task introduces a different window value.
- `cmsne/classifier.py` (`ColourClassifier`) is **not modified** by this plan. New classifiers subclass it or construct-then-override its `extra`/`magnitude_feature` attributes (see Task 2 and Task 4), so the existing, already-tested static classifier and its 146 existing checks are completely unaffected.
- All new tests are appended to `tests/test_cmsne.py` in the existing `print('\n=== ... ===')` + `check(name, cond, extra)` style — no new test file, no pytest.
- No non-SN contaminant classes and no non-Ia lensed signal classes are added anywhere in this plan (per the spec's explicit scope decision).

**User decisions (already made):**
- Background/contaminant scope stays SN-only (Ia + core-collapse); non-SN contaminants (AGN, TDE, nova, variable stars) are out of scope.
- Signal scope stays lensed SN Ia only; no other lensed transient types.
- This is a parallel research pipeline alongside the existing static `ColourClassifier`, not a replacement — whether the project keeps, drops, or merges either approach is decided later from the comparison harness's results (Task 5), not architecturally up front.

---

### Task 1: Trajectory generation layer

**Goal:** Extend `cmsne.multicolour`'s two-point colour generator into an N-point Δt-since-trigger trajectory generator, reusing its OpSim/model simulation plumbing unchanged.

**Files:**
- Modify: `cmsne/multicolour.py` (extract a one-line hook method from `MultiColourGenerator.generate_one`)
- Create: `cmsne/trajectory.py`
- Test: `tests/test_cmsne.py` (append after the existing `=== multicolour: all-band photometry extraction ===` section, i.e. after line 626)

**Acceptance Criteria:**
- [ ] `MultiColourGenerator`'s existing public behaviour (`generate_one`, `generate_many`, their outputs) is byte-for-byte unchanged — the hook's default implementation calls `event_multicolour` exactly as before.
- [ ] `cmsne.trajectory.event_trajectory(lightcurve, grid, window)` returns a dict with one colour key `'{colour}{suffix}'` per `(colour, grid step)` pair and one `'peakmag{suffix}'` per grid step, where `peakmag{suffix}` is the **cumulative** brightest magnitude observed through that step (not the eventual global peak), or `None` if nothing was ever detected.
- [ ] `cmsne.trajectory.TrajectoryGenerator` is a `MultiColourGenerator` subclass that produces trajectory features via the same `generate_one`/`generate_many` call path, with no duplicated simulation code.

**Verify:** `python tests/test_cmsne.py` → all checks pass, in particular the new `=== trajectory: N-point colour-magnitude generation ===` section, and the pre-existing `226 passed, 0 failed`-style summary line count increases with no new failures.

**Steps:**

- [ ] **Step 1: Add the feature-extraction hook to `MultiColourGenerator`**

In `cmsne/multicolour.py`, change `MultiColourGenerator.generate_one` to call a new overridable method instead of `event_multicolour` directly, and add that method:

```python
        feat = self._event_features(lc, evolution_days, window)
        if feat is None:
            return None
        feat['z'] = z
        feat['magnification'] = magnification
        return feat

    def _event_features(self, lc, evolution_days, window):
        """Hook for the feature dict extracted from one light curve.

        Overridden by :class:`cmsne.trajectory.TrajectoryGenerator` to extract
        an N-point trajectory instead of the two-point (trigger,
        trigger+evolution_days) vector. Kept as a method (not a free function
        call) so subclasses can swap the extraction logic without duplicating
        :meth:`generate_one`'s OpSim/model plumbing.
        """
        return event_multicolour(lc, evolution_days, window)
```

(Only the first line inside `generate_one` changes, from `feat = event_multicolour(lc, evolution_days, window)` to `feat = self._event_features(lc, evolution_days, window)`; the `if feat is None: return None` / `feat['z'] = z` / `feat['magnification'] = magnification` / `return feat` lines below it are unchanged. The new `_event_features` method goes immediately after `generate_one` in the class body.)

- [ ] **Step 2: Write the failing tests for the hook + new module**

Append to `tests/test_cmsne.py`, immediately after line 626 (`check('rate_weight dispatches by class', ...)`), before the `=== ColourClassifier ===` section:

```python
_mg = mcol.MultiColourGenerator()
_mg_feat = _mg._event_features(_lc, 15.0, 5.0)
_direct_feat = mcol.event_multicolour(_lc, 15.0, 5.0)
check('MultiColourGenerator._event_features defaults to event_multicolour',
      _mg_feat.keys() == _direct_feat.keys() and
      np.isclose(_mg_feat['gr'], _direct_feat['gr']) and
      np.isclose(_mg_feat['peakmag'], _direct_feat['peakmag']))

print('\n=== trajectory: N-point colour-magnitude generation ===')
from cmsne import trajectory as traj
# g@2,r@3 detected at trigger; g brighter again @20, r follows @23.
_lct = _LCm([2, 3, 20, 23], ['g', 'r', 'g', 'r'], [22.0, 21.5, 20.0, 21.0])
_ft = traj.event_trajectory(_lct, grid=(0.0, 20.0), window=5.0)
check('trigger epoch is first detection', _ft['trigger_day'] == 2.0, _ft['trigger_day'])
check('colour at day 0 matches multicolour.event_multicolour',
      np.isclose(_ft['gr_d0'], 0.5), _ft['gr_d0'])
check('colour at day 20 uses the later epoch', np.isclose(_ft['gr_d20'], -1.0), _ft['gr_d20'])
check('cumulative peakmag at day 0 is the trigger-time brightest',
      np.isclose(_ft['peakmag_d0'], 21.5), _ft['peakmag_d0'])
check('cumulative peakmag at day 20 improves (brighter point included)',
      np.isclose(_ft['peakmag_d20'], 20.0), _ft['peakmag_d20'])
check('no detections -> None',
      traj.event_trajectory(_LCm([1, 2], ['g', 'r'], [np.inf, np.inf])) is None)
check('TRAJECTORY_COLOUR_KEYS has 5 colours x 5 grid steps',
      len(traj.TRAJECTORY_COLOUR_KEYS) == 25, len(traj.TRAJECTORY_COLOUR_KEYS))
check('TRAJECTORY_MAGNITUDE_KEYS has one peakmag per grid step',
      len(traj.TRAJECTORY_MAGNITUDE_KEYS) == 5)

_tg = traj.TrajectoryGenerator()
check('TrajectoryGenerator inherits MultiColourGenerator API',
      hasattr(_tg, 'generate_one') and hasattr(_tg, 'generate_many'))
check('TrajectoryGenerator._event_features uses the trajectory schema',
      set(_tg._event_features(_lct, 15.0, 5.0)) >= {'gr_d0', 'peakmag_d0', 'gr_d60'})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py`
Expected: `ModuleNotFoundError: No module named 'cmsne.trajectory'` (and/or `AttributeError` on `_event_features`), i.e. a clean failure from the missing module/hook, not from a typo.

- [ ] **Step 4: Create `cmsne/trajectory.py`**

```python
"""N-point colour-magnitude trajectories: the time-domain companion to
:mod:`cmsne.multicolour`.

:mod:`cmsne.multicolour` records the colour vector at exactly two epochs
(trigger, trigger+evolution_days). This module generalises that to a fixed
grid of canonical Δt-since-trigger checkpoints, so a classifier can be
trained on -- and a live candidate scored against -- a feature vector that
grows as more epochs are observed. See
docs/superpowers/specs/2026-10-22-sequential-cmd-classifier-design.md.

The time axis is always **days since the candidate's first LSST detection**
(the trigger epoch), never days since true explosion -- the latter is
unobservable for a real candidate (see the design spec, "resolved
methodology questions" #1).
"""

import numpy as np

from .multicolour import MultiColourGenerator, band_photometry, _colours, ADJ_COLOURS

# Canonical Δt-since-trigger checkpoints [observer-days], reusing the exact
# values already validated in the waiting-time study (see
# docs/methods_and_results.md / colour-classifier-findings) so results stay
# comparable to existing findings.
TRAJECTORY_GRID = (0.0, 3.0, 14.0, 30.0, 60.0)


def _step_suffix(day):
    """Feature-key suffix for one grid step, e.g. ``3.0`` -> ``'_d3'``."""
    return f'_d{int(day)}'


def event_trajectory(lightcurve, grid=TRAJECTORY_GRID, window=5.0):
    """N-point colour + cumulative-brightness trajectory for one light curve.

    At each ``Δt`` in ``grid``, records the ugrizy adjacent-band colours
    (NaN where a band was not observed within ``window`` of trigger+Δt) and
    the brightest magnitude seen in any band-photometry snapshot up to and
    including that Δt (not the eventual global peak -- using the global peak
    at an early step would leak information a live candidate could not yet
    have).

    :return: dict with keys ``'{colour}{suffix}'`` (e.g. ``'gr_d14'``) and
        ``'peakmag{suffix}'`` for every grid step, plus ``trigger_day``; or
        ``None`` if nothing was ever detected.
    """
    days = np.asarray(lightcurve.obs_days, dtype=float)
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    det = np.isfinite(mag)
    if not np.any(det):
        return None

    trigger = float(days[det].min())
    feat = {'trigger_day': trigger}
    cumulative_best = np.inf
    for day in grid:
        suffix = _step_suffix(day)
        m = band_photometry(lightcurve, trigger + day, window)
        feat.update(_colours(m, suffix))

        finite_here = [v for v in m.values() if np.isfinite(v)]
        if finite_here:
            cumulative_best = min(cumulative_best, min(finite_here))
        feat[f'peakmag{suffix}'] = cumulative_best if np.isfinite(cumulative_best) else np.nan

    return feat


# Full trajectory feature schema, for cmsne.trajectory.TrajectoryClassifier
# and cmsne.contours (every colour at every grid step; every cumulative
# peakmag at every grid step).
TRAJECTORY_COLOUR_KEYS = [f'{a[-1]}{b[-1]}{_step_suffix(day)}'
                           for day in TRAJECTORY_GRID for a, b in ADJ_COLOURS]
TRAJECTORY_MAGNITUDE_KEYS = tuple(f'peakmag{_step_suffix(day)}' for day in TRAJECTORY_GRID)


class TrajectoryGenerator(MultiColourGenerator):
    """Like :class:`cmsne.multicolour.MultiColourGenerator`, but records the
    full N-point trajectory (:func:`event_trajectory`) instead of a fixed
    two-point colour vector.

    Identical ``generate_one``/``generate_many`` API (inherited unchanged) --
    only the per-event feature extraction differs.
    """

    def __init__(self, source='salt3', grid=TRAJECTORY_GRID):
        super().__init__(source)
        self.grid = grid

    def _event_features(self, lc, evolution_days, window):
        return event_trajectory(lc, self.grid, window)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python tests/test_cmsne.py`
Expected: `PASS` on every check in the new section, overall `0 failed`.

- [ ] **Step 6: Commit**

```bash
git add cmsne/multicolour.py cmsne/trajectory.py tests/test_cmsne.py
git commit -m "feat: add N-point CMD trajectory generation layer"
```

---

### Task 2: Trajectory classifier

**Goal:** Add `TrajectoryClassifier`, a `ColourClassifier` subclass trained on the full N-point trajectory schema, so "updating probability as observations accumulate" is re-scoring one trained model with a longer feature vector rather than chaining independent per-epoch posteriors.

**Files:**
- Modify: `cmsne/trajectory.py` (append the new class)
- Test: `tests/test_cmsne.py` (append after Task 1's trajectory section)

**Acceptance Criteria:**
- [ ] `TrajectoryClassifier` reuses `ColourClassifier.fit/score/probability/rank/predict/recovery_rate` completely unchanged; `cmsne/classifier.py` itself is not modified.
- [ ] `TrajectoryClassifier` trains on `TRAJECTORY_COLOUR_KEYS` + `TRAJECTORY_MAGNITUDE_KEYS` and handles events with NaN at later, not-yet-observed grid steps without error.
- [ ] Scoring the same signal class with more grid steps revealed yields a more confident (higher) posterior on average than with fewer steps revealed, demonstrating the "re-score with a longer vector" update mechanism.

**Verify:** `python tests/test_cmsne.py` → all checks pass, in particular `=== TrajectoryClassifier: cumulative multi-step scoring ===`.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cmsne.py`, after Task 1's trajectory section (after the `TrajectoryGenerator._event_features uses the trajectory schema` check):

```python
print('\n=== TrajectoryClassifier: cumulative multi-step scoring ===')
from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_COLOUR_KEYS, TRAJECTORY_MAGNITUDE_KEYS
_rngt = np.random.default_rng(11)
def _mk_traj(n, mu, n_steps_seen):
    """n events; colours at the first n_steps_seen grid steps set, later NaN."""
    out = []
    for _ in range(n):
        e = {}
        for i, k in enumerate(TRAJECTORY_COLOUR_KEYS):
            step = i // 5     # 5 colours per grid step, grid order
            e[k] = float(_rngt.normal(mu, 0.4)) if step < n_steps_seen else np.nan
        for i, k in enumerate(TRAJECTORY_MAGNITUDE_KEYS):
            e[k] = float(_rngt.normal(22, 1)) if i < n_steps_seen else np.nan
        out.append(e)
    return out
sig_t = _mk_traj(1500, 1.2, 5); bg_t = _mk_traj(3000, 0.0, 5)
tclf = TrajectoryClassifier(target_fpr=0.10).fit(sig_t, bg_t)
rec_t = tclf.recovery_rate(sig_t)
fpr_t = float(np.mean(tclf.predict(bg_t)))
check('trajectory classifier separates a separable signal', rec_t > 0.8, round(rec_t, 3))
check('trajectory classifier threshold pins contamination', abs(fpr_t - 0.10) < 0.06, round(fpr_t, 3))
check('trajectory classifier uses the full step schema',
      tclf.keys == TRAJECTORY_COLOUR_KEYS and tclf.extra == TRAJECTORY_MAGNITUDE_KEYS)

partial = _mk_traj(5, 1.2, 2)
check('partial trajectory (early steps only) scores without crashing',
      np.all(np.isfinite(tclf.score(partial))))

one_step = _mk_traj(200, 1.2, 1)
five_step = _mk_traj(200, 1.2, 5)
p_one = tclf.probability(one_step).mean()
p_five = tclf.probability(five_step).mean()
check('more observed steps -> more confident posterior for true signal',
      p_five > p_one, (round(p_one, 3), round(p_five, 3)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py`
Expected: `ImportError: cannot import name 'TrajectoryClassifier'`.

- [ ] **Step 3: Append `TrajectoryClassifier` to `cmsne/trajectory.py`**

```python
from .classifier import ColourClassifier


class TrajectoryClassifier(ColourClassifier):
    """:class:`cmsne.classifier.ColourClassifier`, but trained on the full
    N-point trajectory schema (:data:`TRAJECTORY_COLOUR_KEYS` +
    :data:`TRAJECTORY_MAGNITUDE_KEYS`) instead of a single colour + one
    magnitude feature.

    "Updating as observations arrive" (see the design spec's resolution of
    sequential non-independence) means re-scoring *this one* trained model
    with a feature vector that has more of its later-Δt entries filled in --
    never chaining independent per-epoch probabilities.

    Reuses :class:`ColourClassifier`'s fit/score/probability/rank/predict/
    recovery_rate unchanged; only the feature schema differs, so
    ``include_peakmag``'s single-feature assumption is bypassed by setting
    ``self.extra`` directly to the full per-step magnitude tuple.
    """

    def __init__(self, keys=TRAJECTORY_COLOUR_KEYS, target_fpr=0.10,
                 calibrate=True, **hgb):
        super().__init__(keys=keys, include_peakmag=False,
                         target_fpr=target_fpr, calibrate=calibrate, **hgb)
        self.magnitude_feature = None
        self.extra = TRAJECTORY_MAGNITUDE_KEYS
```

(Add the `from .classifier import ColourClassifier` import to the top of `cmsne/trajectory.py`, alongside the existing imports; the class body goes at the end of the file.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python tests/test_cmsne.py`
Expected: `PASS` on every check in the new section, overall `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add cmsne/trajectory.py tests/test_cmsne.py
git commit -m "feat: add TrajectoryClassifier for cumulative multi-step scoring"
```

---

### Task 3: Naive sequential-multiplication baseline

**Goal:** Add a deliberately-wrong naive-Bayes odds-multiplication function, used only to quantify (in Task 5) how much that approach overstates confidence relative to the jointly-trained `TrajectoryClassifier`.

**Files:**
- Create: `cmsne/comparison.py`
- Test: `tests/test_cmsne.py` (append after Task 2's section)

**Acceptance Criteria:**
- [ ] `naive_sequential_posterior` combines per-step probabilities via an odds product, skipping (not penalising) NaN/unobserved steps.
- [ ] It never raises a `RuntimeWarning` from `log`/division on NaN input (masked before any `log` call).
- [ ] It accepts a 2D `(n_events, n_steps)` array and returns one combined posterior per event.

**Verify:** `python tests/test_cmsne.py` → all checks pass, in particular `=== comparison: naive sequential-multiplication baseline ===`, with no warnings printed to stderr.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cmsne.py`, after Task 2's `TrajectoryClassifier` section:

```python
print('\n=== comparison: naive sequential-multiplication baseline ===')
from cmsne.comparison import naive_sequential_posterior
p3 = naive_sequential_posterior(np.array([[0.9, 0.9, 0.9]]))
check('three agreeing high-confidence steps combine to near-certainty',
      p3[0] > 0.95, round(float(p3[0]), 4))
p_half = naive_sequential_posterior(np.array([[0.5, 0.5, 0.5]]))
check('all-uninformative (p=0.5) steps leave the posterior at 0.5',
      np.isclose(p_half[0], 0.5), float(p_half[0]))
p_nan = naive_sequential_posterior(np.array([[0.9, np.nan, np.nan]]))
check('NaN (unobserved) steps are skipped, not treated as evidence',
      np.isclose(p_nan[0], 0.9), float(p_nan[0]))
p_batch = naive_sequential_posterior(np.array([[0.9, 0.9], [0.1, 0.1]]))
check('batched 2D input returns one posterior per event',
      p_batch.shape == (2,) and p_batch[0] > 0.5 > p_batch[1], p_batch)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py`
Expected: `ModuleNotFoundError: No module named 'cmsne.comparison'`.

- [ ] **Step 3: Create `cmsne/comparison.py`**

```python
"""Research-comparison utilities for the sequential CMD classifier design.

See docs/superpowers/specs/2026-10-22-sequential-cmd-classifier-design.md,
section D. Benchmarks the trajectory classifier against the existing static
ColourClassifier, and against a deliberately wrong naive-independent-
multiplication baseline, so the comparison -- not an a-priori choice --
decides which approach the project keeps.
"""

import numpy as np


def naive_sequential_posterior(step_probabilities):
    """Chain per-step posteriors by multiplying odds -- the WRONG baseline.

    This treats each grid step's probability as independent evidence and
    combines them via a naive-Bayes odds product. It is included only to
    quantify how badly that assumption overstates confidence relative to the
    jointly trained :class:`cmsne.trajectory.TrajectoryClassifier`
    (resolution #2 in the design spec) -- never use this to make a real
    classification decision.

    :param step_probabilities: array, shape ``(n_events, n_steps)``, each
        entry a per-step P(lensed) in (0, 1); NaN where that step has no
        observation yet (skipped, i.e. treated as uninformative odds=1).
    :return: array of shape ``(n_events,)``, the naively combined posterior.
    """
    p = np.asarray(step_probabilities, dtype=float)
    valid = np.isfinite(p)
    p_clipped = np.clip(p, 1e-12, 1 - 1e-12)

    log_odds_each = np.zeros_like(p_clipped)
    log_odds_each[valid] = np.log(p_clipped[valid] / (1 - p_clipped[valid]))

    combined_log_odds = np.sum(log_odds_each, axis=-1)
    combined_odds = np.exp(combined_log_odds)
    return combined_odds / (1 + combined_odds)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python tests/test_cmsne.py`
Expected: `PASS` on every check in the new section, overall `0 failed`, no warnings on stderr.

- [ ] **Step 5: Commit**

```bash
git add cmsne/comparison.py tests/test_cmsne.py
git commit -m "feat: add naive sequential-multiplication baseline (deliberately wrong, for comparison)"
```

---

### Task 4: CMD contour / posterior-surface layer

**Goal:** Add the "evolving probability density contour" diagnostic: at each Δt grid step, fit a 2-feature classifier on one colour + its matching cumulative magnitude, and evaluate its posterior over a mesh.

**Files:**
- Create: `cmsne/contours.py`
- Test: `tests/test_cmsne.py` (append after Task 3's section)

**Acceptance Criteria:**
- [ ] `posterior_surface` fits without touching `cmsne/classifier.py` (bypasses the `magnitude_feature` validation by setting `clf.extra`/`clf.magnitude_feature` directly, the same pattern as `TrajectoryClassifier`).
- [ ] `posterior_surface` returns `(n, n)` grids with the probability grid bounded in `[0, 1]`.
- [ ] `evolving_contours` produces one surface per requested grid step, keyed by that step's day value.

**Verify:** `python tests/test_cmsne.py` → all checks pass, in particular `=== contours: evolving posterior surface ===`.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cmsne.py`, after Task 3's naive-baseline section:

```python
print('\n=== contours: evolving posterior surface ===')
from cmsne import contours as ctr
_rngc = np.random.default_rng(3)
def _mk_cm(n, mu):
    return [{'gr_d0': float(_rngc.normal(mu, 0.3)), 'peakmag_d0': float(_rngc.normal(22, 1)),
            'gr_d14': float(_rngc.normal(mu, 0.3)), 'peakmag_d14': float(_rngc.normal(22, 1))}
           for _ in range(n)]
sig_c, bg_c = _mk_cm(300, 1.0), _mk_cm(600, 0.0)
cg, mg, pg, cclf = ctr.posterior_surface(sig_c, bg_c, 'gr_d0', 'peakmag_d0', n=20)
check('posterior_surface returns (n,n) grids', cg.shape == (20, 20) and pg.shape == (20, 20), pg.shape)
check('probability grid is bounded in [0,1]', np.all((pg >= 0) & (pg <= 1)))
check('higher colour region is more lensed-like (separable signal)',
      pg[:, -1].mean() > pg[:, 0].mean(), (round(pg[:, -1].mean(), 3), round(pg[:, 0].mean(), 3)))

surfaces = ctr.evolving_contours(sig_c, bg_c, 'gr', grid=(0.0, 14.0), n=15)
check('evolving_contours returns one surface per grid step', set(surfaces) == {0.0, 14.0})
check('each evolving-contour entry has the 4-tuple shape',
      all(len(v) == 4 and v[2].shape == (15, 15) for v in surfaces.values()))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py`
Expected: `ModuleNotFoundError: No module named 'cmsne.contours'`.

- [ ] **Step 3: Create `cmsne/contours.py`**

```python
"""Evolving colour-magnitude probability-density contours.

Generalises the single-epoch "boundary is a gradient, not a line" study
(colour-classifier-findings) across the trajectory's Δt-since-trigger grid
(:mod:`cmsne.trajectory`): at each grid step, fit a 2-feature (one colour +
its matching cumulative magnitude) ColourClassifier and evaluate its
calibrated posterior over a colour-magnitude mesh. This is a diagnostic/
visualisation layer -- it does not feed
:class:`cmsne.trajectory.TrajectoryClassifier`, which trains on the full
feature vector directly (see the design spec, architecture section B vs C).
"""

import numpy as np

from .classifier import ColourClassifier
from .trajectory import TRAJECTORY_GRID, _step_suffix


def posterior_surface(signal_events, background_events, colour_key, magnitude_key,
                      colour_range=None, magnitude_range=None, n=200):
    """Fit a 2-feature classifier and evaluate P(lensed) over a mesh.

    :param signal_events, background_events: event dicts exposing
        ``colour_key`` and ``magnitude_key`` (e.g. from
        :func:`cmsne.trajectory.event_trajectory`).
    :param colour_range, magnitude_range: ``(low, high)``; default to the
        pooled data's min/max.
    :return: ``(colour_grid, magnitude_grid, probability_grid, classifier)``,
        each grid an ``(n, n)`` array from ``np.meshgrid``.
    """
    clf = ColourClassifier(keys=[colour_key], include_peakmag=False, target_fpr=0.10)
    clf.magnitude_feature = magnitude_key
    clf.extra = (magnitude_key,)
    clf.fit(signal_events, background_events)

    pooled = signal_events + background_events
    colours = np.array([e.get(colour_key, np.nan) for e in pooled], dtype=float)
    mags = np.array([e.get(magnitude_key, np.nan) for e in pooled], dtype=float)
    if colour_range is None:
        colour_range = (float(np.nanmin(colours)), float(np.nanmax(colours)))
    if magnitude_range is None:
        magnitude_range = (float(np.nanmin(mags)), float(np.nanmax(mags)))

    colour_grid, magnitude_grid = np.meshgrid(
        np.linspace(*colour_range, n), np.linspace(*magnitude_range, n))
    grid_events = [{colour_key: c, magnitude_key: m}
                  for c, m in zip(colour_grid.ravel(), magnitude_grid.ravel())]
    probability_grid = clf.probability(grid_events).reshape(colour_grid.shape)
    return colour_grid, magnitude_grid, probability_grid, clf


def evolving_contours(signal_trajectories, background_trajectories, pair_key,
                      grid=TRAJECTORY_GRID, n=200):
    """:func:`posterior_surface` at every Δt-since-trigger grid step.

    :param pair_key: adjacent-colour key stem, e.g. ``'gr'`` for g-r.
    :return: dict ``{day: (colour_grid, magnitude_grid, probability_grid, classifier)}``
        keyed by each ``day`` in ``grid``.
    """
    surfaces = {}
    for day in grid:
        suffix = _step_suffix(day)
        surfaces[day] = posterior_surface(
            signal_trajectories, background_trajectories,
            colour_key=f'{pair_key}{suffix}', magnitude_key=f'peakmag{suffix}', n=n)
    return surfaces
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python tests/test_cmsne.py`
Expected: `PASS` on every check in the new section, overall `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add cmsne/contours.py tests/test_cmsne.py
git commit -m "feat: add evolving CMD posterior-surface diagnostic"
```

---

### Task 5: Comparison harness

**Goal:** Benchmark the static `ColourClassifier`, the new `TrajectoryClassifier`, and the naive baseline side by side on the same held-out events, producing the headline numbers the whole design exists to get.

**Files:**
- Modify: `cmsne/comparison.py` (append `compare_classifiers`)
- Test: `tests/test_cmsne.py` (append after Task 4's section)

**Acceptance Criteria:**
- [ ] `compare_classifiers` reports `static_recovery`, `trajectory_recovery` (both at the same `target_fpr`), and `naive_vs_trajectory_mean_abs_diff`.
- [ ] It fits one single-colour `ColourClassifier` per grid step internally (reusing the same construct-then-override pattern as Task 4) to produce the naive baseline's per-step inputs — no duplicated step-classifier logic beyond that.
- [ ] All three numbers are finite and in sane ranges on a separable synthetic population.

**Verify:** `python tests/test_cmsne.py` → all checks pass, in particular `=== comparison: static vs trajectory vs naive harness ===`.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cmsne.py`, after Task 4's contours section:

```python
print('\n=== comparison: static vs trajectory vs naive harness ===')
from cmsne.comparison import compare_classifiers
from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID
from cmsne.classifier import ColourClassifier

_rngh = np.random.default_rng(5)
def _mk_full(n, mu):
    """Events exposing BOTH the static 2-point schema and the trajectory schema."""
    out = []
    for _ in range(n):
        e = {'gr': float(_rngh.normal(mu, 0.4)), 'peakmag': float(_rngh.normal(22, 1))}
        for day in TRAJECTORY_GRID:
            suffix = f'_d{int(day)}'
            e[f'gr{suffix}'] = float(_rngh.normal(mu, 0.4))
            e[f'peakmag{suffix}'] = float(_rngh.normal(22, 1))
        out.append(e)
    return out

sig_h, bg_h = _mk_full(400, 1.2), _mk_full(800, 0.0)
static_clf = ColourClassifier(keys=['gr'], target_fpr=0.10).fit(sig_h, bg_h)
traj_clf = TrajectoryClassifier(target_fpr=0.10).fit(sig_h, bg_h)
result = compare_classifiers(static_clf, traj_clf, sig_h, bg_h, step_colour_key='gr')
check('harness reports all three headline metrics',
      set(result) == {'static_recovery', 'trajectory_recovery', 'naive_vs_trajectory_mean_abs_diff'})
check('both recoveries are plausible for a separable signal',
      result['static_recovery'] > 0.5 and result['trajectory_recovery'] > 0.5, result)
check('naive-vs-trajectory divergence is non-negative and finite',
      np.isfinite(result['naive_vs_trajectory_mean_abs_diff']) and
      result['naive_vs_trajectory_mean_abs_diff'] >= 0.0, result['naive_vs_trajectory_mean_abs_diff'])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python tests/test_cmsne.py`
Expected: `ImportError: cannot import name 'compare_classifiers'`.

- [ ] **Step 3: Append `compare_classifiers` to `cmsne/comparison.py`**

```python
def compare_classifiers(static_clf, trajectory_clf, signal_trajectories, background_trajectories,
                        step_colour_key='gr', target_fpr=0.10):
    """Benchmark the static, trajectory, and naive-multiplication approaches
    on the same held-out trajectory events (design spec section D).

    :param static_clf: a fitted :class:`cmsne.classifier.ColourClassifier`
        (the existing two-point method, unchanged).
    :param trajectory_clf: a fitted
        :class:`cmsne.trajectory.TrajectoryClassifier`.
    :param signal_trajectories, background_trajectories: held-out event
        dicts exposing both the static classifier's keys (``gr``, ...,
        ``peakmag``) and the trajectory schema (``gr_d0``, ...).
    :param step_colour_key: which adjacent colour's per-step probabilities
        feed the naive baseline (e.g. ``'gr'`` -> ``gr_d0``, ``gr_d3``, ...).
    :return: dict of headline metrics: ``static_recovery``,
        ``trajectory_recovery`` (both at ``target_fpr``), and
        ``naive_vs_trajectory_mean_abs_diff`` (how much the naive baseline's
        posterior diverges from the jointly trained classifier's -- the
        quantified overconfidence from resolution #2).
    """
    from .trajectory import TRAJECTORY_GRID, _step_suffix
    from .classifier import ColourClassifier

    static_recovery = static_clf.recovery_rate(signal_trajectories)
    trajectory_recovery = trajectory_clf.recovery_rate(signal_trajectories)

    # Per-step posteriors for the naive baseline: fit one single-colour
    # classifier per grid step (same construct-then-override pattern as
    # cmsne.contours), score each signal event, then combine naively.
    step_probs_sig = np.full((len(signal_trajectories), len(TRAJECTORY_GRID)), np.nan)
    for i, day in enumerate(TRAJECTORY_GRID):
        suffix = _step_suffix(day)
        colour_key, magnitude_key = f'{step_colour_key}{suffix}', f'peakmag{suffix}'
        step_clf = ColourClassifier(keys=[colour_key], include_peakmag=False, target_fpr=target_fpr)
        step_clf.magnitude_feature = magnitude_key
        step_clf.extra = (magnitude_key,)
        step_clf.fit(signal_trajectories, background_trajectories)
        step_probs_sig[:, i] = step_clf.probability(signal_trajectories)

    naive_posterior_sig = naive_sequential_posterior(step_probs_sig)
    trajectory_posterior_sig = trajectory_clf.probability(signal_trajectories)
    naive_vs_trajectory_mean_abs_diff = float(
        np.mean(np.abs(naive_posterior_sig - trajectory_posterior_sig)))

    return {
        'static_recovery': static_recovery,
        'trajectory_recovery': trajectory_recovery,
        'naive_vs_trajectory_mean_abs_diff': naive_vs_trajectory_mean_abs_diff,
    }
```

(Appended at the end of `cmsne/comparison.py`, below `naive_sequential_posterior` from Task 3 -- it is used directly, in the same module, no new import needed for it.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python tests/test_cmsne.py`
Expected: `PASS` on every check in the new section, overall `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add cmsne/comparison.py tests/test_cmsne.py
git commit -m "feat: add static-vs-trajectory-vs-naive comparison harness"
```

---

### Task 6: BlueBEAR trajectory production script

**Goal:** A production-scale trajectory generation script mirroring `scripts/run_multicolour.py`, so the trajectory population can actually be generated at scale on BlueBEAR (no OpSim DB is available locally).

**Files:**
- Create: `scripts/run_trajectory.py`
- Create: `scripts/run_trajectory_bluebear.sh`

**Acceptance Criteria:**
- [ ] `scripts/run_trajectory.py` uses the identical `CLASSES` table as `scripts/run_multicolour.py` (same classes, rates, abs-mag distributions), so the two productions cover the same population and are directly comparable.
- [ ] It uses `cmsne.trajectory.TrajectoryGenerator` in place of `cmsne.multicolour.MultiColourGenerator`, with no other behavioural difference from `run_multicolour.py`'s CLI/manifest/output conventions.
- [ ] The script imports and parses `--help` successfully without requiring an OpSim database connection (DB access only happens inside `run_class`, not at import or arg-parse time).

**Verify:**
- `python scripts/run_trajectory.py --help` → prints usage/help text, exits 0, no OpSim DB required.
- `bash -n scripts/run_trajectory_bluebear.sh` → no output, exit 0 (syntax check).

**Steps:**

- [ ] **Step 1: Create `scripts/run_trajectory.py`**

```python
"""N-point trajectory production run for the sequential CMD classifier design.

Runs each transient class through a real OpSim cadence and records the full
colour trajectory (cmsne.trajectory.event_trajectory) at the canonical
Δt-since-trigger grid, instead of multicolour's fixed two-point vector -- see
docs/superpowers/specs/2026-10-22-sequential-cmd-classifier-design.md.

Classes (10): lensed SN Ia (signal); unlensed SN Ia; and every core-collapse
template, lensed and unlensed -- identical class table to
scripts/run_multicolour.py, so results are directly comparable.

    python scripts/run_trajectory.py --n 100000              # all classes, serial
    python scripts/run_trajectory.py --class-index 0 --n 100000 --out results/traj
"""
import argparse, json, os, pickle, platform, subprocess, sys, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.lsst import contaminant_info
from cmsne.supernovae import redshift_limits
from cmsne.trajectory import TrajectoryGenerator
from cmsne.multicolour import rate_weight

# Class table: (name, sncosmo source, 'lensed'/'unlensed', rate-kind, abs_mag, sigma)
# Identical to scripts/run_multicolour.py's CLASSES, so the two productions
# cover the same population and are directly comparable.
CLASSES = [
    dict(name='lensed_Ia',   source='salt3', kind='lensed',   rate='sig', am=-19.23, sd=0.10),
    dict(name='unlensed_Ia', source='salt3', kind='unlensed', rate='uIa', am=-19.23, sd=0.10),
]
for _am, _sd, _src in contaminant_info:
    _short = _src.replace('nugent-', '')
    CLASSES.append(dict(name=f'unlensed_{_short}', source=_src, kind='unlensed', rate='uCC', am=_am, sd=_sd))
    CLASSES.append(dict(name=f'lensed_{_short}',   source=_src, kind='lensed',   rate='lCC', am=_am, sd=_sd))


def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def run_class(cls, n, seed):
    np.random.seed(seed)
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = TrajectoryGenerator(cls['source'])
    pop = gen.generate_many(n, cls['kind'], z_range, cls['am'], cls['sd'])
    for e in pop:
        e['weight'] = float(rate_weight(cls['rate'], e['z'], e['magnification'], source=cls['source']))
        e['class'] = cls['name']
    return pop, z_range


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=100000, help='attempts per class')
    ap.add_argument('--class-index', type=int, default=None, help='run one class (array mode)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    which = [args.class_index] if args.class_index is not None else list(range(len(CLASSES)))

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = args.out or os.path.join(_ROOT, 'results', f'traj_{stamp}')
    os.makedirs(out, exist_ok=True)
    if args.class_index is None or args.class_index == 0:
        json.dump({'started_utc': stamp, 'git_commit': _git_commit(),
                   'host': platform.node(), 'python': platform.python_version(),
                   'opsim_db': os.environ.get('CMSNE_OPSIM_DB', '(config default)'),
                   'n_per_class': args.n, 'classes': [c['name'] for c in CLASSES]},
                  open(os.path.join(out, 'manifest.json'), 'w'), indent=2)
    print(f'output dir : {out}\ncommit     : {_git_commit()}\n', flush=True)

    run_start = time.time()
    for i in which:
        cls = CLASSES[i]
        t0 = time.time()
        print(f'[{i}] {cls["name"]} ...', flush=True)
        try:
            pop, zr = run_class(cls, args.n, args.seed + i)
        except Exception as e:
            print(f'  FAILED {cls["name"]}: {e!r}', flush=True)
            continue
        with open(os.path.join(out, f'{cls["name"]}.pkl'), 'wb') as f:
            pickle.dump({'class': cls['name'], 'rate': cls['rate'], 'z_range': zr,
                         'n_attempted': args.n, 'events': pop}, f, protocol=pickle.HIGHEST_PROTOCOL)
        row = {'class': cls['name'], 'kept': len(pop), 'n_attempted': args.n,
               'minutes': round((time.time() - t0) / 60, 2)}
        json.dump(row, open(os.path.join(out, f'{cls["name"]}.summary.json'), 'w'), indent=2)
        print(f'  kept {row["kept"]}/{args.n}  [{row["minutes"]:.1f} min]', flush=True)

    print(f'\ntotal {(time.time() - run_start) / 3600:.2f} h  ->  {out}', flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Create `scripts/run_trajectory_bluebear.sh`**

```bash
#!/bin/bash
# N-point trajectory run on BlueBEAR (BEAR-apps module + cmsne-icelake venv).
#
#   sbatch --array=0-9 scripts/run_trajectory_bluebear.sh          # one class per task
#   sbatch scripts/run_trajectory_bluebear.sh --n 20000            # all classes, serial
#
#SBATCH --job-name=cmsne-traj
#SBATCH --output=slurm_logs/traj-%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint=icelake
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge
module load bluebear
module load bear-apps/2021b
module load Python/3.9.6-GCCcore-11.2.0
module load SciPy-bundle/2021.10-foss-2021b
module load astropy/5.0.4-foss-2021b
module load matplotlib/3.4.3-foss-2021b
module load scikit-learn/1.0.1-foss-2021b
source /rds/projects/s/smithgp-lensed-transients/andres/virtual-environments/cmsne-icelake/bin/activate

export CMSNE_OPSIM_DB="${CMSNE_OPSIM_DB:-/rds/projects/s/smithgp-lensed-transients/andres/opsim_data/baseline_v5.3.2_10yrs.db}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "host=$(hostname)  pwd=$(pwd)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo ?)  db=$CMSNE_OPSIM_DB"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    OUT="results/traj_${SLURM_ARRAY_JOB_ID}"
    echo "array task ${SLURM_ARRAY_TASK_ID}"
    exec python scripts/run_trajectory.py --class-index "${SLURM_ARRAY_TASK_ID}" --out "$OUT" "$@"
else
    exec python scripts/run_trajectory.py "$@"
fi
```

- [ ] **Step 3: Verify both scripts**

Run: `python scripts/run_trajectory.py --help`
Expected: argparse usage text printed, exit code 0, no exception (no OpSim DB touched).

Run: `chmod +x scripts/run_trajectory_bluebear.sh && bash -n scripts/run_trajectory_bluebear.sh`
Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_trajectory.py scripts/run_trajectory_bluebear.sh
git commit -m "feat: add BlueBEAR trajectory production script"
```

---

## Follow-up (explicit non-goals for this plan)

- A notebook demo (mirroring `notebooks/06_colour_classifier.ipynb`) showing the three-way comparison end-to-end is natural follow-up work once this lands, not part of this plan.
- Re-tuning the 5-day colour-pairing window, broadening the background/signal class scope, and the Markov transition-density stretch goal are all explicit non-goals carried over from the design spec.
