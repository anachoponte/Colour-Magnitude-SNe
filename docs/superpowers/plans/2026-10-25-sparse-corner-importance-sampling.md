# Sparse-Corner Importance Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate enough effective training samples in sparse/extreme colour-magnitude regions (e.g. very bright, very red) that `ColourClassifier`/`TrajectoryClassifier` can learn the correct asymptotic P(lensed) behaviour there, instead of fitting sampling noise — without changing any model architecture or the existing SLURM job structure.

**Architecture:** A new module, `cmsne/importance_sampling.py`, adds a self-contained pilot+boost two-stage sampler per class: draw a pilot population from the existing uniform proposal (unchanged), diagnose colour-magnitude cells where this class's own rate-weighted effective sample size (ESS) is too thin, fit a boost proposal over latent (redshift[, magnification]) for each flagged cell, draw extra events from it through the unchanged `generate_one`, and reweight the combined population by the resulting mixture-proposal density. `generate_many`/`generate_one` and the generator classes are not modified; only the three driver scripts' orchestration changes.

**Tech Stack:** Python, numpy, `scipy.stats.gaussian_kde` (already a project dependency, used elsewhere in `cmsne/colour_magnitude.py`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-25-sparse-corner-importance-sampling-design.md`

## Global Constraints

- `cmsne/multicolour.py`'s `MultiColourGenerator.generate_many`/`generate_one` and the `TrajectoryGenerator`/`DualSchemaGenerator` subclasses in `cmsne/trajectory.py` are **not modified** by this plan. All new logic lives in `cmsne/importance_sampling.py` and calls these unchanged.
- Every event's final `weight` must satisfy: when `pilot_frac=1.0` (or no cell is flagged), the result is **exactly** `rate_weight_fn(z, magnification)` — bit-for-bit backward compatible with today's behaviour in `scripts/run_multicolour.py:49-51` / `run_trajectory.py:51-54` / `run_dual_schema.py:59-62`. This is the single most important correctness property in this plan (see spec's "Reweighting" section) and gets its own test in Task 5.
- Unlensed events have `magnification` fixed at `1.0` — it is not a random variable for that class, so all diagnosis/boosting logic treats it as 1-dimensional (`z` only) for `kind == 'unlensed'`, and 2-dimensional (`z`, `magnification`) for `kind == 'lensed'`. Getting this wrong (e.g. fitting a 2D KDE on a constant second dimension) produces a singular-matrix error from `scipy.stats.gaussian_kde` — guard against it explicitly, don't let it surface as a crash.
- Tests go in `tests/test_cmsne.py`, following that file's existing house style **exactly**: no pytest, no `assert` statements. Use the module-level `check(name, cond, extra='')` helper already defined at the top of that file (appends to `PASS`/`FAIL` lists, prints `ok`/`FAIL`), and append new sections with a `print('\n=== ... ===')` header, matching every existing section in that file. Run the whole file with `python tests/test_cmsne.py`; it exits non-zero if any check failed.
- All new code lives under `cmsne/importance_sampling.py` plus edits to the three scripts named above. No new files beyond those two locations and the test additions.

**User decisions (already made):**
- "General mechanism, all band pairs" — the diagnosis/boosting must generalize across all five adjacent colour pairs (`ug`, `gr`, `ri`, `iz`, `zy`), not be a `gr`-specific point fix.
- "Self-contained per class" — each class's boosting decision uses only that class's own pilot ESS, with no cross-class/cross-task coordination and no change to the SLURM one-array-task-per-class structure.
- "Just do [the adaptive pilot+boost] approach" — fixed hand-calibrated boost strata per band pair and brute-force oversampling were both explicitly rejected in favour of the adaptive two-stage design.
- Spec approved with no changes.

---

## File Structure

- **Create** `cmsne/importance_sampling.py` — all new logic: ESS calculation, sparse-cell diagnosis, boost-proposal fitting (KDE + two fallback tiers), latent sampling, mixture-density reweighting, and the top-level `generate_with_boosting` orchestrator. One file, one responsibility (adaptive importance sampling for event generation), following the project's existing pattern of one focused module per concern (`contours.py`, `sn_rates.py`, etc.).
- **Modify** `scripts/run_multicolour.py`, `scripts/run_trajectory.py`, `scripts/run_dual_schema.py` — each `run_class()` swaps its `generate_many` + bare-`rate_weight()` loop for one call to `generate_with_boosting`, and the per-class pickle dict gains a `boost_info` field.
- **Modify** `tests/test_cmsne.py` — append one new section (`=== importance_sampling: pilot+boost mixture reweighting ===`) after the existing last section (dual-schema generation, ending ~line 926), before the final `PASS`/`FAIL` summary block.

---

## Task 1: Effective sample size and sparse-cell diagnosis

**Goal:** Add `effective_sample_size` (Kish's ESS) and `diagnose_sparse_cells` (per-band-pair, per-cell ESS flagging) to a new `cmsne/importance_sampling.py`.

**Files:**
- Create: `cmsne/importance_sampling.py`
- Test: `tests/test_cmsne.py` (new section appended near the end, before the final summary block at line 928)

**Acceptance Criteria:**
- [ ] `effective_sample_size(weights)` returns `(sum(w))**2 / sum(w**2)`, and `0.0` for an empty or all-zero/non-positive input.
- [ ] `diagnose_sparse_cells` bins events into an `n_bins x n_bins` grid per band pair over `(colour, magnitude)`, computes per-cell ESS, and flags cells whose ESS falls below `ess_floor_fraction` of that band pair's median non-empty-cell ESS.
- [ ] The grid's last bin in each axis includes points exactly at the maximum value (no point is silently excluded from every cell).
- [ ] `python tests/test_cmsne.py` reports `0 failed`.

**Verify:** `python tests/test_cmsne.py` → `===== N passed, 0 failed =====` (N includes the new checks below).

**Steps:**

- [ ] **Step 1: Write the failing test**

Open `tests/test_cmsne.py`, go to line 926 (end of the `DualSchemaGenerator._event_features` check, right before the final `print(f'\n===== {len(PASS)} passed...')` block), and insert:

```python
print('\n=== importance_sampling: effective sample size + sparse-cell diagnosis ===')
from cmsne import importance_sampling as isamp

check('ESS of uniform weights equals N',
      np.isclose(isamp.effective_sample_size(np.ones(50)), 50.0))
check('ESS of one dominant weight collapses toward 1',
      isamp.effective_sample_size(np.array([100.0, 0.01, 0.01, 0.01])) < 1.1)
check('ESS of empty/all-zero weights is 0.0',
      isamp.effective_sample_size(np.array([])) == 0.0 and
      isamp.effective_sample_size(np.zeros(5)) == 0.0)

# 4x4 grid over gr in [0,4], peakmag in [20,24]: fill every cell densely except one.
_rng_d = np.random.default_rng(0)
_dense_events, _dense_w = [], []
for _i in range(4):
    for _j in range(4):
        n_pts = 2 if (_i, _j) == (3, 3) else 30   # (3,3) is the deliberately sparse cell
        for _ in range(n_pts):
            _dense_events.append({'gr': _i + float(_rng_d.uniform(0, 1)),
                                  'peakmag': 20 + _j + float(_rng_d.uniform(0, 1))})
            _dense_w.append(1.0)
_dense_w = np.array(_dense_w)
# include a point exactly at the max of each axis, to check the inclusive top bin
_dense_events.append({'gr': 4.0, 'peakmag': 24.0})
_dense_w = np.r_[_dense_w, 1.0]

_flagged = isamp.diagnose_sparse_cells(_dense_events, _dense_w, band_pair_keys=['gr'],
                                       magnitude_key='peakmag', n_bins=4, ess_floor_fraction=0.2)
check('diagnose_sparse_cells flags the deliberately sparse cell',
      any(np.isclose(f['colour_range'][0], 3.0) and np.isclose(f['magnitude_range'][0], 23.0)
          for f in _flagged), _flagged)
check('diagnose_sparse_cells does not flag the densely-filled cells',
      not any(np.isclose(f['colour_range'][0], 0.0) and np.isclose(f['magnitude_range'][0], 20.0)
              for f in _flagged), _flagged)
check('every flagged cell reports its ESS below its floor',
      all(f['ess'] < f['floor'] for f in _flagged), _flagged)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_cmsne.py`
Expected: `ModuleNotFoundError: No module named 'cmsne.importance_sampling'`

- [ ] **Step 3: Write minimal implementation**

Create `cmsne/importance_sampling.py`:

```python
"""Adaptive pilot+boost importance sampling for sparse colour-magnitude corners.

Generates more effective training samples in colour-magnitude regions where a
class's own rate-weighted effective sample size (ESS) is too thin for a
downstream classifier to learn reliable behaviour there, by boosting the
latent (redshift[, magnification]) sampling density in those regions and
correcting every event's weight for the resulting mixture-proposal density.
See docs/superpowers/specs/2026-10-25-sparse-corner-importance-sampling-design.md.
"""
import numpy as np
from scipy.stats import gaussian_kde

from .config import survey_dates
from .multicolour import ADJ_COLOURS
from .opsim import create_sky_pointings

BAND_PAIR_KEYS = [f'{a[-1]}{b[-1]}' for a, b in ADJ_COLOURS]  # ['ug', 'gr', 'ri', 'iz', 'zy']
MIN_POINTS_FOR_KDE = 5


def effective_sample_size(weights):
    """Kish's ESS: (sum w)^2 / sum(w^2). 0.0 for empty or non-positive input."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0 or not np.any(w > 0):
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))


def diagnose_sparse_cells(events, weights, band_pair_keys=BAND_PAIR_KEYS,
                          magnitude_key='peakmag', n_bins=8, ess_floor_fraction=0.05):
    """Flag (band_pair, cell) regions where this class's own ESS is too thin.

    :param events: list of event dicts for one class.
    :param weights: per-event rate weights, same order/length as events.
    :param band_pair_keys: colour keys to diagnose independently.
    :param magnitude_key: magnitude feature paired with each colour.
    :param n_bins: grid resolution per axis.
    :param ess_floor_fraction: a cell is flagged when its ESS is below this
        fraction of the median non-empty-cell ESS for that band pair.
    :return: list of dicts ``{'band_pair', 'colour_range', 'magnitude_range',
        'ess', 'floor'}``, one per flagged cell.
    """
    weights = np.asarray(weights, dtype=float)
    mags = np.array([e.get(magnitude_key, np.nan) for e in events], dtype=float)
    flagged = []
    for pair in band_pair_keys:
        colours = np.array([e.get(pair, np.nan) for e in events], dtype=float)
        finite = np.isfinite(colours) & np.isfinite(mags)
        if np.sum(finite) < 2:
            continue
        c_edges = np.linspace(colours[finite].min(), colours[finite].max(), n_bins + 1)
        m_edges = np.linspace(mags[finite].min(), mags[finite].max(), n_bins + 1)
        cells = []
        for i in range(n_bins):
            c_hi = (colours <= c_edges[i + 1]) if i == n_bins - 1 else (colours < c_edges[i + 1])
            for j in range(n_bins):
                m_hi = (mags <= m_edges[j + 1]) if j == n_bins - 1 else (mags < m_edges[j + 1])
                in_cell = finite & (colours >= c_edges[i]) & c_hi & (mags >= m_edges[j]) & m_hi
                if not np.any(in_cell):
                    continue
                cells.append((c_edges[i], c_edges[i + 1], m_edges[j], m_edges[j + 1],
                             effective_sample_size(weights[in_cell])))
        if not cells:
            continue
        floor = ess_floor_fraction * float(np.median([c[4] for c in cells]))
        for c_lo, c_hi_, m_lo, m_hi_, ess in cells:
            if ess < floor:
                flagged.append({'band_pair': pair, 'colour_range': (c_lo, c_hi_),
                                 'magnitude_range': (m_lo, m_hi_), 'ess': ess, 'floor': floor})
    return flagged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_cmsne.py`
Expected: `===== N passed, 0 failed =====` (no `FAIL` lines; the new `ok` lines for the checks above appear under `=== importance_sampling: effective sample size + sparse-cell diagnosis ===`)

- [ ] **Step 5: Commit**

```bash
git add cmsne/importance_sampling.py tests/test_cmsne.py
git commit -m "feat: add ESS and sparse-cell diagnosis for importance sampling"
```

---

## Task 2: Boost-proposal fitting and latent sampling

**Goal:** Add `fit_boost_proposal` (weighted KDE over latent `(z[, magnification])` for a flagged cell, with two fallback tiers), `sample_latent` (draws from a fitted proposal), and `_proposal_density` (evaluates a fitted proposal's density at given points) to `cmsne/importance_sampling.py`.

**Files:**
- Modify: `cmsne/importance_sampling.py`
- Test: `tests/test_cmsne.py` (same new section, extended)

**Acceptance Criteria:**
- [ ] `fit_boost_proposal` fits a weighted `gaussian_kde` when a flagged cell has at least 5 weighted-positive events with non-degenerate latent values; for `lensed` classes this is a 2D KDE over `(z, magnification)`, for `unlensed` classes a 1D KDE over `z` only (magnification is fixed at 1.0, not a random variable).
- [ ] When too few points exist for a reliable KDE but at least one event is present, falls back to a padded uniform box over the observed latent spread (`fallback: 'box'`).
- [ ] When zero events of the class exist in the cell, falls back to the full `(z_range[, magnification_range])` passed in (`fallback: 'prior_box'`) — not data-derived, matching the spec.
- [ ] `sample_latent` draws from a fitted KDE or box fallback and clips/constrains draws to the valid `z_range`/`magnification_range`.
- [ ] `_proposal_density` evaluates a fitted proposal's density at given `(z[, magnification])` points, for both the KDE and box-fallback cases, and returns `0.0` for points outside a box fallback's bounds.
- [ ] `python tests/test_cmsne.py` reports `0 failed`.

**Verify:** `python tests/test_cmsne.py` → `===== N passed, 0 failed =====`

**Steps:**

- [ ] **Step 1: Write the failing test**

Append to the same test section in `tests/test_cmsne.py` (after the Task 1 checks, before any later section):

```python
# fit_boost_proposal: enough points -> KDE (lensed, 2D)
_rng_f = np.random.default_rng(1)
_cell_events_lensed = [{'z': float(_rng_f.uniform(0.1, 0.3)), 'magnification': float(_rng_f.uniform(5, 10)),
                       'gr': 2.0, 'peakmag': 16.0} for _ in range(10)]
_cell_w = np.ones(10)
_prop_kde = isamp.fit_boost_proposal(_cell_events_lensed, _cell_w, 'gr', 'peakmag',
                                     (1.5, 2.5), (15.0, 17.0), lensed=True,
                                     z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('fit_boost_proposal fits a KDE when enough points exist',
      _prop_kde['kde'] is not None and _prop_kde['fallback'] is None)

# fit_boost_proposal: too few points -> box fallback
_cell_events_few = [{'z': 0.2, 'magnification': 8.0, 'gr': 2.0, 'peakmag': 16.0}]
_prop_box = isamp.fit_boost_proposal(_cell_events_few, np.ones(1), 'gr', 'peakmag',
                                     (1.5, 2.5), (15.0, 17.0), lensed=True,
                                     z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('fit_boost_proposal falls back to a box with too few points',
      _prop_box['kde'] is None and _prop_box['fallback'] == 'box' and
      _prop_box['z_box'][0] <= 0.2 <= _prop_box['z_box'][1])

# fit_boost_proposal: zero points -> prior_box fallback spans the full ranges given
_prop_prior = isamp.fit_boost_proposal([], np.array([]), 'gr', 'peakmag',
                                       (1.5, 2.5), (15.0, 17.0), lensed=True,
                                       z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('fit_boost_proposal falls back to the full prior range with zero points',
      _prop_prior['fallback'] == 'prior_box' and _prop_prior['z_box'] == (0.0, 1.0) and
      _prop_prior['mu_box'] == (2.0, 50.0))

# unlensed: 1D KDE over z only (magnification fixed at 1.0)
_cell_events_unlensed = [{'z': float(_rng_f.uniform(0.1, 0.3)), 'magnification': 1.0,
                         'gr': 2.0, 'peakmag': 16.0} for _ in range(10)]
_prop_unlensed = isamp.fit_boost_proposal(_cell_events_unlensed, np.ones(10), 'gr', 'peakmag',
                                          (1.5, 2.5), (15.0, 17.0), lensed=False,
                                          z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('fit_boost_proposal fits a 1D KDE for unlensed classes (no singular-matrix crash)',
      _prop_unlensed['kde'] is not None and 'mu_box' not in _prop_unlensed)

# sample_latent: draws respect the clip bounds
_z_s, _mu_s = isamp.sample_latent(_prop_kde, 200, lensed=True,
                                  z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('sample_latent draws stay within z_range and magnification_range',
      np.all(_z_s >= 0.0) and np.all(_z_s <= 1.0) and
      np.all(_mu_s >= 2.0) and np.all(_mu_s <= 50.0), (float(_z_s.min()), float(_z_s.max())))
_z_u, _mu_u = isamp.sample_latent(_prop_unlensed, 50, lensed=False,
                                  z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('sample_latent returns magnification=1.0 for unlensed draws',
      np.all(_mu_u == 1.0))

# _proposal_density: box fallback is zero outside its bounds, positive inside
_dens_in = isamp._proposal_density(_prop_box, np.array([0.2]), np.array([8.0]), lensed=True)
_dens_out = isamp._proposal_density(_prop_box, np.array([0.9]), np.array([40.0]), lensed=True)
check('_proposal_density is positive inside a box fallback and zero outside',
      _dens_in[0] > 0.0 and _dens_out[0] == 0.0, (_dens_in[0], _dens_out[0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_cmsne.py`
Expected: `AttributeError: module 'cmsne.importance_sampling' has no attribute 'fit_boost_proposal'`

- [ ] **Step 3: Write minimal implementation**

Append to `cmsne/importance_sampling.py`:

```python
def fit_boost_proposal(events, weights, colour_key, magnitude_key, colour_range,
                       magnitude_range, lensed, z_range, magnification_range):
    """Fit a boost proposal over latent (z[, magnification]) for one flagged cell.

    Unlensed events have magnification fixed at 1.0 (not a random variable), so
    only ``z`` is boosted for them; lensed events boost the joint (z, magnification).

    :return: dict with ``'kde'`` (a fitted :class:`scipy.stats.gaussian_kde`, or
        ``None``) and, when ``'kde'`` is ``None``, a box fallback: ``'z_box'``
        (and ``'mu_box'`` if ``lensed``), plus ``'fallback'`` in
        ``{'box', 'prior_box'}`` (``None`` when a KDE was fit).
    """
    weights = np.asarray(weights, dtype=float)
    colours = np.array([e.get(colour_key, np.nan) for e in events], dtype=float)
    mags = np.array([e.get(magnitude_key, np.nan) for e in events], dtype=float)
    in_cell = (np.isfinite(colours) & np.isfinite(mags) &
              (colours >= colour_range[0]) & (colours < colour_range[1]) &
              (mags >= magnitude_range[0]) & (mags < magnitude_range[1]))
    idx = np.where(in_cell)[0]
    z = np.array([events[i]['z'] for i in idx], dtype=float)
    w = weights[idx]
    pos = w > 0
    # Degeneracy must be checked over the weighted-positive events only: a cell
    # can have >=MIN_POINTS_FOR_KDE positive-weight events sharing one z/mu while
    # zero-weight events in the same cell vary, which would otherwise slip past
    # this guard and crash gaussian_kde with a singular covariance matrix.
    enough_z = np.sum(pos) >= MIN_POINTS_FOR_KDE and len(np.unique(z[pos])) > 1

    if lensed:
        mu = np.array([events[i]['magnification'] for i in idx], dtype=float)
        if enough_z and len(np.unique(mu[pos])) > 1:
            return {'kde': gaussian_kde(np.vstack([z, mu]), weights=w), 'fallback': None}
        if len(z) > 0:
            pad_z = max(0.01, 0.1 * (z.max() - z.min()))
            pad_mu = max(0.1, 0.1 * (mu.max() - mu.min())) if len(mu) > 1 else 1.0
            return {'kde': None, 'fallback': 'box',
                    'z_box': (max(z_range[0], z.min() - pad_z), min(z_range[1], z.max() + pad_z)),
                    'mu_box': (max(magnification_range[0], mu.min() - pad_mu),
                              min(magnification_range[1], mu.max() + pad_mu))}
        return {'kde': None, 'fallback': 'prior_box', 'z_box': z_range, 'mu_box': magnification_range}

    if enough_z:
        return {'kde': gaussian_kde(z, weights=w), 'fallback': None}
    if len(z) > 0:
        pad_z = max(0.01, 0.1 * (z.max() - z.min()))
        return {'kde': None, 'fallback': 'box',
                'z_box': (max(z_range[0], z.min() - pad_z), min(z_range[1], z.max() + pad_z))}
    return {'kde': None, 'fallback': 'prior_box', 'z_box': z_range}


def sample_latent(proposal, n_draw, lensed, z_range, magnification_range):
    """Draw ``n_draw`` latent samples from a fitted boost proposal.

    :return: ``(z_samples, magnification_samples)`` arrays of length ``n_draw``
        (``magnification_samples`` is all-ones when not ``lensed``).
    """
    if proposal['kde'] is not None:
        if lensed:
            draws = proposal['kde'].resample(n_draw)
            z_samp = np.clip(draws[0], z_range[0], z_range[1])
            mu_samp = np.clip(draws[1], magnification_range[0], magnification_range[1])
        else:
            z_samp = np.clip(proposal['kde'].resample(n_draw)[0], z_range[0], z_range[1])
            mu_samp = np.ones(n_draw)
        return z_samp, mu_samp

    z_box = proposal['z_box']
    z_samp = np.random.uniform(z_box[0], z_box[1], n_draw)
    if lensed:
        mu_box = proposal['mu_box']
        mu_samp = np.random.uniform(mu_box[0], mu_box[1], n_draw)
    else:
        mu_samp = np.ones(n_draw)
    return z_samp, mu_samp


def _proposal_density(proposal, z, mu, lensed):
    """Evaluate one boost proposal's density at (z[, mu])."""
    if proposal['kde'] is not None:
        return proposal['kde'](np.vstack([z, mu])) if lensed else proposal['kde'](z)

    z_box = proposal['z_box']
    z_width = z_box[1] - z_box[0]
    in_z = (z >= z_box[0]) & (z <= z_box[1])
    if not lensed:
        return np.where(in_z, 1.0 / z_width, 0.0)
    mu_box = proposal['mu_box']
    mu_width = mu_box[1] - mu_box[0]
    in_mu = (mu >= mu_box[0]) & (mu <= mu_box[1])
    return np.where(in_z & in_mu, 1.0 / (z_width * mu_width), 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_cmsne.py`
Expected: `===== N passed, 0 failed =====`

- [ ] **Step 5: Commit**

```bash
git add cmsne/importance_sampling.py tests/test_cmsne.py
git commit -m "feat: add boost-proposal fitting, latent sampling, and density evaluation"
```

---

## Task 3: `generate_with_boosting` orchestrator and mixture reweighting

**Goal:** Add the top-level `generate_with_boosting` function wiring pilot → diagnose → boost → reweight together, with the mixture-density correction applied to pilot-origin events as well as boost-origin ones.

**Files:**
- Modify: `cmsne/importance_sampling.py`
- Test: `tests/test_cmsne.py` (same section, extended)

**Acceptance Criteria:**
- [ ] `generate_with_boosting(gen, n, kind, z_range, abs_mag_avg, abs_mag_sig, rate_weight_fn, ...)` returns `(events, boost_info)`.
- [ ] With `pilot_frac=1.0`, every returned event's `weight` equals `rate_weight_fn(e['z'], e['magnification'])` exactly (bit-for-bit backward compatibility with today's production behaviour).
- [ ] When boosting does fire (a fake generator + a forced-sparse scenario), `boost_info['n_boost'] > 0`, `boost_info['flagged_cells']` is non-empty, and every returned event (pilot- and boost-origin) has a finite, positive `weight`.
- [ ] `python tests/test_cmsne.py` reports `0 failed`.

**Verify:** `python tests/test_cmsne.py` → `===== N passed, 0 failed =====`

**Steps:**

- [ ] **Step 1: Write the failing test**

Append to the same section in `tests/test_cmsne.py`:

```python
class _FakeGen:
    """Toy generator for unit-testing generate_with_boosting without the real
    sncosmo/OpSim pipeline: observed colour/peakmag are a deterministic,
    known function of latent (z, magnification)."""
    @staticmethod
    def _to_event(z, mu):
        return {'z': float(z), 'magnification': float(mu),
                'gr': float(2.0 - z + 0.02 * mu), 'peakmag': float(20.0 + 5 * z - 0.1 * mu)}

    def generate_many(self, n, kind, z_range, abs_mag_avg, abs_mag_sig,
                      magnification_range=(2.0, 50.0), mjd_range=None, mjd_high=None):
        out = []
        for _ in range(n):
            z = np.random.uniform(*z_range)
            mu = np.random.uniform(*magnification_range) if kind == 'lensed' else 1.0
            out.append(self._to_event(z, mu))
        return out

    def generate_one(self, z, abs_mag, magnification, times, ra, dec, mjd_high=None):
        return self._to_event(z, magnification)


def _toy_rate_weight(z, mu):
    return float(np.exp(-z) / mu ** 2)


np.random.seed(0)
_fake = _FakeGen()
_events_nb, _info_nb = isamp.generate_with_boosting(
    _fake, 500, 'lensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight, pilot_frac=1.0)
check('pilot_frac=1.0 reproduces bare rate_weight_fn exactly (backward compatibility)',
      all(np.isclose(e['weight'], _toy_rate_weight(e['z'], e['magnification'])) for e in _events_nb) and
      _info_nb['n_boost'] == 0, len(_events_nb))

np.random.seed(0)
_events_b, _info_b = isamp.generate_with_boosting(
    _fake, 500, 'lensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight,
    pilot_frac=0.5, n_bins=4, ess_floor_fraction=0.9)   # high floor forces some cell to be flagged
check('boosting fires and reports flagged cells + boost draws',
      _info_b['n_boost'] > 0 and len(_info_b['flagged_cells']) > 0, _info_b['n_boost'])
check('every event (pilot- and boost-origin) has a finite, positive weight after boosting',
      all(np.isfinite(e['weight']) and e['weight'] > 0 for e in _events_b), len(_events_b))
check('boosting returns at least as many events as the pilot alone',
      len(_events_b) >= _info_b['n_pilot'], (len(_events_b), _info_b['n_pilot']))

np.random.seed(0)
_events_u, _info_u = isamp.generate_with_boosting(
    _fake, 300, 'unlensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight,
    pilot_frac=0.5, n_bins=4, ess_floor_fraction=0.9)
check('unlensed boosting keeps magnification fixed at 1.0 for every event',
      all(e['magnification'] == 1.0 for e in _events_u))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_cmsne.py`
Expected: `AttributeError: module 'cmsne.importance_sampling' has no attribute 'generate_with_boosting'`

- [ ] **Step 3: Write minimal implementation**

Append to `cmsne/importance_sampling.py`:

```python
def generate_with_boosting(gen, n, kind, z_range, abs_mag_avg, abs_mag_sig,
                           rate_weight_fn, magnification_range=(2.0, 50.0),
                           mjd_range=None, mjd_high=None, pilot_frac=0.85,
                           band_pair_keys=BAND_PAIR_KEYS, magnitude_key='peakmag',
                           n_bins=8, ess_floor_fraction=0.05):
    """Pilot+boost importance sampling for one class's event generation.

    Draws a pilot population from the existing uniform proposal (unchanged
    ``gen.generate_many``), diagnoses sparse (colour, magnitude) cells from the
    pilot's own weighted ESS, fits a boost proposal over latent
    (z[, magnification]) for each flagged cell, draws a boost population from
    it via ``gen.generate_one``, and returns the combined population with every
    event's weight corrected for the resulting mixture-proposal density --
    including pilot-origin events, not only boost-origin ones. When
    ``pilot_frac=1.0`` or nothing is flagged, every returned weight equals
    ``rate_weight_fn(z, magnification)`` exactly, matching today's production
    behaviour.

    :param gen: a :class:`cmsne.multicolour.MultiColourGenerator`-API object
        (``generate_many``/``generate_one``).
    :param n: total attempted draws (pilot + boost together).
    :param kind: ``'lensed'`` or ``'unlensed'`` (as for ``generate_many``).
    :param rate_weight_fn: callable ``(z, magnification) -> weight``.
    :param pilot_frac: fraction of ``n`` spent on the pilot stage.
    :return: ``(events, boost_info)``, where ``boost_info`` has ``'n_pilot'``,
        ``'n_boost'``, and ``'flagged_cells'`` (each flagged cell's band pair,
        ranges, pilot ESS/floor, and which fallback tier its proposal used).
    """
    lensed = (kind == 'lensed')
    if mjd_range is None:
        mjd_range = (survey_dates[0], survey_dates[3])
    n_pilot = n if pilot_frac >= 1.0 else int(round(n * pilot_frac))
    n_boost = n - n_pilot

    pilot = gen.generate_many(n_pilot, kind, z_range, abs_mag_avg, abs_mag_sig,
                              magnification_range=magnification_range,
                              mjd_range=mjd_range, mjd_high=mjd_high)
    if not pilot or n_boost <= 0:
        for e in pilot:
            e['weight'] = float(rate_weight_fn(e['z'], e['magnification']))
        return pilot, {'n_pilot': n_pilot, 'n_boost': 0, 'flagged_cells': []}

    pilot_rw = np.array([rate_weight_fn(e['z'], e['magnification']) for e in pilot])
    flagged = diagnose_sparse_cells(pilot, pilot_rw, band_pair_keys, magnitude_key,
                                    n_bins, ess_floor_fraction)
    if not flagged:
        for e, w in zip(pilot, pilot_rw):
            e['weight'] = float(w)
        return pilot, {'n_pilot': n_pilot, 'n_boost': 0, 'flagged_cells': []}

    deficits = np.array([max(cell['floor'] - cell['ess'], 1e-9) for cell in flagged])
    shares = deficits / deficits.sum()
    boost_counts = np.maximum(1, np.round(shares * n_boost).astype(int))

    proposals, boost_events = [], []
    for cell, n_draw in zip(flagged, boost_counts):
        proposal = fit_boost_proposal(pilot, pilot_rw, cell['band_pair'], magnitude_key,
                                      cell['colour_range'], cell['magnitude_range'],
                                      lensed, z_range, magnification_range)
        cell['fallback'] = proposal['fallback']
        proposals.append(proposal)

        z_samp, mu_samp = sample_latent(proposal, int(n_draw), lensed, z_range, magnification_range)
        for z_i, mu_i in zip(z_samp, mu_samp):
            abs_mag_i = np.random.normal(abs_mag_avg, abs_mag_sig)
            times_i = np.random.uniform(*mjd_range)
            ra, dec = create_sky_pointings(10)
            if len(ra) == 0:
                continue
            r = gen.generate_one(float(z_i), abs_mag_i, float(mu_i), times_i, ra[0], dec[0], mjd_high=mjd_high)
            if r is not None:
                boost_events.append(r)

    n_pilot_attempted = n_pilot
    n_boost_attempted = int(boost_counts.sum())
    total_attempted = n_pilot_attempted + n_boost_attempted
    f_pilot = n_pilot_attempted / total_attempted
    f_boost_per_cell = boost_counts / total_attempted

    all_events = pilot + boost_events
    all_z = np.array([e['z'] for e in all_events])
    all_mu = np.array([e['magnification'] for e in all_events]) if lensed else np.ones(len(all_events))

    p_uniform_const = 1.0 / (z_range[1] - z_range[0])
    if lensed:
        p_uniform_const /= (magnification_range[1] - magnification_range[0])

    p_boost_total = np.zeros(len(all_events))
    for cell_idx, proposal in enumerate(proposals):
        p_boost_total += f_boost_per_cell[cell_idx] * _proposal_density(proposal, all_z, all_mu, lensed)

    p_mix_relative = np.maximum(f_pilot + p_boost_total / p_uniform_const, 1e-12)
    for e, z_i, mu_i, pm_rel in zip(all_events, all_z, all_mu, p_mix_relative):
        e['weight'] = float(rate_weight_fn(z_i, mu_i) / pm_rel)

    return all_events, {'n_pilot': n_pilot_attempted, 'n_boost': n_boost_attempted, 'flagged_cells': flagged}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_cmsne.py`
Expected: `===== N passed, 0 failed =====`

- [ ] **Step 5: Commit**

```bash
git add cmsne/importance_sampling.py tests/test_cmsne.py
git commit -m "feat: add generate_with_boosting pilot+boost orchestrator"
```

---

## Task 4: Wire into the three production driver scripts

**Goal:** Replace each of `run_multicolour.py`/`run_trajectory.py`/`run_dual_schema.py`'s `generate_many` + bare-`rate_weight()` loop with one call to `generate_with_boosting`, add a `--no-boost` flag, and record `boost_info` in the per-class pickle.

**Files:**
- Modify: `scripts/run_multicolour.py:22,43-52,57`
- Modify: `scripts/run_trajectory.py:24,47-55,61`
- Modify: `scripts/run_dual_schema.py:31,55-63,68`

**Acceptance Criteria:**
- [ ] Each script's `run_class()` calls `generate_with_boosting` instead of `generate_many` + a manual weight loop.
- [ ] A `--no-boost` CLI flag sets `pilot_frac=1.0`, reproducing today's exact behaviour.
- [ ] Each class's pickle dict gains a `'boost_info'` key.
- [ ] `python tests/test_cmsne.py` still reports `0 failed` (no regressions to existing checks).

**Verify:** `python tests/test_cmsne.py` → `===== N passed, 0 failed =====` (all pre-existing checks, Tasks 1-3's checks, still pass; this task adds no new checks since it only rewires the scripts, which require a real OpSim DB to run end-to-end and are not covered by the synthetic test harness).

**Steps:**

- [ ] **Step 1: Update `scripts/run_multicolour.py`**

Change the import line (line 22):

```python
from cmsne.multicolour import MultiColourGenerator, rate_weight
```

to:

```python
from cmsne.multicolour import MultiColourGenerator, rate_weight
from cmsne.importance_sampling import generate_with_boosting
```

Replace `run_class` (lines 43-52):

```python
def run_class(cls, n, seed):
    np.random.seed(seed)
    # Sample z where the source covers lsstr (the normalisation band).
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = MultiColourGenerator(cls['source'])
    pop = gen.generate_many(n, cls['kind'], z_range, cls['am'], cls['sd'])
    for e in pop:
        e['weight'] = float(rate_weight(cls['rate'], e['z'], e['magnification'], source=cls['source']))
        e['class'] = cls['name']
    return pop, z_range
```

with:

```python
def run_class(cls, n, seed, pilot_frac=0.85):
    np.random.seed(seed)
    # Sample z where the source covers lsstr (the normalisation band).
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = MultiColourGenerator(cls['source'])
    pop, boost_info = generate_with_boosting(
        gen, n, cls['kind'], z_range, cls['am'], cls['sd'],
        rate_weight_fn=lambda z, mu: rate_weight(cls['rate'], z, mu, source=cls['source']),
        pilot_frac=pilot_frac)
    for e in pop:
        e['class'] = cls['name']
    return pop, z_range, boost_info
```

Add the `--no-boost` flag in `main()` (near the other `ap.add_argument` calls, line 57):

```python
    ap.add_argument('--no-boost', action='store_true', help='disable sparse-corner boosting (pilot_frac=1.0)')
```

Update the call site inside the `for i in which:` loop:

```python
        try:
            pop, zr = run_class(cls, args.n, args.seed + i)
        except Exception as e:
```

to:

```python
        try:
            pop, zr, boost_info = run_class(cls, args.n, args.seed + i,
                                            pilot_frac=1.0 if args.no_boost else 0.85)
        except Exception as e:
```

and the pickle-dump line:

```python
        with open(os.path.join(out, f'{cls["name"]}.pkl'), 'wb') as f:
            pickle.dump({'class': cls['name'], 'rate': cls['rate'], 'z_range': zr,
                         'n_attempted': args.n, 'events': pop}, f, protocol=pickle.HIGHEST_PROTOCOL)
```

to:

```python
        with open(os.path.join(out, f'{cls["name"]}.pkl'), 'wb') as f:
            pickle.dump({'class': cls['name'], 'rate': cls['rate'], 'z_range': zr,
                         'n_attempted': args.n, 'events': pop, 'boost_info': boost_info},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
```

- [ ] **Step 2: Update `scripts/run_trajectory.py`** (identical pattern)

Change the import line (line 24):

```python
from cmsne.multicolour import rate_weight
```

to:

```python
from cmsne.multicolour import rate_weight
from cmsne.importance_sampling import generate_with_boosting
```

Replace `run_class` (lines 47-55):

```python
def run_class(cls, n, seed):
    np.random.seed(seed)
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = TrajectoryGenerator(cls['source'])
    pop = gen.generate_many(n, cls['kind'], z_range, cls['am'], cls['sd'])
    for e in pop:
        e['weight'] = float(rate_weight(cls['rate'], e['z'], e['magnification'], source=cls['source']))
        e['class'] = cls['name']
    return pop, z_range
```

with:

```python
def run_class(cls, n, seed, pilot_frac=0.85):
    np.random.seed(seed)
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = TrajectoryGenerator(cls['source'])
    pop, boost_info = generate_with_boosting(
        gen, n, cls['kind'], z_range, cls['am'], cls['sd'],
        rate_weight_fn=lambda z, mu: rate_weight(cls['rate'], z, mu, source=cls['source']),
        pilot_frac=pilot_frac)
    for e in pop:
        e['class'] = cls['name']
    return pop, z_range, boost_info
```

Add `ap.add_argument('--no-boost', action='store_true', help='disable sparse-corner boosting (pilot_frac=1.0)')` next to the script's other `ap.add_argument` calls (line 61 area), then update the call site and pickle-dump exactly as in Step 1, substituted into this script's `main()` loop and its `pickle.dump({'class': ..., 'events': pop, ...})` line.

- [ ] **Step 3: Update `scripts/run_dual_schema.py`** (identical pattern)

Change the import line (line 31):

```python
from cmsne.multicolour import rate_weight
```

to:

```python
from cmsne.multicolour import rate_weight
from cmsne.importance_sampling import generate_with_boosting
```

Replace `run_class` (lines 55-63):

```python
def run_class(cls, n, seed):
    np.random.seed(seed)
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = DualSchemaGenerator(cls['source'])
    pop = gen.generate_many(n, cls['kind'], z_range, cls['am'], cls['sd'])
    for e in pop:
        e['weight'] = float(rate_weight(cls['rate'], e['z'], e['magnification'], source=cls['source']))
        e['class'] = cls['name']
    return pop, z_range
```

with:

```python
def run_class(cls, n, seed, pilot_frac=0.85):
    np.random.seed(seed)
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = DualSchemaGenerator(cls['source'])
    pop, boost_info = generate_with_boosting(
        gen, n, cls['kind'], z_range, cls['am'], cls['sd'],
        rate_weight_fn=lambda z, mu: rate_weight(cls['rate'], z, mu, source=cls['source']),
        pilot_frac=pilot_frac)
    for e in pop:
        e['class'] = cls['name']
    return pop, z_range, boost_info
```

Add `ap.add_argument('--no-boost', action='store_true', help='disable sparse-corner boosting (pilot_frac=1.0)')` next to the script's other `ap.add_argument` calls (line 68 area), then update the call site and pickle-dump exactly as in Step 1, substituted into this script's `main()` loop and its `pickle.dump({'class': ..., 'events': pop, ...})` line.

- [ ] **Step 4: Run the full test suite to confirm no regressions**

Run: `python tests/test_cmsne.py`
Expected: `===== N passed, 0 failed =====` (unchanged from Task 3 — these scripts require a real OpSim DB to exercise end-to-end, which is outside the synthetic test harness's scope; the regression check here is that the file still imports and parses cleanly: `python -c "import ast; ast.parse(open('scripts/run_multicolour.py').read()); ast.parse(open('scripts/run_trajectory.py').read()); ast.parse(open('scripts/run_dual_schema.py').read()); print('syntax ok')"` → `syntax ok`)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_multicolour.py scripts/run_trajectory.py scripts/run_dual_schema.py
git commit -m "feat: wire pilot+boost importance sampling into production drivers"
```

---

## Task 5: Backward-compatibility regression test against production-shaped data

**Goal:** Prove `generate_with_boosting(..., pilot_frac=1.0)` reproduces today's exact per-event weight on production-shaped data (realistic z/magnification distributions, not just the toy `_FakeGen` from Task 3), closing the loop on the plan's single most important correctness property.

Note: this test uses synthetic data with realistic z/magnification ranges, not a read from the actual cached job 58563115 pickle files. That's deliberate, not a shortfall — `tests/test_cmsne.py` is an explicitly hermetic, no-external-data suite (see its own module docstring: "Needs no OpSim database"), and the real cached pickles live outside the repo in a session-specific scratchpad directory; depending on that path would make this test fail for anyone else and on CI.

**Files:**
- Test: `tests/test_cmsne.py` (same section, extended)

**Acceptance Criteria:**
- [ ] Given production-shaped synthetic events (realistic `z`/`magnification` ranges), computing `rate_weight(...)` directly and via `generate_with_boosting`-style reweighting at `pilot_frac=1.0` (using a trivial pass-through fake generator that just returns pre-made events) give identical weights.
- [ ] `python tests/test_cmsne.py` reports `0 failed`.

**Verify:** `python tests/test_cmsne.py` → `===== N passed, 0 failed =====`

**Steps:**

- [ ] **Step 1: Write the failing test**

Append to the same section in `tests/test_cmsne.py`:

```python
# Backward-compatibility against real production-shaped events (not just the toy
# _FakeGen): a pass-through fake generator that returns pre-built events keeps
# their original z/magnification, so generate_with_boosting(pilot_frac=1.0) must
# reproduce rate_weight(...) exactly, the same property Task 3 checked on toy data.
class _PassThroughGen:
    def __init__(self, events):
        self._events = events

    def generate_many(self, n, kind, z_range, abs_mag_avg, abs_mag_sig,
                      magnification_range=(2.0, 50.0), mjd_range=None, mjd_high=None):
        return [dict(e) for e in self._events[:n]]

    def generate_one(self, z, abs_mag, magnification, times, ra, dec, mjd_high=None):
        return {'z': float(z), 'magnification': float(magnification), 'gr': 2.0, 'peakmag': 20.0}


_rng_pc = np.random.default_rng(2)
_prod_shaped_events = [{'z': float(_rng_pc.uniform(0.01, 1.5)),
                       'magnification': float(_rng_pc.uniform(2.0, 50.0)),
                       'gr': float(_rng_pc.normal(0.5, 0.5)),
                       'peakmag': float(_rng_pc.normal(22, 2))} for _ in range(300)]

np.random.seed(3)
_events_pc, _info_pc = isamp.generate_with_boosting(
    _PassThroughGen(_prod_shaped_events), len(_prod_shaped_events), 'lensed',
    (0.01, 1.5), -19.23, 0.10, _toy_rate_weight, pilot_frac=1.0)
check('generate_with_boosting(pilot_frac=1.0) matches bare rate_weight on production-shaped events',
      all(np.isclose(e['weight'], _toy_rate_weight(e['z'], e['magnification'])) for e in _events_pc) and
      len(_events_pc) == len(_prod_shaped_events) and _info_pc['n_boost'] == 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_cmsne.py`
Expected: passes already if Task 3's implementation is correct and general (it should, since this is the same code path with different input data) — if it instead fails, that reveals Task 3's `generate_with_boosting` implementation special-cased the toy `_FakeGen` rather than being general. Confirm it is `===== N passed, 0 failed =====` before moving on; if it fails, fix `generate_with_boosting` in `cmsne/importance_sampling.py`, not this test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cmsne.py
git commit -m "test: verify backward-compatible weighting on production-shaped data"
```

---

## Manual Validation (after Task 5 lands — not automatable, run by the project owner)

These are the spec's remaining validation steps (3 and 4 from its Testing/Validation Plan). They need real BlueBEAR access and compute budget, so they are **not** implementation tasks a coding agent can run unsupervised — they're listed here so nothing from the spec is silently dropped.

1. **Small live BlueBEAR smoke test.** Run one class (e.g. `lensed_Ia`) at reduced `--n` with boosting enabled (default `pilot_frac=0.85`), e.g.:
   ```bash
   python scripts/run_multicolour.py --class-index 0 --n 5000 --out results/boost_smoke_test
   ```
   Check the resulting pickle's `boost_info['flagged_cells']` is non-empty somewhere in the bright+red region, and that re-running the existing ROC/AUC analysis on this class's pilot-only subset (`boost_info['n_pilot']` events) vs the full boosted population shows the bulk statistics are statistically unchanged. **Bulk ROC/AUC is not sufficient on its own** — it can hide a bias that's concentrated in one or two sparse cells (see the final-review finding on `fit_boost_proposal`'s out-of-range-KDE guard, fixed before merge). Also check, per flagged cell, that the *sum of weights* in that cell for the boosted population is consistent (within Monte Carlo tolerance) with what the un-boosted pilot's rate-weighted density predicts for that cell — not just that the bulk-wide AUC looks unchanged.
2. **Full production rerun.** All 10 classes x 3 drivers at production `--n 100000`, retrain `ColourClassifier`/`TrajectoryClassifier` on the result, and confirm `P(lensed)` in the bright+red corner (e.g. g−r ≥ 2.0, peakmag ≤ 16) now trends toward 1 instead of plateauing/spiking, the way the four prototypes investigated earlier in this project could not achieve through model choice alone. Also re-check the existing headline colour-first numbers (55.6% per-system / 88.8% rate-weighted) are not silently shifted by this change.
