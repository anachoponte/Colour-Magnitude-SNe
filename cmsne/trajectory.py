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

from .lsst import bands as LSST_BANDS
from .multicolour import (MultiColourGenerator, band_photometry, band_photometry_errors,
                          _colours, _colours_errors, ADJ_COLOURS, event_multicolour)
from .classifier import ColourClassifier

# Canonical Δt-since-trigger checkpoints [observer-days], reusing the exact
# values already validated in the waiting-time study (see
# docs/methods_and_results.md / colour-classifier-findings) so results stay
# comparable to existing findings.
TRAJECTORY_GRID = (0.0, 3.0, 14.0, 30.0, 60.0)


def _step_suffix(day):
    """Feature-key suffix for one grid step, e.g. ``3.0`` -> ``'_d3'``."""
    return f'_d{int(day)}'


def matching_band_for_pair(pair_key):
    """Companion magnitude band for a colour pair.

    By standard astronomical CMD convention (e.g. Mane et al. 2024/2025,
    Quimby et al. 2014, Arendse et al. 2025), a colour (c1 - c2) with
    wavelength_1 < wavelength_2 is paired with the redder band c2
    (e.g. 'gr' -> 'r', 'ri' -> 'i', 'iz' -> 'z', 'zy' -> 'y', 'ug' -> 'g').
    """
    return pair_key[-1]


def trajectory_band_magnitude_keys(band, grid=TRAJECTORY_GRID):
    """Per-step cumulative-best magnitude keys for a specific band,
    e.g. 'r' -> ('peakmag_r_d0', 'peakmag_r_d3', ...)."""
    b_name = band[-1]
    return tuple(f'peakmag_{b_name}{_step_suffix(day)}' for day in grid)


def resolve_magnitude_key(pair_key, day, mag_mode='matched', mag_band='auto', sample_event=None):
    """Determine the magnitude feature key and human-readable axis label.

    :param pair_key: colour pair stem, e.g. 'gr'.
    :param day: grid step day (float or int).
    :param mag_mode: 'matched' (default; per-band cumulative best),
                     'snapshot' (per-band single-epoch magnitude),
                     'peakmag' (global all-band cumulative best).
    :param mag_band: 'auto' (use redder band of pair) or explicit band ('u','g','r','i','z','y').
    :param sample_event: optional dict; if passed and candidate key is missing,
                         falls back to 'peakmag{suffix}'.
    :return: (key, label)
    """
    suffix = _step_suffix(day)
    b = matching_band_for_pair(pair_key) if (mag_band is None or mag_band == 'auto') else mag_band[-1]

    if mag_mode == 'matched':
        candidate = f'peakmag_{b}{suffix}'
        label = f'{b} peakmag (cumulative-best)'
    elif mag_mode == 'snapshot':
        candidate = f'{b}{suffix}'
        label = f'{b} magnitude (snapshot)'
    else:
        candidate = f'peakmag{suffix}'
        label = 'peakmag (brightest-observed)'

    if sample_event is not None and candidate not in sample_event:
        fallback = f'peakmag{suffix}'
        if fallback in sample_event:
            return fallback, 'peakmag (all-band fallback)'

    return candidate, label


def event_trajectory(lightcurve, grid=TRAJECTORY_GRID, window=5.0, min_detections=5):
    """N-point colour + cumulative-brightness trajectory for one light curve.

    At each ``Δt`` in ``grid``, records the ugrizy adjacent-band colours
    (NaN where a band was not observed within ``window`` of trigger+Δt) and
    the brightest magnitude seen in any band-photometry snapshot up to and
    including that Δt (not the eventual global peak -- using the global peak
    at an early step would leak information a live candidate could not yet
    have). Also records each colour's and each cumulative magnitude's
    measurement-uncertainty companion (see docs/superpowers/specs/
    2026-10-26-uncertainty-aware-noise-injection-design.md), mirroring
    :func:`cmsne.multicolour.event_multicolour`'s ``*_err`` fields --
    ``peakmag{suffix}_err`` is the error of whichever SINGLE observation
    currently holds the cumulative-best magnitude (tracked explicitly, since
    the cumulative best can come from an earlier step's snapshot).

    Also records per-band snapshot magnitudes (``'{b}{suffix}'``) and per-band
    cumulative-best magnitudes (``'peakmag_{b}{suffix}'``) along with their
    matching measurement-uncertainty companions (``'*_err'``), allowing
    band-matched CMD features (e.g. r vs g-r).

    :param min_detections: minimum number of detected epochs (finite magnitude,
        i.e. SNR > 5) required across all bands. Defaults to 5, matching the
        detection threshold in Ponte Pérez et al. (2025).
    :return: dict with keys ``'{colour}{suffix}'``/``'{colour}{suffix}_err'``,
        ``'peakmag{suffix}'``/``'peakmag{suffix}_err'``,
        ``'{b}{suffix}'``/``'{b}{suffix}_err'``, and
        ``'peakmag_{b}{suffix}'``/``'peakmag_{b}{suffix}_err'`` for every grid step,
        plus ``trigger_day``; or ``None`` if fewer than ``min_detections`` visits
        were detected.
    """
    days = np.asarray(lightcurve.obs_days, dtype=float)
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    det = np.isfinite(mag)
    if np.sum(det) < min_detections:
        return None

    trigger = float(days[det].min())
    feat = {'trigger_day': trigger}
    cumulative_best = np.inf
    cumulative_best_err = np.nan
    cumulative_best_band = {b[-1]: np.inf for b in LSST_BANDS}
    cumulative_best_band_err = {b[-1]: np.nan for b in LSST_BANDS}
    for day in grid:
        suffix = _step_suffix(day)
        m = band_photometry(lightcurve, trigger + day, window)
        err_m = band_photometry_errors(lightcurve, trigger + day, window)
        feat.update(_colours(m, suffix))
        feat.update(_colours_errors(err_m, suffix))

        finite_bands = [b for b, v in m.items() if np.isfinite(v)]
        if finite_bands:
            step_best_band = min(finite_bands, key=lambda b: m[b])
            if m[step_best_band] < cumulative_best:
                cumulative_best = m[step_best_band]
                cumulative_best_err = err_m[step_best_band]
        feat[f'peakmag{suffix}'] = cumulative_best if np.isfinite(cumulative_best) else np.nan
        feat[f'peakmag{suffix}_err'] = cumulative_best_err

        for b in LSST_BANDS:
            b_name = b[-1]
            feat[f'{b_name}{suffix}'] = m[b]
            feat[f'{b_name}{suffix}_err'] = err_m[b]
            if np.isfinite(m[b]) and m[b] < cumulative_best_band[b_name]:
                cumulative_best_band[b_name] = m[b]
                cumulative_best_band_err[b_name] = err_m[b]
            feat[f'peakmag_{b_name}{suffix}'] = (
                cumulative_best_band[b_name] if np.isfinite(cumulative_best_band[b_name]) else np.nan
            )
            feat[f'peakmag_{b_name}{suffix}_err'] = cumulative_best_band_err[b_name]

    return feat


# Full trajectory feature schema, for cmsne.trajectory.TrajectoryClassifier
# and cmsne.contours (every colour at every grid step; every cumulative
# peakmag at every grid step).
TRAJECTORY_COLOUR_KEYS = [f'{a[-1]}{b[-1]}{_step_suffix(day)}'
                           for day in TRAJECTORY_GRID for a, b in ADJ_COLOURS]
TRAJECTORY_MAGNITUDE_KEYS = tuple(f'peakmag{_step_suffix(day)}' for day in TRAJECTORY_GRID)


def reveal_up_to(event, k, grid=TRAJECTORY_GRID):
    """Copy of ``event`` with trajectory keys past grid step ``k`` set to NaN.

    Simulates having observed only the first ``k + 1`` grid steps (``k=0``
    means trigger only) -- used to score a trajectory event as it would have
    looked at an earlier point in its sequence, without re-simulating
    anything. Also NaNs out each hidden step's ``*_err`` companions -- a
    step that hasn't happened yet cannot leak its own measurement
    uncertainty either.

    :param k: last visible grid-step index (0-based).
    :return: a new dict; ``event`` itself is not modified."""
    n_steps = len(grid)
    cols_per_step = len(TRAJECTORY_COLOUR_KEYS) // n_steps
    e = dict(event)
    for step in range(k + 1, n_steps):
        for key in TRAJECTORY_COLOUR_KEYS[step * cols_per_step:(step + 1) * cols_per_step]:
            e[key] = np.nan
            e[f'{key}_err'] = np.nan
        e[TRAJECTORY_MAGNITUDE_KEYS[step]] = np.nan
        e[f'{TRAJECTORY_MAGNITUDE_KEYS[step]}_err'] = np.nan
        step_suffix = _step_suffix(grid[step])
        for b in LSST_BANDS:
            b_name = b[-1]
            for key in (f'{b_name}{step_suffix}', f'peakmag_{b_name}{step_suffix}'):
                if key in e:
                    e[key] = np.nan
                    e[f'{key}_err'] = np.nan
    return e


class TrajectoryGenerator(MultiColourGenerator):
    """Like :class:`cmsne.multicolour.MultiColourGenerator`, but records the
    full N-point trajectory (:func:`event_trajectory`) instead of a fixed
    two-point colour vector.

    Identical ``generate_one``/``generate_many`` API (inherited unchanged) --
    only the per-event feature extraction differs.
    """

    def __init__(self, source='salt3', grid=TRAJECTORY_GRID, min_detections=5):
        super().__init__(source, min_detections=min_detections)
        self.grid = grid

    def _event_features(self, lc, evolution_days, window, min_detections=5):
        return event_trajectory(lc, self.grid, window, min_detections=min_detections)


def event_dual_schema(lightcurve, evolution_days=15.0, grid=TRAJECTORY_GRID, window=5.0, min_detections=5):
    """Both the static two-point schema (:func:`cmsne.multicolour.event_multicolour`)\
    and the full N-point trajectory schema (:func:`event_trajectory`) for the
    same light curve.

    Closes the gap flagged in PR #14: no generator previously produced both
    schemas on the same simulated event, so
    :func:`cmsne.comparison.compare_classifiers` could only be exercised on a
    hand-built synthetic test fixture. The two schemas share exactly one key,
    ``trigger_day``, computed identically by both from the same light curve
    (``trigger = days[detected].min()``), so merging the two dicts is safe --
    no other key collides.

    :return: merged dict, or ``None`` if either half found fewer than
        ``min_detections`` visits.
    """
    static_feat = event_multicolour(lightcurve, evolution_days, window, min_detections=min_detections)
    traj_feat = event_trajectory(lightcurve, grid, window, min_detections=min_detections)
    if static_feat is None or traj_feat is None:
        return None
    return {**static_feat, **traj_feat}


class DualSchemaGenerator(MultiColourGenerator):
    """Like :class:`cmsne.multicolour.MultiColourGenerator`, but records BOTH
    the static two-point schema and the full N-point trajectory schema
    (:func:`event_dual_schema`) for every event -- the production generator
    needed to run :func:`cmsne.comparison.compare_classifiers` on real,
    simulated-at-scale data instead of only a synthetic test fixture.

    Identical ``generate_one``/``generate_many`` API (inherited unchanged) --
    only the per-event feature extraction differs.
    """

    def __init__(self, source='salt3', grid=TRAJECTORY_GRID, min_detections=5):
        super().__init__(source, min_detections=min_detections)
        self.grid = grid

    def _event_features(self, lc, evolution_days, window, min_detections=5):
        return event_dual_schema(lc, evolution_days, self.grid, window, min_detections=min_detections)


class TrajectoryClassifier(ColourClassifier):
    """Like :class:`cmsne.classifier.ColourClassifier`, but trained on the full
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
        # peakmag at each grid step is a CUMULATIVE BEST (running minimum) over
        # time, so noise-injection replicas must stay non-increasing across
        # TRAJECTORY_MAGNITUDE_KEYS' chronological order too -- independently
        # perturbing each step's column could otherwise produce a physically
        # impossible "cumulative best got fainter" feature vector (see
        # docs/superpowers/specs/2026-10-26-uncertainty-aware-noise-injection-design.md).
        # Group any per-band cumulative bests if present in extra too.
        monotonic = [list(TRAJECTORY_MAGNITUDE_KEYS)]
        for b in LSST_BANDS:
            b_keys = [k for k in self.extra if k.startswith(f'peakmag_{b[-1]}_')]
            if len(b_keys) > 1:
                monotonic.append(b_keys)
        self.monotonic_column_groups = monotonic

    def roc_curve_by_step(self, signal_events, background_events,
                          signal_weights=None, background_weights=None,
                          grid=TRAJECTORY_GRID):
        """:meth:`cmsne.classifier.ColourClassifier.roc_curve` at every
        Δt-since-trigger grid step.

        Reveals each event's trajectory feature vector only up to grid step
        ``k`` (:func:`reveal_up_to`) before scoring, so the resulting ROC
        curves show how the recovery/contamination trade-off changes as more
        of the trajectory becomes available -- not just the final, fully-
        revealed snapshot that :meth:`roc_curve` alone would give.

        :return: dict ``{day: (fpr, tpr, thresholds)}`` keyed by each ``day``
            in ``grid``, in the same order as ``grid``.
        """
        out = {}
        for k, day in enumerate(grid):
            sig_partial = [reveal_up_to(e, k, grid) for e in signal_events]
            bg_partial = [reveal_up_to(e, k, grid) for e in background_events]
            out[day] = self.roc_curve(sig_partial, bg_partial, signal_weights, background_weights)
        return out
