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

from .multicolour import MultiColourGenerator, band_photometry, _colours, ADJ_COLOURS, event_multicolour
from .classifier import ColourClassifier

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


def reveal_up_to(event, k, grid=TRAJECTORY_GRID):
    """Copy of ``event`` with trajectory keys past grid step ``k`` set to NaN.

    Simulates having observed only the first ``k + 1`` grid steps (``k=0``
    means trigger only) -- used to score a trajectory event as it would have
    looked at an earlier point in its sequence, without re-simulating
    anything.

    :param k: last visible grid-step index (0-based).
    :return: a new dict; ``event`` itself is not modified.
    """
    n_steps = len(grid)
    cols_per_step = len(TRAJECTORY_COLOUR_KEYS) // n_steps
    e = dict(event)
    for step in range(k + 1, n_steps):
        for key in TRAJECTORY_COLOUR_KEYS[step * cols_per_step:(step + 1) * cols_per_step]:
            e[key] = np.nan
        e[TRAJECTORY_MAGNITUDE_KEYS[step]] = np.nan
    return e


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


def event_dual_schema(lightcurve, evolution_days=15.0, grid=TRAJECTORY_GRID, window=5.0):
    """Both the static two-point schema (:func:`cmsne.multicolour.event_multicolour`)
    and the full N-point trajectory schema (:func:`event_trajectory`) for the
    same light curve.

    Closes the gap flagged in PR #14: no generator previously produced both
    schemas on the same simulated event, so
    :func:`cmsne.comparison.compare_classifiers` could only be exercised on a
    hand-built synthetic test fixture. The two schemas share exactly one key,
    ``trigger_day``, computed identically by both from the same light curve
    (``trigger = days[detected].min()``), so merging the two dicts is safe --
    no other key collides.

    :return: merged dict, or ``None`` if either half found nothing detected.
    """
    static_feat = event_multicolour(lightcurve, evolution_days, window)
    traj_feat = event_trajectory(lightcurve, grid, window)
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

    def __init__(self, source='salt3', grid=TRAJECTORY_GRID):
        super().__init__(source)
        self.grid = grid

    def _event_features(self, lc, evolution_days, window):
        return event_dual_schema(lc, evolution_days, self.grid, window)


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
