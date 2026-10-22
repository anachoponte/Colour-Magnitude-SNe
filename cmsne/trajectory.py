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
