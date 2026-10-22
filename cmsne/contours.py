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
