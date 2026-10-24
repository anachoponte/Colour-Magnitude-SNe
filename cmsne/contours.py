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
from .colour_magnitude import weighted_quantile
from .trajectory import TRAJECTORY_GRID, _step_suffix


def robust_range(values, weights=None, quantiles=(0.01, 0.99)):
    """``(low, high)`` bounds for an axis, robust to rare extreme outliers.

    Rare, near-zero-probability draws (e.g. a lensed SN Ia at implausibly low
    redshift with near-maximal magnification -- physically near-impossible,
    which is exactly why :func:`cmsne.multicolour.rate_weight` assigns them a
    tiny weight) can sit many units away from where the real data density is,
    stretching a plain min/max range so far that most of a plotted mesh ends
    up showing near-empty extrapolation rather than real data (see issue #24).
    Quantiles are far less sensitive to a handful of such points; weighting
    them (when ``weights`` is given) suppresses their influence further,
    consistent with how little they should count everywhere else.

    :param values: per-event values (NaN entries are ignored).
    :param weights: per-event weights, or ``None`` for an unweighted quantile.
    :param quantiles: ``(low, high)`` quantiles in [0, 1].
    :return: ``(low, high)`` tuple of floats.
    """
    values = np.asarray(values, dtype=float)
    lo_q, hi_q = quantiles
    if weights is None:
        finite = values[np.isfinite(values)]
        return (float(np.percentile(finite, 100 * lo_q)),
                float(np.percentile(finite, 100 * hi_q)))
    return (weighted_quantile(values, weights, lo_q),
            weighted_quantile(values, weights, hi_q))


def posterior_surface(signal_events, background_events, colour_key, magnitude_key,
                      colour_range=None, magnitude_range=None, n=200,
                      signal_weights=None, background_weights=None):
    """Fit a 2-feature classifier and evaluate P(lensed) over a mesh.

    :param signal_events, background_events: event dicts exposing
        ``colour_key`` and ``magnitude_key`` (e.g. from
        :func:`cmsne.trajectory.event_trajectory`).
    :param colour_range, magnitude_range: ``(low, high)``; default to
        :func:`robust_range` (1st-99th percentile, rate-weighted if
        ``signal_weights``/``background_weights`` are given) over the pooled
        data, not the full min/max -- see :func:`robust_range`.
    :param signal_weights, background_weights: rate weights per event (see
        :func:`cmsne.multicolour.rate_weight`), used both to fit the
        classifier (matching :meth:`cmsne.classifier.ColourClassifier.fit`)
        and, when the range args are left as ``None``, to compute a
        rate-weighted default range.
    :return: ``(colour_grid, magnitude_grid, probability_grid, classifier)``,
        each grid an ``(n, n)`` array from ``np.meshgrid``.
    """
    clf = ColourClassifier(keys=[colour_key], include_peakmag=False, target_fpr=0.10)
    clf.magnitude_feature = magnitude_key
    clf.extra = (magnitude_key,)
    clf.fit(signal_events, background_events, signal_weights, background_weights)

    pooled = signal_events + background_events
    colours = np.array([e.get(colour_key, np.nan) for e in pooled], dtype=float)
    mags = np.array([e.get(magnitude_key, np.nan) for e in pooled], dtype=float)
    if colour_range is None or magnitude_range is None:
        if signal_weights is None and background_weights is None:
            pooled_weights = None
        else:
            sw = np.ones(len(signal_events)) if signal_weights is None else np.asarray(signal_weights, float)
            bw = np.ones(len(background_events)) if background_weights is None else np.asarray(background_weights, float)
            pooled_weights = np.r_[sw, bw]
        if colour_range is None:
            colour_range = robust_range(colours, pooled_weights)
        if magnitude_range is None:
            magnitude_range = robust_range(mags, pooled_weights)

    colour_grid, magnitude_grid = np.meshgrid(
        np.linspace(*colour_range, n), np.linspace(*magnitude_range, n))
    grid_events = [{colour_key: c, magnitude_key: m}
                  for c, m in zip(colour_grid.ravel(), magnitude_grid.ravel())]
    probability_grid = clf.probability(grid_events).reshape(colour_grid.shape)
    return colour_grid, magnitude_grid, probability_grid, clf


def evolving_contours(signal_trajectories, background_trajectories, pair_key,
                      grid=TRAJECTORY_GRID, n=200,
                      signal_weights=None, background_weights=None):
    """:func:`posterior_surface` at every Δt-since-trigger grid step.

    :param pair_key: adjacent-colour key stem, e.g. ``'gr'`` for g-r.
    :param signal_weights, background_weights: forwarded to every
        :func:`posterior_surface` call (see its docstring).
    :return: dict ``{day: (colour_grid, magnitude_grid, probability_grid, classifier)}``
        keyed by each ``day`` in ``grid``.
    """
    surfaces = {}
    for day in grid:
        suffix = _step_suffix(day)
        surfaces[day] = posterior_surface(
            signal_trajectories, background_trajectories,
            colour_key=f'{pair_key}{suffix}', magnitude_key=f'peakmag{suffix}', n=n,
            signal_weights=signal_weights, background_weights=background_weights)
    return surfaces
