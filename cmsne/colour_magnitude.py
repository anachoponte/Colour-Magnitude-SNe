"""Colour-magnitude analysis helpers.

Utilities to pull matched observations out of a :class:`~cmsne.lightcurve.Lightcurve`,
weight them by the intrinsic (and lensing) SN rates, resample a weighted
population, and fit the decision boundary separating lensed from unlensed events
in colour-magnitude space.

Frame convention
----------------
:func:`individual_observations` converts observation days to the **rest frame**
by dividing by ``(1 + z)``. Every epoch derived downstream (the colour-magnitude
identification epoch, the image-multiplicity epoch) must therefore also be
rest-frame, including the intrinsic light-curve peak day it is measured against.
Mixing the two frames is a bug that has been introduced here more than once.
"""

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18, WMAP9

from .lsst import mask
from .sn_rates import rate_grid, cc_rate_grid


def obs_to_mags(day_band_1, day_band_2, mag_band_1, mag_band_2,
                day_low=-30.0, day_high=150.0, max_separation=3.0,
                select='first'):
    """Find a matched pair of observations in two bands and return their magnitudes.

    Walks the band-1 epochs, pairs each with the *nearest in time* band-2 epoch,
    and keeps pairs where both epochs are inside ``[day_low, day_high]``, are
    separated by no more than ``max_separation`` days, and have finite magnitudes.

    Matching on time matters: pairing the two bands by array position (which an
    earlier version of this function did) pairs magnitudes measured weeks apart
    and produces a colour that corresponds to no single epoch.

    :param day_band_1: rest-frame observation days in band 1
    :param day_band_2: rest-frame observation days in band 2
    :param mag_band_1: observed magnitudes in band 1 (same order as ``day_band_1``)
    :param mag_band_2: observed magnitudes in band 2 (same order as ``day_band_2``)
    :param day_low: earliest rest-frame day to consider
    :param day_high: latest rest-frame day to consider
    :param max_separation: maximum allowed ``|t1 - t2|`` for a pair [days]
    :param select: ``'first'`` returns the earliest valid pair; ``'brightest'``
        returns the pair with the faintest-to-brightest minimum band-1 magnitude.
        ``'first'`` reproduces the historical behaviour -- note that it means the
        returned magnitude is the *first detected* epoch, not the light-curve peak,
        so treating it as a peak magnitude overstates what has been measured.
    :return: ``(band_2_mag, band_1_mag, band_1_day, band_2_day)``, or a tuple of
        four NaNs if no valid pair exists.
    """
    no_match = (np.nan, np.nan, np.nan, np.nan)

    if len(day_band_1) == 0 or len(day_band_2) == 0:
        return no_match

    day_band_1_arr = np.asarray(day_band_1, dtype=float)
    day_band_2_arr = np.asarray(day_band_2, dtype=float)
    mag_band_1_arr = np.asarray(mag_band_1, dtype=float)
    mag_band_2_arr = np.asarray(mag_band_2, dtype=float)

    pairs = []
    for idx1, day_1 in enumerate(day_band_1_arr):
        if not (day_low < day_1 < day_high):
            continue

        idx2 = int(np.argmin(np.abs(day_band_2_arr - day_1)))
        day_2 = day_band_2_arr[idx2]

        if not (day_low < day_2 < day_high):
            continue
        if abs(day_1 - day_2) > max_separation:
            continue
        if not (np.isfinite(mag_band_1_arr[idx1]) and np.isfinite(mag_band_2_arr[idx2])):
            continue

        pairs.append((mag_band_2_arr[idx2], mag_band_1_arr[idx1], day_1, day_2))

    if not pairs:
        return no_match

    if select == 'brightest':
        return min(pairs, key=lambda p: p[1])
    if select == 'first':
        # Earliest epoch at which BOTH bands have been observed, i.e. the earliest
        # the colour can actually be measured. `pairs` is built in band-1 epoch
        # order, so ties fall back to the first-listed pair (matching the old
        # `pairs[0]` behaviour).
        return min(pairs, key=lambda p: max(p[2], p[3]))
    raise ValueError("select must be 'first' or 'brightest', got %r" % (select,))


def individual_observations(band, lightcurve_obj, redshift):
    """Pull the observations taken in ``band`` out of a light curve.

    Days are converted to the **rest frame** (divided by ``1 + z``); see the
    module docstring.

    :param band: sncosmo LSST band name, e.g. ``'lsstg'``
    :param lightcurve_obj: :class:`~cmsne.lightcurve.Lightcurve`
    :param redshift: source redshift
    :return: ``(index, mag, day)`` lists for the matching observations
    """
    index = []
    day = []
    mag = []
    # band[4:] strips the 'lsst' prefix; OpSim filter names are 'g', 'r', ...
    for x in range(len(lightcurve_obj.obs_filters)):
        if lightcurve_obj.obs_filters[x] == band[4:]:
            index.append(x)
            day.append(lightcurve_obj.obs_days[x] / (1 + redshift))
            mag.append(lightcurve_obj.obs_mag[x])
    return index, mag, day


def ul_weight(redshift_val):
    """Intrinsic (unlensed) SN Ia rate weight at ``redshift_val``."""
    redshift, nSN1A = rate_grid()
    index = np.argmin(np.abs(redshift - redshift_val))
    return nSN1A[index]


def cc_ul_weight(redshift_val):
    """Intrinsic (unlensed) core-collapse SN rate weight at ``redshift_val``.

    Uses the star-formation-tracking core-collapse rate
    (:func:`cmsne.sn_rates.cc_rate_grid`), which peaks near z ~ 1.9 -- distinct from
    the delay-time-distribution SN Ia rate (peak z ~ 1). An earlier version reused the
    Ia grid here, which gave the contaminant population the wrong redshift evolution.
    """
    redshift, nCC = cc_rate_grid()
    index = np.argmin(np.abs(redshift - redshift_val))
    return nCC[index]


def cc_l_weight(redshift_val, magnification):
    """Lensed core-collapse SN weight: intrinsic CC rate times the strong-lensing optical depth.

    The intrinsic rate tracks the star-formation history
    (:func:`cmsne.sn_rates.cc_rate_grid`). The ``31e9 pc`` (31 Gpc) scale must match
    the one in :func:`cmsne.sn_rates.prob_` and :func:`l_weight`; they are the same
    normalisation.
    """
    redshift, nCC = cc_rate_grid()
    index = np.argmin(np.abs(redshift - redshift_val))
    z_weight = nCC[index]
    redshift_units = redshift_val * cu.redshift
    distance_unit = redshift_units.to(u.pc, cu.redshift_distance(Planck18, kind="comoving"))
    distance = distance_unit.value
    u_weight = 0.5 * ((distance / 31e9) ** 3) * ((magnification) ** -2)
    return z_weight * u_weight


def l_weight(redshift_val, magnification):
    """Lensed SN Ia weight: intrinsic rate times the strong-lensing optical depth.

    The ``31e9 pc`` (31 Gpc) scale must match the one in
    :func:`cmsne.sn_rates.prob_`; they are the same normalisation.
    """
    redshift, nSN1A = rate_grid()
    index = np.argmin(np.abs(redshift - redshift_val))
    z_weight = nSN1A[index]
    redshift_units = redshift_val * cu.redshift
    distance_unit = redshift_units.to(u.pc, cu.redshift_distance(Planck18, kind="comoving"))
    distance = distance_unit.value
    u_weight = 0.5 * ((distance / 31e9) ** 3) * ((magnification) ** -2)
    total = z_weight * u_weight
    return total


def _angular_diameter_distance_z1z2(cosmo, z1, z2):
    """Angular diameter distance between two redshifts, across astropy versions.

    astropy >= 8 exposes this as the two-argument ``angular_diameter_distance``
    and deprecates ``angular_diameter_distance_z1z2``; astropy < 8 only has the
    latter. Try the modern spelling and fall back, so the package runs unchanged
    locally and on Colab.
    """
    try:
        return cosmo.angular_diameter_distance(z1, z2)
    except TypeError:
        return cosmo.angular_diameter_distance_z1z2(z1, z2)


def time_delay(mu_p, z_s, z_l, lens=2, cosmo=WMAP9):
    """Characteristic lensing time delay between two images [days].

    Scaling relation normalised to a fiducial system, parametrised by lens class.
    ``lens`` selects the deflector scale via the Einstein radius, ellipticity
    ``eta_e`` and convergence ``kappa_e``:

    * ``lens=0`` galaxy-scale  (Einstein radius 1", kappa 0.5)
    * ``lens=1`` group-scale   (Einstein radius 3", kappa 0.65)
    * ``lens=2`` cluster-scale (Einstein radius 10", kappa 0.8) -- the default,
      which reproduces the original single-lens prefactor exactly.

    .. warning::
       The numerical prefactor and the fixed ``0.5`` / ``0.8`` lens parameters are
       carried over unchanged from the original notebook and are **not derived
       here**. In particular the ``(mu_p / 4) ** -3`` magnification scaling differs
       from the ``-1`` power in the superseded version of this expression, and that
       choice sets the shape of every time-delay distribution downstream. Confirm
       it against the reference you are citing before using these numbers.

    :param mu_p: magnification of the image
    :param z_s: source redshift
    :param z_l: lens redshift
    :param lens: lens class index (0 galaxy, 1 group, 2 cluster); default 2
    :param cosmo: astropy cosmology used for the angular diameter distances
    :return: time delay in days
    """
    # Per-lens-class deflector parameters. lens=2 (cluster) reproduces the
    # original hard-coded (10**2, 0.5, 0.8) prefactor bit-for-bit.
    einstein_radius = [1.0, 3.0, 10.0]   # arcsec
    eta_e = [1.0, 0.75, 0.5]             # ellipticity term
    kappa_e = [0.5, 0.65, 0.8]           # convergence

    Dl = cosmo.angular_diameter_distance(z_l)
    Ds = cosmo.angular_diameter_distance(z_s)
    DsDl = _angular_diameter_distance_z1z2(cosmo, z_l, z_s)

    D = (Dl * Ds) / DsDl
    Dt = (3.9 * u.day * (einstein_radius[lens] ** 2) * ((mu_p / 4) ** (-3))
          * ((eta_e[lens]) ** (-2)) * ((kappa_e[lens] / 0.5) ** (-2))
          * (D / (3.3 * u.Gpc)))
    return Dt.to(u.day).value


def time_delay_fourth_lens(mu_p, z_s, z_l, cosmo=WMAP9):
    """Alternative galaxy-scale time delay with a shallower magnification scaling.

    Unlike :func:`time_delay` (which scales as ``(mu/4)**-3``), this model scales
    as ``(mu/4)**-1``. Used as the ``galaxy_2`` lens class.

    .. warning::
       The ``92 day`` prefactor and the ``(mu/4)**-1`` scaling are carried over
       from the summer-project code and are **not derived here**; they need
       independent physical verification (see the reconciliation report).
    """
    Dl = cosmo.angular_diameter_distance(z_l)
    Ds = cosmo.angular_diameter_distance(z_s)
    DsDl = _angular_diameter_distance_z1z2(cosmo, z_l, z_s)
    D = (Dl * Ds) / DsDl
    Dt = (92 * u.day * ((mu_p / 4) ** (-1)) * (D / (3.3 * u.Gpc)))
    return Dt.to(u.day).value


def _weighted_choice(n, q, normalized_weights):
    """Draw ``q`` indices from ``n`` items, **with** replacement, by weight.

    Replacement is not optional here. ``np.random.choice(..., replace=False, p=w)``
    is *sequential* weighted sampling: each draw removes an item and the remaining
    probabilities are renormalised, so heavy items are exhausted early and the
    sample drifts toward the unweighted population as ``q`` approaches ``n``. With
    the strongly skewed rate weights used here that bias is large well before then
    -- drawing 6% of a population already shifts a weight-correlated mean by more
    than half a magnitude, and 20% roughly triples the error.

    Sampling with replacement is the unbiased estimator at every ``q``. For a plain
    statistic (a mean, a fraction, a quantile) prefer :func:`weighted_fraction` or
    :func:`weighted_quantile`, which are exact and need no sampling at all.
    """
    return np.random.choice(n, size=q, replace=True, p=normalized_weights)


def weighted_fraction(condition, weights):
    """Exact rate-weighted fraction of events satisfying ``condition``.

    Equivalent to resampling by weight and taking the fraction, but exact and
    free of Monte-Carlo noise.

    :param condition: boolean array over events
    :param weights: per-event weights
    :return: weighted fraction in [0, 1], or NaN if no weight is available
    """
    condition = np.asarray(condition, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    return float(np.average(condition[good], weights=weights[good]))


def weighted_quantile(values, weights, q=0.5):
    """Exact rate-weighted quantile (median by default).

    :param values: per-event values
    :param weights: per-event weights
    :param q: quantile in [0, 1]
    :return: the weighted quantile, or NaN if nothing usable
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    v, w = values[good], weights[good]
    order = np.argsort(v)
    v, w = v[order], w[order]
    # Midpoint convention: the cumulative weight at the centre of each item.
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def cutting(mags1, mags2, weights):
    """Drop entries where ``mags1`` is NaN, keeping the three arrays aligned."""
    cut_weights = []
    cut_mags1 = []
    cut_mags2 = []
    for i in range(len(mags1)):
        if np.isnan(mags1[i]):
            continue
        else:
            cut_weights.append(weights[i])
            cut_mags1.append(mags1[i])
            cut_mags2.append(mags2[i])
    cut_weights = np.array(cut_weights)
    cut_mags1 = np.array(cut_mags1)
    cut_mags2 = np.array(cut_mags2)
    return cut_weights, cut_mags1, cut_mags2


def detection(mags, band, min_detections=2):
    """Blank out a light curve that never reaches the single-visit depth.

    :param mags: array of magnitudes sampled along the model light curve
    :param band: sncosmo LSST band name, used to look up the depth
    :param min_detections: how many of the sub-sampled epochs must be brighter
        than the limiting magnitude for the event to count as detected
    :return: ``mags`` unchanged if detected, else an all-NaN array of the same shape
    """
    mags = np.asarray(mags, dtype=float)
    mag_condition = mags[0::10]

    if mag_condition.size == 0:
        return np.full_like(mags, np.nan)

    num_detections = np.count_nonzero(mag_condition < mask(band))

    if num_detections >= min_detections:
        return mags
    return np.full_like(mags, np.nan)


def weighted_vals(weights, mags1, mags2, q):
    """Resample ``q`` events from a population, weighted by rate.

    :param weights: per-event weights (same length as ``mags1``)
    :param mags1: per-event band-1 magnitudes (scalar or array per event)
    :param mags2: per-event band-2 magnitudes
    :param q: number of events to draw
    :return: ``(peak_mag1, colour, normalized_weights)``. The first two are empty
        lists when no event has finite magnitudes in both bands.
    """
    new_peak_mag1 = []
    new_color = []
    plottable_data = []
    valid_indices = []

    weights = np.asarray(weights, dtype=float)

    for i in range(len(mags1)):
        mags1_ = mags1[i]
        mags2_ = mags2[i]

        if np.isfinite(np.min(mags1_)) and np.isfinite(np.min(mags2_)):
            plottable_data.append({'peak_mag1': np.min(mags1_), 'color': np.min(mags2_) - np.min(mags1_)})
            valid_indices.append(i)

    # Nothing survived the finiteness cut: np.random.choice would raise on an
    # empty population, and the weight normalisation would divide by zero.
    if not plottable_data:
        return [], [], np.array([])

    plottable_weights = weights[valid_indices]
    sum_weights = np.sum(plottable_weights)
    if not np.isfinite(sum_weights) or sum_weights <= 0:
        return [], [], np.array([])
    normalized_weights = plottable_weights / sum_weights

    chosen_indices = _weighted_choice(len(plottable_data), q, normalized_weights)

    for j in chosen_indices:
        new_peak_mag1.append(plottable_data[j]['peak_mag1'])
        new_color.append(plottable_data[j]['color'])

    return new_peak_mag1, new_color, normalized_weights


def modified_weighted_vals(weights, time_delays, CM_times_rel_peak, CM_times_rel_first,
                           magnifications, redshift, l_colors, ul_colors,
                           l_peak_mags, ul_peak_mags, q):
    """Rate-weighted resampling of the time-delay / epoch / magnification arrays.

    All input sequences must be **index-aligned and the same length** -- they are
    indexed with the same positions. Filtering them independently (e.g. one list
    comprehension per array, each dropping its own NaNs) silently misaligns them
    and pairs each time delay with another event's weight.

    Resampling is done **with replacement** by weight (:func:`_weighted_choice`),
    which reproduces the weighted distribution without bias at any draw size.

    :return: ``(time_delays, CM_times_rel_peak, CM_times_rel_first, magnifications,
        redshift, l_colors, ul_colors, l_peak_mags, ul_peak_mags,
        normalized_weights)`` resampled to ``q`` entries.
    """
    lengths = {len(weights), len(time_delays), len(CM_times_rel_peak),
               len(CM_times_rel_first), len(magnifications), len(redshift),
               len(l_colors), len(ul_colors), len(l_peak_mags), len(ul_peak_mags)}
    if len(lengths) != 1:
        raise ValueError(
            "modified_weighted_vals inputs must be index-aligned and equal length, "
            "got lengths %s" % sorted(lengths))

    empty = ([], [], [], [], [], [], [], [], [], np.array([]))

    new_time_delays = []
    new_CM_times_rel_peak = []
    new_CM_times_rel_first = []
    new_magnifications = []
    new_redshift = []
    new_l_colors = []
    new_ul_colors = []
    new_l_peak_mags = []
    new_ul_peak_mags = []
    plottable_data = []
    valid_indices = []

    weights = np.asarray(weights, dtype=float)

    for i in range(len(time_delays)):
        value_ = time_delays[i]

        if np.isfinite(value_):
            plottable_data.append({'value': value_})
            valid_indices.append(i)

    if not plottable_data:
        return empty

    plottable_weights = weights[valid_indices]
    sum_weights = np.sum(plottable_weights)
    if not np.isfinite(sum_weights) or sum_weights <= 0:
        return empty
    normalized_weights = plottable_weights / sum_weights

    chosen_indices = _weighted_choice(len(plottable_data), q, normalized_weights)

    for j in chosen_indices:
        vi = valid_indices[j]
        new_time_delays.append(plottable_data[j]['value'])
        new_CM_times_rel_peak.append(CM_times_rel_peak[vi])
        new_CM_times_rel_first.append(CM_times_rel_first[vi])
        new_magnifications.append(magnifications[vi])
        new_redshift.append(redshift[vi])
        new_l_colors.append(l_colors[vi])
        new_ul_colors.append(ul_colors[vi])
        new_l_peak_mags.append(l_peak_mags[vi])
        new_ul_peak_mags.append(ul_peak_mags[vi])

    return (new_time_delays, new_CM_times_rel_peak, new_CM_times_rel_first,
            new_magnifications, new_redshift, new_l_colors, new_ul_colors,
            new_l_peak_mags, new_ul_peak_mags, normalized_weights)


def population_arrays(SNe, keys):
    """Stack per-event dictionary entries into arrays.

    :param SNe: list of event dictionaries from a population generator
    :param keys: iterable of dictionary keys to extract
    :return: dict mapping each key to an array over events. Missing keys yield NaN
        so that generators returning different fields can share this helper.
    """
    return {key: np.array([sne.get(key, np.nan) for sne in SNe]) for key in keys}


def full_creation(SNe, p):
    """Rate-weighted colour-magnitude samples for one population.

    :param SNe: population from a generator exposing ``det_mags{1,2}_{ul,l}``
    :param p: number of events to draw
    :return: ``(mag_ul, colour_ul, weights_ul, mag_l, colour_l, weights_l)``
    """
    arrays = population_arrays(SNe, ['weights_ul', 'weights_l', 'det_mags1_ul',
                                     'det_mags2_ul', 'det_mags1_l', 'det_mags2_l'])

    det_mag_ul1, det_mag_ul_color1, norm_weight_ul2 = weighted_vals(
        arrays['weights_ul'], arrays['det_mags1_ul'], arrays['det_mags2_ul'], p)
    det_mag_l1, det_mag_l_color1, norm_weight_l2 = weighted_vals(
        arrays['weights_l'], arrays['det_mags1_l'], arrays['det_mags2_l'], p)

    return (det_mag_ul1, det_mag_ul_color1, norm_weight_ul2,
            det_mag_l1, det_mag_l_color1, norm_weight_l2)


def combine_populations(populations, p):
    """Pool several populations (SN Ia plus contaminants) into one CM sample.

    Replaces the fixed five-population helper this used to be, so adding or
    removing a contaminant class no longer means editing the function body.

    :param populations: iterable of populations, each as returned by a generator's
        ``generate_many``
    :param p: number of events to draw per population
    :return: ``(all_colour_l, all_colour_ul, all_peak_mag_l, all_peak_mag_ul)``
    """
    all_colour_l, all_colour_ul = [], []
    all_peak_mag_l, all_peak_mag_ul = [], []

    for SNe in populations:
        (mag_ul, colour_ul, _, mag_l, colour_l, _) = full_creation(SNe, p)
        all_peak_mag_ul.extend(mag_ul)
        all_colour_ul.extend(colour_ul)
        all_peak_mag_l.extend(mag_l)
        all_colour_l.extend(colour_l)

    return all_colour_l, all_colour_ul, all_peak_mag_l, all_peak_mag_ul


def exponential_regression(weight_mag_x_ul, weight_mag_x_l,
                           weight_mag_colour_ul, weight_mag_colour_l,
                           verbose=False):
    """Fit a logistic classifier and return its decision boundary as a straight line.

    :param weight_mag_x_ul: unlensed peak magnitudes
    :param weight_mag_x_l: lensed peak magnitudes
    :param weight_mag_colour_ul: unlensed colours
    :param weight_mag_colour_l: lensed colours
    :param verbose: print the recovered detection rate and false-positive rate
    :return: ``(x_arr, y_range, m, b)`` -- the boundary sampled over the data range,
        plus its gradient and intercept in colour-magnitude space.
    """
    from sklearn.linear_model import LogisticRegression

    X_blue = np.column_stack((weight_mag_x_ul, weight_mag_colour_ul))
    X_red = np.column_stack((weight_mag_x_l, weight_mag_colour_l))

    X = np.vstack([X_blue, X_red])
    y = np.hstack([np.zeros(len(X_blue)), np.ones(len(X_red))])

    clf = LogisticRegression(class_weight={0: 1, 1: 1})  # Dictates the weighting on false positives or detection rate
    clf.fit(X, y)

    m = -clf.coef_[0, 0] / clf.coef_[0, 1]
    b = -clf.intercept_[0] / clf.coef_[0, 1]

    if verbose:
        y_pred = clf.predict(X)
        false_positives = np.sum((y_pred == 1) & (y == 0))
        true_positives = np.sum((y_pred == 1) & (y == 1))
        print(f"detection rate = {true_positives / len(X_red):.3f}, "
              f"false-positive rate = {false_positives / len(X_blue):.3f}")

    # Create a range of x-values for plotting the regression line
    min_x = min(np.min(weight_mag_x_ul), np.min(weight_mag_x_l))
    max_x = max(np.max(weight_mag_x_ul), np.max(weight_mag_x_l))
    x_arr = np.linspace(min_x, max_x, 1000)

    # Transform back to the original y-space for plotting
    y_range = (m * x_arr + b)
    return x_arr, y_range, m, b


def success_rate(peak_mags, colours, m, b=None):
    """Fraction of events lying above the decision boundary.

    Each event is compared against the boundary evaluated at **its own** peak
    magnitude. Evaluating the boundary on an arbitrary evenly-spaced grid instead
    compares each event to an unrelated magnitude and yields a meaningless rate.

    The boundary can be given either as a straight line (``m`` gradient, ``b``
    intercept, from :func:`exponential_regression`) or as a callable
    ``fitted(x)`` (from :func:`exponential_regression_curvefit`), passed as ``m``
    with ``b`` left as ``None``.

    :param peak_mags: per-event peak magnitudes (the boundary's x axis)
    :param colours: per-event colours (the boundary's y axis)
    :param m: boundary gradient, or a callable boundary ``fitted(x)``
    :param b: boundary intercept (omit when ``m`` is a callable)
    :return: fraction of events above the boundary, or NaN for an empty sample
    """
    peak_mags = np.asarray(peak_mags, dtype=float)
    colours = np.asarray(colours, dtype=float)

    finite = np.isfinite(peak_mags) & np.isfinite(colours)
    if not np.any(finite):
        return np.nan

    if callable(m):
        boundary = np.asarray(m(peak_mags[finite]), dtype=float)
    else:
        boundary = m * peak_mags[finite] + b

    above = colours[finite] > boundary
    return float(np.count_nonzero(above)) / int(np.count_nonzero(finite))


def _robust_exponential_fit(x, y, weights=None, maxfev=40000):
    """
    Shared helper: fits y = a*exp(b*(x - x0)) + c robustly.
    - Centers x on its own mean (x0) so exp(b*x) can't overflow just because x
      itself is a large number (e.g. magnitudes ~15-25) -- b then only has to
      describe the actual curvature, not compensate for the offset.
    - Derives a real starting guess from the data (linear regression in log
      space) instead of a blind [1.0, 0.1, median] guess, which is what was
      causing curve_fit to wander into overflow territory and never converge.
    - Falls back to a smoothed monotonic interpolation if the exponential form
      genuinely doesn't fit, rather than crashing the whole pipeline.
    """
    from scipy.optimize import curve_fit
    from scipy.interpolate import UnivariateSpline

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0 = np.mean(x)
    xs = x - x0

    def model(xs_, a, b, c):
        return a * np.exp(np.clip(b * xs_, -50, 50)) + c

    # Data-driven initial guess: put a floor under the data, then fit a line to
    # log(y - floor) vs xs to get initial (a, b).
    c0 = np.min(y) - 0.1 * (np.ptp(y) + 1e-6)
    shifted = y - c0
    shifted = np.clip(shifted, 1e-6, None)
    b0, log_a0 = np.polyfit(xs, np.log(shifted), 1)
    a0 = np.exp(log_a0)
    p0 = [a0, b0, c0]

    try:
        sigma = 1 / weights if weights is not None else None
        popt, _ = curve_fit(model, xs, y, p0=p0, sigma=sigma,
                            bounds=([-1e4, -20, -1e4], [1e4, 20, 1e4]),
                            maxfev=maxfev)

        def fitted(x_query):
            return model(np.asarray(x_query, dtype=float) - x0, *popt)

        params = (popt[0], popt[1], popt[2], x0)
        return fitted, params, True
    except RuntimeError as e:
        print(f"Exponential fit did not converge ({e}); falling back to a smoothed spline.")
        order = np.argsort(x)
        spline = UnivariateSpline(x[order], y[order], k=min(3, len(x) - 1), s=len(x))

        def fitted(x_query):
            return spline(np.asarray(x_query, dtype=float))

        return fitted, None, False


def exponential_regression_curvefit(weight_mag_x_ul, weight_mag_x_l,
                                    weight_mag_colour_ul, weight_mag_colour_l,
                                    n_bins=15, verbose=False):
    """
    A more rigorous alternative: instead of deriving an 'exponential' shape from a
    linear classifier boundary, this bins the data along x, estimates the colour
    value that best separates the two classes in each bin, and fits a true
    nonlinear exponential model via _robust_exponential_fit (real least-squares
    regression, with a data-driven starting guess so it won't blow up like a
    blind p0 guess can).

    :return: ``(x_arr, y_arr)`` sampling the fitted boundary, or ``(None, None)``
        if no magnitude bin contained both a lensed and an unlensed event.
    """
    x_ul = np.asarray(weight_mag_x_ul)
    x_l = np.asarray(weight_mag_x_l)
    c_ul = np.asarray(weight_mag_colour_ul)
    c_l = np.asarray(weight_mag_colour_l)

    x_all = np.concatenate([x_ul, x_l])
    bin_edges = np.linspace(x_all.min(), x_all.max(), n_bins + 1)
    bin_centers, boundary_colour = [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        ul_in_bin = c_ul[(x_ul >= lo) & (x_ul < hi)]
        l_in_bin = c_l[(x_l >= lo) & (x_l < hi)]
        if len(ul_in_bin) == 0 or len(l_in_bin) == 0:
            continue
        boundary = 0.5 * (np.median(ul_in_bin) + np.median(l_in_bin))
        bin_centers.append(0.5 * (lo + hi))
        boundary_colour.append(boundary)

    bin_centers = np.array(bin_centers)
    boundary_colour = np.array(boundary_colour)

    # A spline/curve fit needs at least a couple of points; with none the caller
    # gets an explicit "no boundary" rather than an exception from polyfit.
    if len(bin_centers) < 2:
        return None, None, None

    fitted, params, converged = _robust_exponential_fit(bin_centers, boundary_colour)
    if verbose and params is not None:
        a, b, c, x0 = params
        print(f"curve_fit exponential params: a={a:.4g}, b={b:.4g}, c={c:.4g} (centered at x0={x0:.3g})")

    x_arr = np.linspace(x_all.min(), x_all.max(), 10000)
    y_arr = fitted(x_arr)

    return x_arr, y_arr, fitted


def contour_plotting(variable_x, variable_y, contour_line, bins=50):
    """Draw a filled KDE map of ``(variable_x, variable_y)`` with one enclosing contour.

    ``contour_line`` is the percentage of the density to enclose (e.g. 10 draws the
    10% contour). Kept as a convenience for interactive notebook use.
    """
    x = np.asarray(variable_x)
    y = np.asarray(variable_y)

    actual_contour_line = 100 - contour_line

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 200),
                         np.linspace(y.min(), y.max(), 200))
    zz = np.reshape(kde(np.vstack([xx.ravel(), yy.ravel()])), xx.shape)

    plt.contour(xx, yy, zz, levels=[np.percentile(zz, actual_contour_line)],
                colors='k', linewidths=2.5)
    plt.contourf(xx, yy, zz, levels=20, cmap='viridis', alpha=0.5)
    plt.title(f'{contour_line}% contour of the 2D data')
    plt.show()


def contour_line(variable_x, variable_y, contour_line):
    """Return the KDE grid ``(xx, yy, zz, level_percentile)`` for a contour overlay.

    Lets a caller draw the contour onto an existing axis instead of a fresh figure.
    """
    x = np.asarray(variable_x)
    y = np.asarray(variable_y)

    actual_contour_line = 100 - contour_line

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 200),
                         np.linspace(y.min(), y.max(), 200))
    zz = np.reshape(kde(np.vstack([xx.ravel(), yy.ravel()])), xx.shape)

    return xx, yy, zz, actual_contour_line
