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
from .contours import robust_range
from .multicolour import ADJ_COLOURS
from .opsim import create_sky_pointings

BAND_PAIR_KEYS = [f'{a[-1]}{b[-1]}' for a, b in ADJ_COLOURS]  # ['ug', 'gr', 'ri', 'iz', 'zy']
MIN_POINTS_FOR_KDE = 5
MIN_IN_RANGE_FRACTION = 0.95
KDE_RANGE_CHECK_DRAWS = 2000


def effective_sample_size(weights):
    """Kish's ESS: (sum w)^2 / sum(w^2). 0.0 for empty or non-positive input.

    Normalizes by the largest absolute weight before squaring -- ESS is
    scale-invariant (``ESS(c*w) == ESS(w)`` for any ``c != 0``), so this is
    mathematically identical to the naive formula for any well-conditioned
    input, but avoids ``w ** 2`` overflowing to ``inf`` (then ``inf / inf``
    -> ``NaN``) when real rate weights span extreme dynamic range -- e.g.
    after :func:`cmsne.classifier._weighted_subsample`'s own mass-rescaling
    draws a sample whose total is many orders of magnitude below the full
    population's, the rescaled weights can individually be large enough that
    squaring them overflows even though neither the weights nor their sum
    alone would (see issue #50: this crashed ``roc_curve()``'s finite-value
    check the first time ``score()`` ran the local-density blend widely
    enough, across thousands of distinct missing-data patterns, to hit it).
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0 or not np.any(w > 0):
        return 0.0
    scale = np.max(np.abs(w))
    if not np.isfinite(scale) or scale <= 0:
        return 0.0
    wn = w / scale
    denom = np.sum(wn ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float(wn.sum() ** 2 / denom)


def weighted_quantile(values, weights, q):
    """Rate-weighted quantile (``q`` in ``[0, 1]``) of ``values``.

    Plain ``np.percentile``/``np.median`` treat every event as equally
    likely, which silently reintroduces the exact bias this module's
    pilot+boost importance sampling corrects for: a deliberately
    oversampled, heavily-downweighted rare event counts the same as a
    common one. Any script that characterizes a rate-weighted Monte Carlo
    sample (e.g. "what does the false-positive population look like")
    must use this -- or :func:`weighted_median` -- instead of a plain
    quantile, or the deliberately non-uniform sampling density shows up as
    if it were the real physical distribution (see Finding 5's own
    high-z-contaminant correction in docs/session_writeup.md for a worked
    example of exactly this mistake).

    :param values: 1D array-like of event values.
    :param weights: per-event rate weights, same length as ``values``.
    :param q: quantile in ``[0, 1]`` (``0.5`` = median).
    :return: the weighted quantile, or ``nan`` if there is no finite,
        positively-weighted data.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    finite = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(finite):
        return float('nan')
    v, w = v[finite], w[finite]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cum, v))


def weighted_median(values, weights):
    """Rate-weighted median; see :func:`weighted_quantile`."""
    return weighted_quantile(values, weights, 0.5)


def weighted_composition(tags, weights):
    """Rate-weighted share of each distinct value in ``tags``.

    Counts each tag by summed rate weight rather than raw event count --
    the count-based version silently reintroduces the same bias
    :func:`weighted_quantile` warns about, since this module's own
    pilot+boost sampling deliberately draws some tags far more densely
    than their true rate.

    :param tags: 1D array-like of hashable labels, one per event.
    :param weights: per-event rate weights, same length as ``tags``.
    :return: ``{tag: weight_fraction}`` over positively-weighted finite
        entries, summing to ``1.0`` (empty dict if none).
    """
    tags = np.asarray(tags)
    w = np.asarray(weights, dtype=float)
    finite = np.isfinite(w) & (w > 0)
    if not np.any(finite):
        return {}
    tags, w = tags[finite], w[finite]
    total = w.sum()
    return {str(tag): float(w[tags == tag].sum() / total) for tag in np.unique(tags)}


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

    Bin edges come from :func:`cmsne.contours.robust_range` (rate-weighted
    1st-99th percentile), not raw min/max. A physically near-impossible event
    with near-zero weight (e.g. near-zero redshift with near-maximal
    magnification -- the same class of outlier ``robust_range`` already
    guards against elsewhere) can otherwise land far outside where the real
    density is and stretch the whole grid, so that genuinely sparse-but-
    physically-plausible cells never get flagged while the outlier's own
    (irrelevant, zero-ESS) cell does.
    """
    weights = np.asarray(weights, dtype=float)
    mags = np.array([e.get(magnitude_key, np.nan) for e in events], dtype=float)
    mag_lo, mag_hi = robust_range(mags, weights)
    flagged = []
    for pair in band_pair_keys:
        colours = np.array([e.get(pair, np.nan) for e in events], dtype=float)
        finite = np.isfinite(colours) & np.isfinite(mags)
        if np.sum(finite) < 2:
            continue
        c_lo, c_hi = robust_range(colours, weights)
        if not np.all(np.isfinite([c_lo, c_hi, mag_lo, mag_hi])) or c_lo >= c_hi or mag_lo >= mag_hi:
            continue
        c_edges = np.linspace(c_lo, c_hi, n_bins + 1)
        m_edges = np.linspace(mag_lo, mag_hi, n_bins + 1)
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


def _kde_in_range_fraction(kde, lensed, z_range, magnification_range,
                           n_draw=KDE_RANGE_CHECK_DRAWS):
    """Monte-Carlo estimate of the fraction of ``kde``'s mass inside the
    valid latent range(s), by resampling from it and checking containment.

    Used to decide whether a fitted KDE is safe to use as a declared
    importance-sampling proposal density: ``sample_latent`` clips draws from
    the KDE to ``z_range``/``magnification_range``, so if a non-trivial
    fraction of the KDE's own mass lies outside those bounds, the clipped
    *sampling* distribution and the unclipped *declared* density
    (``_proposal_density``) diverge, biasing the mixture-weight correction.
    """
    draws = kde.resample(n_draw)
    if lensed:
        z_d, mu_d = draws[0], draws[1]
        in_range = ((z_d >= z_range[0]) & (z_d <= z_range[1]) &
                   (mu_d >= magnification_range[0]) & (mu_d <= magnification_range[1]))
    else:
        z_d = draws[0]
        in_range = (z_d >= z_range[0]) & (z_d <= z_range[1])
    return float(np.mean(in_range))


def fit_boost_proposal(events, weights, colour_key, magnitude_key, colour_range,
                       magnitude_range, lensed, z_range, magnification_range):
    """Fit a boost proposal over latent (z[, magnification]) for one flagged cell.

    Unlensed events have magnification fixed at 1.0 (not a random variable), so
    only ``z`` is boosted for them; lensed events boost the joint (z, magnification).

    :return: dict with ``'kde'`` (a fitted :class:`scipy.stats.gaussian_kde`, or
        ``None``) and, when ``'kde'`` is ``None``, a box fallback: ``'z_box'``
        (and ``'mu_box'`` if ``lensed``), plus ``'fallback'`` in
        ``{'box', 'box_out_of_range', 'prior_box'}`` (``None`` when a KDE was
        fit). ``'box_out_of_range'`` marks a cell where a KDE candidate would
        otherwise have been fit, but too much of its probability mass fell
        outside ``z_range``/``magnification_range`` -- see
        ``MIN_IN_RANGE_FRACTION``; these cells reuse the same padded-box
        fallback as ``'box'`` (too few points) so that ``sample_latent`` and
        ``_proposal_density`` stay self-consistent for them too.
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
    enough_z = np.sum(pos) >= MIN_POINTS_FOR_KDE and len(np.unique(z[pos])) > 1

    if lensed:
        mu = np.array([events[i]['magnification'] for i in idx], dtype=float)
        if enough_z and len(np.unique(mu[pos])) > 1:
            candidate = gaussian_kde(np.vstack([z, mu]), weights=w)
            in_range_frac = _kde_in_range_fraction(candidate, True, z_range, magnification_range)
            if in_range_frac >= MIN_IN_RANGE_FRACTION:
                return {'kde': candidate, 'fallback': None}
            pad_z = max(0.01, 0.1 * (z.max() - z.min()))
            pad_mu = max(0.1, 0.1 * (mu.max() - mu.min())) if len(mu) > 1 else 1.0
            return {'kde': None, 'fallback': 'box_out_of_range',
                    'z_box': (max(z_range[0], z.min() - pad_z), min(z_range[1], z.max() + pad_z)),
                    'mu_box': (max(magnification_range[0], mu.min() - pad_mu),
                              min(magnification_range[1], mu.max() + pad_mu))}
        if len(z) > 0:
            pad_z = max(0.01, 0.1 * (z.max() - z.min()))
            pad_mu = max(0.1, 0.1 * (mu.max() - mu.min())) if len(mu) > 1 else 1.0
            return {'kde': None, 'fallback': 'box',
                    'z_box': (max(z_range[0], z.min() - pad_z), min(z_range[1], z.max() + pad_z)),
                    'mu_box': (max(magnification_range[0], mu.min() - pad_mu),
                              min(magnification_range[1], mu.max() + pad_mu))}
        return {'kde': None, 'fallback': 'prior_box', 'z_box': z_range, 'mu_box': magnification_range}

    if enough_z:
        candidate = gaussian_kde(z, weights=w)
        in_range_frac = _kde_in_range_fraction(candidate, False, z_range, magnification_range)
        if in_range_frac >= MIN_IN_RANGE_FRACTION:
            return {'kde': candidate, 'fallback': None}
        pad_z = max(0.01, 0.1 * (z.max() - z.min()))
        return {'kde': None, 'fallback': 'box_out_of_range',
                'z_box': (max(z_range[0], z.min() - pad_z), min(z_range[1], z.max() + pad_z))}
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
        return pilot, {'n_pilot': n_pilot, 'n_boost': 0, 'n_uniform_topup': 0,
                       'total_attempted': n_pilot, 'flagged_cells': []}

    pilot_rw = np.array([rate_weight_fn(e['z'], e['magnification']) for e in pilot])
    flagged = diagnose_sparse_cells(pilot, pilot_rw, band_pair_keys, magnitude_key,
                                    n_bins, ess_floor_fraction)
    if not flagged:
        # No sparse cells to patch. Spend the remaining budget on more uniform
        # draws rather than returning early: the caller asked for ``n`` attempts
        # and ``rate_weight_fn`` is normalised by ``n``, so stopping at
        # ``n_pilot`` makes every weight too small by ``n_pilot / n`` (15% at
        # the production ``pilot_frac=0.85``). Because whether any cell is
        # flagged varies by class, that error is *asymmetric across classes*
        # and therefore distorts relative rates and composition statistics,
        # not just an overall normalisation.
        topup = gen.generate_many(n_boost, kind, z_range, abs_mag_avg, abs_mag_sig,
                                  magnification_range=magnification_range,
                                  mjd_range=mjd_range, mjd_high=mjd_high)
        all_pilot = pilot + topup
        for e in all_pilot:
            e['weight'] = float(rate_weight_fn(e['z'], e['magnification']))
        return all_pilot, {'n_pilot': n_pilot, 'n_boost': 0,
                           'n_uniform_topup': n_boost,
                           'total_attempted': n_pilot + n_boost,
                           'flagged_cells': []}

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
    # ``rate_weight_fn`` carries 1/(n * p_uniform) from the caller's
    # ``n_attempted = n``, but ``boost_counts = np.maximum(1, ...)`` can sum to
    # more than ``n_boost`` when many cells are flagged, so the attempts
    # actually made may exceed ``n``. Correct to the realised count.
    attempt_correction = n / float(total_attempted)
    for e, z_i, mu_i, pm_rel in zip(all_events, all_z, all_mu, p_mix_relative):
        e['weight'] = float(rate_weight_fn(z_i, mu_i) * attempt_correction / pm_rel)

    return all_events, {'n_pilot': n_pilot_attempted, 'n_boost': n_boost_attempted,
                        'n_uniform_topup': 0, 'total_attempted': total_attempted,
                        'attempt_correction': attempt_correction,
                        'flagged_cells': flagged}
