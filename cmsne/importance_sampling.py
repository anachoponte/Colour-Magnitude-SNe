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
    enough_z = np.sum(w > 0) >= MIN_POINTS_FOR_KDE and len(np.unique(z)) > 1

    if lensed:
        mu = np.array([events[i]['magnification'] for i in idx], dtype=float)
        if enough_z and len(np.unique(mu)) > 1:
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
