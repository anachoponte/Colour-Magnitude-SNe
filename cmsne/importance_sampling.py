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
