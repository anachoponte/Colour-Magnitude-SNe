"""Colour-Magnitude SNe: LSST supernova colour-magnitude analysis toolkit.

Subpackages / modules:

* :mod:`cmsne.config`           - shared constants (OpSim path, survey dates, cosmology).
* :mod:`cmsne.sn_rates`         - cosmic star-formation & SN Ia rate model.
* :mod:`cmsne.opsim`            - loading and querying the LSST OpSim database.
* :mod:`cmsne.observations`     - cadence realisations, time cuts and nightly coadds.
* :mod:`cmsne.lsst`             - LSST band properties, styles and detection limits.
* :mod:`cmsne.lightcurve`       - synthetic light curves from sncosmo + OpSim.
* :mod:`cmsne.colour_magnitude` - colour-magnitude weighting and regression helpers.
* :mod:`cmsne.supernovae`       - SN population generators.
* :mod:`cmsne.plotting`         - light-curve plotting helpers.
"""

__all__ = [
    'config',
    'sn_rates',
    'opsim',
    'observations',
    'lsst',
    'lightcurve',
    'colour_magnitude',
    'supernovae',
    'plotting',
]
