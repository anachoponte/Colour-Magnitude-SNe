"""Cosmic star-formation and Type Ia supernova rate calculations.

This module reproduces the analytic SN Ia rate model from the original
notebook: a Madau-style star-formation history convolved with a delay-time
distribution, plus the observability (probability) weighting used to turn an
intrinsic rate into an observed rate.

The rate grid used by the colour-magnitude weighting functions is expensive to
evaluate (a per-redshift integral), so :func:`rate_grid` computes it once and
caches the result for the lifetime of the process.
"""

import numpy as np
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18

from .config import H0, OMEGA_M, OMEGA_V, N0, N1, MSN

# np.trapezoid is the NumPy >= 2.0 spelling; np.trapz is the older one. Bind once
# so the package runs on either (Colab ships NumPy 2.x, local envs often do not).
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def Mad_son(redshift):
    """Madau & Dickinson-style cosmic star-formation rate density."""
    M_sol = 1.989e30
    return 0.015 * M_sol * ((1 + redshift) ** 2.7 / (1 + ((1 + redshift) / 2.9) ** 5.6))


def HZ(redshift):
    """Hubble parameter H(z) [km/s/Mpc]."""
    # find somewhere to cite these values from
    return H0 * np.sqrt(OMEGA_M * ((1 + redshift) ** 3) + OMEGA_V)


def Z_int(redshift):
    """Integrand ``1 / ((1 + z) H(z))`` (cosmic time per unit redshift)."""
    return 1 / ((1 + redshift) * HZ(redshift))


def SN1A_formation_rate(tim, cosmic_lookback_time, sfr,
                        tmin=0.04, tmax=100, tint=0.5, index_=-1):
    """Convolve the star-formation history with a SN Ia delay-time distribution.

    :param tim: lookback time at which to evaluate the rate
    :param cosmic_lookback_time: array of lookback times matching ``sfr``
    :param sfr: star-formation-rate density evaluated on ``cosmic_lookback_time``
    :return: SN Ia formation rate at lookback time ``tim``
    """
    t_delay = np.linspace(0, tim, 1000)  # duration between light detected and light emitted from distant source
    mask_min = (t_delay >= tmin) & (t_delay <= tint)
    mask_int = (t_delay > tint) & (t_delay <= tmax)
    dtd = np.zeros_like(t_delay)
    dtd[mask_min] = N1
    # At tim == 0 (z == 0) the delay grid collapses to a single point, mask_int is
    # empty, and there is no rate to accumulate. Guarding here avoids evaluating
    # N0 / 0, which warned on every call even though the result was discarded.
    if tim > 0:
        dtd[mask_int] = N0 / tim
    formation_lookback_time = tim + t_delay
    valid_mask = (formation_lookback_time <= np.max(cosmic_lookback_time))
    sn1a_rate = 0.0
    if np.any(valid_mask):
        past_sfr = np.interp(formation_lookback_time[valid_mask], cosmic_lookback_time, sfr)  # past as in slightly further
        sn1a_rate = _trapezoid(past_sfr * dtd[valid_mask], t_delay[valid_mask])
    return sn1a_rate


def compute_rate_grid(redshift=None):
    """Evaluate the intrinsic SN Ia rate on a redshift grid.

    :param redshift: redshift grid to evaluate on. Defaults to ``linspace(0, 10, 10000)``.
    :return: ``(redshift, nSN1A)`` arrays, where ``nSN1A`` is the intrinsic SN Ia rate.
    """
    if redshift is None:
        redshift = np.linspace(0, 10, 10000)

    phi = Mad_son(redshift)
    cosmic_lookback_time = np.array(Planck18.lookback_time(redshift))

    nSN1A_list = [SN1A_formation_rate(tim, cosmic_lookback_time, phi)
                  for tim in cosmic_lookback_time]
    nSN1A = np.array(nSN1A_list) * 6.36e-30
    return redshift, nSN1A


_rate_grid_cache = None


def rate_grid():
    """Return the cached ``(redshift, nSN1A)`` intrinsic rate grid.

    The grid is computed on first use and reused thereafter.
    """
    global _rate_grid_cache
    if _rate_grid_cache is None:
        _rate_grid_cache = compute_rate_grid()
    return _rate_grid_cache


def mag_min(distance):
    """Minimum-flux weighting term for a source at ``distance`` [pc]."""
    msn = MSN + 5 * np.log10(distance / 10)
    return 10 ** (0.4 * (24.7 - msn))


def prob_(redshift):
    """Observability probability as a function of redshift."""
    redshift_units = redshift * cu.redshift
    distance_unit = redshift_units.to(u.pc, cu.redshift_distance(Planck18, kind="comoving"))
    distance = distance_unit.value
    return 0.5 * ((distance / 31e9) ** 3) * ((mag_min(distance)) ** (-2))