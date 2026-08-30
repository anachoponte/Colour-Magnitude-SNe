"""All-band (ugrizy) photometry for the colour-only lensed-SN identifier.

The colour-magnitude study uses one colour at one matched epoch. This module
records the **full colour vector** a real cadence delivers: the nearest detected
magnitude in every LSST band around the trigger (first-detection) epoch, plus the
same 15 observer-days later so the colour *evolution* is captured. No redshift is
used anywhere -- these are pure photometric colours.

:class:`MultiColourGenerator` runs any SN source (SN Ia or a core-collapse
template), lensed or not, through a real OpSim cadence and returns that vector, so
signal (lensed SN Ia) and every contaminant class get identical, cadence-realistic
treatment.
"""

import numpy as np
import sncosmo
from scipy.stats import skewnorm

from .config import survey_dates
from .opsim import initialise_opsim_summary, create_sky_pointings
from .lightcurve import Transient, get_observations
from .lsst import bands as LSST_BANDS
from .colour_magnitude import ul_weight, cc_ul_weight, l_weight, cc_l_weight

# Adjacent-band colours across the ugrizy set.
ADJ_COLOURS = [('lsstu', 'lsstg'), ('lsstg', 'lsstr'), ('lsstr', 'lssti'),
               ('lssti', 'lsstz'), ('lsstz', 'lssty')]


def band_photometry(lightcurve, epoch_obs, window=5.0):
    """Nearest detected magnitude in each LSST band within ``window`` observer-days
    of ``epoch_obs``.

    ``lightcurve.obs_days`` are observer-frame days since this image's t0, and
    ``obs_mag`` is inf below the single-visit depth / nan outside the model's
    wavelength range, so ``finite`` == detected.

    :return: dict ``{band: magnitude or nan}`` over all six LSST bands.
    """
    days = np.asarray(lightcurve.obs_days, dtype=float)
    filt = np.asarray(lightcurve.obs_filters)
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    det = np.isfinite(mag)

    out = {}
    for b in LSST_BANDS:
        m = det & (filt == b[4:]) & (np.abs(days - epoch_obs) <= window)
        if not np.any(m):
            out[b] = np.nan
            continue
        idx = np.where(m)[0]
        nearest = idx[np.argmin(np.abs(days[idx] - epoch_obs))]
        out[b] = float(mag[nearest])
    return out


def _colours(mags, suffix=''):
    return {f'{a[-1]}{b[-1]}{suffix}': (mags[a] - mags[b]) for a, b in ADJ_COLOURS}


def first_detection_per_band(lightcurve, trigger):
    """For each LSST band, the earliest detection: its day relative to ``trigger``
    and its magnitude. Lets a colour vector be reconstructed as a function of how
    long one waits after the trigger to accumulate bands.

    :return: dicts ``(fd_day, fd_mag)`` keyed by band; NaN where a band is never
        detected. ``fd_day`` is observer-days after the trigger (>= 0).
    """
    days = np.asarray(lightcurve.obs_days, dtype=float)
    filt = np.asarray(lightcurve.obs_filters)
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    det = np.isfinite(mag)
    fd_day, fd_mag = {}, {}
    for b in LSST_BANDS:
        m = det & (filt == b[4:])
        if not np.any(m):
            fd_day[b] = np.nan; fd_mag[b] = np.nan; continue
        idx = np.where(m)[0]
        j = idx[np.argmin(days[idx])]          # earliest detection in this band
        fd_day[b] = float(days[j] - trigger)
        fd_mag[b] = float(mag[j])
    return fd_day, fd_mag


def event_multicolour(lightcurve, evolution_days=15.0, window=5.0):
    """Colour feature vector for one light curve.

    The trigger epoch is the **first detection** in any band. Colours are formed at
    the trigger and ``evolution_days`` observer-days later; a colour is NaN when
    either band was not observed within ``window`` of that epoch.

    :return: dict of colours (``gr``, ``ri``, ...), their later-epoch counterparts
        (``gr_d``, ...), the brightest observed magnitude ``peakmag``, the trigger
        epoch, and how many bands were seen at trigger (``n_bands``); or ``None`` if
        nothing was detected.
    """
    days = np.asarray(lightcurve.obs_days, dtype=float)
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    det = np.isfinite(mag)
    if not np.any(det):
        return None

    trigger = float(days[det].min())
    m0 = band_photometry(lightcurve, trigger, window)
    m1 = band_photometry(lightcurve, trigger + evolution_days, window)

    feat = {**_colours(m0), **_colours(m1, '_d')}
    feat['peakmag'] = float(np.nanmin(mag[det]))
    feat['trigger_day'] = trigger
    feat['n_bands'] = int(sum(np.isfinite(v) for v in m0.values()))

    # Per-band first-detection day (relative to trigger) and magnitude, for the
    # recovery-vs-waiting-time analysis.
    fd_day, fd_mag = first_detection_per_band(lightcurve, trigger)
    for b in LSST_BANDS:
        feat[f'fdday_{b[-1]}'] = fd_day[b]
        feat[f'fdmag_{b[-1]}'] = fd_mag[b]
    return feat


class MultiColourGenerator:
    """Run one SN source through the OpSim cadence and extract its colour vector.

    :param source: sncosmo source name (``'salt3'`` for SN Ia, or a Nugent
        core-collapse template such as ``'nugent-sn2p'``).
    """

    def __init__(self, source='salt3'):
        self.source = source
        self.is_ia = (source == 'salt3')

    def generate_one(self, z, abs_mag, magnification, times, ra, dec,
                     mjd_high=None, evolution_days=15.0, window=5.0, verbose=False):
        """One event through the cadence. ``magnification`` = 1 for an unlensed
        event. Returns the colour feature dict (with ``z``, ``magnification``
        attached) or ``None`` if the event is never detected."""
        if mjd_high is None:
            mjd_high = survey_dates[3]

        model = sncosmo.Model(source=self.source)
        if self.is_ia:
            x1 = skewnorm.rvs(-8.24, 1.23, 1.67)
            c = skewnorm.rvs(2.48, -0.089, 0.12)
            model.set(z=z, x1=x1, c=c)
        else:
            model.set(z=z)

        lensed_abs = abs_mag - 2.5 * np.log10(magnification)
        try:
            # Normalise in lsstr; if the model doesn't cover lsstr at this z the
            # event is not usable for a colour trigger anyway.
            model.set_source_peakabsmag(lensed_abs, 'lsstr', 'ab')
        except Exception:
            return None

        gen = initialise_opsim_summary([ra], [dec], verbose=verbose)
        lc = get_observations(Transient(model, times), gen, model, time_delay_=0.0,
                              mjd_high=mjd_high, Show=False, use_previous=False,
                              Verbose=verbose)
        if lc is None:
            return None

        feat = event_multicolour(lc, evolution_days, window)
        if feat is None:
            return None
        feat['z'] = z
        feat['magnification'] = magnification
        return feat

    def generate_many(self, n, kind, z_range, abs_mag_avg, abs_mag_sig,
                      magnification_range=(2.0, 50.0),
                      mjd_range=(survey_dates[0], survey_dates[3]),
                      mjd_high=None, verbose=False):
        """Draw ``n`` events of one class and keep the detected ones.

        :param kind: ``'lensed'`` applies a magnification drawn from
            ``magnification_range``; anything else is unlensed (magnification 1).
        :param z_range: redshift sampling range for this source/class.
        :param abs_mag_avg, abs_mag_sig: peak absolute-magnitude distribution.
        """
        lensed = (kind == 'lensed')
        pop = []
        for _ in range(n):
            z = np.random.uniform(*z_range)
            mu = np.random.uniform(*magnification_range) if lensed else 1.0
            abs_mag = np.random.normal(abs_mag_avg, abs_mag_sig)
            times = np.random.uniform(*mjd_range)
            ra, dec = create_sky_pointings(10)
            if len(ra) == 0:
                continue
            r = self.generate_one(z, abs_mag, mu, times, ra[0], dec[0],
                                  mjd_high=mjd_high, verbose=verbose)
            if r is not None:
                pop.append(r)
        return pop


def rate_weight(kind_source, z, magnification):
    """Per-event rate weight, matching the population it belongs to.

    ``kind_source`` is one of ``'sig'`` (lensed Ia), ``'uIa'`` (unlensed Ia),
    ``'uCC'`` (unlensed core-collapse), ``'lCC'`` (lensed core-collapse).
    """
    if kind_source == 'uIa':
        return ul_weight(z)
    if kind_source == 'uCC':
        return cc_ul_weight(z)
    if kind_source == 'sig':
        return l_weight(z, magnification)
    if kind_source == 'lCC':
        return cc_l_weight(z, magnification)
    raise ValueError(kind_source)
