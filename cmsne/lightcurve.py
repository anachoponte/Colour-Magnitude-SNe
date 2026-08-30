"""Synthetic LSST light curves from sncosmo models and OpSim cadences.

Combines an sncosmo transient model with an OpSim cadence realisation to
produce a :class:`Lightcurve`: per-visit apparent magnitudes (with realistic
flux perturbations and errors) in each observed LSST band.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import sncosmo

from .config import survey_dates
from .lsst import LSSTproperties
from .observations import (opsim_observation, select_observation_time_period,
                           coadds)


@dataclass
class Transient:
    model: sncosmo.Model
    date: float


@dataclass
class Lightcurve:
    # Observer-frame days since this image's t0 (i.e. sncosmo model time), NOT MJD
    # and NOT re-zeroed on the first visit.
    obs_days: np.ndarray
    obs_filters: np.ndarray
    obs_skybrightness: np.ndarray
    obs_lim_mag: np.ndarray
    obs_psf: np.ndarray
    obs_snr: np.ndarray
    obs_N_coadds: np.ndarray
    model_mag: np.ndarray
    obs_mag: np.ndarray
    obs_mag_error: np.ndarray
    app_mag_i_model: np.ndarray
    obs_start: np.ndarray
    coords: np.ndarray
    Nobs_3yr: float
    Nobs_10yr: float


def first_detection_epoch(lightcurve, redshift, rest_frame_peak_day):
    """Rest-frame epoch of the light curve's **first** detection, relative to peak.

    ``obs_days`` are observer-frame days since this image's t0, so they are divided
    by ``(1 + z)`` and referenced to the rest-frame peak, matching the frame
    convention in :mod:`cmsne.colour_magnitude`.

    :return: rest-frame epoch (days) of the earliest finite-magnitude visit, or
        ``np.nan`` if nothing was detected.
    """
    obs_mag = np.asarray(lightcurve.obs_mag)
    obs_days = np.asarray(lightcurve.obs_days)
    detected = np.isfinite(obs_mag)
    # Empty-check BEFORE np.min: np.min on an empty array raises, so the guard has
    # to come first (an earlier version indexed and reduced before checking).
    if not np.any(detected):
        return np.nan
    return np.min(obs_days[detected]) / (1 + redshift) - rest_frame_peak_day


def fitted_peak_magnitude(lightcurve, band=None):
    """Interpolated light-curve **peak** (brightest) magnitude.

    ``obs_to_mags`` returns the *first-detected* magnitude, which is fainter and
    noisier than the true peak. This fits a parabola to the three detections around
    the brightest observation and returns the vertex magnitude — a cleaner
    brightness feature for the classifier.

    :param band: restrict to one LSST band (e.g. ``'lsstr'``); ``None`` uses all
        detected points.
    :return: peak magnitude, or ``np.nan`` if nothing is detected.
    """
    days = np.asarray(lightcurve.obs_days, dtype=float)
    mag = np.asarray(lightcurve.obs_mag, dtype=float)
    det = np.isfinite(mag)
    if band is not None:
        det = det & (np.asarray(lightcurve.obs_filters) == band[4:])
    d, m = days[det], mag[det]
    if len(m) == 0:
        return np.nan
    if len(m) < 3:
        return float(np.min(m))
    i = int(np.argmin(m))                       # brightest detection
    lo, hi = max(0, i - 1), min(len(m), i + 2)  # 3 points bracketing the peak
    dd, mm = d[lo:hi], m[lo:hi]
    if len(np.unique(dd)) < 3:
        return float(np.min(m))
    a, b, _ = np.polyfit(dd, mm, 2)
    if a <= 0:                                  # not concave-up: no interior minimum
        return float(np.min(m))
    t_peak = -b / (2 * a)
    m_peak = float(np.polyval(np.polyfit(dd, mm, 2), t_peak))
    return min(m_peak, float(np.min(m)))        # peak is at least as bright as any sample


def get_app_magnitude(model, day, band, lim_mag):
    """
    Calculate the apparent magnitude + error for each supernova image at a certain time stamp.

    :param model: SNcosmo model for the supernova light curve
    :param day: time stamp of observation
    :param band: bandpass, choose between 'u', 'g', 'r', 'i', 'z', 'y' for LSST.
    :param lim_mag: limiting magnitude of the specific observation in the specific band (takes into account weather)
    :return: app_mag_model: array containing the apparent magnitude from the model
             app_mag_obs: array containing the observed (perturbed) apparent magnitude
             app_mag_error: array containing the apparent magnitude error
             snr: array containing the observation signal-to-noise ratio
    """
    lsst_band = 'lsst' + band
    properties = LSSTproperties.get_properties(lsst_band)
    zeropoint = properties['magnitude_zero_point']

    # Calculate point source flux
    try:
        flux_ps = model.bandflux(lsst_band, time=day, zp=zeropoint, zpsys='ab')
    except ValueError:
        # Bandpass is completely outside the model's spectral range.
        return np.nan, np.nan, np.nan, np.nan

    # Handle non-positive fluxes gracefully
    if flux_ps <= 0:
        return np.nan, np.nan, np.nan, np.nan

    # Calculate limiting flux from zeropoint and limiting magnitude
    lim_flux = 10 ** ((zeropoint - lim_mag) / 2.5)

    flux_error = lim_flux / 5

    # Perturb the flux according to the flux error (from the sky signal)
    flux_perturbation = np.random.normal(loc=0, scale=abs(flux_error))

    new_flux_ps = flux_ps + flux_perturbation
    # If new_flux_ps becomes non-positive after perturbation, treat it as unobservable
    if new_flux_ps <= 0:
        return np.nan, np.nan, np.nan, np.nan

    # Calculate S/N
    snr = new_flux_ps / flux_error

    # Convert to magnitudes
    app_mag_model = zeropoint - 2.5 * np.log10(flux_ps)

    # Ignore divide by zero warnings for robustness, although above checks should prevent most issues
    np.seterr(divide='ignore')

    app_mag_obs = zeropoint - 2.5 * np.log10(new_flux_ps)
    app_mag_obs = np.nan_to_num(app_mag_obs, nan=np.inf)

    # Handle potential division by zero if new_flux_ps is very close to zero
    if new_flux_ps == 0:
        app_mag_error = np.inf
    else:
        app_mag_error = abs(-2.5 * flux_error / (new_flux_ps * np.log(10)))

    if app_mag_obs > lim_mag:
        app_mag_obs = np.inf
        app_mag_error = np.inf  # If unobservable due to limiting magnitude, error is effectively infinite

    # Reset to default error handling
    np.seterr(divide='warn')

    return app_mag_model, app_mag_obs, app_mag_error, snr


def get_observations(transient, gen, model, time_delay_=0.0, obs_upper_limit=1000000,
                     mjd_high=None, Show=False, use_previous=False, Verbose=False):
    """
    Get the observation of the source from an LSST cadence realisation and given supernova source.

    Observations are phased against the transient's **true** ``t0``
    (``transient.date + time_delay_``): a visit at MJD ``t`` is assigned model time
    ``t - t0``. Only visits falling inside the transient's visibility window
    ``t0 + [model.mintime(), model.maxtime()]`` are kept, so a supernova that
    explodes in a cadence gap, or too late in the survey, is genuinely missed and
    the function returns ``None``.

    This replaces an earlier convention that shifted the whole time axis so the
    first visit in the window landed on ``model.mintime()``. That re-zeroing had two
    consequences. Every supernova was caught from its first light, because its phase
    origin was defined by whenever the first visit happened to fall, which biased
    identification rates upward. And for a delayed image it destroyed exactly the
    offset that makes a time delay measurable: the delayed light curve was pushed
    back onto the leading one, so the recovered "delay" was pinned near
    ``model.mintime()`` and showed no correlation with magnification.

    :param transient: Transient dataclass containing instantiated sncosmo model and
        the MJD of ``t = 0`` for the *undelayed* transient
    :param gen: OpSim Summary generator for a given OpSim database and sky pointings
    :param model: SNcosmo model for the supernova light curve
    :param time_delay_: lensing time delay [days] for this image. Its ``t0`` is
        ``transient.date + time_delay_``. Pass the undelayed MJD as
        ``transient.date``; the delay must not also be baked into it.
    :param obs_upper_limit: maximum number of observations to include. Default includes all observations.
    :param mjd_high: end of the observing campaign [MJD]. Defaults to the end of
        year 3. The same calendar cut applies to every image -- it is a property of
        the survey, not of the transient, so a delayed image legitimately gets less
        baseline (and may fall outside the survey entirely).
    :param Show: bool. if True: figures and print statements show the properties of the lensed SN systems
    :param use_previous: bool. if True: use previous instance of OpSim Summary generator
    :param Verbose: bool. if True: gives additional information in print statements
    :return: Lightcurve dataclass, whose ``obs_days`` are observer-frame days since
        this image's ``t0``. Returns ``None`` if the pointing is outside the
        footprint, or if no visit catches the transient while it is visible.
    """

    if use_previous == True:
        observations = opsim_observation(gen, use_previous)
    else:
        observations = opsim_observation(gen)

    if observations is None:  # This check was already here, but now we ensure it returns None
        return None

    Nobs_10yr = len(observations.opsim_times)
    Nobs_3yr = len(observations.opsim_times[observations.opsim_times < survey_dates[3]])

    coords = np.array([observations.ra, observations.dec])

    # Extent of the transient in model time, already observer-frame (sncosmo scales
    # the source's rest-frame phase range by 1 + z).
    start_transient = model.mintime()
    end_transient = model.maxtime()

    # True t0 of this image.
    t0_mjd = transient.date + time_delay_

    if mjd_high is None:
        mjd_high = survey_dates[3]

    # The transient is only observable between first and last light.
    window_low = t0_mjd + start_transient
    window_high = min(t0_mjd + end_transient, mjd_high)

    if Show:
        plt.figure(5)
        plt.hist(observations.opsim_times, bins=100)
        plt.axvline(x=t0_mjd, color='C3', label='t0')
        plt.axvline(x=window_low, color='C0', ls='--')
        plt.axvline(x=window_high, color='C0', ls='--')
        plt.legend()
        plt.show()

    if window_high <= window_low:
        # The transient switches on after the campaign ends.
        if Verbose:
            print("transient falls entirely outside the observing campaign")
        return None

    observations = select_observation_time_period(observations, mjd_low=window_low,
                                                 mjd_high=window_high)

    # No visit caught the transient while it was visible: it is missed, not shifted
    # into view. Without this guard the indexing below raises IndexError.
    if observations.opsim_times.size == 0:
        if Verbose:
            print("no visit caught the transient while it was visible")
        return None

    obs_start = observations.opsim_times[0]      # true MJD of the first usable visit

    # Phase against the true t0. No re-zeroing: the offset between a delayed image
    # and the leading one is exactly what carries the time-delay information.
    observations.opsim_times = observations.opsim_times - t0_mjd

    # Perform nightly coadds
    observations = coadds(observations)

    # Save all important properties
    obs_days = []
    obs_filters = []
    obs_skybrightness = []
    obs_lim_mag = []
    obs_psf = []
    obs_N_coadds = []

    # Save the SN brightness
    model_mag = []  # apparent magnitude without scatter
    obs_mag = []  # apparent magnitude with scatter
    obs_mag_error = []
    app_mag_i_model = []
    obs_snr = []

    for observation in range(obs_upper_limit):

        if observation >= len(observations.opsim_filters):
            break

        day = observations.opsim_times[observation]
        band = observations.opsim_filters[observation]
        lim_mag = observations.opsim_lim_mag[observation]
        psf = observations.opsim_psf[observation]

        lsst_band = 'lsst' + band

        if model.minwave() > sncosmo.get_bandpass(lsst_band).minwave() or model.maxwave() < sncosmo.get_bandpass(lsst_band).maxwave():
            if Verbose:
                print(f"outside of {band}-band bandpasses")
            app_mag_model = app_mag_obs = app_mag_error = snr = np.nan
            app_mag_model_i = np.nan
        else:
            # Calculate apparent magnitudes
            app_mag_model, app_mag_obs, app_mag_error, snr = get_app_magnitude(model, day, band, lim_mag)

            if band == 'i':
                # Calculate apparent magnitudes for i-band with standard properties
                i_properties = LSSTproperties.get_properties('lssti')
                app_mag_model_i, _, _, _ = get_app_magnitude(model, day, band, i_properties['limiting_magnitude'])
            else:
                app_mag_model_i = np.nan

        # Defensive: the window cut above already bounds every day to
        # [mintime, maxtime]; nightly coadding averages timestamps and could in
        # principle nudge the last epoch past the end.
        if day > end_transient:
            break

        obs_days.append(day)
        obs_filters.append(band)
        obs_skybrightness.append(observations.opsim_sky_brightness[observation])
        obs_lim_mag.append(lim_mag)
        obs_psf.append(psf)
        obs_N_coadds.append(observations.N_coadds[observation])

        model_mag.append(np.array(app_mag_model))
        obs_mag.append(np.array(app_mag_obs))
        app_mag_i_model.append(np.array(app_mag_model_i))
        obs_mag_error.append(app_mag_error)
        obs_snr.append(snr)

    obs_days = np.array(obs_days)
    obs_filters = np.array(obs_filters)
    obs_skybrightness = np.array(obs_skybrightness)
    obs_lim_mag = np.array(obs_lim_mag)
    obs_psf = np.array(obs_psf)
    obs_snr = np.array(obs_snr)
    obs_N_coadds = np.array(obs_N_coadds)

    model_mag = np.array(model_mag)
    obs_mag = np.array(obs_mag)
    app_mag_i_model = np.array(app_mag_i_model)
    obs_mag_error = np.array(obs_mag_error)

    obs_mag = obs_mag[:len(obs_days)]
    model_mag = model_mag[:len(obs_days)]

    return Lightcurve(obs_days, obs_filters, obs_skybrightness, obs_lim_mag, obs_psf, obs_snr, obs_N_coadds, model_mag, obs_mag, obs_mag_error, app_mag_i_model, obs_start, coords, Nobs_3yr, Nobs_10yr)
