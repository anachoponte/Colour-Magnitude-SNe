"""Supernova population generators for colour-magnitude studies.

Three generators draw random SN populations (redshift, stretch, colour, absolute
magnitude, lensing magnification) and return the quantities needed to build
colour-magnitude diagrams:

* :class:`Supernovae` samples full LSST cadence realisations per event, including
  a time-delayed second lensed image (realistic, slow).
* :class:`Supernovae2` uses the model light curve directly with a simple
  detection cut (faster, idealised).
* :class:`Nugent` does the same for the core-collapse contaminant templates.

Frame convention
----------------
Observation days come out of :func:`cmsne.colour_magnitude.individual_observations`
already divided by ``(1 + z)``, i.e. in the rest frame. Any epoch measured against
the intrinsic light-curve peak must divide that peak day by ``(1 + z)`` too. Both
:attr:`time_of_SN` and :attr:`time_delay` below do so, which is what makes them
directly comparable when they are histogrammed together.
"""

import numpy as np
import sncosmo
from scipy.stats import skewnorm

from .config import survey_dates
from .opsim import initialise_opsim_summary, create_sky_pointings
from .lightcurve import Transient, get_observations, first_detection_epoch
from .colour_magnitude import (obs_to_mags, individual_observations,
                               ul_weight, cc_ul_weight, cc_l_weight, l_weight,
                               detection, time_delay, time_delay_fourth_lens)

# Time-delay model per lens class. Keyed so a population can carry all four in
# parallel; galaxy_1/group/cluster share the (mu/4)**-3 scaling with different
# deflector scales, galaxy_2 uses the shallower (mu/4)**-1 model.
LENS_MODELS = ('galaxy_1', 'galaxy_2', 'group', 'cluster')


def _covers(model, *bands):
    """True if the model's spectral range spans every one of ``bands``.

    Checked at both ends for every band. Testing only one band's blue edge against
    another band's red edge (as an earlier version did) passes models that do not
    actually cover the bands being measured, which then silently return NaN
    magnitudes instead of the event being rejected.
    """
    for band in bands:
        bandpass = sncosmo.get_bandpass(band)
        if model.minwave() > bandpass.minwave() or model.maxwave() < bandpass.maxwave():
            return False
    return True


def redshift_limits(source, *bands):
    """Redshift range over which ``source`` spectrally covers all of ``bands``.

    A source of rest-frame extent ``[smin, smax]`` observed at redshift ``z`` spans
    ``[smin (1+z), smax (1+z)]``, so covering a bandpass ``[bmin, bmax]`` requires
    ``smin (1+z) <= bmin`` and ``smax (1+z) >= bmax``. The blue edge therefore sets
    an upper limit on ``z`` and the red edge a lower one.

    Sampling redshift over this range instead of a fixed wide interval matters twice
    over. A range that is too wide wastes most draws on events rejected before any
    observation is simulated; a single global cap chosen for one band silently
    truncates the redder pairs, which tolerate far higher redshift (for ``salt3``,
    ``lsstz-lssty`` reaches z = 3.01 where ``lsstg-lsstr`` stops at 0.93).

    :param source: sncosmo source name, Source, or Model
    :param bands: sncosmo bandpass names that must be covered
    :return: ``(z_min, z_max)``, clipped so ``z_min >= 0``
    """
    if isinstance(source, str):
        src = sncosmo.get_source(source)
    elif hasattr(source, 'source'):            # an sncosmo.Model
        src = source.source
    else:
        src = source

    band_mins = [sncosmo.get_bandpass(b).minwave() for b in bands]
    band_maxs = [sncosmo.get_bandpass(b).maxwave() for b in bands]

    z_max = min(band_mins) / src.minwave() - 1.0
    z_min = max(0.0, max(band_maxs) / src.maxwave() - 1.0)
    return z_min, z_max


class Supernovae:
    """Lensed SN Ia population sampled through real LSST cadence realisations.

    Each event yields three light curves at the same sky position: the unlensed
    source, the magnified leading image, and a magnified image delayed by
    :func:`~cmsne.colour_magnitude.time_delay`.
    """

    def __init__(self, model_name='salt3'):
        self.model = sncosmo.Model(source=model_name)

    def _measure_time_delay(self, gen, times, delay, lc_ul_obs_mags, lc_ul_obs_days,
                            rest_frame_peak_day, z, followup_mjd_high, verbose=False):
        """Rest-frame epoch (relative to peak) at which the trailing image is first
        detected *after* the leading image's brightest visit.

        The trailing image is delayed by ``delay`` days and gets the longer
        follow-up window (discovery is already secured by the leading image, so
        recovering the second one later does not enlarge the discovery sample).

        All epochs are rest-frame. ``obs_days`` are observer-frame days since t0, so
        the trailing epochs -- put back on the leading image's clock with ``+ delay``
        -- and the leading-image ``benchmark`` are BOTH divided by ``(1 + z)`` before
        being compared. (An earlier version left the benchmark in the observer frame
        while the trailing epochs were rest-frame, mixing the two frames.)

        :return: rest-frame epoch in days, or ``np.nan`` if the second image is
            never recovered.
        """
        event_tim = Transient(self.model, times)
        lc_tim = get_observations(event_tim, gen, self.model, time_delay_=delay,
                                  mjd_high=followup_mjd_high,
                                  Show=False, use_previous=False, Verbose=verbose)
        if lc_tim is None:
            return np.nan

        obs_days_tim = (np.asarray(lc_tim.obs_days) + delay) / (1 + z)
        obs_mag_tim = np.asarray(lc_tim.obs_mag)

        detected_1 = np.isfinite(obs_mag_tim)         # trailing-image detections
        detected_2 = np.isfinite(lc_ul_obs_mags)      # leading-image detections

        if detected_2.sum() < 3 or detected_1.sum() < 1:
            return np.nan

        # Rest-frame day of the leading image's brightest detection.
        benchmark = lc_ul_obs_days[detected_2][np.argmin(lc_ul_obs_mags[detected_2])] / (1 + z)
        later_days = obs_days_tim[detected_1][obs_days_tim[detected_1] > benchmark]

        if len(later_days) == 0:
            return np.nan
        return np.min(later_days) - rest_frame_peak_day

    def generate_one(self, z, x1, abs_mag, c, magnification, times, ra, dec,
                     band1, band2, required_model='cluster', lens_models=LENS_MODELS,
                     ref_band='bessellb', min_angular_separation=0.8,
                     discovery_mjd_high=None, followup_mjd_high=None,
                     verbose=False):
        """Generate one lensed SN Ia observed through an LSST cadence.

        :param z: source redshift
        :param x1: SALT3 stretch parameter
        :param abs_mag: unlensed peak absolute magnitude
        :param c: SALT3 colour parameter
        :param magnification: lensing magnification of the images
        :param times: MJD of the transient's ``t = 0`` (undelayed)
        :param ra: right ascension of the sky pointing [deg]
        :param dec: declination of the sky pointing [deg]
        :param band1: sncosmo band for the magnitude axis, e.g. ``'lsstr'``
        :param band2: sncosmo band for the colour's blue side
        :param required_model: lens class whose time delay must be measurable for
            the event to be kept (``'galaxy_1'``, ``'galaxy_2'``, ``'group'`` or
            ``'cluster'``). ``None`` accepts the event regardless. Default
            ``'cluster'`` reproduces the original single-lens acceptance.
        :param lens_models: lens classes to measure a delay for. Each is measured
            independently and reported in the result dict.
        :param ref_band: **rest-frame** band whose light-curve peak defines the epoch
            zero point. Default ``'bessellb'`` follows the usual SN convention.
        :param min_angular_separation: image separation below which the pair is
            unresolvable and no time delay is measured [arcsec]
        :param discovery_mjd_high: end of the window in which the **leading** image
            must be found. Defaults to the end of year 3. This bounds the discovery
            sample: a system only counts if its first image is detected here.
        :param followup_mjd_high: end of the window in which the **trailing** image
            may be caught. Defaults to the end of year 5. Follow-up may run past the
            discovery window, so a system discovered in year 3 can still have its
            second image recovered later; the leading image is unaffected.
        :param verbose: forwarded to the OpSim generator and light-curve builder
        :return: dict of event properties, or ``None`` if the event is unusable
        """
        if discovery_mjd_high is None:
            discovery_mjd_high = survey_dates[3]
        if followup_mjd_high is None:
            followup_mjd_high = survey_dates[5]
        if isinstance(lens_models, str):
            lens_models = (lens_models,)

        self.model.set(z=z, x1=x1, c=c)

        if not _covers(self.model, band1, band2):
            # A science band is outside the model's spectral range, so this event
            # cannot be measured. The reference band is deliberately NOT part of
            # this test -- see below.
            return None

        # Epoch zero point: the rest-frame phase of peak, straight from the source.
        # peakphase is already a rest-frame phase, is independent of z, needs no
        # observer-frame coverage, and still responds to x1 (see git history for why
        # this replaced scanning an observer-frame lsstr light curve).
        rest_frame_peak_day = self.model.source.peakphase(ref_band)

        self.band1 = band1
        self.band2 = band2

        # One generator, one pull per component at the same pointing: unlensed,
        # leading, and one trailing image per lens model measured. Same sky position,
        # so the OpSim query (and its BallTree lookup) is not rebuilt per component.
        n_pulls = 2 + len(lens_models)
        gen = initialise_opsim_summary([ra] * n_pulls, [dec] * n_pulls, verbose=verbose)

        # --- Unlensed component -------------------------------------------------
        # set_source_peakabsmag rescales the whole source amplitude, so the second
        # call overwrites the first and the normalisation ends up defined in band1
        # alone. Kept as-is to preserve the existing calibration.
        self.model.set_source_peakabsmag(abs_mag, band2, 'ab')
        self.model.set_source_peakabsmag(abs_mag, band1, 'ab')

        event_ul = Transient(self.model, times)
        lc_ul = get_observations(event_ul, gen, self.model, time_delay_=0.0,
                                 mjd_high=discovery_mjd_high,
                                 Show=False, use_previous=False, Verbose=verbose)
        weights_ul = ul_weight(z)
        if lc_ul is None:
            return None

        _, mag_band_1_ul, day_band_1_ul = individual_observations(band1, lc_ul, z)
        _, mag_band_2_ul, day_band_2_ul = individual_observations(band2, lc_ul, z)
        band_2_mag_ul, band_1_mag_ul, band_1_days_ul, band_2_days_ul = obs_to_mags(
            day_band_1_ul, day_band_2_ul, mag_band_1_ul, mag_band_2_ul)

        if np.isnan(band_1_mag_ul):
            return None

        # --- Lensed, undelayed (leading) component -----------------------------
        lensed_abs_mag = abs_mag - 2.5 * np.log10(magnification)
        self.model.set_source_peakabsmag(lensed_abs_mag, band2, 'ab')
        self.model.set_source_peakabsmag(lensed_abs_mag, band1, 'ab')

        event_l = Transient(self.model, times)
        lc_l = get_observations(event_l, gen, self.model, time_delay_=0.0,
                                mjd_high=discovery_mjd_high,
                                Show=False, use_previous=False, Verbose=verbose)
        weights_l = l_weight(z, magnification)
        if lc_l is None:
            return None

        lc_ul_obs_days = np.asarray(lc_ul.obs_days)
        lc_ul_obs_mags = np.asarray(lc_ul.obs_mag)

        _, mag_band_1_l, day_band_1_l = individual_observations(band1, lc_l, z)
        _, mag_band_2_l, day_band_2_l = individual_observations(band2, lc_l, z)
        band_2_mag_l, band_1_mag_l, band_1_days_l, band_2_days_l = obs_to_mags(
            day_band_1_l, day_band_2_l, mag_band_1_l, mag_band_2_l)

        if np.isnan(band_1_mag_l):
            return None

        # Colour is a difference of magnitudes at a matched epoch. The magnitudes
        # are NOT rescaled by (1 + z): a magnitude is logarithmic, so dividing it
        # by (1 + z) is not a K-correction, it just compresses the colour axis.
        lensed_colour = band_2_mag_l - band_1_mag_l

        # Identification epoch (rest-frame day of the matched colour pair). Both
        # band_*_days_l are already rest-frame (individual_observations divides by
        # 1 + z), so id_day is rest-frame too.
        id_day = max(band_1_days_l, band_2_days_l)
        first_alert_day = first_detection_epoch(lc_l, z, rest_frame_peak_day)
        time_of_SN_rel_peak = id_day - rest_frame_peak_day
        time_of_SN_rel_first = (id_day - rest_frame_peak_day) - first_alert_day

        # Number of detections (alerts) before the colour method identifies the
        # system. obs_days are observer-frame days since t0, so convert them to the
        # rest frame before comparing against the rest-frame identification day.
        # (An earlier version compared observer-frame obs_days against a rest-frame,
        # peak-referenced epoch, mixing frames and references.)
        detected_l = np.isfinite(np.asarray(lc_l.obs_mag))
        alert_days = np.asarray(lc_l.obs_days)[detected_l] / (1 + z)
        n_alerts_before_id = int(np.count_nonzero(alert_days < id_day))

        # --- Lensed, delayed component(s): one time delay per lens class -------
        z_l = z / 2                                              # ad-hoc lens redshift
        ang_sep = 10 / (magnification * 0.5 * 0.8 * 0.2)         # ad-hoc separation proxy

        timdel = {
            'galaxy_1': time_delay(magnification, z, z_l, 0),
            'galaxy_2': time_delay_fourth_lens(magnification, z, z_l),
            'group':    time_delay(magnification, z, z_l, 1),
            'cluster':  time_delay(magnification, z, z_l, 2),
        }

        result_time_delay = {name: np.nan for name in lens_models}
        if ang_sep > min_angular_separation:
            for name in lens_models:
                result_time_delay[name] = self._measure_time_delay(
                    gen, times, timdel[name], lc_ul_obs_mags, lc_ul_obs_days,
                    rest_frame_peak_day, z, followup_mjd_high, verbose=verbose)
                # Re-pin the model to the lensed leading state for the next pull.
                self.model.set(z=z, x1=x1, c=c)
                self.model.set_source_peakabsmag(lensed_abs_mag, band1, 'ab')

        # required_model controls acceptance: np.isnan (not `is np.nan`) so a
        # computed NaN is caught, not just the literal sentinel.
        if required_model is not None and np.isnan(result_time_delay.get(required_model, np.nan)):
            return None

        # Legacy aliases point at the required lens model (default 'cluster', the
        # original single-lens prefactor) so existing notebooks/tests keep working.
        legacy_key = required_model if required_model in result_time_delay else (
            'cluster' if 'cluster' in timdel else lens_models[0])

        return {
            'z': z,
            'x1': x1,
            'c': c,
            'magnification': magnification,
            'weights_ul': weights_ul,
            'weights_l': weights_l,
            'band_2_mag_ul': band_2_mag_ul,
            'band_1_mag_ul': band_1_mag_ul,
            'band_2_mag_l': band_2_mag_l,
            'band_1_mag_l': band_1_mag_l,
            'rest_frame_peak_day': rest_frame_peak_day,
            'timdel_galaxy_1': timdel['galaxy_1'],
            'timdel_galaxy_2': timdel['galaxy_2'],
            'timdel_group': timdel['group'],
            'timdel_cluster': timdel['cluster'],
            'time_delay_galaxy_1': result_time_delay.get('galaxy_1', np.nan),
            'time_delay_galaxy_2': result_time_delay.get('galaxy_2', np.nan),
            'time_delay_group': result_time_delay.get('group', np.nan),
            'time_delay_cluster': result_time_delay.get('cluster', np.nan),
            'band_1_days_ul': band_1_days_ul,
            'band_2_days_ul': band_2_days_ul,
            'band_1_days_l': band_1_days_l,
            'band_2_days_l': band_2_days_l,
            'lensed_colour': lensed_colour,
            'time_of_SN_rel_peak': time_of_SN_rel_peak,
            'time_of_SN_rel_first': time_of_SN_rel_first,
            'first_alert_day': first_alert_day,
            'n_alerts_before_id': n_alerts_before_id,
            # Legacy keys (single-lens schema).
            'timdel': timdel[legacy_key],
            'time_delay': result_time_delay.get(legacy_key, np.nan),
            'time_of_SN': time_of_SN_rel_peak,
        }

    def generate_many(self, n, band1, band2, required_model='cluster',
                      lens_models=LENS_MODELS, z_range=None,
                      mjd_range=(survey_dates[0], survey_dates[3]),
                      magnification_range=(2.0, 50.0),
                      discovery_mjd_high=None, followup_mjd_high=None,
                      verbose=False):
        """Draw ``n`` candidate events and keep the usable ones.

        :param n: number of events to *attempt*. Events that produce no matched
            observation or no measurable time delay are dropped, so the returned
            population is smaller than ``n``.
        :param required_model: lens class that must yield a delay (see
            :meth:`generate_one`).
        :param lens_models: lens classes to measure delays for.
        :param z_range: redshift sampling range. Defaults to exactly the range over
            which the model covers both bands (:func:`redshift_limits`), which is
            per-band-pair -- so the redder pairs get their full reach instead of
            being clipped to a bluer band's limit.
        :param mjd_range: range of ``t = 0`` MJDs to sample. Keep this inside the
            discovery window: it is when the *leading* image goes off. Sampled from
            this parameter (an earlier version hard-coded the year 1-3 MJDs, which
            silently ignored a changed baseline or a passed-in range).
        :param discovery_mjd_high: end of the leading-image window (default: year 3)
        :param followup_mjd_high: end of the trailing-image window (default: year 5)
        :param magnification_range: range of magnifications to sample
        :param verbose: forwarded to :meth:`generate_one`
        """
        if z_range is None:
            z_range = redshift_limits(self.model, band1, band2)

        population = []
        for _ in range(n):
            z = np.random.uniform(*z_range)
            x1 = skewnorm.rvs(-8.24, 1.23, 1.67)
            abs_mag = np.random.normal(-19.23, 0.1)
            c = skewnorm.rvs(2.48, -0.089, 0.12)
            magnification = np.random.uniform(*magnification_range)
            times = np.random.uniform(*mjd_range)

            # Oversample: create_sky_pointings applies a declination cut, so asking
            # for one point sometimes returns none.
            ra, dec = create_sky_pointings(10)
            if len(ra) == 0:
                continue

            result = self.generate_one(z, x1, abs_mag, c, magnification, times,
                                       ra[0], dec[0], band1, band2,
                                       required_model=required_model,
                                       lens_models=lens_models,
                                       discovery_mjd_high=discovery_mjd_high,
                                       followup_mjd_high=followup_mjd_high,
                                       verbose=verbose)
            if result is not None:
                population.append(result)

        return population


class Supernovae2:
    """SN Ia population evaluated straight from the model light curve.

    No cadence realisation: magnitudes are sampled on a fixed time grid and cut on
    the single-visit depth. Much faster than :class:`Supernovae`, and used for the
    dense background population in the colour-magnitude diagrams.
    """

    def __init__(self, model_name='salt3', times=np.linspace(54950, 55100, 1000)):
        self.model = sncosmo.Model(source=model_name)
        self.times = times

    def generate_one(self, z, x1, abs_mag, c, magnification, band1, band2,
                     t0=55000, min_detections=2):
        self.model.set(z=z, t0=t0, x1=x1, c=c)
        self.model.set_source_peakabsmag(abs_mag, band2, 'ab')
        self.model.set_source_peakabsmag(abs_mag, band1, 'ab')
        self.band1 = band1
        self.band2 = band2

        if not _covers(self.model, band1, band2):
            return None

        mags1 = self.model.bandmag(band1, 'ab', self.times)
        mags2 = self.model.bandmag(band2, 'ab', self.times)
        weights_ul = ul_weight(z)
        det_mags1_ul = detection(mags1, band1, min_detections)
        det_mags2_ul = detection(mags2, band2, min_detections)

        lensed_abs_mag = abs_mag - 2.5 * np.log10(magnification)
        self.model.set_source_peakabsmag(lensed_abs_mag, band2, 'ab')
        self.model.set_source_peakabsmag(lensed_abs_mag, band1, 'ab')

        mags_l1 = self.model.bandmag(band1, 'ab', self.times)
        mags_l2 = self.model.bandmag(band2, 'ab', self.times)
        weights_l = l_weight(z, magnification)
        det_mags1_l = detection(mags_l1, band1, min_detections)
        det_mags2_l = detection(mags_l2, band2, min_detections)

        return {
            'z': z,
            'x1': x1,
            'c': c,
            't0': t0,
            'magnification': magnification,
            'weights_ul': weights_ul,
            'weights_l': weights_l,
            'mags1': mags1,
            'mags2': mags2,
            'mags_l1': mags_l1,
            'mags_l2': mags_l2,
            'det_mags1_ul': det_mags1_ul,
            'det_mags2_ul': det_mags2_ul,
            'det_mags1_l': det_mags1_l,
            'det_mags2_l': det_mags2_l,
        }

    def generate_many(self, n, band1, band2, z_range=None,
                      magnification_range=(2.0, 50.0), min_detections=2):
        """Draw ``n`` events. ``z_range`` defaults to the model's coverage range for
        this band pair (:func:`redshift_limits`)."""
        if z_range is None:
            z_range = redshift_limits(self.model, band1, band2)

        population = []
        for _ in range(n):
            z = np.random.uniform(*z_range)
            x1 = skewnorm.rvs(-8.24, 1.23, 1.67)
            abs_mag = np.random.normal(-19.23, 0.1)
            c = skewnorm.rvs(2.48, -0.089, 0.12)
            magnification = np.random.uniform(*magnification_range)
            result = self.generate_one(z, x1, abs_mag, c, magnification,
                                       band1, band2, min_detections=min_detections)
            if result is not None:
                population.append(result)
        return population


class Nugent:
    """Core-collapse contaminant populations from the Nugent templates.

    Same idealised treatment as :class:`Supernovae2`, but the template (and its
    absolute-magnitude distribution) is chosen per call so a single instance can
    generate SN Ibc, IIL, IIn and IIP populations. These are the contaminants the
    colour-magnitude cut has to reject.
    """

    def __init__(self, times=np.linspace(54990, 55100, 1000)):
        self.times = times

    def generate_one(self, z, abs_mag, magnification, band1, band2, contam_model,
                     t0=55000, min_detections=2):
        self.model = sncosmo.Model(source=contam_model)
        self.model.set(z=z, t0=t0)
        self.model.set_source_peakabsmag(abs_mag, band2, 'ab')
        self.model.set_source_peakabsmag(abs_mag, band1, 'ab')
        self.band1 = band1
        self.band2 = band2

        if not _covers(self.model, band1, band2):
            return None

        mags1 = self.model.bandmag(band1, 'ab', self.times)
        mags2 = self.model.bandmag(band2, 'ab', self.times)
        weights_ul = cc_ul_weight(z)
        det_mags1_ul = detection(mags1, band1, min_detections)
        det_mags2_ul = detection(mags2, band2, min_detections)

        lensed_abs_mag = abs_mag - 2.5 * np.log10(magnification)
        self.model.set_source_peakabsmag(lensed_abs_mag, band2, 'ab')
        self.model.set_source_peakabsmag(lensed_abs_mag, band1, 'ab')

        mags_l1 = self.model.bandmag(band1, 'ab', self.times)
        mags_l2 = self.model.bandmag(band2, 'ab', self.times)
        weights_l = cc_l_weight(z, magnification)
        det_mags1_l = detection(mags_l1, band1, min_detections)
        det_mags2_l = detection(mags_l2, band2, min_detections)

        return {
            'z': z,
            'x1': np.nan,     # not a SALT model; kept so populations share a schema
            'c': np.nan,
            't0': t0,
            'magnification': magnification,
            'weights_ul': weights_ul,
            'weights_l': weights_l,
            'det_mags1_ul': det_mags1_ul,
            'det_mags2_ul': det_mags2_ul,
            'det_mags1_l': det_mags1_l,
            'det_mags2_l': det_mags2_l,
        }

    def generate_many(self, n, band1, band2, abs_mag_avg, abs_mag_sig, contam_model,
                      z_range=None, magnification_range=(2.0, 50.0),
                      min_detections=2):
        """Draw ``n`` events from ``contam_model``. ``z_range`` defaults to that
        template's coverage range for this band pair (:func:`redshift_limits`).

        The Nugent templates span 1000-25000 A rest-frame, far wider than ``salt3``,
        so contaminants reach much higher redshift than the SN Ia population for the
        same band pair -- which is the point: they have to be rejected out there too.
        """
        if z_range is None:
            z_range = redshift_limits(contam_model, band1, band2)

        population = []
        for _ in range(n):
            z = np.random.uniform(*z_range)
            abs_mag = np.random.normal(abs_mag_avg, abs_mag_sig)
            magnification = np.random.uniform(*magnification_range)
            result = self.generate_one(z, abs_mag, magnification, band1, band2,
                                       contam_model, min_detections=min_detections)
            if result is not None:
                population.append(result)
        return population
