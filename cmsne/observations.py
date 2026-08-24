"""Turning raw OpSim cadence realisations into tidy observation records.

This module defines the :class:`Observations` container plus the helpers that
draw a single cadence realisation from the OpSim generator, trim it to a time
window, and combine same-night visits into nightly coadds.
"""

from dataclasses import dataclass

import numpy as np

from .config import survey_dates


class GeneratorWrapper:
    def __init__(self, generator):
        """
        This class acts as a wrapper around a generator to allow flexibility with selecting which instance (current
        or next) of the generator to obtain.
        :param generator: The generator object to be wrapped.
        """

        self.generator = generator
        self.current = None
        self._advance()  # Advance to the first value

    def _advance(self):
        try:
            self.current = next(self.generator)
        except StopIteration:
            self.current = None

    def get_current(self):
        return self.current

    def get_next(self):
        self._advance()
        return self.current


# This dataclass will store the relevant metadata for each opsim pointing
@dataclass
class Observations:
    ra: float
    dec: float
    opsim_times: np.ndarray
    opsim_filters: np.ndarray
    opsim_psf: np.ndarray
    opsim_lim_mag: np.ndarray
    opsim_sky_brightness: np.ndarray
    # Populated by :func:`coadds`; declared here so the attribute is discoverable
    # rather than being attached to the instance out of nowhere.
    N_coadds: np.ndarray = None


def opsim_observation(gen_input, use_previous=False):
    """
    Function to draw one random cadence realisation for 1 sky position from the OpSim database.
    If a sky pointing is outside of the LSST footprint, the function continues until the point is in the footprint.
    :param gen_input: Either a raw generator (from initialise_opsim_summary) OR a GeneratorWrapper instance.
    :param use_previous: Bool. If True, and `gen_input` is a GeneratorWrapper, its current state is used. If False, the next item from `gen_input` is taken.
                         NOTE: If `gen_input` is a raw generator, `use_previous` is effectively ignored, and `next(gen_input)` is always called.
    :return: Observations dataclass containing 2 floats with the right ascension and declination of the observation, and 5 arrays containing the observation times (in MJD), filters, PSF FWHM ('seeingFwhmGeom'), limiting magnitude, and sky brightness for 10 years of LSST observations.
    """
    obs = None

    if isinstance(gen_input, GeneratorWrapper):
        if use_previous == False:
            obs = gen_input.get_next()
        else:  # use_previous == True
            obs = gen_input.get_current()
    else:  # gen_input is a raw generator (e.g., from initialise_opsim_summary)
        # In this case, use_previous is not relevant; we just get the next item from the raw generator.
        try:
            obs = next(gen_input)
        except StopIteration:
            obs = None

    if obs is None or obs.empty:  # Handle cases where no observations are returned or DataFrame is empty
        return None

    obs = obs.sort_values(by=['expMJD'])

    opsim_ra = np.mean(obs['fieldRA'])
    opsim_dec = np.mean(obs['fieldDec'])
    opsim_times = np.array(obs['expMJD'])
    opsim_filters = np.array(obs['filter'])
    opsim_psf = np.array(obs['seeingFwhmGeom'])
    opsim_lim_mag = np.array(obs['fiveSigmaDepth'])
    opsim_sky_brightness = np.array(obs['filtSkyBrightness'])

    observations = Observations(opsim_ra, opsim_dec, opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness)

    return observations


def select_observation_time_period(observations, mjd_low, mjd_high=survey_dates[3]):
    """
    Function to limit the LSST observations to a shorter time duration. The default selection excludes everything
    after the third year of observations (MJD = 61325).

    .. note::
       This mutates ``observations`` in place (and returns it). Draw a fresh
       cadence realisation per transient component rather than reusing one
       :class:`Observations` instance for several cuts.

    :param observations: Observations dataclass
    :param mjd_low: lower threshold (everything before this date will be discarded)
    :param mjd_high: upper threshold (everything after this date will be discarded). Default: end of year 3
    :return: the same Observations dataclass, limited to dates between mjd_low and mjd_high
    """

    indices = (observations.opsim_times > mjd_low) & (observations.opsim_times < mjd_high)

    observations.opsim_times = observations.opsim_times[indices]
    observations.opsim_filters = observations.opsim_filters[indices]
    observations.opsim_psf = observations.opsim_psf[indices]
    observations.opsim_lim_mag = observations.opsim_lim_mag[indices]
    observations.opsim_sky_brightness = observations.opsim_sky_brightness[indices]

    return observations


def coadds(observations):
    """
    Calculates nightly coadds if observations are taken on the same day in the same filter.
    :param observations: Observations dataclass
    :return: updated Observations dataclass arrays of observation times, filters, psf, limiting magnitudes, sky brightness using nightly coadds.
    """

    coadd_times, coadd_filters, coadd_psf, coadd_lim_mag, coadd_sky_brightness = [], [], [], [], []
    n_obs = len(observations.opsim_times)

    N_coadds = []
    # Boolean flags rather than a growing `ID_list` that was searched with `in`:
    # the membership test made this O(N^2) in the number of visits, which dominates
    # the runtime when generating large populations.
    already_coadded = np.zeros(n_obs, dtype=bool)

    for t1 in range(n_obs):

        # Check if observation was already coadded
        if already_coadded[t1]:
            continue

        # Add limiting magnitude, time, and ID
        lim_mag_list = [observations.opsim_lim_mag[t1]]
        times_list = [observations.opsim_times[t1]]
        already_coadded[t1] = True

        for t2 in range(t1 + 1, n_obs):

            # Same day?
            if observations.opsim_times[t2] - observations.opsim_times[t1] >= 1:
                break

            # Same filter?
            if observations.opsim_filters[t1] == observations.opsim_filters[t2]:
                lim_mag_list.append(observations.opsim_lim_mag[t2])
                times_list.append(observations.opsim_times[t2])
                already_coadded[t2] = True

        # Perform coadds
        coadd_lim_mag.append(calculate_coadd(lim_mag_list))
        coadd_times.append(np.mean(times_list))
        N_coadds.append(len(times_list))
        coadd_filters.append(observations.opsim_filters[t1])
        coadd_psf.append(observations.opsim_psf[t1])
        coadd_sky_brightness.append(observations.opsim_sky_brightness[t1])

    observations.opsim_lim_mag = np.array(coadd_lim_mag)
    observations.opsim_times = np.array(coadd_times)
    observations.opsim_filters = np.array(coadd_filters)
    observations.opsim_psf = np.array(coadd_psf)
    observations.opsim_sky_brightness = np.array(coadd_sky_brightness)
    observations.N_coadds = np.array(N_coadds)

    return observations


def calculate_coadd(lim_mag_list):
    """
    Calculates the new limiting magnitude by combining the limiting magnitudes in lim_mag_list.
    Formula from https://smtn-016.lsst.io

    :param lim_mag_list: list containing the limiting magnitudes that need to be coadded
    :return: the coadded limiting magnitude (float)
    """

    lim_mag_array = np.array(lim_mag_list)
    lim_mag_new = 1.25 * np.log10(np.sum(10 ** (0.8 * lim_mag_array)))
    return lim_mag_new