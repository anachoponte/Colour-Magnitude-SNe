"""Loading and querying the LSST OpSim database via OpSimSummaryV2.

This module wraps :mod:`opsimsummaryv2` so the rest of the project can request
random sky pointings and their observing cadence using the same column names as
the original (v1) OpSimSummary tutorial.
"""

import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import opsimsummaryv2 as ossv2

from .config import MY_OPSIM_DB

# Unlike OpSimSummary v1 (which queried the database lazily), OpSimSummaryV2 reads
# the whole OpSim table into memory and builds a BallTree when the OpSimSurvey
# object is constructed. We therefore build it once and cache it, rather than
# rebuilding it for every set of sky pointings.
_opsim_survey_cache = {}


def load_opsim_survey(db_path=MY_OPSIM_DB, verbose=True):
    """
    Build (and cache) an OpSimSummaryV2 OpSimSurvey object for a given OpSim db.

    :param db_path: path to the OpSim .db file
    :param verbose: if False, suppress the one-off "loading database" message
    :return: opsimsummaryv2.OpSimSurvey object (loaded once per db_path per session)
    """
    if db_path not in _opsim_survey_cache:
        if verbose:
            print(f"Loading OpSim database {db_path} (this happens once per session)...")
        _opsim_survey_cache[db_path] = ossv2.OpSimSurvey(db_path)
    return _opsim_survey_cache[db_path]


# OpSimSummaryV2 names two columns differently to the original OpSimSummary. We
# rename them back so the rest of this tutorial can keep using the v1 names:
#   observationStartMJD -> expMJD
#   skyBrightness       -> filtSkyBrightness
# (fieldRA, fieldDec, filter, seeingFwhmGeom and fiveSigmaDepth are unchanged.)
_V2_COLUMN_RENAME = {'observationStartMJD': 'expMJD', 'skyBrightness': 'filtSkyBrightness'}


def create_sky_pointings(N, dec_low=-90, dec_high=40):
    """
    Creates random points on a sphere (limited between dec_low and dec_high).
    Acception fraction of points is around 2/3, so sample ~1.6 times as many points as you require.

    :param N: number of points sampled. Roughly 2/3 of these will lie inside the LSST footprint for the default parameters.
    :param dec_low: lower declination limit
    :param dec_high: upper declination limit
    :return: two arrays containing the x-coordinates (right ascension) and y-coordinates (declination) of random sky pointings
    """

    sample_number = N

    ra_points = np.random.uniform(low=0, high=360, size=sample_number)
    dec_points = np.arcsin(2 * np.random.uniform(size=sample_number) - 1) / np.pi * 180

    dec_selection = (dec_points > dec_low) & (dec_points < dec_high)
    ra_points = ra_points[dec_selection]
    dec_points = dec_points[dec_selection]

    return ra_points, dec_points


def initialise_opsim_summary(ra_pointings, dec_pointings, db_path=MY_OPSIM_DB,
                             verbose=True):
    """
    Initialise the generator that draws cadence realisations from the OpSim database.

    This is the OpSimSummaryV2 equivalent of v1's
    SynOpSim.fromOpSimDB(...).pointingsEnclosing(...). It yields one observation
    table (pandas DataFrame) per input sky pointing, in the same order as the
    input arrays. Pointings outside the LSST footprint yield an empty DataFrame
    (so np.mean(obs['fieldRA']) returns NaN, exactly as in the v1 version).

    :param ra_pointings: array (or scalar) of right ascensions of sky pointings [deg]
    :param dec_pointings: array (or scalar) of declinations of sky pointings [deg]
    :param db_path: path to the OpSim .db file
    :param verbose: if False, stay silent. Population generators call this once per
        event, so the default chatter is worth turning off in loops rather than
        muting ``sys.stdout`` wholesale.
    :return: generator yielding one OpSim observation DataFrame per sky pointing
    """

    if verbose:
        print("Setting up OpSimSummaryV2 generator...")
    survey = load_opsim_survey(db_path, verbose=verbose)

    # get_obs_from_coords expects arrays; wrap scalars so single-pointing calls work.
    ra_pointings = np.atleast_1d(ra_pointings).astype(float)
    dec_pointings = np.atleast_1d(dec_pointings).astype(float)

    # is_deg=True: coordinates are in degrees.
    # formatobs=False: return the raw OpSim columns. (formatobs=True would instead
    # return SNANA-style PSF/ZPT/SKYSIG; this tutorial uses seeingFwhmGeom and its
    # own LSST zeropoints, so we keep the raw columns.)
    raw_gen = survey.get_obs_from_coords(ra_pointings, dec_pointings, is_deg=True,
                                         formatobs=False)

    # Rename the two changed columns on the fly so downstream code is unchanged.
    return (obs.rename(columns=_V2_COLUMN_RENAME) for obs in raw_gen)


def get_Nobs_MJD(ra, dec, gen, MJD):
    """
    Calculates important properties for any dates < MJD
    :param ra: input right ascension
    :param dec: input declination
    :param gen: OpSim generator
    :param MJD: cutoff date (single value or list of values)
    returns: dictionary or list of dictionaries containing opsim_ra (opsim visit right ascensions), opsim_dec (opsim visit declinations), Nobs (Number of observations up to MJD), Nobs_10 (Number of observations in 10 years), orig_ra (the original input right ascension), orig_dec (the original input declination)
    """

    # Check if MJD is a list/array or single value
    is_list = isinstance(MJD, (list, np.ndarray))
    if not is_list:
        MJD = [MJD]  # Convert single value to list for uniform processing

    # Initialize result containers for each MJD
    results = []
    for _ in MJD:
        results.append({
            'opsim_ra_list': [],
            'opsim_dec_list': [],
            'Nobs': [],
            'Nobs_10': [],
            'orig_ra': [],
            'orig_dec': []
        })

    # Single iteration over the generator
    for p in range(len(ra)):
        obs = next(gen)
        obs = obs.sort_values(by=['expMJD'])

        opsim_ra = np.mean(obs['fieldRA'])
        opsim_dec = np.mean(obs['fieldDec'])
        opsim_mjd = obs['expMJD']

        if np.isnan(opsim_ra) or np.isnan(opsim_dec):
            continue

        # Calculate values for all MJD cutoffs
        for i, mjd_val in enumerate(MJD):
            indices = opsim_mjd < mjd_val

            if len(opsim_mjd[indices]) > 0.0:
                results[i]['Nobs'].append(len(opsim_mjd[indices]))
                results[i]['Nobs_10'].append(len(opsim_mjd))
                results[i]['opsim_ra_list'].append(opsim_ra)
                results[i]['opsim_dec_list'].append(opsim_dec)
                results[i]['orig_ra'].append(ra[p])
                results[i]['orig_dec'].append(dec[p])

    # Process results and convert to final format
    final_results = []
    for result in results:
        Nobs = np.array(result['Nobs'])
        Nobs_10 = np.array(result['Nobs_10'])
        opsim_ra_list = np.array(result['opsim_ra_list'])
        opsim_dec_list = np.array(result['opsim_dec_list'])

        opsim_ra1 = coord.Angle(opsim_ra_list * u.degree)
        opsim_dec_coord = coord.Angle(opsim_dec_list * u.degree)
        opsim_ra = - opsim_ra1.wrap_at(180 * u.degree)

        orig_ra_list = np.array(result['orig_ra'])
        orig_dec_list = np.array(result['orig_dec'])
        orig_ra1 = coord.Angle(orig_ra_list * u.degree)
        orig_dec_coord = coord.Angle(orig_dec_list * u.degree)
        orig_ra_converted = - orig_ra1.wrap_at(180 * u.degree)

        result_dict = {
            'opsim_ra': opsim_ra,
            'opsim_dec': opsim_dec_coord,
            'Nobs': Nobs,
            'Nobs_10': Nobs_10,
            'orig_ra': orig_ra_converted,
            'orig_dec': orig_dec_coord
        }

        final_results.append(result_dict)

    # Return single dictionary if input was single value, otherwise return list
    if not is_list:
        return final_results[0]
    else:
        return final_results
