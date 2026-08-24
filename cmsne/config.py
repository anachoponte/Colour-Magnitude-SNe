"""Project-wide constants shared across the ``cmsne`` package.

Edit the values here (in particular :data:`MY_OPSIM_DB`) to match your own
environment rather than hard-coding paths inside the analysis modules.
"""

import os

# ---------------------------------------------------------------------------
# OpSim database
# ---------------------------------------------------------------------------
# Path to the LSST OpSim ``.db`` file used by :mod:`cmsne.opsim`.
#
# Set the CMSNE_OPSIM_DB environment variable to override this without editing
# the file, so the same checkout works on a local machine and on Colab:
#     export CMSNE_OPSIM_DB=~/opsim_data/baseline_v5.3.5_10yrs.db
# The default remains the Colab Drive path (now the v5.3.5 baseline the survey
# timeline below was derived from).
DEFAULT_OPSIM_DB = '/content/drive/MyDrive/baseline_v5.3.5_10yrs.db'

MY_OPSIM_DB = os.path.expanduser(os.environ.get('CMSNE_OPSIM_DB', DEFAULT_OPSIM_DB))

# ---------------------------------------------------------------------------
# Survey timeline
# ---------------------------------------------------------------------------
# MJD of the start of each LSST observing year (year 0 ... year 10).
survey_dates = [61208, 61573, 61939, 62304, 62669, 63034,
                63400, 63765, 64130, 64495, 64860]

# ---------------------------------------------------------------------------
# Cosmology / SN Ia rate constants
# ---------------------------------------------------------------------------
H0 = 70          # Hubble constant [km/s/Mpc]
OMEGA_M = 0.3    # matter density parameter
OMEGA_V = 0.7    # dark-energy density parameter

# Delay-time-distribution normalisations used in the SN Ia rate integral.
N0 = 1.02e-4
N1 = 1.22e-3

# Fiducial absolute magnitude of a SN Ia.
MSN = -19.36