# Colour-Magnitude-SNe

Simulating LSST observations of (lensed and unlensed) Type Ia supernovae and
building colour-magnitude diagrams from them.

The original prototype (`Summer Project.ipynb`) has been split into a reusable
`cmsne` package plus a set of task notebooks.

## Layout

```
cmsne/                     # reusable library
  config.py                # constants: OpSim path, survey dates, cosmology
  sn_rates.py              # cosmic star-formation & SN Ia rate model
  opsim.py                 # loading/querying the LSST OpSim database
  observations.py          # cadence realisations, time cuts, nightly coadds
  lsst.py                  # LSST band properties, plot styles, detection limits
  lightcurve.py            # synthetic light curves (sncosmo + OpSim)
  colour_magnitude.py      # colour-magnitude weighting, boundaries & time delays
  supernovae.py            # SN population generators (Ia + contaminants)
  plotting.py              # light-curve plotting helpers

notebooks/                 # one task per notebook
  01_sn_rates.ipynb        # cosmic SN Ia rates & observability
  02_opsim_footprint.ipynb # OpSim survey timeline, footprint, N_obs maps
  03_lightcurves.ipynb     # build & plot a single supernova light curve
  04_colour_magnitude.ipynb# colour-magnitude diagrams for every band pair   (GENERATED)
  05_time_delays.ipynb     # identification epoch vs image-multiplicity epoch (GENERATED)

scripts/
  build_notebooks.py       # regenerates notebooks 04 and 05
  extract_pointings.py     # export the per-visit OpSim pointing table to CSV

tests/
  test_cmsne.py            # unit checks on a synthetic cadence (no database needed)
  verify_real_opsim.py     # end-to-end checks against a real OpSim database
```

## Tests

```bash
python tests/test_cmsne.py          # 99 checks, needs no database
python tests/verify_real_opsim.py   # 36 checks, needs CMSNE_OPSIM_DB
```

Both exit non-zero on failure. `test_cmsne.py` stubs `opsimsummaryv2` and feeds a
synthetic visit table, so it runs anywhere and is the one to run routinely.
`verify_real_opsim.py` **skips** (exit 0) when `CMSNE_OPSIM_DB` is unset, points at
a missing file, or `opsimsummaryv2` is absent.

Many of these checks pin specific bugs that have occurred in this code — mixed
observer/rest frames, magnitudes divided by `1 + z`, a re-zeroed observation time
axis destroying the lensing delay, biased weighted resampling, and per-band-pair
redshift limits. Run them after changing anything in `cmsne/`.

> **Notebooks 04 and 05 are generated.** Edit `scripts/build_notebooks.py` and re-run
> it rather than editing the `.ipynb` files, or the two will silently diverge.
> Notebooks 01–03 are hand-maintained.

## Frame conventions

Observation days are converted to the **rest frame** (divided by `1 + z`) by
`colour_magnitude.individual_observations`. Every epoch derived from them must
therefore also be rest-frame, including the intrinsic light-curve peak day they
are measured against — otherwise the colour-method epoch (`time_of_SN`) and the
image-multiplicity epoch (`time_delay`) are not comparable, and they are plotted
against each other in notebook 05.

Magnitudes are **never** rescaled by `1 + z`. A magnitude is logarithmic, so
dividing one by `1 + z` is not a K-correction; it just compresses the colour axis
by a redshift-dependent factor.

## Setup

### Local (conda)

`opsimsummaryv2` needs `numpy>=1.24`, `healpy` and `sqlalchemy`, so it is worth a
dedicated environment rather than sharing one with another project:

```bash
conda create -y -n cmsne-env -c conda-forge python=3.11 \
  numpy scipy pandas matplotlib astropy scikit-learn healpy sqlalchemy \
  sncosmo pyarrow iminuit jupyter ipykernel
conda activate cmsne-env
# --no-deps so pip cannot clobber the conda-installed compiled stack
pip install --no-deps git+https://github.com/LSSTDESC/OpSimSummaryV2.git
```

### OpSim database

Baselines live at
`https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/`
(the 10-year files are ~715 MB):

```bash
mkdir -p ~/opsim_data && cd ~/opsim_data
curl -LO https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/baseline_v5.3.5_10yrs.db
```

Point the package at it with an environment variable — no need to edit any file,
so the same checkout works locally and on Colab:

```bash
export CMSNE_OPSIM_DB=~/opsim_data/baseline_v5.3.5_10yrs.db
```

`cmsne/config.py` falls back to the Colab Drive path when the variable is unset.
Set it **before** importing `cmsne`: the value is bound as a function default at
import time.

> **Changing OpSim version:** `config.survey_dates` is hardcoded from the survey
> start MJD. Re-derive it (notebook 02 prints the true earliest/latest MJD) if you
> switch baselines, or every time cut silently shifts.

## Usage

Run the notebooks in order (they assume the `cmsne` package one directory up,
which the setup cell adds to `sys.path`). Import the library directly with, e.g.:

```python
from cmsne.opsim import initialise_opsim_summary
from cmsne.lightcurve import Transient, get_observations
```
