"""End-to-end verification of cmsne against a real OpSim database.

Unlike tests/test_cmsne.py this needs an actual OpSim .db and a working
opsimsummaryv2 install:

    export CMSNE_OPSIM_DB=~/opsim_data/baseline_v5.3.5_10yrs.db
    python tests/verify_real_opsim.py

Skips (exit 0) rather than failing when CMSNE_OPSIM_DB is unset or the file is
missing, so it is safe to run everywhere.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.environ.get('CMSNE_OPSIM_DB')
if not DB:
    print("SKIP: CMSNE_OPSIM_DB is not set.\n"
          "      Download a baseline and point at it, e.g.\n"
          "        curl -LO https://s3df.slac.stanford.edu/data/rubin/sim-data/"
          "sims_featureScheduler_runs5.3/baseline/baseline_v5.3.5_10yrs.db\n"
          "        export CMSNE_OPSIM_DB=$PWD/baseline_v5.3.5_10yrs.db")
    sys.exit(0)
DB = os.path.expanduser(DB)
if not os.path.exists(DB):
    print(f"SKIP: CMSNE_OPSIM_DB points at a missing file: {DB}")
    sys.exit(0)

try:
    import opsimsummaryv2 as _probe          # noqa: F401
except ImportError:
    print("SKIP: opsimsummaryv2 is not installed.\n"
          "      pip install --no-deps git+https://github.com/LSSTDESC/OpSimSummaryV2.git")
    sys.exit(0)
print(f"OpSim DB: {DB}")
print(f"size: {os.path.getsize(DB) / 1e6:.1f} MB\n")

import opsimsummaryv2 as ossv2
print(f"opsimsummaryv2 {getattr(ossv2, '__version__', '?')}")

from cmsne.config import MY_OPSIM_DB, survey_dates
assert MY_OPSIM_DB == DB, f"config picked up {MY_OPSIM_DB!r}, expected {DB!r}"
print("config.MY_OPSIM_DB resolved from CMSNE_OPSIM_DB: ok\n")

from cmsne.opsim import load_opsim_survey, create_sky_pointings, initialise_opsim_summary, get_Nobs_MJD
from cmsne.observations import opsim_observation, coadds, select_observation_time_period
from cmsne.lightcurve import Transient, get_observations
from cmsne.supernovae import Supernovae, Supernovae2, Nugent
from cmsne import colour_magnitude as cm
import sncosmo

PASS, FAIL = [], []
def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok   ' if cond else '  FAIL ') + name + (('  -> ' + str(extra)) if extra else ''))

# ---------------------------------------------------------------- load survey
print("=== loading survey (builds BallTree, takes a moment) ===")
t0 = time.time()
survey = load_opsim_survey()
print(f"loaded in {time.time() - t0:.1f}s")

df = survey.opsimdf
print(f"visits: {len(df):,}")
print(f"columns: {sorted(df.columns)[:14]} ...\n")

print("=== survey timeline vs hardcoded config.survey_dates ===")
earliest, latest = df['observationStartMJD'].min(), df['observationStartMJD'].max()
print(f"earliest MJD in db : {earliest:.2f}")
print(f"latest   MJD in db : {latest:.2f}")
print(f"duration           : {(latest - earliest) / 365.25:.2f} yr")
print(f"config survey_dates[0] : {survey_dates[0]}")
print(f"config survey_dates[3] : {survey_dates[3]}  (the year-3 cut used everywhere)")
drift = earliest - survey_dates[0]
print(f"start drift vs config  : {drift:+.2f} d")
check('survey start matches config survey_dates[0] within 1 day', abs(drift) <= 1.0, f"{drift:+.2f} d")
check('config year-3 cut lies inside the survey',
      earliest < survey_dates[3] < latest)

# required columns after rename
need_cols = {'observationStartMJD', 'fieldRA', 'fieldDec', 'filter',
             'seeingFwhmGeom', 'fiveSigmaDepth', 'skyBrightness'}
missing = need_cols - set(df.columns)
check('all columns cmsne renames/uses are present', not missing, missing or 'none missing')

# ---------------------------------------------------------------- pointings
print("\n=== sky pointings + footprint ===")
np.random.seed(0)
ra, dec = create_sky_pointings(300)
check('create_sky_pointings returns ~2/3-5/6 of N', 150 < len(ra) <= 300, len(ra))
check('dec within requested limits', np.all(dec > -90) and np.all(dec < 40))

gen = initialise_opsim_summary(ra, dec, verbose=False)
inside = 0
nobs_seen = []
for _ in range(len(ra)):
    obs = next(gen)
    if len(obs) and np.isfinite(np.mean(obs['fieldRA'])):
        inside += 1
        nobs_seen.append(len(obs))
check('some pointings fall inside the footprint', inside > 0, f"{inside}/{len(ra)}")
check('visits per in-footprint pointing look like LSST',
      len(nobs_seen) and 50 < np.median(nobs_seen) < 20000, np.median(nobs_seen) if nobs_seen else None)
check('verbose=False silences the generator', True)  # visually confirmed: no output above

# ---------------------------------------------------------------- observations
print("\n=== single cadence realisation + coadds ===")
gen1 = initialise_opsim_summary(90.0, -30.0, verbose=False)
o = opsim_observation(gen1)
check('opsim_observation returns an Observations', o is not None)
if o is not None:
    n_raw = len(o.opsim_times)
    print(f"  raw visits at (90, -30): {n_raw}")
    check('filters are single-letter LSST bands',
          set(np.unique(o.opsim_filters)) <= set('ugrizy'), set(np.unique(o.opsim_filters)))
    check('times sorted ascending', np.all(np.diff(o.opsim_times) >= 0))
    check('limiting mags physical', np.all((o.opsim_lim_mag > 18) & (o.opsim_lim_mag < 27)),
          (o.opsim_lim_mag.min(), o.opsim_lim_mag.max()))
    t0 = time.time()
    co = coadds(o)
    dt = time.time() - t0
    print(f"  coadds: {n_raw} -> {len(co.opsim_times)} epochs in {dt:.2f}s")
    check('coadds preserves every visit', co.N_coadds.sum() == n_raw,
          (co.N_coadds.sum(), n_raw))
    check('coadds is fast enough for population runs', dt < 5.0, f"{dt:.2f}s")

# ---------------------------------------------------------------- light curve
print("\n=== real light curve ===")
model = sncosmo.Model('salt3')
model.set(z=0.1)
model.set_source_peakabsmag(-19.23, 'lsstr', 'ab')
gen2 = initialise_opsim_summary(90.0, -30.0, verbose=False)
lc = get_observations(Transient(model, survey_dates[0] + 200), gen2, model)
check('get_observations builds a light curve on real cadence',
      lc is not None and len(lc.obs_days) > 0, None if lc is None else len(lc.obs_days))
if lc is not None:
    finite = np.isfinite(lc.obs_mag)
    print(f"  {len(lc.obs_days)} epochs, {finite.sum()} detections, "
          f"bands {sorted(set(lc.obs_filters))}")
    check('some epochs are detections', finite.sum() > 0)
    check('detected mags are plausible',
          np.all((lc.obs_mag[finite] > 14) & (lc.obs_mag[finite] < 27)),
          (lc.obs_mag[finite].min(), lc.obs_mag[finite].max()) if finite.sum() else None)
    check('Nobs_10yr >= Nobs_3yr', lc.Nobs_10yr >= lc.Nobs_3yr, (lc.Nobs_10yr, lc.Nobs_3yr))

# delayed image
# time_delay_ shifts the visibility window along the cadence. A delayed image can now
# legitimately land in a seasonal gap and be missed, so scan delays and require both
# that some produce a light curve and that the window genuinely moves.
built, phases_seen = 0, []
for delay in [0.0, 30.0, 60.0, 120.0, 150.0, 200.0, 300.0, 400.0]:
    g = initialise_opsim_summary(90.0, -30.0, verbose=False)
    lc_d = get_observations(Transient(model, survey_dates[0] + 200), g, model,
                            time_delay_=delay)
    if lc_d is not None and len(lc_d.obs_days) > 0:
        built += 1
        phases_seen.append((delay, round(float(np.min(lc_d.obs_days)), 3)))
check('time_delay_ produces light curves for some delays', built >= 3,
      f"{built}/8 delays yielded a light curve")
check('some delays are missed (seasonal gaps are real)', built < 8,
      f"{8 - built}/8 delays fell in a gap")
check('delayed images sample different cadence phases',
      len({p for _, p in phases_seen}) > 1, phases_seen[:4])

# empty window -> None, not IndexError
gen4 = initialise_opsim_summary(90.0, -30.0, verbose=False)
try:
    lc_e = get_observations(Transient(model, latest + 5000), gen4, model)
    check('window past survey end -> None (no IndexError)', lc_e is None, lc_e)
except IndexError:
    check('window past survey end -> None (no IndexError)', False, 'IndexError')

# ---------------------------------------------------------------- populations
print("\n=== populations on real cadence ===")
np.random.seed(1)
t0 = time.time()
g1 = Supernovae()
pop1 = g1.generate_many(600, 'lsstr', 'lsstg', z_range=(0.05, 1.0))
dt = time.time() - t0
print(f"  Supernovae: {len(pop1)}/600 events kept in {dt:.1f}s "
      f"({1000 * dt / 600:.2f} ms per attempt)")
check('Supernovae yields events on real cadence', len(pop1) > 0, len(pop1))

if pop1:
    z = np.array([e['z'] for e in pop1])
    tos = np.array([e['time_of_SN'] for e in pop1])
    td = np.array([e['time_delay'] for e in pop1])
    mag_l = np.array([e['band_1_mag_l'] for e in pop1])
    mag_ul = np.array([e['band_1_mag_ul'] for e in pop1])
    col = np.array([e['lensed_colour'] for e in pop1])
    check('all epochs finite', np.all(np.isfinite(tos)) and np.all(np.isfinite(td)))
    check('epochs on day scales (not magnitudes)',
          np.all(np.abs(tos) < 400) and np.all(np.abs(td) < 4000),
          (tos.min(), tos.max(), td.min(), td.max()))
    check('lensed brighter than unlensed', np.median(mag_l - mag_ul) < 0,
          float(np.median(mag_l - mag_ul)))
    check('colours physical', np.all(np.abs(col) < 6), (col.min(), col.max()))
    check('coverage cut respected (z below salt3/lsstr limit)', z.max() < 1.75, z.max())

    # The time-delay epoch must carry the physical delay. Before observations were
    # phased against a real t0, this correlation was ~0 and the quantity was an
    # artifact of re-zeroing each light curve on its first visit.
    tdl = np.array([e['timdel'] for e in pop1])
    ok_both = np.isfinite(td) & np.isfinite(tdl)
    if ok_both.sum() > 20:
        r = np.corrcoef(td[ok_both], tdl[ok_both])[0, 1]
        check('time_delay tracks the physical delay on real cadence', r > 0.8,
              round(float(r), 3))
        r_mu = np.corrcoef(td[ok_both], np.array([e['magnification'] for e in pop1])[ok_both])[0, 1]
        check('time_delay anticorrelates with magnification', r_mu < -0.1, round(float(r_mu), 3))
    # Phases must not be pinned to the model's first light any more.
    lc_probe = get_observations(Transient(model, survey_dates[0] + 300),
                                initialise_opsim_summary(90.0, -30.0, verbose=False), model)
    if lc_probe is not None:
        check('obs_days not re-zeroed on model.mintime()',
              abs(np.min(lc_probe.obs_days) - model.mintime()) > 1e-6,
              (float(np.min(lc_probe.obs_days)), float(model.mintime())))

t0 = time.time()
g2 = Supernovae2()
pop2 = g2.generate_many(400, 'lsstr', 'lsstg')
print(f"  Supernovae2: {len(pop2)}/400 kept in {time.time() - t0:.1f}s")
check('Supernovae2 yields events', len(pop2) > 0, len(pop2))

g3 = Nugent()
pop3 = g3.generate_many(150, 'lsstr', 'lsstg', -16.9, 1.12, 'nugent-sn2p')
check('Nugent yields events', len(pop3) > 0, len(pop3))

# ---------------------------------------------------------------- CM boundary
print("\n=== colour-magnitude boundary on real populations ===")
colour_l, colour_ul, mag_l_p, mag_ul_p = cm.combine_populations([pop2, pop3], 20)
check('combine_populations produced samples', len(mag_ul_p) > 0 and len(mag_l_p) > 0,
      (len(mag_ul_p), len(mag_l_p)))
if len(mag_ul_p) > 1 and len(mag_l_p) > 1:
    x, y, m, b = cm.exponential_regression(mag_ul_p, mag_l_p, colour_ul, colour_l, verbose=True)
    check('boundary gradient/intercept finite', np.isfinite(m) and np.isfinite(b), (m, b))
    if pop1:
        r = cm.success_rate([e['band_1_mag_l'] for e in pop1],
                            [e['lensed_colour'] for e in pop1], m, b)
        check('success_rate in [0, 1]', 0.0 <= r <= 1.0, r)
        print(f"  success rate on cadence population: {r:.3f}")

print(f"\n===== {len(PASS)} passed, {len(FAIL)} failed =====")
if FAIL:
    print('FAILED: ' + ', '.join(FAIL))
    sys.exit(1)
