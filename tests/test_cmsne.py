"""Unit checks for the cmsne package, run against a synthetic OpSim cadence.

Needs no OpSim database and no opsimsummaryv2 install -- the module is stubbed and
a synthetic visit table is fed in, so the full light-curve path still executes.

    python tests/test_cmsne.py

Exits non-zero if any check fails. For checks against a real database see
tests/verify_real_opsim.py.
"""
import sys, types, os
import numpy as np
import pandas as pd

# --- stub opsimsummaryv2 so cmsne.opsim imports -----------------------------
stub = types.ModuleType('opsimsummaryv2')
class _FakeSurvey:
    def __init__(self, path): pass
stub.OpSimSurvey = _FakeSurvey
sys.modules['opsimsummaryv2'] = stub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, so `import cmsne` works

import sncosmo
from cmsne import colour_magnitude as cm
from cmsne import supernovae as sn
from cmsne import observations as obsmod
from cmsne import lightcurve as lcmod
from cmsne.config import survey_dates

PASS, FAIL = [], []
def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok   ' if cond else '  FAIL ') + name + (('  -> ' + str(extra)) if extra else ''))

print('\n=== obs_to_mags: epoch matching ===')
# band1 epochs 0,10,20 ; band2 epochs 10.5, 40 -> only day 10 pairs (|dt|=0.5<=3)
d1, d2 = [0.0, 10.0, 20.0], [10.5, 40.0]
m1, m2 = [21.0, 20.0, 22.0], [20.4, 25.0]
b2m, b1m, b1d, b2d = cm.obs_to_mags(d1, d2, m1, m2)
check('pairs nearest-in-time epoch', (b1d, b2d) == (10.0, 10.5), (b1d, b2d))
check('returns that pair\'s mags', (b1m, b2m) == (20.0, 20.4), (b1m, b2m))
check('rejects separations > max_separation',
      all(np.isnan(v) for v in cm.obs_to_mags([0.0], [40.0], [20.0], [21.0])))
check('empty input -> NaNs', all(np.isnan(v) for v in cm.obs_to_mags([], [], [], [])))
check('non-finite mags rejected',
      all(np.isnan(v) for v in cm.obs_to_mags([0.0], [0.5], [np.inf], [21.0])))
check('outside day window rejected',
      all(np.isnan(v) for v in cm.obs_to_mags([500.0], [500.2], [20.0], [21.0])))
# duplicate day values must not collapse (old .index(j) bug)
b2m2, b1m2, _, _ = cm.obs_to_mags([5.0, 5.0], [5.0], [22.0, 19.0], [20.0])
check('duplicate days handled (first pair)', b1m2 == 22.0, b1m2)
bb2, bb1, _, _ = cm.obs_to_mags([5.0, 5.0], [5.0, 5.0], [22.0, 19.0], [20.0, 20.0],
                                select='brightest')
check('select="brightest" picks brightest band1', bb1 == 19.0, bb1)

print('\n=== detection ===')
mags_bright = np.full(1000, 20.0)
mags_faint = np.full(1000, 30.0)
check('bright curve passes', np.allclose(cm.detection(mags_bright, 'lsstr'), 20.0))
check('faint curve blanked', np.all(np.isnan(cm.detection(mags_faint, 'lsstr'))))
check('empty input safe', np.all(np.isnan(cm.detection(np.array([]), 'lsstr'))) or
      cm.detection(np.array([]), 'lsstr').size == 0)
one = np.full(1000, 30.0); one[0] = 20.0
check('1 detection fails min_detections=2', np.all(np.isnan(cm.detection(one, 'lsstr'))))
two = np.full(1000, 30.0); two[0] = 20.0; two[10] = 20.0
check('2 detections pass', np.allclose(cm.detection(two, 'lsstr'), two))

print('\n=== weighted_vals guards ===')
r = cm.weighted_vals(np.array([1.0, 2.0]), [np.nan, np.nan], [np.nan, np.nan], 5)
check('all-NaN population returns empty (no crash)', r[0] == [] and r[1] == [])
r = cm.weighted_vals(np.array([0.0, 0.0]), [20.0, 21.0], [21.0, 22.0], 5)
check('zero-sum weights returns empty (no /0)', r[0] == [])
r = cm.weighted_vals(np.array([1.0, 3.0]), [20.0, 21.0], [21.0, 22.0], 4)
check('oversampling uses replacement', len(r[0]) == 4, len(r[0]))

print('\n=== weighted sampling is unbiased at every draw size ===')
# Strongly skewed weights, and an observable correlated with weight, so any drift
# toward the unweighted population shows up immediately.
rng_w = np.random.default_rng(3)
n_pop = 4000
w_skew = rng_w.pareto(1.1, n_pop) + 1e-3
x_obs = -2.5 * np.log10(w_skew / w_skew.max()) + rng_w.normal(0, 0.3, n_pop)
exact_mean = np.average(x_obs, weights=w_skew)
unweighted = x_obs.mean()
check('test fixture is discriminating (weighted != unweighted)',
      abs(exact_mean - unweighted) > 1.0, (round(exact_mean, 2), round(unweighted, 2)))

# q/n from 0.5% up to 25%: the bias used to grow monotonically with this ratio.
worst = 0.0
for q in [20, 200, 500, 1000]:
    draws = []
    for s in range(8):
        np.random.seed(s)
        draws.append(np.mean(cm.weighted_vals(w_skew, x_obs, x_obs, q)[0]))
    bias = abs(np.mean(draws) - exact_mean)
    worst = max(worst, bias)
    check(f'weighted_vals unbiased at q/n={q / n_pop:.1%}', bias < 0.35, round(float(bias), 3))
check('bias does not grow with draw size', worst < 0.35, round(float(worst), 3))

print('\n=== exact weighted statistics (no resampling) ===')
frac = cm.weighted_fraction(x_obs > exact_mean, w_skew)
mc = []
for s in range(8):
    np.random.seed(s)
    idx = np.random.choice(n_pop, 4000, replace=True, p=w_skew / w_skew.sum())
    mc.append(np.mean(x_obs[idx] > exact_mean))
check('weighted_fraction matches weighted resampling', abs(frac - np.mean(mc)) < 0.02,
      (round(frac, 3), round(float(np.mean(mc)), 3)))
check('weighted_fraction handles all-zero weights', np.isnan(cm.weighted_fraction([True, False], [0.0, 0.0])))
check('weighted_fraction trivial cases',
      cm.weighted_fraction([True, True], [1.0, 1.0]) == 1.0 and
      cm.weighted_fraction([False, False], [1.0, 2.0]) == 0.0)

med = cm.weighted_quantile(x_obs, w_skew)
mc_med = np.median([x_obs[np.random.default_rng(s).choice(n_pop, 20000, replace=True,
                                                          p=w_skew / w_skew.sum())].mean()
                    for s in range(3)])
check('weighted_quantile median is finite and inside the range',
      np.isfinite(med) and x_obs.min() <= med <= x_obs.max(), round(float(med), 3))
check('weighted_quantile equals plain median for equal weights',
      abs(cm.weighted_quantile([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1]) - 2.5) < 1e-9,
      cm.weighted_quantile([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1]))
check('weighted_quantile respects weights',
      cm.weighted_quantile([0.0, 10.0], [1000.0, 1.0]) < 1.0,
      cm.weighted_quantile([0.0, 10.0], [1000.0, 1.0]))
check('weighted_quantile all-zero weights -> NaN',
      np.isnan(cm.weighted_quantile([1.0, 2.0], [0.0, 0.0])))

print('\n=== modified_weighted_vals alignment ===')
try:
    cm.modified_weighted_vals(np.array([1.0, 1.0]), [1.0], [1.0], [1.0], [1.0],
                              [1.0], [1.0], [1.0], [1.0], [1.0], 1)
    check('misaligned inputs raise', False)
except ValueError as e:
    check('misaligned inputs raise ValueError', True)
w = np.array([1.0, 1.0, 1.0])
out = cm.modified_weighted_vals(
    w, [1.0, 2.0, np.nan], [10.0, 20.0, 30.0], [11.0, 21.0, 31.0],
    [5.0, 6.0, 7.0], [0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.4, 0.5, 0.6],
    [22.0, 23.0, 24.0], [21.0, 22.0, 23.0], 4)
# out = (time_delays, CM_times_rel_peak, CM_times_rel_first, mag, z,
#        l_colors, ul_colors, l_peak_mags, ul_peak_mags, weights)
check('NaN delay dropped, others stay paired',
      set(zip(out[0], out[1])) <= {(1.0, 10.0), (2.0, 20.0)}, set(zip(out[0], out[1])))
check('modified_weighted_vals carries all aligned arrays',
      len(out) == 10 and all(len(out[k]) == len(out[0]) for k in range(9)),
      [len(out[k]) for k in range(9)])

print('\n=== l_weight normalisation scale ===')
import astropy.units as u, astropy.cosmology.units as cu
from astropy.cosmology import Planck18
z_t, mu_t = 0.5, 10.0
lw = cm.l_weight(z_t, mu_t)
uw = cm.ul_weight(z_t)
d_pc = (z_t * cu.redshift).to(u.pc, cu.redshift_distance(Planck18, kind='comoving')).value
expect = uw * 0.5 * ((d_pc / 31e9) ** 3) * mu_t ** -2
check('l_weight uses the 31 Gpc (=31e9 pc) scale', np.isclose(lw, expect, rtol=1e-12),
      (lw, expect))
# the pre-fix code divided by 31 instead of 31e9 -> 1e27 too large
check('l_weight is not the /31 variant', not np.isclose(lw, expect * 1e27, rtol=1e-6))
check('l_weight finite and positive', np.isfinite(lw) and lw > 0, lw)

print('\n=== time_delay ===')
td = cm.time_delay(10.0, 1.0, 0.5)
check('time_delay finite positive', np.isfinite(td) and td > 0, td)

print('\n=== exponential_regression / success_rate ===')
rng = np.random.default_rng(0)
x_ul, c_ul = rng.normal(22, 1, 200), rng.normal(0.0, 0.2, 200)
x_l, c_l = rng.normal(22, 1, 200), rng.normal(1.5, 0.2, 200)
xa, ya, m, b = cm.exponential_regression(x_ul, x_l, c_ul, c_l, verbose=True)
check('returns 4-tuple with finite m,b', np.isfinite(m) and np.isfinite(b), (m, b))
sr_l = cm.success_rate(x_l, c_l, m, b)
sr_ul = cm.success_rate(x_ul, c_ul, m, b)
check('separable data: lensed rate high', sr_l > 0.9, sr_l)
check('separable data: unlensed rate low', sr_ul < 0.1, sr_ul)
check('success_rate empty -> NaN', np.isnan(cm.success_rate([], [], m, b)))
xc, yc, fitted = cm.exponential_regression_curvefit(x_ul, x_l, c_ul, c_l)
check('curvefit boundary returned', xc is not None and np.all(np.isfinite(yc)))
check('curvefit returns a callable boundary', callable(fitted))
check('success_rate accepts a fitted callable',
      abs(cm.success_rate(x_l, c_l, fitted) - cm.success_rate(x_l, c_l, m, b)) < 0.5)
check('curvefit degenerate input -> (None, None)',
      cm.exponential_regression_curvefit([1.0], [1.0], [0.0], [1.0], n_bins=15)[0] is None)

print('\n=== Supernovae2 / Nugent populations ===')
g2 = sn.Supernovae2()
pop2 = g2.generate_many(60, 'lsstr', 'lsstg', z_range=(0.0, 1.0))
check('Supernovae2 yields events', len(pop2) > 0, len(pop2))
check('Supernovae2 schema', all(k in pop2[0] for k in
      ['z', 'det_mags1_ul', 'det_mags2_l', 'weights_l', 'magnification']))
gn = sn.Nugent()
popn = gn.generate_many(40, 'lsstr', 'lsstg', -16.9, 1.12, 'nugent-sn2p', z_range=(0.0, 0.5))
check('Nugent yields events', len(popn) > 0, len(popn))
check('Nugent schema matches Supernovae2',
      set(['z', 'x1', 'c', 't0', 'magnification', 'weights_ul', 'weights_l',
           'det_mags1_ul', 'det_mags2_ul', 'det_mags1_l', 'det_mags2_l']) <= set(popn[0]))
# coverage check must reject high z rather than silently returning NaN
check('high-z rejected by coverage check',
      g2.generate_one(3.0, 0.0, -19.2, 0.0, 10.0, 'lsstr', 'lsstg') is None)

print('\n=== redshift_limits: per-band-pair coverage ===')
zg = sn.redshift_limits('salt3', 'lsstg', 'lsstr')
zr = sn.redshift_limits('salt3', 'lsstr', 'lssti')
zi = sn.redshift_limits('salt3', 'lssti', 'lsstz')
zz = sn.redshift_limits('salt3', 'lsstz', 'lssty')
check('g-r capped near 0.93', abs(zg[1] - 0.933) < 0.01, zg)
check('r-i capped near 1.68', abs(zr[1] - 1.685) < 0.01, zr)
check('i-z capped near 2.38', abs(zi[1] - 2.380) < 0.01, zi)
check('z-y capped near 3.01', abs(zz[1] - 3.015) < 0.01, zz)
check('redder pairs reach higher z than bluer (the whole point)',
      zg[1] < zr[1] < zi[1] < zz[1])
check('a global 1.75 cap would truncate i-z and z-y', zi[1] > 1.75 and zz[1] > 1.75)
check('z_min is 0 for salt3 (red edge never binds)', zg[0] == 0.0 and zz[0] == 0.0)
check('Nugent templates reach far higher z than salt3',
      sn.redshift_limits('nugent-sn2p', 'lsstg', 'lsstr')[1] > 2.8,
      sn.redshift_limits('nugent-sn2p', 'lsstg', 'lsstr'))
# limits must be consistent with the runtime coverage guard
m_probe = sncosmo.Model('salt3')
for lo_hi, pair in [(zi, ('lssti', 'lsstz')), (zz, ('lsstz', 'lssty'))]:
    m_probe.set(z=lo_hi[1] - 0.02)
    ok_inside = sn._covers(m_probe, *pair)
    m_probe.set(z=lo_hi[1] + 0.05)
    ok_outside = sn._covers(m_probe, *pair)
    check(f'{pair[0]}-{pair[1]}: _covers agrees with redshift_limits',
          ok_inside and not ok_outside, (ok_inside, ok_outside))

print('\n=== rest-frame peak reference (peakphase) ===')
m_pk = sncosmo.Model('salt3')
m_pk.set(z=0.0, x1=0.0, c=0.0)
p0 = m_pk.source.peakphase('bessellb')
m_pk.set(z=2.5)
p_hi = m_pk.source.peakphase('bessellb')
check('peakphase is z-independent', abs(p0 - p_hi) < 1e-9, (p0, p_hi))
check('peakphase near 0 d (not the old +49 d artefact)', abs(p0) < 5.0, p0)
m_pk.set(z=0.0, x1=2.0)
check('peakphase responds to x1', abs(m_pk.source.peakphase('bessellb') - p0) > 1e-3)
# a red pair above the old lsstr cap must now be generatable
g2_hi = sn.Supernovae2()
hi = g2_hi.generate_one(2.2, 0.0, -19.23, 0.0, 10.0, 'lsstz', 'lssti')
check('z=2.2 i/z event now accepted (was blocked by lsstr ref band)', hi is not None)

print('\n=== full_creation / combine_populations ===')
fc = cm.full_creation(pop2, 5)
check('full_creation returns 6 items', len(fc) == 6)
combo = cm.combine_populations([pop2, popn], 5)
check('combine_populations pools both', len(combo[3]) == 10, len(combo[3]))

print('\n=== synthetic OpSim: full light-curve + Supernovae path ===')
def fake_gen(ra_pointings, dec_pointings, db_path=None, verbose=True):
    """One DataFrame per pointing: 3 years of visits cycling through g,r,i."""
    ra_arr = np.atleast_1d(ra_pointings); dec_arr = np.atleast_1d(dec_pointings)
    def _iter():
        for ra, dec in zip(ra_arr, dec_arr):
            n = 900
            mjd = survey_dates[0] + np.sort(np.random.uniform(0, 1100, n))
            filt = np.array(['g', 'r', 'i'] * (n // 3))
            yield pd.DataFrame({
                'expMJD': mjd, 'fieldRA': np.full(n, float(ra)),
                'fieldDec': np.full(n, float(dec)), 'filter': filt,
                'seeingFwhmGeom': np.full(n, 0.8),
                'fiveSigmaDepth': np.full(n, 24.5),
                'filtSkyBrightness': np.full(n, 21.0)})
    return _iter()

def fake_gen_long(ra_pointings, dec_pointings, db_path=None, verbose=True):
    """As fake_gen, but the cadence runs to year 6 so follow-up windows past
    year 3 have visits to find."""
    ra_arr = np.atleast_1d(ra_pointings); dec_arr = np.atleast_1d(dec_pointings)
    span = survey_dates[6] - survey_dates[0]
    def _iter():
        for ra, dec in zip(ra_arr, dec_arr):
            n = 2199                      # divisible by 3 so the filter cycle fits
            mjd = survey_dates[0] + np.sort(np.random.uniform(0, span, n))
            filt = np.array(['g', 'r', 'i'] * (n // 3))
            yield pd.DataFrame({
                'expMJD': mjd, 'fieldRA': np.full(n, float(ra)),
                'fieldDec': np.full(n, float(dec)), 'filter': filt,
                'seeingFwhmGeom': np.full(n, 0.8),
                'fiveSigmaDepth': np.full(n, 24.5),
                'filtSkyBrightness': np.full(n, 21.0)})
    return _iter()

sn.initialise_opsim_summary = fake_gen
np.random.seed(1)

lc = lcmod.get_observations(lcmod.Transient(sncosmo.Model('salt3'), survey_dates[0]),
                            fake_gen(90, -30), sncosmo.Model('salt3'))
check('get_observations builds a light curve', lc is not None and len(lc.obs_days) > 0,
      None if lc is None else len(lc.obs_days))

# time_delay_ must shift the window, not crash
lc_d = lcmod.get_observations(lcmod.Transient(sncosmo.Model('salt3'), survey_dates[0]),
                              fake_gen(90, -30), sncosmo.Model('salt3'), time_delay_=200.0)
check('time_delay_ accepted', lc_d is not None and len(lc_d.obs_days) > 0)

# empty window must return None rather than IndexError
try:
    lc_e = lcmod.get_observations(lcmod.Transient(sncosmo.Model('salt3'), 99999.0),
                                  fake_gen(90, -30), sncosmo.Model('salt3'))
    check('empty observing window -> None (no IndexError)', lc_e is None, lc_e)
except IndexError as e:
    check('empty observing window -> None (no IndexError)', False, 'IndexError raised')

g1 = sn.Supernovae()
np.random.seed(7)
pop1 = g1.generate_many(40, 'lsstr', 'lsstg', z_range=(0.05, 0.8))
check('Supernovae yields events through cadence', len(pop1) > 0, len(pop1))
if pop1:
    ev = pop1[0]
    need = ['z', 'magnification', 'weights_ul', 'weights_l', 'band_1_mag_ul',
            'band_2_mag_ul', 'band_1_mag_l', 'band_2_mag_l', 'lensed_colour',
            'time_of_SN', 'time_delay', 'timdel', 'rest_frame_peak_day']
    check('Supernovae schema complete (Sam parity)', all(k in ev for k in need),
          [k for k in need if k not in ev])
    check('lensed_colour == band2 - band1 (no /(1+z) on mags)',
          np.isclose(ev['lensed_colour'], ev['band_2_mag_l'] - ev['band_1_mag_l']))
    check('time_of_SN finite', np.isfinite(ev['time_of_SN']), ev['time_of_SN'])
    check('time_delay finite (days, not a magnitude)', np.isfinite(ev['time_delay']),
          ev['time_delay'])
    tds = np.array([e['time_delay'] for e in pop1])
    tos = np.array([e['time_of_SN'] for e in pop1])
    check('both epochs on comparable day scales',
          np.all(np.abs(tds) < 1e4) and np.all(np.abs(tos) < 1e4),
          (float(np.nanmin(tds)), float(np.nanmax(tds)),
           float(np.nanmin(tos)), float(np.nanmax(tos))))
    # lensed image must be brighter than unlensed for magnification > 1
    dm = np.array([e['band_1_mag_l'] - e['band_1_mag_ul'] for e in pop1])
    check('lensed images brighter than unlensed', np.nanmedian(dm) < 0, np.nanmedian(dm))

print('\n=== time handling: true t0 phasing, no re-zeroing ===')

def gen_at(mjds, ra=90.0, dec=-30.0, filt='r'):
    """Generator yielding one table with visits at exactly the given MJDs."""
    mjds = np.asarray(mjds, dtype=float)
    n = len(mjds)
    def _iter():
        yield pd.DataFrame({
            'expMJD': mjds, 'fieldRA': np.full(n, ra), 'fieldDec': np.full(n, dec),
            'filter': np.array([filt] * n), 'seeingFwhmGeom': np.full(n, 0.8),
            'fiveSigmaDepth': np.full(n, 24.5), 'filtSkyBrightness': np.full(n, 21.0)})
    return _iter()

mdl = sncosmo.Model('salt3'); mdl.set(z=0.05)   # z>0: peakabsmag needs a distance
mdl.set_source_peakabsmag(-19.23, 'lsstr', 'ab')
T0 = 61300.0
mn, mx = mdl.mintime(), mdl.maxtime()
# Visits every 5 days across the visibility window (spaced so coadding cannot merge).
visit_mjds = T0 + np.arange(mn + 1, mx - 1, 5.0)
lc_t = lcmod.get_observations(lcmod.Transient(mdl, T0), gen_at(visit_mjds), mdl)
check('light curve built from in-window visits', lc_t is not None)
if lc_t is not None:
    expected = visit_mjds - T0
    got = np.sort(np.asarray(lc_t.obs_days))
    check('obs_days == visit_MJD - t0 (true phasing)',
          len(got) == len(expected) and np.allclose(got, np.sort(expected), atol=1e-6),
          (got[:3], np.sort(expected)[:3]))
    check('first obs_day is NOT pinned to model.mintime()',
          abs(got[0] - mn) > 0.5, (got[0], mn))

# Shifting t0 later must shift the phases, not re-zero them. Compare only the visits
# that survive BOTH windows -- the window edge moves with t0, so the earliest
# surviving visit legitimately differs between the two runs.
lc_t2 = lcmod.get_observations(lcmod.Transient(mdl, T0 + 3.0), gen_at(visit_mjds), mdl)
if lc_t is not None and lc_t2 is not None:
    mjd_a = np.round(np.asarray(lc_t.obs_days) + T0, 6)
    mjd_b = np.round(np.asarray(lc_t2.obs_days) + T0 + 3.0, 6)
    common = np.intersect1d(mjd_a, mjd_b)
    check('a given visit keeps its MJD under a t0 change (phase shifts by -3 d)',
          len(common) > 3 and np.allclose(
              np.sort([m - T0 for m in common]),
              np.sort([m - (T0 + 3.0) for m in common]) + 3.0, atol=1e-6),
          len(common))

# A supernova exploding in a cadence gap must be missed, not dragged into view.
gap_visits = np.array([T0 - 400.0, T0 - 350.0, T0 + 400.0, T0 + 450.0])
check('SN in a cadence gap is missed (returns None)',
      lcmod.get_observations(lcmod.Transient(mdl, T0), gen_at(gap_visits), mdl) is None)
# ... and one right at the survey end is truncated away.
check('SN after the campaign cut is missed',
      lcmod.get_observations(lcmod.Transient(mdl, survey_dates[3] + 500),
                             gen_at(visit_mjds), mdl) is None)

# A delayed image must be phased on its own t0, offset from the leading one.
lc_lead = lcmod.get_observations(lcmod.Transient(mdl, T0), gen_at(visit_mjds), mdl)
lc_del = lcmod.get_observations(lcmod.Transient(mdl, T0), gen_at(visit_mjds + 40.0),
                                mdl, time_delay_=40.0)
if lc_lead is not None and lc_del is not None:
    check('delayed image phases match the leading image (its own visits shifted too)',
          np.isclose(np.min(lc_del.obs_days), np.min(lc_lead.obs_days), atol=1e-6),
          (np.min(lc_del.obs_days), np.min(lc_lead.obs_days)))
# With the visit list fixed, a delay makes the image sample a DIFFERENT set of visits
# (its window has moved down the cadence), which is the physically meaningful effect.
lc_del2 = lcmod.get_observations(lcmod.Transient(mdl, T0), gen_at(visit_mjds), mdl,
                                 time_delay_=40.0)
if lc_lead is not None and lc_del2 is not None:
    mjd_lead = np.round(np.asarray(lc_lead.obs_days) + T0, 6)
    mjd_del2 = np.round(np.asarray(lc_del2.obs_days) + T0 + 40.0, 6)
    check('a delayed image samples different visits than the leading one',
          not np.array_equal(np.sort(mjd_lead), np.sort(mjd_del2)),
          (mjd_lead.min(), mjd_del2.min()))
    check('delayed image visits all lie after the leading image switches on',
          mjd_del2.min() > mjd_lead.min(), (mjd_lead.min(), mjd_del2.min()))

print('\n=== time_delay now tracks the physical delay ===')
sn.initialise_opsim_summary = fake_gen
np.random.seed(11)
pop_d = g1.generate_many(600, 'lsstr', 'lsstg', z_range=(0.05, 0.6))
check('population generated for delay check', len(pop_d) > 20, len(pop_d))
if len(pop_d) > 20:
    td_ = np.array([e['time_delay'] for e in pop_d])
    tdl = np.array([e['timdel'] for e in pop_d])
    mu_ = np.array([e['magnification'] for e in pop_d])
    z_ = np.array([e['z'] for e in pop_d])
    r_td = np.corrcoef(td_, tdl)[0, 1]
    # The physical delay is the dominant term, so the recovered epoch must track it.
    check('corr(time_delay, timdel) is strong', r_td > 0.8, round(float(r_td), 3))
    # Physical delay scales as mu^-3, so it must anticorrelate with magnification.
    r_mu = np.corrcoef(td_, mu_)[0, 1]
    check('time_delay anticorrelates with magnification', r_mu < -0.15, round(float(r_mu), 3))
    # Events can still sit near minphase - peakphase, but now for a physical reason:
    # an image cannot be detected before its own first light, so near-simultaneous
    # images land there. The discriminating test is that those events are the
    # SMALL-delay ones -- under the old artifact, floor proximity was independent of
    # the delay entirely.
    floor = sncosmo.get_source('salt3').minphase() - 0.4314976373122497
    near = np.abs(td_ - floor) < 2.0
    if near.sum() > 5 and (~near).sum() > 5:
        check('near-floor events are the small-delay ones (physical bound, not artifact)',
              np.median(tdl[near]) < np.median(tdl[~near]),
              (round(float(np.median(tdl[near])), 3), round(float(np.median(tdl[~near])), 3)))
        check('near-floor events are the highly magnified ones (delay ~ mu^-3)',
              np.median(mu_[near]) > np.median(mu_[~near]),
              (round(float(np.median(mu_[near])), 1), round(float(np.median(mu_[~near])), 1)))
    # Sanity: the recovered epoch should sit near timdel/(1+z) plus a light-curve phase.
    resid = td_ - tdl / (1 + z_)
    check('time_delay - timdel/(1+z) is a light-curve phase (|.|<120 d)',
          np.all(np.abs(resid) < 120), (float(resid.min()), float(resid.max())))

print('\n=== asymmetric discovery / follow-up windows ===')
check('defaults are year 3 discovery, year 5 follow-up',
      survey_dates[3] < survey_dates[5], (survey_dates[3], survey_dates[5]))

# get_observations must honour mjd_high for a delayed image independently.
mdl_w = sncosmo.Model('salt3'); mdl_w.set(z=0.05)
mdl_w.set_source_peakabsmag(-19.23, 'lsstr', 'ab')
T = survey_dates[3] - 40.0                      # leading image near the year-3 edge
DELAY = 500.0                                    # trailing image lands in year 4
visits = np.arange(survey_dates[0] + 10, survey_dates[6], 4.0)
lc_short = lcmod.get_observations(lcmod.Transient(mdl_w, T), gen_at(visits), mdl_w,
                                  time_delay_=DELAY, mjd_high=survey_dates[3])
lc_long = lcmod.get_observations(lcmod.Transient(mdl_w, T), gen_at(visits), mdl_w,
                                 time_delay_=DELAY, mjd_high=survey_dates[5])
check('trailing image missed under a year-3 cap', lc_short is None)
check('trailing image recovered under a year-5 cap',
      lc_long is not None and len(lc_long.obs_days) > 0,
      None if lc_long is None else len(lc_long.obs_days))

# The leading image must be untouched by the follow-up extension. Test that through
# generate_one (where the two windows are wired separately) rather than through
# get_observations, whose mjd_high legitimately truncates whatever it is given.
sn.initialise_opsim_summary = fake_gen_long
common = dict(z=0.3, x1=0.0, abs_mag=-19.23, c=0.0, magnification=6.0,
              times=survey_dates[0] + 300.0, ra=90.0, dec=-30.0,
              band1='lsstr', band2='lsstg')
np.random.seed(5)
ev_a = g1.generate_one(**common, followup_mjd_high=survey_dates[3])
np.random.seed(5)
ev_b = g1.generate_one(**common, followup_mjd_high=survey_dates[5])
if ev_a is not None and ev_b is not None:
    # Magnitudes and epochs are bit-identical; rest_frame_peak_day can differ in the
    # last couple of bits because sncosmo's peakphase runs a numerical search on the
    # shared model instance, whose amplitude the previous call left set differently.
    # That propagates ~1e-15 into any epoch measured against it.
    check('leading-image magnitudes bit-identical under follow-up extension',
          (ev_a['band_1_mag_ul'] == ev_b['band_1_mag_ul']
           and ev_a['band_1_mag_l'] == ev_b['band_1_mag_l']
           and ev_a['lensed_colour'] == ev_b['lensed_colour']))
    check('leading-image epoch unchanged to float precision',
          abs(ev_a['time_of_SN'] - ev_b['time_of_SN']) < 1e-9,
          abs(ev_a['time_of_SN'] - ev_b['time_of_SN']))
else:
    check('leading-image quantities unchanged by follow-up extension',
          ev_a is None and ev_b is None, 'both rejected identically')

# A leading image that only fits inside year 5 must still be rejected.
T_late = survey_dates[3] + 200.0
check('leading image past the discovery window is rejected',
      lcmod.get_observations(lcmod.Transient(mdl_w, T_late), gen_at(visits), mdl_w,
                             mjd_high=survey_dates[3]) is None)

# End to end, on a cadence that actually extends past year 3: a longer follow-up
# window must admit longer delays and recover at least as many systems.
np.random.seed(21)
pop_a = g1.generate_many(500, 'lsstr', 'lsstg', z_range=(0.05, 0.6),
                         followup_mjd_high=survey_dates[3])
np.random.seed(21)
pop_b = g1.generate_many(500, 'lsstr', 'lsstg', z_range=(0.05, 0.6),
                         followup_mjd_high=survey_dates[5])
if pop_a and pop_b:
    d_a = np.array([e['timdel'] for e in pop_a])
    d_b = np.array([e['timdel'] for e in pop_b])
    check('longer follow-up recovers at least as many systems',
          len(pop_b) >= len(pop_a) * 0.98, (len(pop_a), len(pop_b)))
    check('longer follow-up admits longer delays',
          np.percentile(d_b, 99) > np.percentile(d_a, 99),
          (round(float(np.percentile(d_a, 99)), 1), round(float(np.percentile(d_b, 99)), 1)))
    cap_a = survey_dates[3] - survey_dates[0]
    check('year-3 follow-up cannot admit a delay longer than the window',
          d_a.max() <= cap_a + 1, (round(float(d_a.max()), 1), cap_a))

print('\n=== coadds parity + perf ===')
np.random.seed(3)
n = 400
o = obsmod.Observations(90.0, -30.0,
                        np.sort(np.random.uniform(0, 50, n)),
                        np.array(['g', 'r'] * (n // 2)),
                        np.full(n, 0.8), np.full(n, 24.5), np.full(n, 21.0))
co = obsmod.coadds(o)
check('coadds produces N_coadds', co.N_coadds is not None and co.N_coadds.sum() == n,
      None if co.N_coadds is None else co.N_coadds.sum())
check('coadds arrays stay consistent',
      len(co.opsim_times) == len(co.opsim_filters) == len(co.opsim_lim_mag) == len(co.N_coadds))
check('coadded depth >= single-visit depth', np.all(co.opsim_lim_mag >= 24.5 - 1e-9))

print('\n=== integration: multi-lens time_delay + fourth lens ===')
td_cluster = cm.time_delay(10.0, 1.0, 0.5, lens=2)
check('default lens (=2, cluster) reproduces original prefactor',
      np.isclose(cm.time_delay(10.0, 1.0, 0.5), td_cluster), (cm.time_delay(10.0, 1.0, 0.5), td_cluster))
td_gal = cm.time_delay(10.0, 1.0, 0.5, lens=0)
td_grp = cm.time_delay(10.0, 1.0, 0.5, lens=1)
check('galaxy < group < cluster delay (Einstein-radius scaling)',
      td_gal < td_grp < td_cluster, (td_gal, td_grp, td_cluster))
td4 = cm.time_delay_fourth_lens(10.0, 1.0, 0.5)
check('time_delay_fourth_lens finite positive', np.isfinite(td4) and td4 > 0, td4)
# (mu/4)**-1 scaling: halving mu should ~double the fourth-lens delay
check('fourth lens scales as (mu/4)**-1',
      np.isclose(cm.time_delay_fourth_lens(5.0, 1.0, 0.5),
                 2 * td4, rtol=1e-9), (cm.time_delay_fourth_lens(5.0, 1.0, 0.5), 2 * td4))

print('\n=== integration: core-collapse weights ===')
check('cc_ul_weight finite positive', np.isfinite(cm.cc_ul_weight(0.5)) and cm.cc_ul_weight(0.5) > 0)
lw_cc = cm.cc_l_weight(0.5, 10.0)
import astropy.units as _u, astropy.cosmology.units as _cu
from astropy.cosmology import Planck18 as _P18
_d = (0.5 * _cu.redshift).to(_u.pc, _cu.redshift_distance(_P18, kind='comoving')).value
check('cc_l_weight uses the 31 Gpc scale',
      np.isclose(lw_cc, cm.cc_ul_weight(0.5) * 0.5 * (_d / 31e9) ** 3 * 10.0 ** -2, rtol=1e-12))

print('\n=== integration: first_detection_epoch bug fix ===')
class _LC:
    def __init__(self, days, mags):
        self.obs_days = np.array(days); self.obs_mag = np.array(mags)
check('no detections -> NaN (empty-check before np.min)',
      np.isnan(lcmod.first_detection_epoch(_LC([1., 2.], [np.inf, np.inf]), 0.5, 0.0)))
fde = lcmod.first_detection_epoch(_LC([10., 20., 5.], [np.inf, 20.0, 21.0]), 1.0, 0.0)
check('first detection epoch = min detected day / (1+z)', np.isclose(fde, 5.0 / 2.0), fde)

print('\n=== integration: Supernovae multi-lens schema + required_model ===')
sn.initialise_opsim_summary = fake_gen
np.random.seed(21)
g_ml = sn.Supernovae()
pop_ml = g_ml.generate_many(60, 'lsstr', 'lsstg', z_range=(0.05, 0.6))
check('multi-lens population generated', len(pop_ml) > 5, len(pop_ml))
if pop_ml:
    ev = pop_ml[0]
    newkeys = ['time_delay_galaxy_1', 'time_delay_galaxy_2', 'time_delay_group',
               'time_delay_cluster', 'timdel_galaxy_1', 'timdel_cluster',
               'time_of_SN_rel_peak', 'time_of_SN_rel_first', 'first_alert_day',
               'n_alerts_before_id']
    check('multi-lens schema present', all(k in ev for k in newkeys),
          [k for k in newkeys if k not in ev])
    check('n_alerts_before_id is a non-negative int',
          isinstance(ev['n_alerts_before_id'], int) and ev['n_alerts_before_id'] >= 0,
          ev['n_alerts_before_id'])
    check('legacy time_delay alias == required (cluster) model',
          np.isclose(ev['time_delay'], ev['time_delay_cluster']),
          (ev['time_delay'], ev['time_delay_cluster']))
    # every kept event must have a finite delay for the required model
    check('required_model="cluster" guarantees a finite cluster delay',
          all(np.isfinite(e['time_delay_cluster']) for e in pop_ml))

print(f'\n===== {len(PASS)} passed, {len(FAIL)} failed =====')
if FAIL:
    print('FAILED: ' + ', '.join(FAIL)); sys.exit(1)
