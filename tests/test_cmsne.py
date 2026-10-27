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

def _auc(fpr, tpr):
    """Trapezoidal AUC -- avoids numpy.trapz/trapezoid's rename across versions."""
    fpr, tpr = np.asarray(fpr, float), np.asarray(tpr, float)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))

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
expect = uw * ((d_pc / 31e9) ** 3) * (mu_t ** -3)
check('l_weight uses the 31 Gpc (=31e9 pc) scale and mu^-3 optical depth', np.isclose(lw, expect, rtol=1e-12),
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

print('\n=== integration: core-collapse weights (SFR-tracking rate) ===')
check('cc_ul_weight finite positive', np.isfinite(cm.cc_ul_weight(0.5)) and cm.cc_ul_weight(0.5) > 0)
lw_cc = cm.cc_l_weight(0.5, 10.0)
import astropy.units as _u, astropy.cosmology.units as _cu
from astropy.cosmology import Planck18 as _P18
_d = (0.5 * _cu.redshift).to(_u.pc, _cu.redshift_distance(_P18, kind='comoving')).value
check('cc_l_weight uses the 31 Gpc scale and mu^-3 optical depth',
      np.isclose(lw_cc, cm.cc_ul_weight(0.5) * ((_d / 31e9) ** 3) * (10.0 ** -3), rtol=1e-12))

from cmsne.sn_rates import cc_rate_grid, rate_grid, cosmic_sfr_density
z_cc, n_cc = cc_rate_grid()
z_ia, n_ia = rate_grid()
z_peak_cc = z_cc[np.argmax(n_cc)]
z_peak_ia = z_ia[np.argmax(n_ia)]
check('CC rate peaks near the SFR peak (z ~ 1.9)', 1.6 < z_peak_cc < 2.2, round(float(z_peak_cc), 2))
# The Ia DTD shifts rate toward lower z, so CC gives relatively MORE weight to high z.
r_cc = cm.cc_ul_weight(2.0) / cm.cc_ul_weight(0.3)
r_ia = cm.ul_weight(2.0) / cm.ul_weight(0.3)
check('CC up-weights high-z relative to the DTD-convolved Ia rate',
      r_cc > r_ia, (round(float(r_cc), 2), round(float(r_ia), 2)))
check('CC rate tracks the cosmic SFR shape',
      np.allclose(n_cc / n_cc.max(),
                  cosmic_sfr_density(z_cc) / cosmic_sfr_density(z_cc).max(), atol=1e-9))
# the whole point: the CC weight is no longer just the Ia weight rescaled
check('cc_ul_weight is NOT proportional to ul_weight across z',
      not np.isclose(cm.cc_ul_weight(2.0) / cm.cc_ul_weight(0.3),
                     cm.ul_weight(2.0) / cm.ul_weight(0.3), rtol=0.05),
      (cm.cc_ul_weight(2.0) / cm.cc_ul_weight(0.3), cm.ul_weight(2.0) / cm.ul_weight(0.3)))

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

print('\n=== multicolour: all-band photometry extraction ===')
from cmsne import multicolour as mcol
class _LCm:
    def __init__(s, d, f, m, e=None):
        s.obs_days = np.array(d, float); s.obs_filters = np.array(f); s.obs_mag = np.array(m, float)
        s.obs_mag_error = np.full(len(m), 0.1) if e is None else np.array(e, float)
# g@2, r@3 detected; r@4 fainter; i@20 below depth (inf). Trigger = first detection (day 2).
# Errors: g@2=0.05, r@3=0.08, i@20=inf (undetected anyway), r@4=0.12.
_lc = _LCm([2, 3, 20, 4], ['g', 'r', 'i', 'r'], [22.0, 21.5, np.inf, 21.7], [0.05, 0.08, np.inf, 0.12])
_bp = mcol.band_photometry(_lc, 2.0, window=5.0)
check('band_photometry picks nearest detection', np.isclose(_bp['lsstg'], 22.0) and np.isclose(_bp['lsstr'], 21.5),
      (_bp['lsstg'], _bp['lsstr']))
check('band_photometry NaN for undetected band', np.isnan(_bp['lssti']) and np.isnan(_bp['lsstu']))
_bpe = mcol.band_photometry_errors(_lc, 2.0, window=5.0)
check('band_photometry_errors matches the SAME nearest-detection index as band_photometry',
      np.isclose(_bpe['lsstg'], 0.05) and np.isclose(_bpe['lsstr'], 0.08), (_bpe['lsstg'], _bpe['lsstr']))
check('band_photometry_errors NaN for undetected band', np.isnan(_bpe['lssti']) and np.isnan(_bpe['lsstu']))
_f = mcol.event_multicolour(_lc)
check('trigger epoch is first detection', _f['trigger_day'] == 2.0, _f['trigger_day'])
check('colour = band mag difference', np.isclose(_f['gr'], 0.5), _f['gr'])
check('colour NaN when a band is outside the window', np.isnan(_f['ri']))
check('peakmag is brightest detection', np.isclose(_f['peakmag'], 21.5), _f['peakmag'])
check('peakmag_firstdet is the earliest detection', np.isclose(_f['peakmag_firstdet'], 22.0), _f['peakmag_firstdet'])
check('peakmag_fit falls back to brightest when <3 band points',
      np.isclose(_f['peakmag_fit'], 21.5), _f['peakmag_fit'])
check('n_bands counts bands at trigger', _f['n_bands'] == 2, _f['n_bands'])
check('event_multicolour: gr_err is the quadrature sum of the two bands\' own errors',
      np.isclose(_f['gr_err'], np.sqrt(0.05 ** 2 + 0.08 ** 2)), _f['gr_err'])
check('event_multicolour: colour error is NaN exactly where the colour itself is NaN',
      np.isnan(_f['ri']) and np.isnan(_f['ri_err']))
check('event_multicolour: peakmag_err is the error of whichever observation achieved peakmag',
      np.isclose(_f['peakmag_err'], 0.08), _f['peakmag_err'])
check('event_multicolour: peakmag_firstdet_err is the error of the first-detection observation',
      np.isclose(_f['peakmag_firstdet_err'], 0.05), _f['peakmag_firstdet_err'])
check('event_multicolour: no peakmag_fit_err field is added (explicitly out of scope)',
      'peakmag_fit_err' not in _f)
check('no detections -> None', mcol.event_multicolour(_LCm([1, 2], ['g', 'r'], [np.inf, np.inf])) is None)
check('rate_weight dispatches by class',
      mcol.rate_weight('uIa', 0.3, 1.0) > 0 and mcol.rate_weight('sig', 0.3, 10.0) > 0)
_mg = mcol.MultiColourGenerator()
_mg_feat = _mg._event_features(_lc, 15.0, 5.0)
_direct_feat = mcol.event_multicolour(_lc, 15.0, 5.0)
check('MultiColourGenerator._event_features defaults to event_multicolour',
      _mg_feat.keys() == _direct_feat.keys() and
      np.isclose(_mg_feat['gr'], _direct_feat['gr']) and
      np.isclose(_mg_feat['peakmag'], _direct_feat['peakmag']))

print('\n=== trajectory: N-point colour-magnitude generation ===')
from cmsne import trajectory as traj
# g@2,r@3 detected at trigger; g brighter again @20, r follows @23.
# Errors: g@2=0.05, r@3=0.08, g@20=0.03, r@23=0.06.
_lct = _LCm([2, 3, 20, 23], ['g', 'r', 'g', 'r'], [22.0, 21.5, 20.0, 21.0],
            [0.05, 0.08, 0.03, 0.06])
_ft = traj.event_trajectory(_lct, grid=(0.0, 20.0), window=5.0)
check('trigger epoch is first detection', _ft['trigger_day'] == 2.0, _ft['trigger_day'])
check('colour at day 0 matches multicolour.event_multicolour',
      np.isclose(_ft['gr_d0'], 0.5), _ft['gr_d0'])
check('colour at day 20 uses the later epoch', np.isclose(_ft['gr_d20'], -1.0), _ft['gr_d20'])
check('cumulative peakmag at day 0 is the trigger-time brightest',
      np.isclose(_ft['peakmag_d0'], 21.5), _ft['peakmag_d0'])
check('cumulative peakmag at day 20 improves (brighter point included)',
      np.isclose(_ft['peakmag_d20'], 20.0), _ft['peakmag_d20'])
check('no detections -> None',
      traj.event_trajectory(_LCm([1, 2], ['g', 'r'], [np.inf, np.inf])) is None)

check('event_trajectory: gr_d0_err is the quadrature sum of the trigger-snapshot bands\' own errors',
      np.isclose(_ft['gr_d0_err'], np.sqrt(0.05 ** 2 + 0.08 ** 2)), _ft['gr_d0_err'])
check('event_trajectory: gr_d20_err uses the day-20-snapshot bands\' own errors',
      np.isclose(_ft['gr_d20_err'], np.sqrt(0.03 ** 2 + 0.06 ** 2)), _ft['gr_d20_err'])
check('event_trajectory: peakmag_d0_err is the error of whichever observation achieved '
      'the cumulative-best magnitude at day 0 (r@3, the brighter of g@2/r@3)',
      np.isclose(_ft['peakmag_d0_err'], 0.08), _ft['peakmag_d0_err'])
check('event_trajectory: peakmag_d20_err tracks the NEW observation once it supersedes '
      'the running best (g@20=20.0 is brighter than the day-0 best of 21.5)',
      np.isclose(_ft['peakmag_d20_err'], 0.03), _ft['peakmag_d20_err'])
check('TRAJECTORY_COLOUR_KEYS has 5 colours x 5 grid steps',
      len(traj.TRAJECTORY_COLOUR_KEYS) == 25, len(traj.TRAJECTORY_COLOUR_KEYS))
check('TRAJECTORY_MAGNITUDE_KEYS has one peakmag per grid step',
      len(traj.TRAJECTORY_MAGNITUDE_KEYS) == 5)

_tg = traj.TrajectoryGenerator()
check('TrajectoryGenerator inherits MultiColourGenerator API',
      hasattr(_tg, 'generate_one') and hasattr(_tg, 'generate_many'))
check('TrajectoryGenerator._event_features uses the trajectory schema',
      set(_tg._event_features(_lct, 15.0, 5.0)) >= {'gr_d0', 'peakmag_d0', 'gr_d60'})

print('\n=== TrajectoryClassifier: cumulative multi-step scoring ===')
from cmsne.trajectory import (TrajectoryClassifier, TRAJECTORY_COLOUR_KEYS,
                              TRAJECTORY_MAGNITUDE_KEYS, TRAJECTORY_GRID, reveal_up_to)
_rngt = np.random.default_rng(11)
def _mk_traj(n, mu, n_steps_seen):
    """n events; colours at the first n_steps_seen grid steps set, later NaN."""
    out = []
    for _ in range(n):
        e = {}
        for i, k in enumerate(TRAJECTORY_COLOUR_KEYS):
            step = i // 5     # 5 colours per grid step, grid order
            e[k] = float(_rngt.normal(mu, 0.4)) if step < n_steps_seen else np.nan
        for i, k in enumerate(TRAJECTORY_MAGNITUDE_KEYS):
            e[k] = float(_rngt.normal(22, 1)) if i < n_steps_seen else np.nan
        out.append(e)
    return out
sig_t = _mk_traj(1500, 0.3, 5); bg_t = _mk_traj(3000, 0.0, 5)
tclf = TrajectoryClassifier(target_fpr=0.10).fit(sig_t, bg_t)
rec_t = tclf.recovery_rate(sig_t)
fpr_t = float(np.mean(tclf.predict(bg_t)))
check('trajectory classifier separates a separable signal', rec_t > 0.8, round(rec_t, 3))
check('trajectory classifier threshold pins contamination', abs(fpr_t - 0.10) < 0.06, round(fpr_t, 3))
check('trajectory classifier uses the full step schema',
      tclf.keys == TRAJECTORY_COLOUR_KEYS and tclf.extra == TRAJECTORY_MAGNITUDE_KEYS)

partial = _mk_traj(5, 0.3, 2)
check('partial trajectory (early steps only) scores without crashing',
      np.all(np.isfinite(tclf.score(partial))))

one_step = _mk_traj(200, 0.3, 1)
five_step = _mk_traj(200, 0.3, 5)
p_one = tclf.probability(one_step).mean()
p_five = tclf.probability(five_step).mean()
check('more observed steps -> more confident posterior for true signal',
      p_five > p_one, (round(p_one, 3), round(p_five, 3)))

revealed = reveal_up_to(sig_t[0], 1, TRAJECTORY_GRID)
check('reveal_up_to keeps steps 0-1 finite',
      np.isfinite(revealed['gr_d0']) and np.isfinite(revealed['gr_d3']))
check('reveal_up_to NaNs out steps beyond k',
      np.isnan(revealed['gr_d14']) and np.isnan(revealed['peakmag_d14']) and
      np.isnan(revealed['gr_d60']) and np.isnan(revealed['peakmag_d60']))
check('reveal_up_to does not mutate the original event',
      np.isfinite(sig_t[0]['gr_d14']))

_sig_t_err = dict(sig_t[0])
for _k in TRAJECTORY_COLOUR_KEYS:
    _sig_t_err[f'{_k}_err'] = 0.2
for _k in TRAJECTORY_MAGNITUDE_KEYS:
    _sig_t_err[f'{_k}_err'] = 0.1
_revealed_err = reveal_up_to(_sig_t_err, 1, TRAJECTORY_GRID)
check('reveal_up_to also NaNs out *_err keys for steps beyond k (not just the values) '
      '-- a hidden future step must not leak its own measurement uncertainty either',
      np.isnan(_revealed_err['gr_d14_err']) and np.isnan(_revealed_err['peakmag_d14_err']) and
      np.isnan(_revealed_err['gr_d60_err']) and np.isnan(_revealed_err['peakmag_d60_err']))
check('reveal_up_to keeps *_err keys for steps 0-1 finite',
      np.isfinite(_revealed_err['gr_d0_err']) and np.isfinite(_revealed_err['gr_d3_err']))

fpr_r, tpr_r, thr_r = tclf.roc_curve(sig_t, bg_t)
check('roc_curve returns matching-length fpr/tpr/thresholds arrays',
      len(fpr_r) == len(tpr_r) == len(thr_r))
check('roc_curve fpr/tpr bounded in [0,1] and fpr non-decreasing',
      np.all((fpr_r >= 0) & (fpr_r <= 1)) and np.all((tpr_r >= 0) & (tpr_r <= 1)) and
      np.all(np.diff(fpr_r) >= 0))
auc_r = _auc(fpr_r, tpr_r)
check('roc_curve AUC is high for a separable trajectory signal', auc_r > 0.9, round(auc_r, 3))

by_step = tclf.roc_curve_by_step(sig_t, bg_t, grid=TRAJECTORY_GRID)
check('roc_curve_by_step returns one entry per grid step', set(by_step) == set(TRAJECTORY_GRID))
aucs_by_step = [_auc(by_step[d][0], by_step[d][1]) for d in TRAJECTORY_GRID]
check('AUC with only the trigger step revealed is already well above chance',
      aucs_by_step[0] > 0.6, round(aucs_by_step[0], 3))
check('revealing more steps does not collapse the AUC (final >= first step)',
      aucs_by_step[-1] >= aucs_by_step[0], [round(a, 3) for a in aucs_by_step])

print('\n=== ColourClassifier: multi-colour flexible boundary ===')
from cmsne.classifier import ColourClassifier, colour_feature_matrix
_rng = np.random.default_rng(7)
def _mk(n, mu):                                   # events with a 5-colour vector, some NaN
    out = []
    for _ in range(n):
        e = {k: float(_rng.normal(mu, 0.4)) for k in ['ug', 'gr', 'ri', 'iz', 'zy']}
        e['peakmag'] = float(_rng.normal(22, 1))
        for k in _rng.choice(['ug', 'gr', 'ri', 'iz', 'zy'], 2, replace=False):
            e[k] = np.nan                         # emulate missing bands
        out.append(e)
    return out
sig_ev = _mk(1500, 1.2); bg_ev = _mk(3000, 0.0)   # signal redder than background
Xm, cols = colour_feature_matrix(sig_ev[:3])
check('feature matrix shape + NaNs', Xm.shape == (3, 6) and np.isnan(Xm).any(), (Xm.shape,))
clf = ColourClassifier(target_fpr=0.10).fit(sig_ev, bg_ev)
rec = clf.recovery_rate(sig_ev)
fpr = float(np.mean(clf.predict(bg_ev)))
check('separable colours: high recovery', rec > 0.8, round(rec, 3))
check('threshold pins contamination near target', abs(fpr - 0.10) < 0.06, round(fpr, 3))
fpr_roc, tpr_roc, thr_roc = clf.roc_curve(sig_ev, bg_ev)
check('roc_curve returns matching-length fpr/tpr/thresholds arrays',
      len(fpr_roc) == len(tpr_roc) == len(thr_roc))
check('roc_curve fpr/tpr bounded in [0,1] and fpr non-decreasing',
      np.all((fpr_roc >= 0) & (fpr_roc <= 1)) and np.all((tpr_roc >= 0) & (tpr_roc <= 1)) and
      np.all(np.diff(fpr_roc) >= 0))
auc = _auc(fpr_roc, tpr_roc)
check('separable signal gives a high ROC AUC', auc > 0.9, round(auc, 3))
fpr_w, tpr_w, _ = clf.roc_curve(sig_ev, bg_ev,
      signal_weights=np.abs(_rng.normal(1, 0.3, len(sig_ev))),
      background_weights=np.abs(_rng.normal(1, 0.3, len(bg_ev))))
auc_w = _auc(fpr_w, tpr_w)
check('rate-weighted roc_curve also gives a high AUC', auc_w > 0.85, round(auc_w, 3))
check('NaN colours handled (no crash, finite scores)', np.all(np.isfinite(clf.score(sig_ev))))
check('default magnitude feature is brightest-observed peakmag',
      clf.magnitude_feature == 'peakmag' and clf.extra == ('peakmag',), (clf.magnitude_feature, clf.extra))
check('magnitude_feature is selectable', ColourClassifier(magnitude_feature='peakmag_fit').extra == ('peakmag_fit',))
check('include_peakmag=False -> colour-only', ColourClassifier(include_peakmag=False).extra == ())
try:
    ColourClassifier(magnitude_feature='nope'); _bad = False
except ValueError:
    _bad = True
check('invalid magnitude_feature rejected', _bad)
# a rate-weighted recovery call runs and stays in [0,1]
rw = clf.recovery_rate(sig_ev, weights=np.abs(_rng.normal(1, 0.3, len(sig_ev))))
check('weighted recovery in range', 0.0 <= rw <= 1.0, round(rw, 3))
# probability output + prior adjustment + ranking
p_bal = clf.probability(sig_ev)
check('probability in [0,1]', np.all((p_bal >= 0) & (p_bal <= 1)))
check('signal more lensed-like than background (mean prob)',
      p_bal.mean() > clf.probability(bg_ev).mean(), (round(p_bal.mean(), 2),))
p_true = clf.probability(sig_ev, prior=1e-3)
check('lower prior never raises the posterior', np.all(p_true <= p_bal + 1e-9))
# at the true prior, an unlensed-like (background) object collapses toward the prior
check('background posterior collapses at the true prior',
      clf.probability(bg_ev, prior=1e-3).mean() < 0.01,
      round(float(clf.probability(bg_ev, prior=1e-3).mean()), 5))
check('prior adjustment is monotonic in the score',
      np.all(np.argsort(p_true) == np.argsort(p_bal)))
order = clf.rank(sig_ev + bg_ev)
check('rank returns a full permutation, most-lensed first',
      len(order) == len(sig_ev) + len(bg_ev) and clf.score(sig_ev + bg_ev)[order[0]] ==
      clf.score(sig_ev + bg_ev).max())

print('\n=== cleanups: CC sub-type fractions + fitted peak ===')
from cmsne.lsst import CC_FRACTIONS, CC_TO_IA_RATE
from cmsne.multicolour import rate_weight
check('CC_FRACTIONS sum to 1', abs(sum(CC_FRACTIONS.values()) - 1.0) < 1e-9, sum(CC_FRACTIONS.values()))
check('common CC (II-P) weighted above rare (II-L)',
      rate_weight('uCC', 0.3, 1.0, 'nugent-sn2p') > rate_weight('uCC', 0.3, 1.0, 'nugent-sn2l'))
check('fraction scales the CC weight',
      np.isclose(rate_weight('uCC', 0.3, 1.0, 'nugent-sn2p'),
                 cm.cc_ul_weight(0.3) * CC_FRACTIONS['nugent-sn2p']))
check('Ia weight ignores source arg',
      rate_weight('uIa', 0.3, 1.0, 'nugent-sn2p') == rate_weight('uIa', 0.3, 1.0))

pk_days = np.array([-8., -4., 0., 4., 8.]); pk_mag = 0.02 * pk_days ** 2 + 21.0   # min 21.0 at day 0
class _LCp:
    obs_days = pk_days; obs_mag = pk_mag; obs_filters = np.array(['r'] * 5)
peak = lcmod.fitted_peak_magnitude(_LCp())
check('fitted peak recovers the parabola vertex', abs(peak - 21.0) < 0.05, round(peak, 3))
check('fitted peak brighter than first detection', peak < pk_mag[0], (round(peak, 2), pk_mag[0]))
check('fitted peak NaN when nothing detected',
      np.isnan(lcmod.fitted_peak_magnitude(type('L', (), {'obs_days': np.array([1.]), 'obs_mag': np.array([np.inf]), 'obs_filters': np.array(['r'])})())))

print('\n=== comparison: naive sequential-multiplication baseline ===')
from cmsne.comparison import naive_sequential_posterior
p3 = naive_sequential_posterior(np.array([[0.9, 0.9, 0.9]]))
check('three agreeing high-confidence steps combine to near-certainty',
      p3[0] > 0.95, round(float(p3[0]), 4))
p_half = naive_sequential_posterior(np.array([[0.5, 0.5, 0.5]]))
check('all-uninformative (p=0.5) steps leave the posterior at 0.5',
      np.isclose(p_half[0], 0.5), float(p_half[0]))
p_nan = naive_sequential_posterior(np.array([[0.9, np.nan, np.nan]]))
check('NaN (unobserved) steps are skipped, not treated as evidence',
      np.isclose(p_nan[0], 0.9), float(p_nan[0]))
p_batch = naive_sequential_posterior(np.array([[0.9, 0.9], [0.1, 0.1]]))
check('batched 2D input returns one posterior per event',
      p_batch.shape == (2,) and p_batch[0] > 0.5 > p_batch[1], p_batch.tolist())

print('\n=== contours: evolving posterior surface ===')
from cmsne import contours as ctr
_rngc = np.random.default_rng(3)
def _mk_cm(n, mu):
    return [{'gr_d0': float(_rngc.normal(mu, 0.3)), 'peakmag_d0': float(_rngc.normal(22, 1)),
            'gr_d14': float(_rngc.normal(mu, 0.3)), 'peakmag_d14': float(_rngc.normal(22, 1))}
           for _ in range(n)]
sig_c, bg_c = _mk_cm(300, 1.0), _mk_cm(600, 0.0)
cg, mg, pg, cclf = ctr.posterior_surface(sig_c, bg_c, 'gr_d0', 'peakmag_d0', n=20)
check('posterior_surface returns (n,n) grids', cg.shape == (20, 20) and pg.shape == (20, 20), pg.shape)
check('probability grid is bounded in [0,1]', np.all((pg >= 0) & (pg <= 1)))
check('higher colour region is more lensed-like (separable signal)',
      pg[:, -1].mean() > pg[:, 0].mean(), (round(pg[:, -1].mean(), 3), round(pg[:, 0].mean(), 3)))

surfaces = ctr.evolving_contours(sig_c, bg_c, 'gr', grid=(0.0, 14.0), n=15)
check('evolving_contours returns one surface per grid step', set(surfaces) == {0.0, 14.0})

# robust_range: outlier-insensitive, and weight-aware
_vals = np.concatenate([_rngc.normal(0, 1, 500), [500.0, -500.0]])  # 2 extreme outliers
_lo_mm, _hi_mm = float(np.min(_vals)), float(np.max(_vals))
_lo_rr, _hi_rr = ctr.robust_range(_vals)
check('robust_range ignores rare extreme outliers unlike min/max',
      _hi_rr < _hi_mm / 10 and _lo_rr > _lo_mm / 10, ((_lo_rr, _hi_rr), (_lo_mm, _hi_mm)))
_w_suppress = np.ones(len(_vals)); _w_suppress[-2:] = 1e-6  # near-zero-weight the outliers
_lo_rrw, _hi_rrw = ctr.robust_range(_vals, weights=_w_suppress, quantiles=(0.001, 0.999))
check('robust_range is weight-aware: near-zero weight on outliers keeps range tight',
      _hi_rrw < 10 and _lo_rrw > -10, (_lo_rrw, _hi_rrw))

# posterior_surface/evolving_contours accept and use weights
_sig_w = np.ones(len(sig_c)); _bg_w = np.ones(len(bg_c))
_bg_w[:300] = 1e-6   # down-weight half the background so it barely influences the fit
cg_w, mg_w, pg_w, clf_w = ctr.posterior_surface(
    sig_c, bg_c, 'gr_d0', 'peakmag_d0', n=20, signal_weights=_sig_w, background_weights=_bg_w)
check('posterior_surface accepts weights and still returns valid grids',
      pg_w.shape == (20, 20) and np.all((pg_w >= 0) & (pg_w <= 1)))
check('posterior_surface weights actually change the fit (differs from unweighted)',
      not np.allclose(pg_w, pg), float(np.mean(np.abs(pg_w - pg))))

surfaces_w = ctr.evolving_contours(sig_c, bg_c, 'gr', grid=(0.0, 14.0), n=15,
                                   signal_weights=_sig_w, background_weights=_bg_w)
check('evolving_contours forwards weights to every step',
      not np.allclose(surfaces_w[0.0][2], surfaces[0.0][2]))
check('each evolving-contour entry has the 4-tuple shape',
      all(len(v) == 4 and v[2].shape == (15, 15) for v in surfaces.values()))

print('\n=== comparison: static vs trajectory vs naive harness ===')
from cmsne.comparison import compare_classifiers
from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID
from cmsne.classifier import ColourClassifier

_rngh = np.random.default_rng(5)
def _mk_full(n, mu):
    """Events exposing BOTH the static 2-point schema and the trajectory schema."""
    out = []
    for _ in range(n):
        e = {'gr': float(_rngh.normal(mu, 0.4)), 'peakmag': float(_rngh.normal(22, 1))}
        for day in TRAJECTORY_GRID:
            suffix = f'_d{int(day)}'
            for colour in ('ug', 'gr', 'ri', 'iz', 'zy'):
                e[f'{colour}{suffix}'] = float(_rngh.normal(mu, 0.4))
            e[f'peakmag{suffix}'] = float(_rngh.normal(22, 1))
        out.append(e)
    return out

sig_h, bg_h = _mk_full(400, 1.2), _mk_full(800, 0.0)
static_clf = ColourClassifier(keys=['gr'], target_fpr=0.10).fit(sig_h, bg_h)
traj_clf = TrajectoryClassifier(target_fpr=0.10).fit(sig_h, bg_h)
result = compare_classifiers(static_clf, traj_clf, sig_h, bg_h, step_colour_key='gr')
check('harness reports all three headline metrics',
      set(result) == {'static_recovery', 'trajectory_recovery', 'naive_vs_trajectory_mean_abs_diff'})
check('both recoveries are plausible for a separable signal',
      result['static_recovery'] > 0.5 and result['trajectory_recovery'] > 0.5, result)
check('naive-vs-trajectory divergence is non-negative and finite',
      np.isfinite(result['naive_vs_trajectory_mean_abs_diff']) and
      result['naive_vs_trajectory_mean_abs_diff'] >= 0.0, result['naive_vs_trajectory_mean_abs_diff'])

print('\n=== trajectory: dual-schema generation (static + trajectory merged) ===')
_lcd = _LCm([2, 3, 20, 23], ['g', 'r', 'g', 'r'], [22.0, 21.5, 20.0, 21.0])
_fd = traj.event_dual_schema(_lcd, evolution_days=15.0, grid=(0.0, 20.0), window=5.0)
check('dual schema has the static multicolour keys',
      'peakmag' in _fd and 'gr' in _fd and 'n_bands' in _fd)
check('dual schema has the trajectory keys', 'gr_d0' in _fd and 'peakmag_d20' in _fd)
check('shared trigger_day key matches both halves',
      _fd['trigger_day'] == mcol.event_multicolour(_lcd, 15.0, 5.0)['trigger_day'] ==
      traj.event_trajectory(_lcd, (0.0, 20.0), 5.0)['trigger_day'])
check('static gr colour matches standalone event_multicolour',
      np.isclose(_fd['gr'], mcol.event_multicolour(_lcd, 15.0, 5.0)['gr']))
check('trajectory gr_d0 colour matches standalone event_trajectory',
      np.isclose(_fd['gr_d0'], traj.event_trajectory(_lcd, (0.0, 20.0), 5.0)['gr_d0']))
check('static gr (epoch=trigger) matches trajectory gr_d0 (same epoch)',
      np.isclose(_fd['gr'], _fd['gr_d0']))
check('no detections -> None',
      traj.event_dual_schema(_LCm([1, 2], ['g', 'r'], [np.inf, np.inf])) is None)

_dg = traj.DualSchemaGenerator()
check('DualSchemaGenerator inherits MultiColourGenerator API',
      hasattr(_dg, 'generate_one') and hasattr(_dg, 'generate_many'))
check('DualSchemaGenerator._event_features uses the dual schema',
      set(_dg._event_features(_lcd, 15.0, 5.0)) >= {'gr', 'peakmag', 'gr_d0', 'peakmag_d60'})

print('\n=== importance_sampling: effective sample size + sparse-cell diagnosis ===')
from cmsne import importance_sampling as isamp

check('ESS of uniform weights equals N',
      np.isclose(isamp.effective_sample_size(np.ones(50)), 50.0))
check('ESS of one dominant weight collapses toward 1',
      isamp.effective_sample_size(np.array([100.0, 0.01, 0.01, 0.01])) < 1.1)
check('ESS of empty/all-zero weights is 0.0',
      isamp.effective_sample_size(np.array([])) == 0.0 and
      isamp.effective_sample_size(np.zeros(5)) == 0.0)

# 4x4 grid over gr in [0,4], peakmag in [20,24]: fill every cell densely except one.
_rng_d = np.random.default_rng(0)
_dense_events, _dense_w = [], []
for _i in range(4):
    for _j in range(4):
        n_pts = 2 if (_i, _j) == (3, 3) else 30   # (3,3) is the deliberately sparse cell
        for _ in range(n_pts):
            _dense_events.append({'gr': _i + float(_rng_d.uniform(0, 1)),
                                  'peakmag': 20 + _j + float(_rng_d.uniform(0, 1))})
            _dense_w.append(1.0)
_dense_w = np.array(_dense_w)

_flagged = isamp.diagnose_sparse_cells(_dense_events, _dense_w, band_pair_keys=['gr'],
                                       magnitude_key='peakmag', n_bins=4, ess_floor_fraction=0.2)
check('diagnose_sparse_cells flags the deliberately sparse cell',
      any(f['colour_range'][0] > 2.5 and f['magnitude_range'][0] > 22.5 for f in _flagged), _flagged)
check('diagnose_sparse_cells does not flag the densely-filled cells',
      not any(f['colour_range'][1] < 1.5 and f['magnitude_range'][1] < 21.5 for f in _flagged), _flagged)
check('every flagged cell reports its ESS below its floor',
      all(f['ess'] < f['floor'] for f in _flagged), _flagged)

# Regression: a real BlueBEAR smoke-test run surfaced this before robust_range
# was used for bin edges -- a physically near-impossible event with near-zero
# weight (e.g. near-zero redshift + near-maximal magnification, which
# rate_weight already assigns weight=0.0 to) must not stretch the grid and
# hijack which cells get flagged, the same class of outlier
# cmsne.contours.robust_range already guards against elsewhere.
_outlier_events = list(_dense_events) + [{'gr': 500.0, 'peakmag': -500.0} for _ in range(5)]
_outlier_w = np.r_[_dense_w, np.zeros(5)]
_flagged_outlier = isamp.diagnose_sparse_cells(_outlier_events, _outlier_w, band_pair_keys=['gr'],
                                               magnitude_key='peakmag', n_bins=4, ess_floor_fraction=0.2)
check('a zero-weight extreme outlier does not stretch the grid away from the real data',
      all(-2.0 <= f['colour_range'][0] <= 6.0 and 18.0 <= f['magnitude_range'][0] <= 26.0
          for f in _flagged_outlier), _flagged_outlier)
check('the outlier scenario still flags the same deliberately sparse cell as before',
      any(f['colour_range'][0] > 2.5 and f['magnitude_range'][0] > 22.5 for f in _flagged_outlier),
      _flagged_outlier)

# fit_boost_proposal: enough points -> KDE (lensed, 2D)
_rng_f = np.random.default_rng(1)
_cell_events_lensed = [{'z': float(_rng_f.uniform(0.1, 0.3)), 'magnification': float(_rng_f.uniform(5, 10)),
                       'gr': 2.0, 'peakmag': 16.0} for _ in range(10)]
_cell_w = np.ones(10)
_prop_kde = isamp.fit_boost_proposal(_cell_events_lensed, _cell_w, 'gr', 'peakmag',
                                     (1.5, 2.5), (15.0, 17.0), lensed=True,
                                     z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('fit_boost_proposal fits a KDE when enough points exist',
      _prop_kde['kde'] is not None and _prop_kde['fallback'] is None)

# fit_boost_proposal: too few points -> box fallback
_cell_events_few = [{'z': 0.2, 'magnification': 8.0, 'gr': 2.0, 'peakmag': 16.0}]
_prop_box = isamp.fit_boost_proposal(_cell_events_few, np.ones(1), 'gr', 'peakmag',
                                     (1.5, 2.5), (15.0, 17.0), lensed=True,
                                     z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('fit_boost_proposal falls back to a box with too few points',
      _prop_box['kde'] is None and _prop_box['fallback'] == 'box' and
      _prop_box['z_box'][0] <= 0.2 <= _prop_box['z_box'][1])

# fit_boost_proposal: zero points -> prior_box fallback spans the full ranges given
_prop_prior = isamp.fit_boost_proposal([], np.array([]), 'gr', 'peakmag',
                                       (1.5, 2.5), (15.0, 17.0), lensed=True,
                                       z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('fit_boost_proposal falls back to the full prior range with zero points',
      _prop_prior['fallback'] == 'prior_box' and _prop_prior['z_box'] == (0.0, 1.0) and
      _prop_prior['mu_box'] == (2.0, 50.0))

# unlensed: 1D KDE over z only (magnification fixed at 1.0)
_cell_events_unlensed = [{'z': float(_rng_f.uniform(0.1, 0.3)), 'magnification': 1.0,
                         'gr': 2.0, 'peakmag': 16.0} for _ in range(10)]
_prop_unlensed = isamp.fit_boost_proposal(_cell_events_unlensed, np.ones(10), 'gr', 'peakmag',
                                          (1.5, 2.5), (15.0, 17.0), lensed=False,
                                          z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('fit_boost_proposal fits a 1D KDE for unlensed classes (no singular-matrix crash)',
      _prop_unlensed['kde'] is not None and 'mu_box' not in _prop_unlensed)

# sample_latent: draws respect the clip bounds
_z_s, _mu_s = isamp.sample_latent(_prop_kde, 200, lensed=True,
                                  z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('sample_latent draws stay within z_range and magnification_range',
      np.all(_z_s >= 0.0) and np.all(_z_s <= 1.0) and
      np.all(_mu_s >= 2.0) and np.all(_mu_s <= 50.0), (float(_z_s.min()), float(_z_s.max())))
_z_u, _mu_u = isamp.sample_latent(_prop_unlensed, 50, lensed=False,
                                  z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check('sample_latent returns magnification=1.0 for unlensed draws',
      np.all(_mu_u == 1.0))

# _proposal_density: box fallback is zero outside its bounds, positive inside
_dens_in = isamp._proposal_density(_prop_box, np.array([0.2]), np.array([8.0]), lensed=True)
_dens_out = isamp._proposal_density(_prop_box, np.array([0.9]), np.array([40.0]), lensed=True)
check('_proposal_density is positive inside a box fallback and zero outside',
      _dens_in[0] > 0.0 and _dens_out[0] == 0.0, (_dens_in[0], _dens_out[0]))

# _proposal_density: KDE branch gives finite, non-negative densities (lensed and unlensed)
_dens_kde = isamp._proposal_density(_prop_kde, np.array([0.2, 0.25]), np.array([7.0, 8.0]), lensed=True)
check('_proposal_density returns finite non-negative values for a lensed KDE',
      np.all(np.isfinite(_dens_kde)) and np.all(_dens_kde >= 0), str(_dens_kde))
_dens_kde_u = isamp._proposal_density(_prop_unlensed, np.array([0.2, 0.25]), np.ones(2), lensed=False)
check('_proposal_density returns finite non-negative values for an unlensed KDE',
      np.all(np.isfinite(_dens_kde_u)) and np.all(_dens_kde_u >= 0), str(_dens_kde_u))

# regression: weighted-positive events share one z (and magnification) while zero-weight
# events in the same cell differ -> must NOT attempt a singular KDE; falls back to a box
_reg_events = ([{'z': 0.25, 'magnification': float(_rng_f.uniform(5, 10)), 'gr': 2.0, 'peakmag': 16.0}
                for _ in range(5)] +
               [{'z': float(z0), 'magnification': float(_rng_f.uniform(5, 10)), 'gr': 2.0, 'peakmag': 16.0}
                for z0 in (0.1, 0.15, 0.2, 0.3, 0.35)])
_reg_w = np.r_[np.ones(5), np.zeros(5)]
_reg_ok, _reg_prop = True, None
try:
    _reg_prop = isamp.fit_boost_proposal(_reg_events, _reg_w, 'gr', 'peakmag',
                                         (1.5, 2.5), (15.0, 17.0), lensed=True,
                                         z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
except Exception as _e:
    _reg_ok = False
check('fit_boost_proposal does not crash on degenerate-positive lensed cell (zero-weight spread ignored)',
      _reg_ok and _reg_prop['kde'] is None and _reg_prop['fallback'] == 'box', _reg_prop)
_reg_events_u = [{'z': 0.25, 'magnification': 1.0, 'gr': 2.0, 'peakmag': 16.0} for _ in range(5)] + \
                [{'z': float(z0), 'magnification': 1.0, 'gr': 2.0, 'peakmag': 16.0}
                 for z0 in (0.1, 0.15, 0.2, 0.3, 0.35)]
_reg_ok_u, _reg_prop_u = True, None
try:
    _reg_prop_u = isamp.fit_boost_proposal(_reg_events_u, _reg_w, 'gr', 'peakmag',
                                           (1.5, 2.5), (15.0, 17.0), lensed=False,
                                           z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
except Exception as _e:
    _reg_ok_u = False
check('fit_boost_proposal does not crash on degenerate-positive unlensed cell',
      _reg_ok_u and _reg_prop_u['kde'] is None and _reg_prop_u['fallback'] == 'box', _reg_prop_u)

# fit_boost_proposal: enough points for a KDE, but the weighted-positive z values
# cluster right against the lower edge of z_range with enough spread that the
# KDE's own bandwidth smooths real density below z_range[0] -- sample_latent
# would clip those draws back onto the boundary while _proposal_density (pre-fix)
# kept evaluating the unclipped KDE, so the declared and actual sampling
# distributions would disagree. The guard must reject this KDE and fall back to
# the (self-consistent) padded box, tagged 'box_out_of_range' -- not silently
# accept it (fallback is None) and not confuse it with the too-few-points 'box'.
_rng_oob = np.random.default_rng(42)
_n_oob = 20
_z_oob = np.clip(_rng_oob.normal(0.02, 0.05, _n_oob), 0.0, None)   # test setup only: events
_mu_oob = _rng_oob.uniform(5.0, 10.0, _n_oob)                      # themselves stay physical (>=0)
_events_oob = [{'z': float(_z_oob[i]), 'magnification': float(_mu_oob[i]),
               'gr': 2.0, 'peakmag': 16.0} for i in range(_n_oob)]
_prop_oob = isamp.fit_boost_proposal(_events_oob, np.ones(_n_oob), 'gr', 'peakmag',
                                     (1.5, 2.5), (15.0, 17.0), lensed=True,
                                     z_range=(0.0, 1.0), magnification_range=(2.0, 50.0))
check("fit_boost_proposal rejects a KDE whose mass leaks past z_range's lower edge",
      _prop_oob['kde'] is None and _prop_oob['fallback'] == 'box_out_of_range', _prop_oob)

class _FakeGen:
    """Toy generator for unit-testing generate_with_boosting without the real
    sncosmo/OpSim pipeline: observed colour/peakmag are a deterministic,
    known function of latent (z, magnification)."""
    @staticmethod
    def _to_event(z, mu):
        return {'z': float(z), 'magnification': float(mu),
                'gr': float(2.0 - z + 0.02 * mu), 'peakmag': float(20.0 + 5 * z - 0.1 * mu)}

    def generate_many(self, n, kind, z_range, abs_mag_avg, abs_mag_sig,
                      magnification_range=(2.0, 50.0), mjd_range=None, mjd_high=None):
        out = []
        for _ in range(n):
            z = np.random.uniform(*z_range)
            mu = np.random.uniform(*magnification_range) if kind == 'lensed' else 1.0
            out.append(self._to_event(z, mu))
        return out

    def generate_one(self, z, abs_mag, magnification, times, ra, dec, mjd_high=None):
        return self._to_event(z, magnification)


def _toy_rate_weight(z, mu):
    return float(np.exp(-z) / mu ** 2)


np.random.seed(0)
_fake = _FakeGen()
_events_nb, _info_nb = isamp.generate_with_boosting(
    _fake, 500, 'lensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight, pilot_frac=1.0)
check('pilot_frac=1.0 reproduces bare rate_weight_fn exactly (backward compatibility)',
      all(np.isclose(e['weight'], _toy_rate_weight(e['z'], e['magnification'])) for e in _events_nb) and
      _info_nb['n_boost'] == 0, len(_events_nb))

np.random.seed(0)
_events_b, _info_b = isamp.generate_with_boosting(
    _fake, 500, 'lensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight,
    pilot_frac=0.5, n_bins=4, ess_floor_fraction=0.9)   # high floor forces some cell to be flagged
check('boosting fires and reports flagged cells + boost draws',
      _info_b['n_boost'] > 0 and len(_info_b['flagged_cells']) > 0, _info_b['n_boost'])
check('every event (pilot- and boost-origin) has a finite, positive weight after boosting',
      all(np.isfinite(e['weight']) and e['weight'] > 0 for e in _events_b), len(_events_b))
check('boosting returns at least as many events as the pilot alone',
      len(_events_b) >= _info_b['n_pilot'], (len(_events_b), _info_b['n_pilot']))

np.random.seed(0)
_events_u, _info_u = isamp.generate_with_boosting(
    _fake, 300, 'unlensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight,
    pilot_frac=0.5, n_bins=4, ess_floor_fraction=0.9)
check('unlensed boosting keeps magnification fixed at 1.0 for every event',
      all(e['magnification'] == 1.0 for e in _events_u))

# Quantitative unbiasedness stress check (what the design spec actually asked for):
# with this (z_range, n_bins, ess_floor_fraction) combination, the pilot's sparse
# cell sits right against z_range's lower edge (z_range=(0.0, 1.0)), so boosting
# routinely needs the KDE-acceptance guard above to stay unbiased -- confirmed by
# inspecting flagged_cells' 'fallback' across the trial seeds below. Reusing the
# existing _fake/_toy_rate_weight/(pilot_frac=0.5, n_bins=4, ess_floor_fraction=0.9)
# pattern, compare the mean total weighted mass with boosting on against the mean
# with boosting off (pilot_frac=1.0), same n and same seeds: an unbiased scheme
# should make these two means agree up to Monte Carlo noise, regardless of how the
# weight is redistributed between pilot- and boost-origin events.
# 20 trials of n=500 takes well under a second here, so no need to shrink it.
_n_trials_oob = 20
_sums_boost, _sums_nb, _saw_box_out_of_range = [], [], False
for _seed in range(_n_trials_oob):
    np.random.seed(_seed)
    _ev_b, _info_b2 = isamp.generate_with_boosting(
        _fake, 500, 'lensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight,
        pilot_frac=0.5, n_bins=4, ess_floor_fraction=0.9)
    _sums_boost.append(sum(e['weight'] for e in _ev_b))
    if any(c.get('fallback') == 'box_out_of_range' for c in _info_b2['flagged_cells']):
        _saw_box_out_of_range = True

    np.random.seed(_seed)
    _ev_nb2, _ = isamp.generate_with_boosting(
        _fake, 500, 'lensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight, pilot_frac=1.0)
    _sums_nb.append(sum(e['weight'] for e in _ev_nb2))

_mean_boost, _mean_nb = float(np.mean(_sums_boost)), float(np.mean(_sums_nb))
_rel_diff = abs(_mean_boost - _mean_nb) / _mean_nb
check('this stress scenario actually exercises the box_out_of_range guard',
      _saw_box_out_of_range)
check('boosted and un-boosted total weighted mass agree within 15% on average '
      '(quantitative unbiasedness check)',
      _rel_diff < 0.15, (_mean_boost, _mean_nb, _rel_diff))


class _FakeGenWithDrops(_FakeGen):
    """Like _FakeGen, but generate_one returns None on every 3rd call, to
    exercise dropped boost draws."""
    def __init__(self):
        self.calls = 0

    def generate_one(self, z, abs_mag, magnification, times, ra, dec, mjd_high=None):
        self.calls += 1
        if self.calls % 3 == 0:
            return None
        return self._to_event(z, magnification)


np.random.seed(0)
_events_d, _info_d = isamp.generate_with_boosting(
    _FakeGenWithDrops(), 500, 'lensed', (0.0, 1.0), -19.0, 0.1, _toy_rate_weight,
    pilot_frac=0.5, n_bins=4, ess_floor_fraction=0.9)
_attempted_total = _info_d['n_pilot'] + _info_d['n_boost']
check('dropped boost draws are handled: function completes and population is smaller than attempted',
      len(_events_d) < _attempted_total and len(_events_d) >= _info_d['n_pilot'],
      (len(_events_d), _attempted_total))
check('every event still has a finite, positive weight when boost draws are dropped',
      all(np.isfinite(e['weight']) and e['weight'] > 0 for e in _events_d), len(_events_d))
check('flagged-cell bookkeeping stays consistent when boost draws are dropped',
      len(_info_d['flagged_cells']) > 0 and
      all('fallback' in c and c['band_pair'] for c in _info_d['flagged_cells']),
      len(_info_d['flagged_cells']))


# Backward-compatibility against real production-shaped events (not just the toy
# _FakeGen): a pass-through fake generator that returns pre-built events keeps
# their original z/magnification, so generate_with_boosting(pilot_frac=1.0) must
# reproduce rate_weight(...) exactly, the same property Task 3 checked on toy data.
class _PassThroughGen:
    def __init__(self, events):
        self._events = events

    def generate_many(self, n, kind, z_range, abs_mag_avg, abs_mag_sig,
                      magnification_range=(2.0, 50.0), mjd_range=None, mjd_high=None):
        return [dict(e) for e in self._events[:n]]

    def generate_one(self, z, abs_mag, magnification, times, ra, dec, mjd_high=None):
        return {'z': float(z), 'magnification': float(magnification), 'gr': 2.0, 'peakmag': 20.0}


_rng_pc = np.random.default_rng(2)
_prod_shaped_events = [{'z': float(_rng_pc.uniform(0.01, 1.5)),
                       'magnification': float(_rng_pc.uniform(2.0, 50.0)),
                       'gr': float(_rng_pc.normal(0.5, 0.5)),
                       'peakmag': float(_rng_pc.normal(22, 2))} for _ in range(300)]

np.random.seed(3)
_events_pc, _info_pc = isamp.generate_with_boosting(
    _PassThroughGen(_prod_shaped_events), len(_prod_shaped_events), 'lensed',
    (0.01, 1.5), -19.23, 0.10, _toy_rate_weight, pilot_frac=1.0)
check('generate_with_boosting(pilot_frac=1.0) matches bare rate_weight on production-shaped events',
      all(np.isclose(e['weight'], _toy_rate_weight(e['z'], e['magnification'])) for e in _events_pc) and
      len(_events_pc) == len(_prod_shaped_events) and _info_pc['n_boost'] == 0)

print('\n=== local density-ratio blend: kernel/bandwidth helpers ===')
from cmsne.classifier import _weighted_std, _scott_bandwidths, _kernel_weighted_values

_ws_vals = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
_ws_wts = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
check('_weighted_std ignores NaN and matches plain std of the finite subset',
      np.isclose(_weighted_std(_ws_vals, _ws_wts), np.std([1.0, 2.0, 3.0, 4.0]), atol=1e-9),
      _weighted_std(_ws_vals, _ws_wts))
check('_weighted_std returns the safe fallback 1.0 for a degenerate (all-equal) column',
      _weighted_std(np.array([5.0, 5.0, 5.0]), np.array([1.0, 1.0, 1.0])) == 1.0)
check('_weighted_std returns the safe fallback 1.0 for fewer than 2 finite entries',
      _weighted_std(np.array([5.0, np.nan]), np.array([1.0, 1.0])) == 1.0)

_bw_X = np.column_stack([np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                         np.array([10.0, 10.0, 10.0, 10.0, 10.0])])
_bw_w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
_bw = _scott_bandwidths(_bw_X, _bw_w)
check('_scott_bandwidths returns one positive, finite bandwidth per column',
      _bw.shape == (2,) and np.all(_bw > 0) and np.all(np.isfinite(_bw)), str(_bw))
check('_scott_bandwidths gives the degenerate (constant) column the smaller bandwidth',
      _bw[1] < _bw[0], str(_bw))

_kw_X = np.array([[0.0], [1.0], [10.0]])
_kw_w = np.array([1.0, 1.0, 1.0])
_kw_h = np.array([1.0])
_kw_dims = np.array([0])
_kw_vals = _kernel_weighted_values(np.array([0.0]), _kw_X, _kw_w, _kw_h, _kw_dims)
check('_kernel_weighted_values peaks at the exact-match training row',
      _kw_vals[0] > _kw_vals[1] > _kw_vals[2], str(_kw_vals))
check('_kernel_weighted_values matches weight*1.0 exactly at zero distance',
      np.isclose(_kw_vals[0], 1.0), _kw_vals[0])
check('_kernel_weighted_values returns zeros for empty dims',
      np.all(_kernel_weighted_values(np.array([0.0]), _kw_X, _kw_w, _kw_h, np.array([], dtype=int)) == 0.0))
check('_kernel_weighted_values returns zeros for zero training rows',
      np.all(_kernel_weighted_values(np.array([0.0]), np.zeros((0, 1)), np.zeros(0), _kw_h, _kw_dims) == 0.0))

print('\n=== uncertainty-aware noise injection: primitives ===')
from cmsne.classifier import (colour_feature_sigma_matrix, _replicate_with_uncertainty,
                              _grouped_effective_sample_size, DEFAULT_N_UNCERTAINTY_REPLICATES)
from cmsne.importance_sampling import effective_sample_size as _ess_ref2

check('DEFAULT_N_UNCERTAINTY_REPLICATES is a sane positive default', DEFAULT_N_UNCERTAINTY_REPLICATES >= 2,
      DEFAULT_N_UNCERTAINTY_REPLICATES)

_sig_events = [{'gr': 1.0, 'peakmag': 20.0, 'gr_err': 0.1},
               {'gr': 2.0, 'peakmag': 21.0}]  # second event has NO gr_err/peakmag_err at all
_sig_matrix = colour_feature_sigma_matrix(_sig_events, keys=['gr'], extra=('peakmag',))
check('colour_feature_sigma_matrix reads the *_err field when present',
      np.isclose(_sig_matrix[0, 0], 0.1), _sig_matrix[0, 0])
check('colour_feature_sigma_matrix is NaN when no *_err field exists',
      np.isnan(_sig_matrix[0, 1]) and np.isnan(_sig_matrix[1, 0]) and np.isnan(_sig_matrix[1, 1]))

# No-op: sigma entirely non-finite over dims -> unchanged, no RNG draws.
_rwu_X = np.array([[1.0, 2.0], [3.0, 4.0]])
_rwu_w = np.array([0.5, 0.5])
_rwu_sigma_none = np.full_like(_rwu_X, np.nan)
_rwu_Xr, _rwu_wr, _rwu_id = _replicate_with_uncertainty(_rwu_X, _rwu_sigma_none, _rwu_w, np.array([0, 1]), 8)
check('_replicate_with_uncertainty is a no-op when sigma is entirely non-finite',
      np.array_equal(_rwu_Xr, _rwu_X) and np.array_equal(_rwu_wr, _rwu_w) and
      np.array_equal(_rwu_id, np.arange(2)))

# Weight preservation: each original event's replica weights sum back exactly.
_rwu_sigma = np.array([[0.2, np.nan], [np.nan, 0.5]])
_rwu_Xr2, _rwu_wr2, _rwu_id2 = _replicate_with_uncertainty(_rwu_X, _rwu_sigma, _rwu_w, np.array([0, 1]), 8)
check('_replicate_with_uncertainty expands each row into n_replicates copies',
      len(_rwu_Xr2) == 2 * 8, len(_rwu_Xr2))
check('_replicate_with_uncertainty preserves each original event\'s total weight exactly',
      np.allclose(np.bincount(_rwu_id2, weights=_rwu_wr2, minlength=2), _rwu_w))
check('_replicate_with_uncertainty leaves a dimension unperturbed when that row\'s sigma is non-finite',
      np.all(_rwu_Xr2[_rwu_id2 == 0, 1] == 2.0) and np.all(_rwu_Xr2[_rwu_id2 == 1, 0] == 3.0))

# Perturbation: with a large, finite sigma, replicas measurably scatter.
_rwu_sigma_big = np.array([[5.0, np.nan]])
_rwu_Xr3, _, _ = _replicate_with_uncertainty(np.array([[10.0, 0.0]]), _rwu_sigma_big, np.array([1.0]),
                                              np.array([0]), 2000, seed=1)
check('_replicate_with_uncertainty replicas scatter with roughly the given sigma',
      abs(np.std(_rwu_Xr3[:, 0]) - 5.0) < 0.5, np.std(_rwu_Xr3[:, 0]))

# Second-pass review finding: two columns of the SAME row that are the literal same
# underlying observation (e.g. two trajectory grid steps whose nearest-detection
# window picked the identical measurement) end up with IDENTICAL (value, sigma) by
# construction -- they must move together under perturbation (one shared noise
# draw), not drift apart as if independently measured.
_dup_X = np.array([[3.0, 3.0, 7.0]])       # cols 0,1 are the SAME duplicated observation; col 2 is unrelated
_dup_sigma = np.array([[0.5, 0.5, 0.5]])   # identical sigma too (same underlying error)
_dup_Xr, _, _ = _replicate_with_uncertainty(_dup_X, _dup_sigma, np.array([1.0]), np.array([0, 1, 2]), 50, seed=7)
check('_replicate_with_uncertainty draws the SAME noise offset for two columns sharing '
      'the identical (value, sigma) pair (the duplicate-observation case)',
      np.array_equal(_dup_Xr[:, 0] - _dup_X[0, 0], _dup_Xr[:, 1] - _dup_X[0, 1]))
check('_replicate_with_uncertainty still draws INDEPENDENT noise for an unrelated column '
      '(not grouped just because it shares the same sigma value)',
      not np.allclose(_dup_Xr[:, 0] - _dup_X[0, 0], _dup_Xr[:, 2] - _dup_X[0, 2]))
check('_replicate_with_uncertainty grouping does not change the no-duplicate case: two '
      'genuinely distinct values (even with equal sigma) scatter independently, matching '
      'pre-grouping behaviour for the static schema (values essentially never collide)',
      np.std(_dup_Xr[:, 2]) > 0.1, np.std(_dup_Xr[:, 2]))

# Second-pass review finding: a cumulative-minimum column sequence (e.g. trajectory
# peakmag at increasing Δt) must stay non-increasing per replica even after
# independent perturbation -- a classifier should never see a physically impossible
# "cumulative best got fainter" feature vector.
_mono_X = np.array([[22.0, 21.0, 20.0]])    # already non-increasing (brighter over time)
_mono_sigma = np.array([[0.5, 0.5, 0.5]])
_mono_Xr, _, _ = _replicate_with_uncertainty(
    _mono_X, _mono_sigma, np.array([1.0]), np.array([0, 1, 2]), 500, seed=9,
    monotonic_groups=[[0, 1, 2]])
check('_replicate_with_uncertainty with monotonic_groups keeps every replica non-increasing '
      'across the grouped columns, even though each was perturbed independently',
      np.all(np.diff(_mono_Xr, axis=1) <= 0))
_mono_Xr_nogroup, _, _ = _replicate_with_uncertainty(
    _mono_X, _mono_sigma, np.array([1.0]), np.array([0, 1, 2]), 500, seed=9)
check('without monotonic_groups (the default, every existing caller), replicas are free to '
      'violate the ordering -- proving the constraint above comes from monotonic_groups, '
      'not some other implicit property of this data/seed',
      not np.all(np.diff(_mono_Xr_nogroup, axis=1) <= 0))

# The ESS-inflation bug, and the fix: 5 replicas of ONE real point must not look
# like 5 independent points.
_grp_event_id = np.array([0, 0, 0, 0, 0])
_grp_g = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
check('naive (ungrouped) ESS is fooled into thinking 5 replicas are 5 independent samples',
      np.isclose(_ess_ref2(_grp_g), 5.0), _ess_ref2(_grp_g))
check('_grouped_effective_sample_size correctly reports ESS~1 for 5 replicas of ONE real point',
      np.isclose(_grouped_effective_sample_size(_grp_g, _grp_event_id, 1), 1.0),
      _grouped_effective_sample_size(_grp_g, _grp_event_id, 1))

_grp_event_id2 = np.array([0, 0, 0, 1, 1, 1])
_grp_g2 = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
_grp_expected = _ess_ref2(np.array([0.3, 0.15]))
check('_grouped_effective_sample_size matches a direct group-sum-then-ESS calculation',
      np.isclose(_grouped_effective_sample_size(_grp_g2, _grp_event_id2, 2), _grp_expected),
      (_grouped_effective_sample_size(_grp_g2, _grp_event_id2, 2), _grp_expected))
check('_grouped_effective_sample_size reduces to plain ESS for a trivial (non-replicated) event_id',
      np.isclose(_grouped_effective_sample_size(_grp_g2, np.arange(6), 6), _ess_ref2(_grp_g2)))

print('\n=== local density-ratio blend: density ratio and blend weight ===')
from cmsne.classifier import _local_density_ratio, _blend_weight

_ldr_h = np.array([1.0])
_ldr_dims = np.array([0])
# Signal clustered at x=0; background far away (20 bandwidths), contributes
# essentially nothing to the kernel sum there.
_ldr_Xs = np.array([[0.0], [0.1], [-0.1]])
_ldr_ws = np.array([1e-5, 1e-5, 1e-5])
_ldr_Xb = np.array([[20.0], [21.0], [19.0]])
_ldr_wb = np.array([1e-5, 1e-5, 1e-5])
_p_kde, _ess_s, _ess_b = _local_density_ratio(
    np.array([0.0]), _ldr_Xs, _ldr_ws, _ldr_Xb, _ldr_wb, _ldr_h, _ldr_dims)
check('_local_density_ratio -> 1 when signal is local and background is many bandwidths away',
      _p_kde is not None and np.isclose(_p_kde, 1.0, atol=1e-6), _p_kde)
check('_local_density_ratio reports lower local ESS for the far-away class than the local class',
      _ess_b < _ess_s, (_ess_b, _ess_s))

# Symmetric case: swap roles.
_p_kde2, _, _ = _local_density_ratio(
    np.array([0.0]), _ldr_Xb, _ldr_wb, _ldr_Xs, _ldr_ws, _ldr_h, _ldr_dims)
check('_local_density_ratio -> 0 when background is local and signal is far away',
      _p_kde2 is not None and np.isclose(_p_kde2, 0.0, atol=1e-6), _p_kde2)

# Degenerate case: both classes far from x.
_p_kde3, _, _ = _local_density_ratio(
    np.array([1000.0]), _ldr_Xs, _ldr_ws, _ldr_Xb, _ldr_wb, _ldr_h, _ldr_dims)
check('_local_density_ratio returns None when both classes are far from x',
      _p_kde3 is None, _p_kde3)

check('_blend_weight is near 1.0 when both local ESS values are far above threshold',
      _blend_weight(1e6, 1e6, ess_threshold=10.0, log_width=1.0) > 0.999)
check('_blend_weight is near 0.0 when either local ESS value is far below threshold',
      _blend_weight(1000.0, 0.01, ess_threshold=10.0, log_width=1.0) < 0.001)
check('_blend_weight is driven by the MINIMUM of the two local ESS values, not their average',
      _blend_weight(1000.0, 0.01, ess_threshold=10.0, log_width=1.0) ==
      _blend_weight(0.01, 0.01, ess_threshold=10.0, log_width=1.0))


print('\n=== local density-ratio blend: wired into ColourClassifier ===')
from cmsne.classifier import ColourClassifier
from cmsne.trajectory import TrajectoryClassifier

# Dense bulk: plenty of both classes everywhere in a shared box -> alpha
# should be close to 1, blended probability() close to tree-only score().
_rng_dense = np.random.default_rng(3)
_n_dense = 1500
_dense_sig = [{'gr': float(_rng_dense.normal(1.0, 0.3)), 'peakmag': float(_rng_dense.normal(22.0, 1.0))}
             for _ in range(_n_dense)]
_dense_bg = [{'gr': float(_rng_dense.normal(1.2, 0.3)), 'peakmag': float(_rng_dense.normal(22.0, 1.0))}
            for _ in range(_n_dense)]
_dense_sw = np.full(_n_dense, 1e-5)
_dense_bw = np.full(_n_dense, 1e-5)
_clf_dense = ColourClassifier(keys=['gr'], include_peakmag=True)
_clf_dense.fit(_dense_sig, _dense_bg, _dense_sw, _dense_bw)
_dense_pts = [{'gr': 1.0, 'peakmag': 22.0}, {'gr': 1.1, 'peakmag': 21.7}]
_dense_blended = _clf_dense.probability(_dense_pts)
_dense_tree_only = np.clip(_clf_dense.score(_dense_pts), 1e-12, 1 - 1e-12)
check('dense region: blended probability() matches tree-only score() closely',
      np.allclose(_dense_blended, _dense_tree_only, atol=1e-3),
      (_dense_blended, _dense_tree_only))

# Sparse corner: background NEVER appears past gr=3.0, signal has a few rare
# points out there -- mimics the real bright+red bug at small scale.
_rng_sparse = np.random.default_rng(4)
_n_sparse = 1500
_sparse_sig_bulk = [{'gr': float(_rng_sparse.normal(1.0, 0.3)), 'peakmag': float(_rng_sparse.normal(22.0, 1.0))}
                    for _ in range(_n_sparse)]
_sparse_sig_corner = [{'gr': float(_rng_sparse.normal(4.0, 0.1)), 'peakmag': float(_rng_sparse.normal(16.0, 0.1))}
                      for _ in range(3)]
_sparse_sig = _sparse_sig_bulk + _sparse_sig_corner
_sparse_bg = [{'gr': float(_rng_sparse.normal(1.2, 0.3)), 'peakmag': float(_rng_sparse.normal(22.0, 1.0))}
             for _ in range(_n_sparse)]
_sparse_sw = np.full(len(_sparse_sig), 1e-5)
_sparse_bw = np.full(_n_sparse, 1e-5)
_clf_sparse = ColourClassifier(keys=['gr'], include_peakmag=True)
_clf_sparse.fit(_sparse_sig, _sparse_bg, _sparse_sw, _sparse_bw)
_corner_pt = [{'gr': 4.0, 'peakmag': 16.0}]
_corner_blended = _clf_sparse.probability(_corner_pt)[0]
_corner_tree_only = np.clip(_clf_sparse.score(_corner_pt), 1e-12, 1 - 1e-12)[0]
check('sparse corner: blended probability() is materially higher than tree-only score()',
      _corner_blended > _corner_tree_only + 0.05,
      (_corner_blended, _corner_tree_only))
check('sparse corner: blended probability() is clearly above the balanced-prior midpoint',
      _corner_blended > 0.7, _corner_blended)

# No finite feature columns at all -> falls back to p_tree unchanged.
_nan_pt = [{'gr': np.nan, 'peakmag': np.nan}]
_nan_blended = _clf_dense.probability(_nan_pt)
_nan_tree_only = np.clip(_clf_dense.score(_nan_pt), 1e-12, 1 - 1e-12)
check('event with no finite columns falls back to tree-only probability unchanged',
      np.allclose(_nan_blended, _nan_tree_only))

# Two distinct finite-column patterns in the same call.
_clf_multi = ColourClassifier(keys=['gr', 'ri'], include_peakmag=True)
_multi_sig = [{'gr': float(_rng_dense.normal(1.0, 0.3)), 'ri': float(_rng_dense.normal(0.5, 0.2)),
              'peakmag': float(_rng_dense.normal(22.0, 1.0))} for _ in range(500)]
_multi_bg = [{'gr': float(_rng_dense.normal(1.2, 0.3)), 'ri': float(_rng_dense.normal(0.6, 0.2)),
             'peakmag': float(_rng_dense.normal(22.0, 1.0))} for _ in range(500)]
_clf_multi.fit(_multi_sig, _multi_bg, np.full(500, 1e-5), np.full(500, 1e-5))
_mixed_batch = [{'gr': 1.0, 'ri': 0.5, 'peakmag': 22.0}, {'gr': 1.0, 'peakmag': 22.0}]
_mixed_result = _clf_multi.probability(_mixed_batch)
check('a batch with two distinct finite-column patterns does not crash and is finite',
      len(_mixed_result) == 2 and np.all(np.isfinite(_mixed_result)), str(_mixed_result))

# TrajectoryClassifier inherits the blend via inheritance, no extra code.
# Every one of TrajectoryClassifier's ~30 default feature columns (25
# TRAJECTORY_COLOUR_KEYS + 5 TRAJECTORY_MAGNITUDE_KEYS) gets a value for
# every event below -- an all-NaN column crashes
# HistGradientBoostingClassifier's binning step (sklearn's sliding-window
# bin-edge computation can't run on a column with zero finite entries), so
# this must not leave any column entirely NaN across the whole population.
from cmsne.trajectory import TRAJECTORY_COLOUR_KEYS, TRAJECTORY_MAGNITUDE_KEYS


def _tc_event(colour_val, mag_val):
    e = {k: colour_val for k in TRAJECTORY_COLOUR_KEYS}
    e.update({k: mag_val for k in TRAJECTORY_MAGNITUDE_KEYS})
    return e


_tc_key = TRAJECTORY_COLOUR_KEYS[0]
_tc_mag_key = TRAJECTORY_MAGNITUDE_KEYS[0]
_tc_sig_bulk = [_tc_event(float(_rng_sparse.normal(1.0, 0.3)), float(_rng_sparse.normal(22.0, 1.0)))
               for _ in range(_n_sparse)]
_tc_sig_corner = [_tc_event(float(_rng_sparse.normal(4.0, 0.1)), float(_rng_sparse.normal(16.0, 0.1)))
                  for _ in range(3)]
_tc_sig = _tc_sig_bulk + _tc_sig_corner
_tc_bg = [_tc_event(float(_rng_sparse.normal(1.2, 0.3)), float(_rng_sparse.normal(22.0, 1.0)))
         for _ in range(_n_sparse)]
_tc = TrajectoryClassifier()
_tc.fit(_tc_sig, _tc_bg, np.full(len(_tc_sig), 1e-5), np.full(_n_sparse, 1e-5))
_tc_corner_pt = [_tc_event(4.0, 16.0)]
_tc_corner_blended = _tc.probability(_tc_corner_pt)[0]
_tc_corner_tree_only = np.clip(_tc.score(_tc_corner_pt), 1e-12, 1 - 1e-12)[0]
check('TrajectoryClassifier inherits the sparse-corner blend with no extra code',
      _tc_corner_blended > _tc_corner_tree_only + 0.05,
      (_tc_corner_blended, _tc_corner_tree_only))


print('\n=== uncertainty-aware noise injection: tree-side wiring ===')

check('ColourClassifier stores n_uncertainty_replicates',
      ColourClassifier(n_uncertainty_replicates=3).n_uncertainty_replicates == 3)

# A population with real, large gr_err values trains and predicts without error.
_rng_unc = np.random.default_rng(11)
_n_unc = 300
_unc_sig = [{'gr': float(_rng_unc.normal(1.0, 0.3)), 'peakmag': float(_rng_unc.normal(22.0, 1.0)),
             'gr_err': float(abs(_rng_unc.normal(0.3, 0.1))), 'peakmag_err': 0.1}
            for _ in range(_n_unc)]
_unc_bg = [{'gr': float(_rng_unc.normal(1.3, 0.3)), 'peakmag': float(_rng_unc.normal(22.0, 1.0)),
            'gr_err': float(abs(_rng_unc.normal(0.3, 0.1))), 'peakmag_err': 0.1}
           for _ in range(_n_unc)]
_unc_w = np.full(_n_unc, 1e-5)
_clf_unc = ColourClassifier(keys=['gr'], include_peakmag=True, n_uncertainty_replicates=4)
_clf_unc.fit(_unc_sig, _unc_bg, _unc_w, _unc_w)
_unc_probe = _clf_unc.score([{'gr': 1.1, 'peakmag': 22.0, 'gr_err': 0.2}])
check('a classifier trained with real uncertainty data produces a finite, calibrated-range score',
      np.isfinite(_unc_probe[0]) and 0.0 <= _unc_probe[0] <= 1.0, _unc_probe[0])

# With NO uncertainty data anywhere (training OR query -- reusing the
# error-free _dense_sig/_dense_bg/_dense_sw/_dense_bw fixtures from the
# "wired into ColourClassifier" section above), score() is numerically
# identical to predict_proba directly: replication is a true no-op start to
# finish, not just at the query.
_clf_noerr = ColourClassifier(keys=['gr'], include_peakmag=True)
_clf_noerr.fit(_dense_sig, _dense_bg, _dense_sw, _dense_bw)
_X_direct, _ = colour_feature_matrix([{'gr': 1.1, 'peakmag': 22.0}], ['gr'], ('peakmag',))
check('score() with no *_err data anywhere matches predict_proba directly '
      '(replication is a true no-op, training and query alike)',
      np.isclose(_clf_noerr.score([{'gr': 1.1, 'peakmag': 22.0}])[0],
                 _clf_noerr.clf.predict_proba(_X_direct)[0, 1]))


print('\n=== local density-ratio blend: backward compatibility and kde_max_events ===')

# Production-shaped: wide dynamic range of rate weights (spanning several
# orders of magnitude, like real rate_weight() output), no sparse corners --
# the blend must leave probability() unchanged here.
_rng_bc = np.random.default_rng(5)
_n_bc = 2000
_bc_sig = [{'gr': float(_rng_bc.normal(1.0, 0.3)), 'peakmag': float(_rng_bc.normal(22.0, 1.0))}
          for _ in range(_n_bc)]
_bc_bg = [{'gr': float(_rng_bc.normal(1.2, 0.3)), 'peakmag': float(_rng_bc.normal(22.0, 1.0))}
         for _ in range(_n_bc)]
# Realistic order-of-magnitude spread, same distribution for both classes so
# neither is artificially starved -- this is specifically testing "dense and
# well-sampled", not sparsity.
_bc_sw = 10.0 ** _rng_bc.uniform(-7, -5, _n_bc)
_bc_bw = 10.0 ** _rng_bc.uniform(-7, -5, _n_bc)
_clf_bc = ColourClassifier(keys=['gr'], include_peakmag=True)
_clf_bc.fit(_bc_sig, _bc_bg, _bc_sw, _bc_bw)
_bc_pts = [{'gr': 1.0, 'peakmag': 22.0}, {'gr': 1.1, 'peakmag': 21.5}, {'gr': 0.9, 'peakmag': 22.5}]
_bc_blended = _clf_bc.probability(_bc_pts)
_bc_tree_only = np.clip(_clf_bc.score(_bc_pts), 1e-12, 1 - 1e-12)
check('production-shaped dense region: blended probability() unchanged from tree-only score()',
      np.allclose(_bc_blended, _bc_tree_only, atol=1e-3), (_bc_blended, _bc_tree_only))

# kde_max_events now caps each missing-data PATTERN's own subsample, lazily
# in _kde_subset_for_pattern -- not the globally stored training set, which
# is now the full (positive-weight) population (see
# docs/superpowers/specs/2026-10-26-kde-pattern-aware-subsampling-design.md).
_rng_cap = np.random.default_rng(6)
_n_cap = 500
_cap_events = [{'gr': float(_rng_cap.normal(0, 1))} for _ in range(_n_cap)]
_cap_weights = np.abs(_rng_cap.normal(1e-6, 1e-7, _n_cap))
_clf_cap = ColourClassifier(keys=['gr'], include_peakmag=False, kde_max_events=50)
_clf_cap.fit(_cap_events, _cap_events, _cap_weights, _cap_weights)
check('_kde_Xs now stores the full positive-weight population, unaffected by kde_max_events',
      _clf_cap._kde_Xs.shape[0] == np.sum(_cap_weights > 0), _clf_cap._kde_Xs.shape[0])
_subset_capped = _clf_cap._kde_subset_for_pattern((True,), np.array([0]))
check('kde_max_events caps the PER-PATTERN subsample (via _kde_subset_for_pattern) to at most the cap',
      _subset_capped.Xs.shape[0] <= 50, _subset_capped.Xs.shape[0])
check('kde_max_events preserves total rate mass after per-pattern subsampling (within 1%)',
      abs(_subset_capped.ws.sum() - _cap_weights.sum()) / _cap_weights.sum() < 0.01,
      (_subset_capped.ws.sum(), _cap_weights.sum()))

# Below the cap: no subsampling, nothing changes at the pattern level either.
_clf_nocap = ColourClassifier(keys=['gr'], include_peakmag=False, kde_max_events=10000)
_clf_nocap.fit(_cap_events, _cap_events, _cap_weights, _cap_weights)
_subset_nocap = _clf_nocap._kde_subset_for_pattern((True,), np.array([0]))
check('kde_max_events above the actual event count leaves the per-pattern subset untouched',
      _subset_nocap.Xs.shape[0] == np.sum(_cap_weights > 0), _subset_nocap.Xs.shape[0])

print('\n=== KDE pattern-aware subsampling: budget is not wasted on rows a pattern cannot use ===')
# Mimics the real bug at small scale: 950/1000 rows are missing column 1
# (e.g. peakmag never measured for most events), 50 have both columns.
# With the OLD design (subsample kde_max_events=10 from all 1000 BEFORE
# filtering), the all-finite pattern would expect only ~0.5 of those 10 to
# actually be finite in both columns -- almost certainly 0. The fix filters
# per pattern FIRST, so the all-finite pattern's own cap is reached in full,
# and a second, disjoint pattern reaches its own cap independently (patterns
# do not compete for one shared, order-dependent draw).
_rng_waste = np.random.default_rng(7)
_n_both = 50
_n_col0_only = 950
_Xs_waste = np.vstack([
    np.column_stack([_rng_waste.normal(0, 1, _n_both), _rng_waste.normal(20, 1, _n_both)]),
    np.column_stack([_rng_waste.normal(0, 1, _n_col0_only), np.full(_n_col0_only, np.nan)]),
])
_ws_waste = np.full(_n_both + _n_col0_only, 1e-6)
_clf_waste = ColourClassifier(keys=['gr'], include_peakmag=True, kde_max_events=10)
_clf_waste.columns = ['gr', 'peakmag']
_sigma_waste = np.full_like(_Xs_waste, np.nan)  # no uncertainty data -- isolates the subsampling behaviour
_clf_waste._store_kde_training_data(_Xs_waste, _ws_waste, _Xs_waste, _ws_waste, _sigma_waste, _sigma_waste)
check('_store_kde_training_data retains the FULL population (no global subsampling)',
      _clf_waste._kde_Xs.shape[0] == _n_both + _n_col0_only, _clf_waste._kde_Xs.shape[0])
_subset_both = _clf_waste._kde_subset_for_pattern((True, True), np.array([0, 1]))
check('the all-finite pattern reaches its FULL kde_max_events cap, drawn only from rows it can use '
      '(not starved by 950 unrelated-pattern rows competing for a shared budget)',
      _subset_both.Xs.shape[0] == 10, _subset_both.Xs.shape[0])
_subset_col0 = _clf_waste._kde_subset_for_pattern((True, False), np.array([0]))
check('a second, disjoint pattern independently reaches its own full cap too (patterns do not starve each other)',
      _subset_col0.Xs.shape[0] == 10, _subset_col0.Xs.shape[0])


print('\n=== adaptive KDE bandwidth: _widened_kernel_values ===')
from cmsne.classifier import _widened_kernel_values, DEFAULT_KDE_MIN_ESS
from cmsne.importance_sampling import effective_sample_size as _ess_ref

# Dense case: ess already >= min_ess at scale=1 -> no-op.
_wk_h = np.array([1.0])
_wk_dims = np.array([0])
_wk_X_dense = np.array([[0.0], [0.1], [-0.1], [0.05], [-0.05],
                        [0.2], [-0.2], [0.15], [-0.15], [0.02]])
_wk_w_dense = np.full(10, 1e-5)
_g_unwidened = _kernel_weighted_values(np.array([0.0]), _wk_X_dense, _wk_w_dense, _wk_h, _wk_dims)
_g_widened, _ess_widened = _widened_kernel_values(
    np.array([0.0]), _wk_X_dense, _wk_w_dense, _wk_h, _wk_dims, min_ess=5.0)
check('_widened_kernel_values is a no-op when min_ess is already satisfied at scale=1',
      np.allclose(_g_widened, _g_unwidened), (_g_widened, _g_unwidened))

# Sparse case: one near point dominates (ESS~1 at scale=1); several farther
# points only start contributing once the kernel widens.
_wk_X_sparse = np.array([[2.0], [10.0], [10.0], [10.0]])
_wk_w_sparse = np.full(4, 1e-5)
_g_fixed = _kernel_weighted_values(np.array([0.0]), _wk_X_sparse, _wk_w_sparse, _wk_h, _wk_dims)
_ess_fixed = _ess_ref(_g_fixed)
_g_w, _ess_w = _widened_kernel_values(
    np.array([0.0]), _wk_X_sparse, _wk_w_sparse, _wk_h, _wk_dims, min_ess=2.0)
check('_widened_kernel_values raises local ESS above the unwidened (single-point-dominated) value',
      _ess_w > _ess_fixed and _ess_fixed < 1.5, (_ess_w, _ess_fixed))
check('_widened_kernel_values reaches the requested min_ess in this scenario',
      _ess_w >= 2.0 - 1e-6, _ess_w)

# Ceiling/plateau case: only 2 points have positive weight -- no amount of
# widening can push ESS above effective_sample_size(weights) itself.
_wk_X_ceiling = np.array([[50.0], [50.1]])
_wk_w_ceiling = np.array([1e-5, 1e-5])
_ceiling = _ess_ref(_wk_w_ceiling)
_g_c, _ess_c = _widened_kernel_values(
    np.array([0.0]), _wk_X_ceiling, _wk_w_ceiling, _wk_h, _wk_dims, min_ess=100.0, max_iter=20)
check('_widened_kernel_values stops at the class ceiling rather than looping uselessly to max_iter',
      np.isclose(_ess_c, _ceiling, atol=1e-3), (_ess_c, _ceiling))

print('\n=== adaptive KDE bandwidth: reproduces and fixes the real corner bug ===')
from cmsne.classifier import _local_density_ratio

# Mimics the real bright+red corner failure: each class has one dominant
# near point at the fixed bandwidth (local ESS~1 for both), but background's
# near point carries more weight at a slightly closer distance -- so the
# UNWIDENED ratio wrongly favours background. Each class also has a larger,
# farther-out population that should correctly swing the ratio toward
# signal once the kernel widens enough to actually see it.
_bug_h = np.array([1.0])
_bug_dims = np.array([0])
_bug_Xs = np.array([[3.0]] + [[6.0]] * 30)
_bug_ws = np.full(31, 1e-5)
_bug_Xb = np.array([[2.9]] + [[30.0]] * 30)
_bug_wb = np.array([5e-5] + [1e-5] * 30)

_p_unwidened, _, _ = _local_density_ratio(
    np.array([0.0]), _bug_Xs, _bug_ws, _bug_Xb, _bug_wb, _bug_h, _bug_dims, min_kde_ess=1.0)
check('without meaningful widening, the corner-mimicking scenario wrongly favours background',
      _p_unwidened is not None and _p_unwidened < 0.3, _p_unwidened)

_p_widened, _ess_sig_check, _ess_bg_check = _local_density_ratio(
    np.array([0.0]), _bug_Xs, _bug_ws, _bug_Xb, _bug_wb, _bug_h, _bug_dims, min_kde_ess=10.0)
check('with adaptive widening (min_kde_ess=10), the same scenario correctly favours signal',
      _p_widened is not None and _p_widened > 0.5, _p_widened)

_fixed_ess_sig_expected = _ess_ref(_kernel_weighted_values(
    np.array([0.0]), _bug_Xs, _bug_ws, _bug_h, _bug_dims))
_fixed_ess_bg_expected = _ess_ref(_kernel_weighted_values(
    np.array([0.0]), _bug_Xb, _bug_wb, _bug_h, _bug_dims))
check('local_ess_sig/local_ess_bg (the blend-weight inputs) are identical regardless of min_kde_ess',
      np.isclose(_ess_sig_check, _fixed_ess_sig_expected) and
      np.isclose(_ess_bg_check, _fixed_ess_bg_expected),
      (_ess_sig_check, _fixed_ess_sig_expected, _ess_bg_check, _fixed_ess_bg_expected))

print('\n=== adaptive KDE bandwidth: cached ess_ceiling args avoid redundant recomputation ===')
# ess_ceiling_sig/ess_ceiling_bg must actually be USED when passed, not silently
# recomputed from ws/wb -- otherwise caching them at the call site (as
# ColourClassifier._kde_subset_for_pattern now does) would be a no-op.
# min_kde_ess=10.0 here would normally widen both classes (their real
# ceilings are well above 10), favouring signal as in the widened check
# above. Passing explicit ceilings below min_kde_ess must skip widening for
# both classes instead, reproducing the UNWIDENED (wrong-favouring) result.
_p_forced_unwidened, _, _ = _local_density_ratio(
    np.array([0.0]), _bug_Xs, _bug_ws, _bug_Xb, _bug_wb, _bug_h, _bug_dims,
    min_kde_ess=10.0, ess_ceiling_sig=0.0, ess_ceiling_bg=0.0)
check('an explicit ess_ceiling below min_kde_ess is honoured (skips widening) even though the '
      'true ceiling would allow it',
      _p_forced_unwidened is not None and _p_forced_unwidened < 0.3, _p_forced_unwidened)

_clf_cache = ColourClassifier(keys=['gr'], include_peakmag=False)
_clf_cache.fit(_dense_sig, _dense_bg, _dense_sw, _dense_bw)
_pattern_dense = (True,)
_dims_dense = np.array([0])
_subset = _clf_cache._kde_subset_for_pattern(_pattern_dense, _dims_dense)
check('_kde_subset_for_pattern now returns a _KDEPatternSubset with 10 fields '
      '(Xs, ws, Xb, wb, ess_ceiling_sig, ess_ceiling_bg, event_id_s, event_id_b, n_orig_s, n_orig_b)',
      len(_subset) == 10, len(_subset))
_expected_ceil_sig = _ess_ref(_subset.ws)
_expected_ceil_bg = _ess_ref(_subset.wb)
check('_kde_subset_for_pattern caches effective_sample_size(ws)/effective_sample_size(wb) correctly '
      '(no uncertainty data here, so ws/wb are unreplicated and this matches directly)',
      np.isclose(_subset.ess_ceiling_sig, _expected_ceil_sig) and
      np.isclose(_subset.ess_ceiling_bg, _expected_ceil_bg),
      (_subset.ess_ceiling_sig, _expected_ceil_sig, _subset.ess_ceiling_bg, _expected_ceil_bg))
check('_kde_subset_for_pattern caches the same object on a repeated call (no recomputation)',
      _clf_cache._kde_subset_for_pattern(_pattern_dense, _dims_dense) is _subset)

print('\n=== uncertainty-aware noise injection: KDE-side wiring ===')
# Grouped-ESS integration: _widened_kernel_values and _local_density_ratio must
# not be fooled by replicas of one real point, end to end (not just via the
# standalone _grouped_effective_sample_size helper tested in Task 2).
_gi_h = np.array([1.0])
_gi_dims = np.array([0])
_gi_X_one_point_x5 = np.array([[3.0], [3.0], [3.0], [3.0], [3.0]])  # 5 "replicas" of ONE real point
_gi_w_one_point_x5 = np.full(5, 1e-5 / 5)                           # each at weight/5
_gi_event_id = np.array([0, 0, 0, 0, 0])
_gi_g_ungrouped = _kernel_weighted_values(np.array([0.0]), _gi_X_one_point_x5, _gi_w_one_point_x5, _gi_h, _gi_dims)
check('WITHOUT the event_id/n_original fix, 5 replicas of one point look like ESS~5',
      _ess_ref(_gi_g_ungrouped) > 4.0, _ess_ref(_gi_g_ungrouped))
_, _gi_ess_grouped = _widened_kernel_values(
    np.array([0.0]), _gi_X_one_point_x5, _gi_w_one_point_x5, _gi_h, _gi_dims, min_ess=10.0,
    event_id=_gi_event_id, n_original=1)
check('_widened_kernel_values with event_id/n_original correctly reports ESS~1 for the same 5 replicas',
      _gi_ess_grouped < 1.5, _gi_ess_grouped)

# Issue #37 smoking-gun reproduction, at the _local_density_ratio level (the
# same style as PR #36's own bug-reproduction test, _bug_Xs/_bug_ws above):
# one background point sitting RIGHT AT the query with a large, unmodelled
# uncertainty (mirrors the real gr=6.16 event -- a near-detection-limit
# colour) single-handedly dominates the density estimate WITHOUT noise
# injection, exactly as observed on real data, and is correctly smeared back
# down WITH it. Signal sits far enough away (3 bandwidths) that its 20 points
# don't outweigh background's single exact-match point until that point's own
# uncertainty is taken into account.
_sg_h = np.array([1.0])
_sg_dims = np.array([0])
_sg_Xs = np.array([[3.0]] * 20)       # signal: 20 points, each 3 bandwidths from the query
_sg_ws = np.full(20, 1e-5)
_sg_Xb = np.array([[0.0]])            # background: ONE point exactly at the query
_sg_wb = np.array([1e-5])
_sg_sigma_b = np.array([[5.0]])       # ...with a large, unmodelled measurement uncertainty

_p_sg_noinj, _, _ = _local_density_ratio(np.array([0.0]), _sg_Xs, _sg_ws, _sg_Xb, _sg_wb, _sg_h, _sg_dims)
check('WITHOUT noise injection, the single exact-match high-uncertainty background point '
      'dominates the query, as observed in the real issue #37 finding',
      _p_sg_noinj is not None and _p_sg_noinj < 0.3, _p_sg_noinj)

_sg_Xb_rep, _sg_wb_rep, _sg_event_id_b = _replicate_with_uncertainty(
    _sg_Xb, _sg_sigma_b, _sg_wb, _sg_dims, n_replicates=500, seed=3)
_p_sg_inj, _, _ = _local_density_ratio(
    np.array([0.0]), _sg_Xs, _sg_ws, _sg_Xb_rep, _sg_wb_rep, _sg_h, _sg_dims,
    event_id_b=_sg_event_id_b, n_orig_b=1)
check('WITH noise injection, that point\'s own large uncertainty smears its contribution and '
      'the probability moves back toward the confident signal population',
      _p_sg_inj is not None and _p_sg_inj > _p_sg_noinj, (_p_sg_noinj, _p_sg_inj))

print('\n=== uncertainty-aware noise injection: integration ===')
from cmsne.trajectory import TrajectoryClassifier

_rank_sig = [{'gr': float(_rng_unc.normal(1.0, 0.3)), 'peakmag': float(_rng_unc.normal(22.0, 1.0)),
              'gr_err': 0.2} for _ in range(50)]
_rank_bg = [{'gr': float(_rng_unc.normal(1.5, 0.3)), 'peakmag': float(_rng_unc.normal(22.0, 1.0)),
             'gr_err': 0.2} for _ in range(50)]
_rank_w = np.full(50, 1e-5)
_clf_rank = ColourClassifier(keys=['gr'], include_peakmag=True, n_uncertainty_replicates=4)
_clf_rank.fit(_rank_sig, _rank_bg, _rank_w, _rank_w)
_rank_query = [{'gr': float(_rng_unc.normal(1.2, 0.3)), 'peakmag': 22.0, 'gr_err': 0.2} for _ in range(10)]
check('rank() orders by score() with no separate change needed',
      np.array_equal(_clf_rank.rank(_rank_query), np.argsort(_clf_rank.score(_rank_query))[::-1]))


# TrajectoryClassifier hardcodes include_peakmag=False internally and sets its
# own self.extra -- follow the SAME construction the existing "inherits the
# sparse-corner blend" test above already uses (_tc_event/TRAJECTORY_COLOUR_KEYS/
# TRAJECTORY_MAGNITUDE_KEYS/_tc_key/_rng_sparse), only adding an *_err field;
# passing keys=.../include_peakmag=... here would collide with its __init__.
_tc_unc_sig = [_tc_event(float(_rng_sparse.normal(1.0, 0.3)), float(_rng_sparse.normal(22.0, 1.0)))
              for _ in range(50)]
_tc_unc_bg = [_tc_event(float(_rng_sparse.normal(1.3, 0.3)), float(_rng_sparse.normal(22.0, 1.0)))
             for _ in range(50)]
for _e in _tc_unc_sig + _tc_unc_bg:
    _e[f'{_tc_key}_err'] = 0.2
_tc_unc_w = np.full(50, 1e-5)
_traj_clf_unc = TrajectoryClassifier(n_uncertainty_replicates=4)
_traj_clf_unc.fit(_tc_unc_sig, _tc_unc_bg, _tc_unc_w, _tc_unc_w)
_tc_unc_query = [_tc_event(1.1, 22.0)]
_tc_unc_query[0][f'{_tc_key}_err'] = 0.2
_traj_probe = _traj_clf_unc.probability(_tc_unc_query)
check('TrajectoryClassifier inherits noise injection with no extra code',
      np.all(np.isfinite(_traj_probe)) and np.all((_traj_probe >= 0) & (_traj_probe <= 1)), _traj_probe)

print('\n=== second-pass review fix: trajectory monotonicity + duplicate-observation '
      'correlation, on REAL event_trajectory output ===')
# A light curve engineered so trigger+0d and trigger+3d's +-5d windows BOTH pick the
# SAME single g@1/r@1 observation pair (window=5 heavily overlaps a 3-day grid gap,
# unlike event_multicolour's disjoint trigger/trigger+evolution_days snapshots) --
# the real-data mechanism the second-pass review flagged, not a contrived case.
# A brighter detection at day 12 (trigger+14's window) and brighter still at day 35
# (trigger+30's window) exercises genuine cumulative-best (peakmag) improvement.
_lct_dup = _LCm([1, 1, 12, 35, 35], ['g', 'r', 'g', 'g', 'r'],
                [22.0, 21.0, 20.0, 19.0, 19.5],
                [0.05, 0.08, 0.03, 0.02, 0.025])
_ft_dup = traj.event_trajectory(_lct_dup, grid=traj.TRAJECTORY_GRID, window=5.0)
check('fixture check: gr_d0 and gr_d3 really do come from the identical underlying '
      'observation pair (both value AND error match exactly), confirming this is the '
      'real mechanism, not assumed',
      _ft_dup['gr_d0'] == _ft_dup['gr_d3'] and _ft_dup['gr_d0_err'] == _ft_dup['gr_d3_err'],
      (_ft_dup['gr_d0'], _ft_dup['gr_d3'], _ft_dup['gr_d0_err'], _ft_dup['gr_d3_err']))
check('fixture check: peakmag genuinely improves (gets brighter) over the grid, so the '
      'monotonicity check below is not vacuous',
      _ft_dup['peakmag_d60'] < _ft_dup['peakmag_d14'] < _ft_dup['peakmag_d0'],
      (_ft_dup['peakmag_d0'], _ft_dup['peakmag_d14'], _ft_dup['peakmag_d60']))

_dup_X_full, _dup_cols = colour_feature_matrix([_ft_dup], TRAJECTORY_COLOUR_KEYS, TRAJECTORY_MAGNITUDE_KEYS)
_dup_sigma_full = colour_feature_sigma_matrix([_ft_dup], TRAJECTORY_COLOUR_KEYS, TRAJECTORY_MAGNITUDE_KEYS)
_dup_mono_groups = [[_dup_cols.index(k) for k in TRAJECTORY_MAGNITUDE_KEYS]]
_dup_Xrep, _, _ = _replicate_with_uncertainty(
    _dup_X_full, _dup_sigma_full, np.array([1.0]), np.arange(_dup_X_full.shape[1]), 500, seed=13,
    monotonic_groups=_dup_mono_groups)
_gr_d0_idx, _gr_d3_idx = _dup_cols.index('gr_d0'), _dup_cols.index('gr_d3')
check('on REAL event_trajectory output run through _replicate_with_uncertainty: the two '
      'columns sharing the identical underlying observation produce IDENTICAL perturbed '
      'values across every one of 500 replicas (not just similar -- exactly equal, since '
      'they share one noise draw)',
      np.array_equal(_dup_Xrep[:, _gr_d0_idx], _dup_Xrep[:, _gr_d3_idx]))
_mag_cols_idx = [_dup_cols.index(k) for k in TRAJECTORY_MAGNITUDE_KEYS]
check('on REAL event_trajectory output: every one of 500 replicas stays non-increasing '
      'across the cumulative-peakmag grid-step sequence, even though peakmag_d14/peakmag_d30 '
      'etc. were perturbed independently from DIFFERENT underlying observations',
      np.all(np.diff(_dup_Xrep[:, _mag_cols_idx], axis=1) <= 0))

# End-to-end smoke test: TrajectoryClassifier trained/queried on a population built from
# REAL event_trajectory calls (not the toy single-column _tc_event fixture), confirming
# the fixes above compose correctly through fit()/probability(), not just in isolation.
def _mk_real_traj_event(rng, g_mean, r_mean):
    # One g/r pair near each of the 5 grid steps (0, 3, 14, 30, 60d post-trigger,
    # each within window=5.0) so every gr_d*/peakmag_d* column is finite for every
    # event -- an all-NaN column crashes HistGradientBoostingClassifier's binning
    # (same gotcha the existing _tc_event-based test already works around).
    offsets = [0.0, 13.0, 31.0, 61.0]
    days, filters, mags, errs = [], [], [], []
    for k, off in enumerate(offsets):
        jitter = float(rng.uniform(0, 1.5))
        days += [off + jitter, off + jitter]
        filters += ['g', 'r']
        mags += [float(rng.normal(g_mean - 0.3 * k, 0.2)), float(rng.normal(r_mean - 0.3 * k, 0.2))]
        errs += [float(abs(rng.normal(0.1, 0.03))), float(abs(rng.normal(0.1, 0.03)))]
    return traj.event_trajectory(_LCm(days, filters, mags, errs),
                                 grid=traj.TRAJECTORY_GRID, window=5.0)
_rng_real_traj = np.random.default_rng(17)
_real_traj_sig = [_mk_real_traj_event(_rng_real_traj, 21.0, 20.0) for _ in range(60)]
_real_traj_bg = [_mk_real_traj_event(_rng_real_traj, 22.0, 20.5) for _ in range(60)]
_real_traj_w = np.full(60, 1e-5)
# Restricted to the gr-trajectory columns -- this fixture's light curves only ever
# detect g/r, so every OTHER band pair's column is entirely NaN across the whole
# population, which crashes HistGradientBoostingClassifier's binning step (same
# known gotcha as the existing "inherits the sparse-corner blend" test above).
# self.monotonic_column_groups still covers the full TRAJECTORY_MAGNITUDE_KEYS
# regardless of this `keys` restriction (set unconditionally from self.extra in
# __init__), so the monotonicity fix is still exercised end to end here too.
_gr_traj_keys = [f'gr{traj._step_suffix(d)}' for d in traj.TRAJECTORY_GRID]
_real_traj_clf = TrajectoryClassifier(keys=_gr_traj_keys, n_uncertainty_replicates=4)
_real_traj_clf.fit(_real_traj_sig, _real_traj_bg, _real_traj_w, _real_traj_w)
_real_traj_probe = _real_traj_clf.probability(_real_traj_sig[:5])
check('TrajectoryClassifier fit+probability on a population of REAL event_trajectory '
      'events (with real duplicate-observation and cumulative-peakmag structure) produces '
      'finite, in-range output end to end',
      np.all(np.isfinite(_real_traj_probe)) and np.all((_real_traj_probe >= 0) & (_real_traj_probe <= 1)),
      _real_traj_probe.tolist())

# M6: verify the monotonicity invariant survives INSIDE the actual KDE cache
# (_kde_subset_for_pattern's stored, post-replication subset), not just inferred
# from "probability() didn't crash and stayed in [0,1]".
from cmsne.classifier import _finite_pattern
_real_traj_Xq, _ = colour_feature_matrix(_real_traj_sig[:1], _real_traj_clf.keys, _real_traj_clf.extra)
_real_traj_pattern = _finite_pattern(_real_traj_Xq[0])
_real_traj_dims = np.where(np.asarray(_real_traj_pattern))[0]
_real_traj_subset = _real_traj_clf._kde_subset_for_pattern(_real_traj_pattern, _real_traj_dims)
_real_traj_mag_idx = [_real_traj_clf.columns.index(k) for k in TRAJECTORY_MAGNITUDE_KEYS]
check('inside the REAL, post-replication KDE-cached training subset, every cumulative-'
      'peakmag row (signal side) stays non-increasing across the grid-step sequence',
      np.all(np.diff(_real_traj_subset.Xs[:, _real_traj_mag_idx], axis=1) <= 0))
check('inside the REAL, post-replication KDE-cached training subset, every cumulative-'
      'peakmag row (background side) stays non-increasing across the grid-step sequence',
      np.all(np.diff(_real_traj_subset.Xb[:, _real_traj_mag_idx], axis=1) <= 0))


print('\n=== final review fix: _replicate_with_uncertainty is batch/row-order independent ===')
# Finding from the final whole-branch review: _replicate_with_uncertainty drew noise
# from one shared rng stream across the WHOLE call's perturbable rows, so a given
# row's draws depended on how many OTHER rows preceded it in the same call -- i.e.
# the same logical event (same X, same sigma) got DIFFERENT replicated values
# depending on what else shared its batch, so score()/predict() on a query event
# could change depending on what else was in the list passed alongside it.
_bi_sigma_row = np.array([[2.0]])
_bi_X_row = np.array([[5.0]])
_bi_Xr_alone, _, _ = _replicate_with_uncertainty(_bi_X_row, _bi_sigma_row, np.array([1.0]), np.array([0]), 8)

_bi_sigma_pair = np.array([[2.0], [2.0]])
_bi_X_pair = np.array([[3.0], [5.0]])  # the row of interest (5.0) now has ONE other row before it
_bi_Xr_pair, _, _bi_eid_pair = _replicate_with_uncertainty(
    _bi_X_pair, _bi_sigma_pair, np.array([1.0, 1.0]), np.array([0]), 8)
_bi_row5_in_pair = np.sort(_bi_Xr_pair[_bi_eid_pair == 1, 0])
check('the SAME logical row\'s replicated values are identical whether it is replicated '
      'alone or preceded by another row in the same call (previously depended on batch order)',
      np.allclose(np.sort(_bi_Xr_alone[:, 0]), _bi_row5_in_pair),
      (np.sort(_bi_Xr_alone[:, 0]), _bi_row5_in_pair))

# Same fix must also decorrelate independent rows: signal and background training
# rows replicated in separate calls with the same default seed must NOT receive
# the same underlying noise sequence (previously true since both calls freshly
# seeded the SAME global seed=0).
_bi_sigma_dummy = np.full((5, 1), 0.2)
_bi_X_a = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
_bi_X_b = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
_bi_Xa_rep, _, _ = _replicate_with_uncertainty(_bi_X_a, _bi_sigma_dummy, np.full(5, 1.0), np.array([0]), 8)
_bi_Xb_rep, _, _ = _replicate_with_uncertainty(_bi_X_b, _bi_sigma_dummy, np.full(5, 1.0), np.array([0]), 8)
_bi_noise_a = (_bi_Xa_rep[:, 0] - np.repeat(_bi_X_a[:, 0], 8))
_bi_noise_b = (_bi_Xb_rep[:, 0] - np.repeat(_bi_X_b[:, 0], 8))
check('two independent arrays replicated with the same default seed get DIFFERENT '
      'noise (previously identical, since both reused the same fresh seed=0 stream)',
      not np.allclose(_bi_noise_a, _bi_noise_b))

print('\n=== final review fix: replication does not trigger spurious early-stopping ===')
# Finding: HistGradientBoostingClassifier's early_stopping='auto' (the default)
# engages once n_samples > 10,000. Replication multiplies the row count the
# tree actually trains on by n_uncertainty_replicates, so a population whose
# ORIGINAL size would never have triggered auto early-stopping can cross that
# threshold once replicated -- a replication-induced behaviour change, not
# just added noise. Worse, HGB's internal train/validation split can then put
# near-duplicate replicas of the SAME event on both sides, so validation loss
# looks artificially good. early_stopping must be decided from the ORIGINAL
# (pre-replication) sample count, matching what sklearn's own 'auto' would
# have decided with no replication in play.
_es_rng = np.random.default_rng(31)
_n_es = 700  # 700 signal + 700 background = 1400 original rows, well under 10,000 alone
_es_sig = [{'gr': float(_es_rng.normal(1.0, 0.3)), 'peakmag': float(_es_rng.normal(22.0, 1.0)),
            'gr_err': 0.2} for _ in range(_n_es)]
_es_bg = [{'gr': float(_es_rng.normal(1.3, 0.3)), 'peakmag': float(_es_rng.normal(22.0, 1.0)),
           'gr_err': 0.2} for _ in range(_n_es)]
_es_w = np.full(_n_es, 1e-5)
# n_uncertainty_replicates=8 (default) x 1400 = 11,200 > 10,000 -- sklearn's own
# early_stopping='auto' heuristic only sees this REPLICATED count and would enable
# it, even though the "samples" it would validate against are near-duplicate
# (noise-perturbed) copies of only 1,400 distinct original events -- siblings can
# land on both sides of its internal train/validation split, so validation loss
# doesn't hold anything genuinely out. Test the DECISION directly (not n_iter_,
# which this exact leakage can make an unreliable proxy either direction).
_es_clf = ColourClassifier(keys=['gr'], include_peakmag=True, calibrate=False)
_es_clf.fit(_es_sig, _es_bg, _es_w, _es_w)
check('early_stopping is explicitly disabled when noise-injection replication actually '
      'occurred, overriding sklearn\'s own auto-heuristic which only sees the inflated '
      'REPLICATED sample count',
      _es_clf.clf.get_params()['early_stopping'] is False)

# No-op case (no uncertainty data anywhere) must be entirely unchanged: early_stopping
# stays at sklearn's own default exactly as before this fix.
_es_clf_noerr = ColourClassifier(keys=['gr'], include_peakmag=True, calibrate=False)
_es_clf_noerr.fit(_dense_sig, _dense_bg, _dense_sw, _dense_bw)
check('early_stopping is left at sklearn\'s own default when no replication occurred '
      '(true no-op, unaffected by this fix)',
      _es_clf_noerr.clf.get_params()['early_stopping'] == 'auto')

# An explicit user override of early_stopping must always be respected, whether or
# not replication occurs.
_es_clf_override = ColourClassifier(keys=['gr'], include_peakmag=True, calibrate=False,
                                    early_stopping=True)
_es_clf_override.fit(_es_sig, _es_bg, _es_w, _es_w)
check('an explicit early_stopping=True passed by the caller is never overridden by this fix',
      _es_clf_override.clf.get_params()['early_stopping'] is True)

print('\n=== second-pass review fix: early_stopping decision does not depend on whether '
      'replication happened, only on ORIGINAL row count ===')
# Finding: the FIRST fix decided early_stopping purely from "did replication occur",
# always forcing False when it did. On real BlueBEAR production data the ORIGINAL
# population already exceeds 10,000 rows (sklearn's own 'auto' threshold), so this
# made n_uncertainty_replicates=1 (no-op, early_stopping stays 'auto' -> effectively
# True) and n_uncertainty_replicates=8 (forced False) train under DIFFERENT
# early_stopping regimes purely as an accident of replication, confounding the
# Manual Validation sweep's n_uncertainty_replicates axis with an unrelated
# early-stopping/capacity change, not noise injection alone. (The fix landed here --
# always False, with max_iter pinned to a reference fit -- was itself superseded by
# the third-pass fix further below, once matching the FLAG was found not to fix the
# underlying leak; this section's own check now asserts the CURRENT, final
# behaviour, not the intermediate one.)
_eso_rng = np.random.default_rng(32)
_n_eso = 6000  # 6000 signal + 6000 background = 12000 original rows, ABOVE 10,000 alone
_eso_sig = [{'gr': float(_eso_rng.normal(1.0, 0.3)), 'peakmag': float(_eso_rng.normal(22.0, 1.0)),
             'gr_err': 0.2} for _ in range(_n_eso)]
_eso_bg = [{'gr': float(_eso_rng.normal(1.3, 0.3)), 'peakmag': float(_eso_rng.normal(22.0, 1.0)),
            'gr_err': 0.2} for _ in range(_n_eso)]
_eso_w = np.full(_n_eso, 1e-5)
_eso_clf_rep = ColourClassifier(keys=['gr'], include_peakmag=True, calibrate=False,
                                n_uncertainty_replicates=8)
_eso_clf_rep.fit(_eso_sig, _eso_bg, _eso_w, _eso_w)
# Superseded by the third-pass fix below: matching the FLAG alone (early_stopping=True
# for both K=1 and K=8) turned out not to fix the real leak (HGB's internal validation
# split has no concept of replica siblings), so the implementation now always disables
# early_stopping and pins max_iter to a reference fit's discovered value whenever
# replication occurs with no explicit override -- never leaves it as True itself.
check('early_stopping is explicitly False (with max_iter pinned, not sklearn\'s own '
      'decision) once replication is active and the caller has not set it themselves, '
      'regardless of whether the ORIGINAL row count was above or below 10,000',
      _eso_clf_rep.clf.get_params()['early_stopping'] is False,
      _eso_clf_rep.clf.get_params()['early_stopping'])

check('min_samples_leaf is scaled by n_uncertainty_replicates when replication actually '
      'occurs, so a "leaf" still requires ~20 REAL distinct events\' worth of replica rows '
      '(sklearn\'s own default of 20 would otherwise count replica rows, making a leaf only '
      '20/K real events -- a tree-capacity increase stacking on top of noise smoothing, not '
      'noise injection alone)',
      _eso_clf_rep.clf.get_params()['min_samples_leaf'] == 20 * 8,
      _eso_clf_rep.clf.get_params()['min_samples_leaf'])

_eso_clf_noinj = ColourClassifier(keys=['gr'], include_peakmag=True, calibrate=False,
                                  n_uncertainty_replicates=1)
_eso_clf_noinj.fit(_eso_sig, _eso_bg, _eso_w, _eso_w)
check('min_samples_leaf is left at sklearn\'s own default (20) when no replication '
      'occurs (n_uncertainty_replicates=1, true no-op)',
      _eso_clf_noinj.clf.get_params()['min_samples_leaf'] == 20,
      _eso_clf_noinj.clf.get_params()['min_samples_leaf'])

print('\n=== third-pass review fix: K=1 and K=8 get IDENTICAL effective n_iter_, not just '
      'matching early_stopping flags ===')
# Finding: matching the early_stopping FLAG (the fix above) does not fix the leak --
# HistGradientBoostingClassifier's own internal validation split has no concept of
# "replica siblings", so at K=8 near-duplicate replicas can land on BOTH sides of
# that split, letting validation loss track training loss almost exactly and never
# triggering a stop -- n_iter_ silently diverges toward max_iter even with
# early_stopping=True on both sides, while K=1 (no replication, no such leakage)
# genuinely stops early. Fixed by precomputing n_iter_ from a one-time reference fit
# on the ORIGINAL data, then pinning the replicated fit's max_iter to that exact
# value with early_stopping explicitly off.
# Fourth-pass review finding (C1): an earlier version of this fixture (gr means
# 4 stds apart) was perfectly separable enough that validation loss plateaus
# IDENTICALLY whether or not replica siblings leak across HGB's internal split --
# round-2's own buggy (flag-only) code ALSO landed on the same n_iter_ there,
# making that test unable to tell fixed from broken (verified directly: the
# old flag-only logic, reconstructed on that fixture's own replicated data,
# gave 43 -- the SAME as this fix's 43). This fixture (means only 0.3 apart,
# l2_regularization=0 -- the default 1.0 keeps the loss improving past
# max_iter=300 regardless of separation) is weakly-enough separable that the
# leak DOES change the outcome. Rather than trust one specific seed/gap to stay
# discriminating forever, the test PROVES it discriminates every run: it
# reconstructs the OLD (flag-only, pre-this-fix) logic directly on the SAME
# replicated data and asserts it actually diverges from K=1's own n_iter_,
# before checking that the ACTUAL (fixed) code does NOT diverge.
_niter_rng = np.random.default_rng(1)
_n_niter = 6000
_niter_sig = [{'gr': float(_niter_rng.normal(1.0, 0.3)), 'gr_err': 0.05} for _ in range(_n_niter)]
_niter_bg = [{'gr': float(_niter_rng.normal(1.3, 0.3)), 'gr_err': 0.05} for _ in range(_n_niter)]
_niter_w = np.full(_n_niter, 1e-5)
_niter_clf_k1 = ColourClassifier(keys=['gr'], include_peakmag=False, calibrate=False,
                                 n_uncertainty_replicates=1, l2_regularization=0.0)
_niter_clf_k1.fit(_niter_sig, _niter_bg, _niter_w, _niter_w)
_niter_clf_k8 = ColourClassifier(keys=['gr'], include_peakmag=False, calibrate=False,
                                 n_uncertainty_replicates=8, l2_regularization=0.0)
_niter_clf_k8.fit(_niter_sig, _niter_bg, _niter_w, _niter_w)
check('fixture check: early stopping genuinely fires for K=1 on this fixture '
      '(n_iter_ well below max_iter=300), so the comparisons below are not vacuous',
      _niter_clf_k1.clf.n_iter_ < 300, _niter_clf_k1.clf.n_iter_)

from sklearn.ensemble import HistGradientBoostingClassifier as _HGBCForTest
_niter_Xs, _ = colour_feature_matrix(_niter_sig, ['gr'], ())
_niter_Xb, _ = colour_feature_matrix(_niter_bg, ['gr'], ())
_niter_sigma_s = colour_feature_sigma_matrix(_niter_sig, ['gr'], ())
_niter_sigma_b = colour_feature_sigma_matrix(_niter_bg, ['gr'], ())
_niter_dims = np.arange(1)
_niter_Xs_rep, _niter_ws_rep, _ = _replicate_with_uncertainty(
    _niter_Xs, _niter_sigma_s, _niter_w, _niter_dims, 8)
_niter_Xb_rep, _niter_wb_rep, _ = _replicate_with_uncertainty(
    _niter_Xb, _niter_sigma_b, _niter_w, _niter_dims, 8)
_niter_X_old = np.vstack([_niter_Xs_rep, _niter_Xb_rep])
_niter_y_old = np.r_[np.ones(len(_niter_Xs_rep)), np.zeros(len(_niter_Xb_rep))]
_niter_w_old = np.r_[_niter_ws_rep, _niter_wb_rep]
_niter_old_style = _HGBCForTest(max_depth=4, learning_rate=0.1, max_iter=300, l2_regularization=0.0,
                                min_samples_leaf=160, random_state=0, early_stopping=True).fit(
    _niter_X_old, _niter_y_old, sample_weight=_niter_w_old)
check('fixture check: this fixture genuinely DISCRIMINATES -- the OLD (flag-only, '
      'round-2) logic reconstructed on the SAME replicated data actually diverges '
      'from K=1\'s own n_iter_, proving the leak this fix addresses is real for this '
      'fixture (not a case where the bug and the fix coincidentally agree)',
      _niter_old_style.n_iter_ != _niter_clf_k1.clf.n_iter_,
      (_niter_clf_k1.clf.n_iter_, _niter_old_style.n_iter_))
check('with the ACTUAL fix: K=1 (no-op, sklearn\'s own \'auto\' on the original data) '
      'and K=8 (replicated, pinned to a reference fit\'s discovered n_iter_) train for '
      'the EXACT SAME number of effective boosting iterations -- K=8 does NOT diverge '
      'the way the reconstructed OLD logic (checked above) does on this SAME fixture',
      _niter_clf_k1.clf.n_iter_ == _niter_clf_k8.clf.n_iter_,
      (_niter_clf_k1.clf.n_iter_, _niter_clf_k8.clf.n_iter_))

# calibrate=True (the actual default): CalibratedClassifierCV's own cv=3 folds each
# train on a different ~2/3 subset, so K=1 itself has NO single canonical n_iter_ to
# match in the first place -- its own 3 folds naturally land on 3 different values.
# The exact-match guarantee above is therefore specific to calibrate=False; under
# calibrate=True it reduces to a documented, weaker one (see the KNOWN LIMITATION
# comment in fit()). What IS cleanly, deterministically true regardless -- and
# tested here -- is that this fix's own pinning is internally CONSISTENT: every one
# of K>1's CalibratedClassifierCV folds gets the SAME n_iter_ (all pinned to the one
# reference value), not three different ones the way K=1's folds naturally do.
_niter_cal_rng = np.random.default_rng(1)
_n_niter_cal = 15000  # 15000+15000=30000 total, comfortably above both the 10k and
                      # 15k thresholds the calibrate=True KNOWN LIMITATION concerns
_niter_cal_sig = [{'gr': float(_niter_cal_rng.normal(1.0, 0.3)), 'gr_err': 0.05}
                  for _ in range(_n_niter_cal)]
_niter_cal_bg = [{'gr': float(_niter_cal_rng.normal(1.3, 0.3)), 'gr_err': 0.05}
                 for _ in range(_n_niter_cal)]
_niter_cal_w = np.full(_n_niter_cal, 1e-5)
_niter_cal_clf = ColourClassifier(keys=['gr'], include_peakmag=False, calibrate=True,
                                  n_uncertainty_replicates=8, l2_regularization=0.0)
_niter_cal_clf.fit(_niter_cal_sig, _niter_cal_bg, _niter_cal_w, _niter_cal_w)
_niter_cal_fold_iters = [cc.estimator.n_iter_ for cc in _niter_cal_clf.clf.calibrated_classifiers_]
check('fixture check: early stopping genuinely fires under calibrate=True too (fold '
      'n_iter_ well below max_iter=300), so the consistency check below could not '
      'pass merely because every fold trivially hit the same ceiling',
      _niter_cal_fold_iters[0] < 300, _niter_cal_fold_iters)
check('under calibrate=True (the actual default) at production-like scale, every '
      'CalibratedClassifierCV fold is pinned to the IDENTICAL n_iter_ (internal '
      'consistency of this fix\'s own pinning, regardless of the documented K=1-vs-'
      'K>1 fold-variance limitation above)',
      len(set(_niter_cal_fold_iters)) == 1, _niter_cal_fold_iters)

_m2_clf = ColourClassifier(keys=['gr'], include_peakmag=True, calibrate=False,
                           n_uncertainty_replicates=8, early_stopping=True)
_m2_clf.fit(_eso_sig, _eso_bg, _eso_w, _eso_w)
check('min_samples_leaf is still scaled by K even when the caller explicitly sets '
      'early_stopping themselves (previously nested inside the same guard, so an '
      'explicit early_stopping override silently lost the min_samples_leaf scaling too)',
      _m2_clf.clf.get_params()['min_samples_leaf'] == 20 * 8,
      _m2_clf.clf.get_params()['min_samples_leaf'])
check('an explicit early_stopping=True from the caller is still never overridden by '
      'the reference-fit mechanism',
      _m2_clf.clf.get_params()['early_stopping'] is True)

print('\n=== final review fix: threshold calibration matches what predict() actually compares against ===')
# Finding: self.threshold was calibrated from raw self.clf.predict_proba(Xb), but
# predict()/recovery_rate()/roc_curve() all compare it against self.score(), which
# Task 3 redefined to average over each query event's OWN replicas -- a different
# quantity whenever real uncertainty data is present. Tested directly against the
# implementation (self.threshold must reproduce weighted_quantile(self.score(...)))
# rather than via the realized FPR, which this project's isotonic calibration can
# make a flaky/confounded statistic (large plateaus of tied calibrated values mean
# weighted_quantile's exact-quantile convention doesn't land cleanly on target_fpr
# even when calibrated against the CORRECT quantity -- a separate, pre-existing
# property of CalibratedClassifierCV + weighted_quantile, not something this plan
# introduced or is in scope to fix).
_th_rng = np.random.default_rng(3)
_n_th = 1500
_th_sig = [{'gr': float(_th_rng.normal(1.0, 0.3)), 'peakmag': float(_th_rng.normal(22.0, 1.0)),
            'gr_err': 0.3} for _ in range(_n_th)]
_th_bg = [{'gr': float(_th_rng.normal(1.3, 0.3)), 'peakmag': float(_th_rng.normal(22.0, 1.0)),
           'gr_err': 0.3} for _ in range(_n_th)]
_th_w = np.full(_n_th, 1e-5)
_th_clf = ColourClassifier(keys=['gr'], include_peakmag=True, target_fpr=0.10)
_th_clf.fit(_th_sig, _th_bg, _th_w, _th_w)
_th_expected_threshold = cm.weighted_quantile(_th_clf.score(_th_bg), _th_w, q=1.0 - _th_clf.target_fpr)
check('self.threshold is computed from self.score(background_events) -- the SAME '
      'quantity predict()/recovery_rate()/roc_curve() apply it against -- not from '
      'raw predict_proba, a different quantity once real uncertainty data makes '
      'score() average over replicas',
      np.isclose(_th_clf.threshold, _th_expected_threshold),
      (_th_clf.threshold, _th_expected_threshold))



print('\n=== band-matched CMD features: per-band keys, resolving, and classifiers ===')
from cmsne.trajectory import matching_band_for_pair, resolve_magnitude_key, trajectory_band_magnitude_keys
from cmsne.lsst import bands as LSST_BANDS

# 1. matching_band_for_pair
check('matching_band_for_pair pairs adjacent colours with the redder band',
      matching_band_for_pair('gr') == 'r' and
      matching_band_for_pair('ri') == 'i' and
      matching_band_for_pair('iz') == 'z' and
      matching_band_for_pair('zy') == 'y' and
      matching_band_for_pair('ug') == 'g')

# 2. trajectory_band_magnitude_keys
_r_keys = trajectory_band_magnitude_keys('r')
check('trajectory_band_magnitude_keys generates per-step peakmag_r keys',
      _r_keys == ('peakmag_r_d0', 'peakmag_r_d3', 'peakmag_r_d14', 'peakmag_r_d30', 'peakmag_r_d60'))

# 3. resolve_magnitude_key
_k_m, _lbl_m = resolve_magnitude_key('gr', 0.0, mag_mode='matched')
check('resolve_magnitude_key matched mode returns peakmag_{b}_d0', _k_m == 'peakmag_r_d0', _k_m)
_k_s, _lbl_s = resolve_magnitude_key('gr', 14.0, mag_mode='snapshot')
check('resolve_magnitude_key snapshot mode returns {b}_d14', _k_s == 'r_d14', _k_s)
_k_p, _lbl_p = resolve_magnitude_key('gr', 14.0, mag_mode='peakmag')
check('resolve_magnitude_key peakmag mode returns peakmag_d14', _k_p == 'peakmag_d14', _k_p)
_k_custom, _ = resolve_magnitude_key('gr', 14.0, mag_mode='matched', mag_band='i')
check('resolve_magnitude_key honours explicit mag_band override', _k_custom == 'peakmag_i_d14', _k_custom)
_k_fb, _ = resolve_magnitude_key('gr', 0.0, mag_mode='matched', sample_event={'peakmag_d0': 20.0})
check('resolve_magnitude_key falls back to peakmag_d0 when per-band key is absent in sample_event',
      _k_fb == 'peakmag_d0', _k_fb)

# 4. event_multicolour per-band features
_f_bm = mcol.event_multicolour(_lc)
check('event_multicolour includes per-band trigger and evolution magnitudes',
      all(b[-1] in _f_bm and f"{b[-1]}_err" in _f_bm and f"{b[-1]}_d" in _f_bm and f"{b[-1]}_d_err" in _f_bm
          for b in LSST_BANDS))
check('event_multicolour per-band values match detected observations',
      np.isclose(_f_bm['g'], 22.0) and np.isclose(_f_bm['g_err'], 0.05) and
      np.isclose(_f_bm['r'], 21.5) and np.isclose(_f_bm['r_err'], 0.08) and
      np.isnan(_f_bm['i']) and np.isnan(_f_bm['i_err']))

# 5. event_trajectory per-band features and errors
_ft_bm = traj.event_trajectory(_lct, grid=(0.0, 20.0), window=5.0)
check('event_trajectory includes per-band snapshot and cumulative-best keys',
      np.isclose(_ft_bm['g_d0'], 22.0) and np.isclose(_ft_bm['g_d0_err'], 0.05) and
      np.isclose(_ft_bm['r_d0'], 21.5) and np.isclose(_ft_bm['r_d0_err'], 0.08) and
      np.isclose(_ft_bm['peakmag_g_d0'], 22.0) and np.isclose(_ft_bm['peakmag_g_d20'], 20.0) and
      np.isclose(_ft_bm['peakmag_g_d20_err'], 0.03) and
      np.isclose(_ft_bm['peakmag_r_d0'], 21.5) and np.isclose(_ft_bm['peakmag_r_d20'], 21.0) and
      np.isclose(_ft_bm['peakmag_r_d20_err'], 0.06))

# 6. reveal_up_to masks future per-band keys
_revealed_bm = traj.reveal_up_to(_ft_bm, 0, grid=(0.0, 20.0))
check('reveal_up_to NaNs out future per-band values and errors beyond step k',
      np.isnan(_revealed_bm['r_d20']) and np.isnan(_revealed_bm['r_d20_err']) and
      np.isnan(_revealed_bm['peakmag_r_d20']) and np.isnan(_revealed_bm['peakmag_r_d20_err']) and
      np.isfinite(_revealed_bm['r_d0']) and np.isfinite(_revealed_bm['peakmag_r_d0']))

# 7. ColourClassifier accepts band-matched magnitude features
_clf_r = ColourClassifier(keys=['gr'], magnitude_feature='r')
check('ColourClassifier accepts single-band magnitude feature',
      _clf_r.magnitude_feature == 'r' and _clf_r.extra == ('r',))
_clf_pk_r = ColourClassifier(keys=['gr'], magnitude_feature='peakmag_r')
check('ColourClassifier accepts band cumulative peakmag feature',
      _clf_pk_r.magnitude_feature == 'peakmag_r' and _clf_pk_r.extra == ('peakmag_r',))

# 8. evolving_contours supports matched magnitude_mode
_sig_c_bm = [{'gr_d0': float(_rngc.normal(1.0, 0.3)), 'peakmag_r_d0': float(_rngc.normal(21.0, 1)),
              'gr_d14': float(_rngc.normal(1.0, 0.3)), 'peakmag_r_d14': float(_rngc.normal(20.5, 1))}
             for _ in range(200)]
_bg_c_bm = [{'gr_d0': float(_rngc.normal(0.0, 0.3)), 'peakmag_r_d0': float(_rngc.normal(22.5, 1)),
             'gr_d14': float(_rngc.normal(0.0, 0.3)), 'peakmag_r_d14': float(_rngc.normal(22.0, 1))}
            for _ in range(400)]
_surfaces_bm = ctr.evolving_contours(_sig_c_bm, _bg_c_bm, 'gr', grid=(0.0, 14.0), n=10, magnitude_mode='matched')
check('evolving_contours with magnitude_mode="matched" returns valid probability surfaces',
      set(_surfaces_bm) == {0.0, 14.0} and _surfaces_bm[0.0][2].shape == (10, 10) and
      np.all((_surfaces_bm[0.0][2] >= 0) & (_surfaces_bm[0.0][2] <= 1)))


print(f'\n===== {len(PASS)} passed, {len(FAIL)} failed =====')
if FAIL:
    print('FAILED: ' + ', '.join(FAIL)); sys.exit(1)

print('\n=== absolute rates framework (LTT adaptation) ===')
from cmsne.rates import (
    MadauDickinsonSFRD, CustomRateDistribution, LensingOpticalDepth,
    RateCalculator, DetectionMatrix, check_images_resolved,
    sn_ia_rate_distribution, sn_cc_rate_distribution,
    sn_ia_rate_calculator, sn_cc_rate_calculator,
    absolute_rate_weight, LSST_SURVEY_AREA_DEG2, LSST_SURVEY_SOLID_ANGLE
)

# Test SFRD
sfrd = MadauDickinsonSFRD(local_rate=2.4e-5, alpha=2.7, beta=2.9, z_pivot=1.9)
check('MadauDickinsonSFRD local rate matches R0', np.isclose(sfrd.rate_density(0.0), 2.4e-5))
check('MadauDickinsonSFRD rate increases towards peak z', sfrd.rate_density(1.5) > sfrd.rate_density(0.0))

# Test Optical depth
check('LensingOpticalDepth dtau_dmu_dz scales as mu^-3',
      np.isclose(LensingOpticalDepth.dtau_dmu_dz(1.0, 10.0) / LensingOpticalDepth.dtau_dmu_dz(1.0, 20.0), 8.0))
tau_1 = LensingOpticalDepth.tau(1.0, mu_min=2.0, mu_max=50.0)
check('LensingOpticalDepth.tau is finite and positive', np.isfinite(tau_1) and tau_1 > 0, tau_1)

# Test RateCalculator
calc_ia = sn_ia_rate_calculator()
calc_cc = sn_cc_rate_calculator()
n_occ_ia = calc_ia.sources_in_redshift_bin(0.01, 1.5)
n_occ_cc = calc_cc.sources_in_redshift_bin(0.01, 1.5)
check('expected occurring SN Ia in LSST footprint ~ 1e7/yr', 5e6 < n_occ_ia < 2e7, round(n_occ_ia, 1))
check('expected occurring CC in LSST footprint ~ 3e7/yr', 1e7 < n_occ_cc < 5e7, round(n_occ_cc, 1))

# Test absolute_rate_weight
w_bare_ia = absolute_rate_weight('uIa', 0.5)
w_mc_ia = absolute_rate_weight('uIa', 0.5, z_range=(0.0, 1.5), n_attempted=100000)
check('absolute_rate_weight bare is source_rate_density',
      np.isclose(w_bare_ia, calc_ia.source_rate_density(0.5)))
check('absolute_rate_weight MC scales by delta_z / N',
      np.isclose(w_mc_ia, w_bare_ia * 1.5 / 100000))

# Lensed Ia weight
w_mc_sig = absolute_rate_weight('sig', 0.5, magnification=10.0,
                                z_range=(0.0, 1.5), magnification_range=(2.0, 50.0),
                                n_attempted=100000)
expected_sig = calc_ia.source_rate_density(0.5) * LensingOpticalDepth.dtau_dmu_dz(0.5, 10.0) * (1.5 * 48.0 / 100000)
check('absolute_rate_weight MC lensed scales by delta_z * delta_mu / N',
      np.isclose(w_mc_sig, expected_sig))

# Check resolution helper
check('check_images_resolved respects seeing',
      check_images_resolved(1.2, 0.7) and not check_images_resolved(0.5, 0.7))
