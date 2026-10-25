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

print('\n=== integration: core-collapse weights (SFR-tracking rate) ===')
check('cc_ul_weight finite positive', np.isfinite(cm.cc_ul_weight(0.5)) and cm.cc_ul_weight(0.5) > 0)
lw_cc = cm.cc_l_weight(0.5, 10.0)
import astropy.units as _u, astropy.cosmology.units as _cu
from astropy.cosmology import Planck18 as _P18
_d = (0.5 * _cu.redshift).to(_u.pc, _cu.redshift_distance(_P18, kind='comoving')).value
check('cc_l_weight uses the 31 Gpc scale',
      np.isclose(lw_cc, cm.cc_ul_weight(0.5) * 0.5 * (_d / 31e9) ** 3 * 10.0 ** -2, rtol=1e-12))

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
    def __init__(s, d, f, m):
        s.obs_days = np.array(d, float); s.obs_filters = np.array(f); s.obs_mag = np.array(m, float)
# g@2, r@3 detected; r@4 fainter; i@20 below depth (inf). Trigger = first detection (day 2).
_lc = _LCm([2, 3, 20, 4], ['g', 'r', 'i', 'r'], [22.0, 21.5, np.inf, 21.7])
_bp = mcol.band_photometry(_lc, 2.0, window=5.0)
check('band_photometry picks nearest detection', np.isclose(_bp['lsstg'], 22.0) and np.isclose(_bp['lsstr'], 21.5),
      (_bp['lsstg'], _bp['lsstr']))
check('band_photometry NaN for undetected band', np.isnan(_bp['lssti']) and np.isnan(_bp['lsstu']))
_f = mcol.event_multicolour(_lc)
check('trigger epoch is first detection', _f['trigger_day'] == 2.0, _f['trigger_day'])
check('colour = band mag difference', np.isclose(_f['gr'], 0.5), _f['gr'])
check('colour NaN when a band is outside the window', np.isnan(_f['ri']))
check('peakmag is brightest detection', np.isclose(_f['peakmag'], 21.5), _f['peakmag'])
check('peakmag_firstdet is the earliest detection', np.isclose(_f['peakmag_firstdet'], 22.0), _f['peakmag_firstdet'])
check('peakmag_fit falls back to brightest when <3 band points',
      np.isclose(_f['peakmag_fit'], 21.5), _f['peakmag_fit'])
check('n_bands counts bands at trigger', _f['n_bands'] == 2, _f['n_bands'])
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
_lct = _LCm([2, 3, 20, 23], ['g', 'r', 'g', 'r'], [22.0, 21.5, 20.0, 21.0])
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
                 cm.cc_ul_weight(0.3) * CC_FRACTIONS['nugent-sn2p'] * CC_TO_IA_RATE))
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
# include a point exactly at the max of each axis, to check the inclusive top bin
_dense_events.append({'gr': 4.0, 'peakmag': 24.0})
_dense_w = np.r_[_dense_w, 1.0]

_flagged = isamp.diagnose_sparse_cells(_dense_events, _dense_w, band_pair_keys=['gr'],
                                       magnitude_key='peakmag', n_bins=4, ess_floor_fraction=0.2)
check('diagnose_sparse_cells flags the deliberately sparse cell',
      any(np.isclose(f['colour_range'][0], 3.0, atol=0.05) and np.isclose(f['magnitude_range'][0], 23.0, atol=0.05)
          for f in _flagged), _flagged)
check('diagnose_sparse_cells does not flag the densely-filled cells',
      not any(np.isclose(f['colour_range'][0], 0.0, atol=0.05) and np.isclose(f['magnitude_range'][0], 20.0, atol=0.05)
              for f in _flagged), _flagged)
check('every flagged cell reports its ESS below its floor',
      all(f['ess'] < f['floor'] for f in _flagged), _flagged)

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


print(f'\n===== {len(PASS)} passed, {len(FAIL)} failed =====')
if FAIL:
    print('FAILED: ' + ', '.join(FAIL)); sys.exit(1)
