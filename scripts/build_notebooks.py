"""Regenerate notebooks/04_colour_magnitude.ipynb and notebooks/05_time_delays.ipynb.

These two notebooks are GENERATED. Edit this script and re-run it rather than
editing the .ipynb files directly, or the two will silently diverge:

    python scripts/build_notebooks.py

Notebooks 01-03 are hand-maintained and untouched by this script.
"""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


SETUP = '''
# If running in Colab, install the dependencies (uncomment):
# !pip install sncosmo
# !pip install git+https://github.com/LSSTDESC/OpSimSummaryV2.git

# Make the `cmsne` package importable when this notebook lives in notebooks/.
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
'''

# ---------------------------------------------------------------- notebook 04
nb04 = notebook([
    md("""
# 04 - Colour-magnitude diagrams

Builds a colour-magnitude diagram for every LSST band pair, showing:

* a dense **SN Ia** background plus four **core-collapse contaminant** populations
  (Ibc, IIL, IIn, IIP), each drawn twice - unlensed (black-edged) and lensed;
* the **logistic decision boundary** separating lensed from unlensed events;
* lensed events sampled through **real LSST cadence realisations**, coloured by the
  rest-frame epoch at which the colour method could identify them.

All populations are resampled with rate weights, so what is plotted is a fair draw
from the expected observed population rather than from the raw simulation.

Uses `cmsne.supernovae`, `cmsne.colour_magnitude` and `cmsne.lsst`.
"""),
    code(SETUP + '''
from cmsne.lsst import band_pairs, contaminant_info
from cmsne.supernovae import Supernovae, Supernovae2, Nugent
from cmsne.colour_magnitude import (full_creation, combine_populations,
                                    exponential_regression, success_rate)
'''),
    md("""
## Sample sizes

Sized for an overnight run: roughly **4.2 hours** on measured per-event costs of
4.5 ms (cadence path), 15.5 ms (SN Ia background) and 14.1 ms (per contaminant
class), across 10 band pairs and 4 contaminant classes.

Cadence-path acceptance is **7-14%**. It was ~35% higher before observations were
phased against a real `t0`: supernovae exploding in a cadence gap, or too late in
the survey, are now genuinely missed rather than having their time axis shifted so
the first visit became first light. The per-event cost roughly halved at the same
time, because the visibility window is far narrower than the old three-year cut.

Redshift is **not** capped globally. Each generator defaults to
`redshift_limits(...)`, the exact range over which its model covers that band pair,
so the redder pairs keep their full reach (`lsstz-lssty` runs to z = 3.01 where
`lsstg-lsstr` stops at 0.93) while nothing is wasted above the limit. That change
alone lifted cadence-path acceptance from ~2%, and took the two fast generators
to 100%.

Memory: the fast generators hold a 1000-point light curve per event, so
`N_BACKGROUND` costs ~62 KB/event and each contaminant class ~31 KB/event - about
1.7 GB live per band pair, released between pairs.
"""),
    code('''
N_BACKGROUND  = 10000   # SN Ia events per band pair (model-only, fast)
N_CONTAMINANT = 8000    # events per contaminant class (model-only, fast)
N_CADENCE     = 200000  # events through a real OpSim cadence (slow, the scarce one)
N_DRAW        = 600     # rate-weighted draws per population, for the fit and the plot
PLOT_MAX_CADENCE = 2500 # cap on cadence points *drawn* per panel (stats use all)

Generator  = Supernovae()    # cadence realisations, incl. delayed second image
Generator2 = Supernovae2()   # SN Ia background
Generator3 = Nugent()        # core-collapse contaminants
'''),
    md("""
## Generate the populations

One pass per band pair. Each band pair is `(bluer, redder)`; the redder band is the
magnitude axis and the colour is `bluer - redder`.
"""),
    code('''
import time
from cmsne.supernovae import redshift_limits

plot_data_per_band = []

# Collected across all band pairs, for the shared colourbar and for notebook 05.
all_l_times, all_time_delays, all_timdel = [], [], []
all_redshifts, all_magnifications, all_weights_l = [], [], []

run_start = time.time()

for pair_index, pair in enumerate(band_pairs, start=1):
    pair_start = time.time()
    blue_band, red_band = pair
    band1, band2 = red_band, blue_band   # band1 = magnitude axis, band2 = colour's blue side

    z_lo, z_hi = redshift_limits('salt3', band1, band2)
    print(f"[{pair_index}/{len(band_pairs)}] {blue_band}-{red_band}  "
          f"SN Ia z range {z_lo:.2f}-{z_hi:.2f}", flush=True)

    # --- fast model-only populations: SN Ia plus contaminants ----------------
    SNe_Ia = Generator2.generate_many(N_BACKGROUND, band1, band2)
    contaminants = [
        Generator3.generate_many(N_CONTAMINANT, band1, band2, abs_mag, sigma, source)
        for abs_mag, sigma, source in contaminant_info
    ]

    # Rate-weighted draws for each population, plotted separately.
    drawn = {'Ia': full_creation(SNe_Ia, N_DRAW)}
    for (_, _, source), pop in zip(contaminant_info, contaminants):
        drawn[source] = full_creation(pop, N_DRAW)

    # Decision boundary fitted on all populations pooled together.
    all_colour_l, all_colour_ul, all_peak_mag_l, all_peak_mag_ul = combine_populations(
        [SNe_Ia] + contaminants, N_DRAW)
    x_range, y_range, m_, b_ = exponential_regression(
        all_peak_mag_ul, all_peak_mag_l, all_colour_ul, all_colour_l)

    # --- slow path: events observed through a real LSST cadence -------------
    SNe_data = Generator.generate_many(N_CADENCE, band1, band2)

    l_peak_mags, l_colors, l_times = [], [], []
    for sne_event in SNe_data:
        l_peak_mags.append(sne_event['band_1_mag_l'])
        l_colors.append(sne_event['lensed_colour'])
        l_times.append(sne_event['time_of_SN'])

        all_l_times.append(sne_event['time_of_SN'])
        all_time_delays.append(sne_event['time_delay'])
        all_timdel.append(sne_event['timdel'])          # physical lensing delay
        all_redshifts.append(sne_event['z'])
        all_magnifications.append(sne_event['magnification'])
        all_weights_l.append(sne_event['weights_l'])

    # Each event is compared against the boundary at its OWN peak magnitude.
    rate = success_rate(l_peak_mags, l_colors, m_, b_)

    plot_data_per_band.append({
        'band_pair': pair, 'drawn': drawn,
        'x_range': x_range, 'y_range': y_range,
        'l_peak_mags': l_peak_mags, 'l_colors': l_colors, 'l_times': l_times,
        'success_rate': rate, 'n_cadence_events': len(SNe_data),
    })

    elapsed = time.time() - pair_start
    eta = (time.time() - run_start) / pair_index * (len(band_pairs) - pair_index)
    print(f"      {len(SNe_data)} cadence events kept of {N_CADENCE}"
          f" ({100 * len(SNe_data) / N_CADENCE:.1f}%), success rate = {rate:.3f}"
          f"  [{elapsed / 60:.1f} min, ETA {eta / 60:.0f} min]", flush=True)

print(f"\\ntotal: {(time.time() - run_start) / 3600:.2f} h")
'''),
    md("""
## Plot the grid
"""),
    code('''
STYLES = {
    'Ia':            ('tab:gray', 'SN Ia'),
    'nugent-sn1bc':  ('b',        'SN Ibc'),
    'nugent-sn2l':   ('r',        'SN IIL'),
    'nugent-sn2n':   ('c',        'SN IIn'),
    'nugent-sn2p':   ('g',        'SN IIP'),
}

# Shared colour scale for the epoch colouring.
valid_l_times = np.array([t for t in all_l_times if np.isfinite(t)])
if valid_l_times.size:
    vmin_time, vmax_time = valid_l_times.min(), valid_l_times.max()
else:
    vmin_time, vmax_time = 0.0, 1.0
    print("Warning: no lensed SNe with valid epochs; colourbar scale is arbitrary.")

ncols = 4
nrows = (len(plot_data_per_band) + ncols - 1) // ncols
# constrained_layout reserves room for the two-line titles and the colourbar.
# The previous subplots_adjust + fig.add_axes combination left no vertical gap
# between rows, so each title overlapped the axis label of the panel above.
fig, ax = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.4 * nrows),
                       constrained_layout=True)
ax_flat = np.atleast_1d(ax).flatten()

# Drop unused panels before the colourbar is attached to the survivors.
for k in range(len(plot_data_per_band), nrows * ncols):
    fig.delaxes(ax_flat[k])
used_axes = list(ax_flat[:len(plot_data_per_band)])

for n, data in enumerate(plot_data_per_band):
    current_ax = ax_flat[n]
    current_ax.plot(data['x_range'], data['y_range'], color='black', lw=1.5)

    for key, (colour, label) in STYLES.items():
        mag_ul, colour_ul, _, mag_l, colour_l, _ = data['drawn'][key]
        # Unlensed: black-edged. Lensed: plain fill, and the one that gets a label.
        current_ax.scatter(mag_ul, colour_ul, c=colour, edgecolors='black',
                           s=14, linewidths=.4, alpha=.75)
        current_ax.scatter(mag_l, colour_l, c=colour, label=label, s=14, alpha=.75)

    # Statistics use every cadence event; the overlay is subsampled so a panel with
    # tens of thousands of them stays readable rather than saturating to a blob.
    pm = np.asarray(data['l_peak_mags'], dtype=float)
    cl = np.asarray(data['l_colors'], dtype=float)
    tm = np.asarray(data['l_times'], dtype=float)
    if pm.size > PLOT_MAX_CADENCE:
        sel = np.random.choice(pm.size, PLOT_MAX_CADENCE, replace=False)
        pm, cl, tm = pm[sel], cl[sel], tm[sel]

    scat = current_ax.scatter(pm, cl, c=tm, cmap='spring',
                              vmin=vmin_time, vmax=vmax_time, s=9, alpha=.6,
                              linewidths=0)

    blue_band, red_band = data['band_pair']
    current_ax.set_title(f"{blue_band} vs {red_band}\\n"
                         f"success rate = {data['success_rate']:.2f}", fontsize=11)
    current_ax.set_xlabel(f"{red_band} peak magnitude")
    current_ax.set_ylabel(f"{blue_band} - {red_band}")
    if n == 0:
        current_ax.legend(loc='upper left', fontsize=8)

norm = plt.Normalize(vmin=vmin_time, vmax=vmax_time)
sm = cm.ScalarMappable(cmap='spring', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=used_axes, fraction=0.02, pad=0.015)
cbar.set_label('Rest-frame epoch of identification (days from peak)')

plt.show()
'''),
    md("""
## Success rate per band pair

Fraction of cadence-realisation lensed events that fall above the decision
boundary, i.e. that the colour method would flag.
"""),
    code('''
for data in plot_data_per_band:
    blue_band, red_band = data['band_pair']
    print(f"{blue_band:>6} - {red_band:<6}  "
          f"success rate = {data['success_rate']:.3f}  "
          f"(n = {data['n_cadence_events']})")
'''),
    md("""
## Save the cadence-realisation sample for notebook 05

The time-delay diagnostics reuse this population rather than regenerating it.
"""),
    code('''
np.savez('cadence_population.npz',
         l_times=np.array(all_l_times, dtype=float),
         time_delays=np.array(all_time_delays, dtype=float),
         timdel=np.array(all_timdel, dtype=float),
         redshifts=np.array(all_redshifts, dtype=float),
         magnifications=np.array(all_magnifications, dtype=float),
         weights_l=np.array(all_weights_l, dtype=float))
print(f"saved {len(all_time_delays)} events to cadence_population.npz")
'''),
    md("""
## Beyond the single-colour boundary

This notebook fits a straight line in one colour vs magnitude. The boundary and
multi-colour studies (see the results write-up) show that a **flexible boundary on
the full ugrizy colour vector** separates lensed SN Ia from the contaminant
background far better, especially once a real cadence limits which bands you have.

That identifier now lives in the package:

```python
from cmsne.classifier import ColourClassifier
clf = ColourClassifier(target_fpr=0.10).fit(signal_events, background_events,
                                             signal_weights, background_weights)
recovery = clf.recovery_rate(signal_events, weights=signal_weights)
```

For production, generate the populations with `scripts/run_full.py` (colour-magnitude
grid) and `scripts/run_multicolour.py` (all-band photometry), then analyse offline.
"""),
])

# ---------------------------------------------------------------- notebook 05
nb05 = notebook([
    md("""
# 05 - Time-delay and identification-epoch distributions

Compares the two epochs the analysis produces for each lensed system:

* **`time_of_SN`** - the rest-frame epoch (relative to the intrinsic peak) at which
  the *colour method* could flag the system from a single magnified image;
* **`time_delay`** - the rest-frame epoch at which the *delayed second image* is
  first detected, i.e. when image multiplicity becomes apparent.

Both are measured against the same rest-frame intrinsic peak day, which is what
makes them directly comparable. If the colour method fires systematically earlier,
that is the case for using it as a trigger.

Run notebook 04 first to produce `cadence_population.npz`.
"""),
    code(SETUP + '''
from cmsne.colour_magnitude import (modified_weighted_vals, weighted_fraction,
                                    weighted_quantile)
'''),
    code('''
data = np.load('cadence_population.npz')
CM_times      = data['l_times']
time_delays   = data['time_delays']     # epoch the 2nd image is first detected
timdel        = data['timdel']          # physical lensing delay, for validation
redshifts     = data['redshifts']
magnifications = data['magnifications']
weights_l     = data['weights_l']

print(f"{len(time_delays)} events loaded")

# Keep every array index-aligned: build ONE mask over all of them rather than
# filtering each array separately, which would silently pair a time delay with a
# different event's weight.
finite = (np.isfinite(CM_times) & np.isfinite(time_delays) & np.isfinite(redshifts)
          & np.isfinite(magnifications) & np.isfinite(weights_l) & np.isfinite(timdel))
print(f"{finite.sum()} events with all quantities finite")

CM_times, time_delays = CM_times[finite], time_delays[finite]
redshifts, magnifications = redshifts[finite], magnifications[finite]
weights_l, timdel = weights_l[finite], timdel[finite]
'''),
    md("""
## Validate before interpreting

The second-image epoch is only meaningful if it actually carries the physical delay.
An earlier version of the pipeline re-zeroed each light curve's time axis on its first
visit, which erased the delay and pinned this quantity near the model's minimum phase
-- it correlated with magnification at -0.09 and differed by only 2 days between
mu<10 and mu>40, despite the delay scaling as mu^-3. Check that before reading
anything off the histograms.
"""),
    code('''
r_delay = np.corrcoef(time_delays, timdel)[0, 1]
r_mu    = np.corrcoef(time_delays, magnifications)[0, 1]
print(f"corr(second-image epoch, physical delay) = {r_delay:+.4f}   (want >> 0)")
print(f"corr(second-image epoch, magnification)  = {r_mu:+.4f}   (want < 0, delay ~ mu^-3)")

lo, hi = magnifications < 10, magnifications > 40
print(f"median epoch, mu<10 : {np.median(time_delays[lo]):+7.2f} d  (n={lo.sum()})")
print(f"median epoch, mu>40 : {np.median(time_delays[hi]):+7.2f} d  (n={hi.sum()})")
print(f"physical delay range: {timdel.min():.3f} to {timdel.max():.1f} d")

if r_delay < 0.5:
    print("\\nWARNING: the epoch does not track the physical delay - do not interpret "
          "the histograms below.")
else:
    print("\\nOK: the epoch tracks the physical delay.")
'''),
    md("""
## Rate-weighted resampling
"""),
    code('''
N_DRAW = 1000

(weight_time_delays, new_CM_times, new_magnifications,
 new_redshift, normalized_weights) = modified_weighted_vals(
    weights_l, time_delays, CM_times, magnifications, redshifts, N_DRAW)

print(f"resampled to {len(weight_time_delays)} events")
'''),
    md("""
## Identification epoch vs image-multiplicity epoch
"""),
    code('''
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-40, 150, 60)
ax.hist(new_CM_times, bins=bins, alpha=0.55, label='colour method (`time_of_SN`)')
ax.hist(weight_time_delays, bins=bins, alpha=0.55, label='second image (`time_delay`)')
ax.set_xlabel('Rest-frame epoch relative to intrinsic peak (days)')
ax.set_ylabel('Rate-weighted counts')
ax.legend()
plt.show()

# Headline numbers come from the exact weighted estimators on the FULL population,
# not from the resampled draw above. Resampling is only needed to draw the
# histogram; a mean, a fraction or a quantile can be weighted directly, which is
# exact and free of Monte-Carlo noise.
print(f"median colour-method epoch : {weighted_quantile(CM_times, weights_l):+.2f} d")
print(f"median second-image epoch  : {weighted_quantile(time_delays, weights_l):+.2f} d")
print(f"colour method earlier in "
      f"{100 * weighted_fraction(CM_times < time_delays, weights_l):.1f}% of systems")
print(f"  (unweighted, for comparison: "
      f"{100 * np.mean(CM_times < time_delays):.1f}%)")
'''),
    md("""
## How many of these delays are actually measurable?

The magnification prior is flat in mu over 2-50, but the delay scales as
`(mu/4)**-3`, so most of that range maps to sub-day delays. For those systems the
"second image" is detected essentially simultaneously with the first, which is not
an image-multiplicity detection in any observational sense -- you would need to
resolve the pair spatially, and the angular-separation cut in `generate_one`
(`125 / mu` arcsec against a 0.8 arcsec threshold) never fires for mu <= 50.

Read the epoch comparison above against this table before drawing conclusions.
"""),
    code('''
print(f"{'delay cut':>12} {'share':>8} {'colour earlier':>16} {'median 2nd image':>18}")
for thr in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
    m = timdel > thr
    if m.sum() < 50:
        continue
    frac = weighted_fraction(CM_times[m] < time_delays[m], weights_l[m])
    med = weighted_quantile(time_delays[m], weights_l[m])
    print(f"{thr:>10.0f} d {100 * m.mean():>7.1f}% {100 * frac:>15.1f}% {med:>16.2f} d")

print(f"\\nmedian physical delay: {np.median(timdel):.2f} d; "
      f"{100 * np.mean(timdel < 1):.0f}% are under 1 day")
'''),
    md("""
## Per-system offset between the two identification epochs

The two epochs above are marginal distributions. The quantity that actually matters
operationally is the **paired** difference for each system:

    lead = time_delay - time_of_SN

i.e. how many rest-frame days earlier the colour method flags a system than its second
image becomes detectable. Positive means the colour method gets there first.

Read the regime split alongside it. Where the delay exceeds the supernova's own visible
lifetime the colour method wins by construction -- the second image simply has not
arrived yet -- so the overall fraction is dominated by systems where multiplicity is not
yet available, rather than by the colour method beating a live alternative.
"""),
    code('''
import sncosmo

lead = time_delays - CM_times          # + => colour method identifies first

# Classify by how long the delay is relative to the SN's own visible lifetime,
# since that is what sets the sign of the offset.
src = sncosmo.get_source('salt3')
visible = (src.maxphase() - src.minphase()) * (1 + redshifts)
ratio = timdel / visible

regimes = [('delay > light curve',    ratio >= 1,                        '#2c7d5c'),
           ('10-100% of light curve', (ratio >= .1) & (ratio < 1),       '#8ab17d'),
           ('1 d - 10%',              (timdel >= 1) & (ratio < .1),      '#e0a851'),
           ('delay < 1 d',            timdel < 1,                        '#b1372b')]

LO, HI, NBINS = -60, 200, 66
edges = np.linspace(LO, HI, NBINS)
clipped = np.clip(lead, LO + 1e-9, HI - 1e-9)   # outermost bins are overflow

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={'width_ratios': [2, 1]})

ax = axes[0]
counts, _, patches = ax.hist(
    [clipped[m] for _, m, _ in regimes], bins=edges, stacked=True,
    weights=[weights_l[m] / weights_l.sum() for _, m, _ in regimes],
    color=[c for _, _, c in regimes], label=[l for l, _, _ in regimes],
    edgecolor='white', linewidth=.2)

# The end bins pile up everything beyond the axis; hatch them so the spike is not
# mistaken for real structure at +200 d.
for group in patches:
    for bar in (group[0], group[-1]):
        bar.set_hatch('///')
        bar.set_edgecolor('0.35')

ax.axvline(0, color='k', lw=1.4)
med = weighted_quantile(lead, weights_l)
ax.axvline(med, color='#b5179e', lw=1.6, ls='--', label=f'weighted median {med:+.0f} d')
frac = 100 * weighted_fraction(lead > 0, weights_l)
ax.annotate(f'colour method first\\n{frac:.1f}%', xy=(.60, .93), xycoords='axes fraction',
            ha='center', fontsize=11, color='#2c7d5c', fontweight='bold')
ax.annotate(f'multiplicity first\\n{100 - frac:.1f}%', xy=(.16, .93), xycoords='axes fraction',
            ha='center', fontsize=11, color='#b1372b', fontweight='bold')
over = 100 * weighted_fraction(lead >= HI, weights_l)
under = 100 * weighted_fraction(lead <= LO, weights_l)
ax.set_xlim(LO, HI)
ax.set_xlabel('Colour method lead time (rest-frame days)\\n'
              'time_delay - time_of_SN;  positive = colour method identifies first')
ax.set_ylabel('Rate-weighted fraction per bin')
ax.set_title('Offset between the two identification epochs\\n'
             f'hatched end bins are overflow: {under:.1f}% below, {over:.1f}% above',
             fontsize=11)
ax.legend(fontsize=8.5, loc='center right')

ax = axes[1]
labels = [l for l, _, _ in regimes]
shares = [100 * weighted_fraction(m, weights_l) for _, m, _ in regimes]
meds = [weighted_quantile(lead[m], weights_l[m]) for _, m, _ in regimes]
ypos = np.arange(len(labels))
ax.barh(ypos, meds, color=[c for _, _, c in regimes], height=.62)
ax.axvline(0, color='k', lw=1.2)
ax.set_yticks(ypos)
ax.set_yticklabels([f'{l}\\n({s:.0f}% of systems)' for l, s in zip(labels, shares)], fontsize=8.5)
ax.invert_yaxis()
for y, m_ in zip(ypos, meds):
    ax.annotate(f'{m_:+.0f} d', xy=(m_, y), xytext=(6 if m_ >= 0 else -6, 0),
                textcoords='offset points', va='center',
                ha='left' if m_ >= 0 else 'right', fontsize=9)
ax.set_xlabel('Median lead time (rest-frame days)')
ax.set_title('...split by how long the delay is', fontsize=11)
span = max(abs(min(meds)), abs(max(meds)))
ax.set_xlim(-0.35 * span, 1.30 * span)   # room for the value labels at both ends

plt.tight_layout()
plt.show()

print(f"weighted median lead      : {med:+.2f} d")
print(f"weighted IQR              : "
      f"{weighted_quantile(lead, weights_l, .25):+.1f} to "
      f"{weighted_quantile(lead, weights_l, .75):+.1f} d")
print(f"colour method identifies first in {frac:.1f}% of systems\\n")
print(f"{'regime':<24}{'share':>8}{'median lead':>14}{'colour first':>14}")
for (lbl, m, _), share in zip(regimes, shares):
    print(f"{lbl:<24}{share:>7.1f}%{weighted_quantile(lead[m], weights_l[m]):>+13.1f} d"
          f"{100 * weighted_fraction(lead[m] > 0, weights_l[m]):>13.1f}%")
'''),
    md("""
## Dependence on redshift and magnification
"""),
    code('''
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].hist2d(new_CM_times, new_redshift, bins=50)
axes[0, 0].set_xlabel('Colour-method epoch (days)'); axes[0, 0].set_ylabel('Redshift')

axes[0, 1].hist2d(weight_time_delays, new_redshift, bins=50)
axes[0, 1].set_xlabel('Second-image epoch (days)'); axes[0, 1].set_ylabel('Redshift')

axes[1, 0].hist2d(new_magnifications, new_redshift, bins=50)
axes[1, 0].set_xlabel('Magnification'); axes[1, 0].set_ylabel('Redshift')

axes[1, 1].hist2d(new_CM_times, new_magnifications, bins=50)
axes[1, 1].set_xlabel('Colour-method epoch (days)'); axes[1, 1].set_ylabel('Magnification')

plt.tight_layout()
plt.show()
'''),
])

for name, nb in [("04_colour_magnitude.ipynb", nb04), ("05_time_delays.ipynb", nb05)]:
    path = os.path.join(REPO, "notebooks", name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print("wrote", path)
