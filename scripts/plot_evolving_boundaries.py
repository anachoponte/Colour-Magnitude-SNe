"""Evolving colour-magnitude decision boundary: how P(lensed) moves across
the trajectory's Δt-since-trigger grid (issue #18).

`cmsne.contours` already computes this (`posterior_surface`/`evolving_contours`,
added in PR #14) -- it has just never been run on real data or plotted. This
script does both: loads a real dual-schema production population, fits one
2-feature classifier per (band pair, grid step) via `posterior_surface`, and
renders four figures per band pair:

  - `boundary_<pair>.png` -- the P(lensed Ia) heatmap alone, one panel per
    grid step.
  - `boundary_<pair>_sources.png` -- a corner-plot-style companion: the
    discrete per-class events behind the heatmap, weight-rated (each point's
    alpha reflects its own rate weight, not just its class membership) plus
    weighted marginal histograms of every class above (magnitude) and to the
    right (colour).
  - `boundary_<pair>_corner.png` -- the same corner layout, with the
    posterior surface heatmap underlaid in each main panel.
  - `boundary_<pair>_resampled.png` -- the same corner layout again, but
    plotting a weighted *resample* of each class rather than every raw event
    at weight-scaled alpha: point density, not opacity, then encodes each
    class's weighted distribution (see `plot_resampled_sources_corner`).

A class's discrete events are NOT uniformly realistic: `cmsne.multicolour`'s
rate_weight() assigns per-event weights spanning many orders of magnitude
(for the real production population behind these figures, lensed Ia weights
range from 0 to ~2e-7, with roughly half the events below 0.1% of the
maximum -- implausibly low-redshift, near-maximally-magnified draws that are
essentially never realised). An earlier version of the sources figure
scattered every event at the same alpha regardless of weight, which made it
inconsistent with the (correctly weighted, since issue #24) heatmap next to
it: outliers that the heatmap's fit all but ignores looked just as prominent
as the realistic bulk of the population. `_weight_alpha` below fixes that by
alpha-encoding each point from its own weight.

`posterior_surface` is called directly (via `compute_surfaces`) rather than
through `evolving_contours` so a single colour/magnitude axis range can be
shared across all of a figure's panels -- otherwise each panel auto-scales
to its own step's finite-value range (which drifts slightly step to step),
making the boundary's motion hard to read side by side. The shared range is
rate-weighted and percentile-based (`cmsne.contours.robust_range`, issue
#24) rather than plain min/max, for the same outlier reason as above. Each
band pair's posterior surfaces are fitted once (`compute_surfaces`) and
reused by both the heatmap-only and the combined corner+heatmap figure, so
the (expensive) classifier fit never runs twice for the same band pair.

Every heatmap's contourf call passes explicit bin edges spanning the full
[0, 1] probability range (`levels=np.linspace(0, 1, 21)`), not a bare
integer level count: matplotlib's `levels=N` autoscales those bins to the
*data's* actual range when N is an int, which silently clipped the colorbar
to whatever maximum a given grid step/band pair happened to reach (e.g.
0-0.5) instead of the fixed 0-1 scale every panel should share.

Axes follow the lensing-literature convention: magnitude on x, colour on y
(not the stellar-CMD convention of colour on x, magnitude on y).

    python scripts/plot_evolving_boundaries.py --data-dir results/dual_58563115 --out-dir docs/figures
"""
import argparse, os, pickle, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.multicolour import ADJ_COLOURS
from cmsne.trajectory import TRAJECTORY_GRID, _step_suffix
from cmsne.contours import posterior_surface, robust_range

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']

# Adjacent-colour pair keys in trajectory feature-name order, e.g. ('lsstu','lsstg') -> 'ug'.
PAIR_KEYS = [f'{a[-1]}{b[-1]}' for a, b in ADJ_COLOURS]

N_MARGINAL_BINS = 30
HEATMAP_LEVELS = np.linspace(0, 1, 21)
N_RESAMPLE = 2000
RESAMPLE_SEED = 0


def shared_ranges(signal, background, signal_weights, background_weights,
                  pair_key, grid=TRAJECTORY_GRID):
    """Pooled, rate-weighted, outlier-robust colour/magnitude range across
    every grid step, for one shared axis range across all of a figure's
    panels (see :func:`cmsne.contours.robust_range`)."""
    colours, mags, weights = [], [], []
    pooled = signal + background
    pooled_weights = np.r_[signal_weights, background_weights]
    for day in grid:
        suffix = _step_suffix(day)
        colour_key, magnitude_key = f'{pair_key}{suffix}', f'peakmag{suffix}'
        colours.extend(e.get(colour_key, np.nan) for e in pooled)
        mags.extend(e.get(magnitude_key, np.nan) for e in pooled)
        weights.extend(pooled_weights)
    colours, mags = np.asarray(colours, float), np.asarray(mags, float)
    weights = np.asarray(weights, float)
    if not np.any(np.isfinite(colours)) or not np.any(np.isfinite(mags)):
        raise ValueError(
            f"'{pair_key}' has no finite colour/magnitude values at any grid step "
            "-- nothing observed this band pair in the given population")
    return robust_range(colours, weights), robust_range(mags, weights)


def compute_surfaces(signal, background, signal_weights, background_weights,
                     pair_key, colour_range, magnitude_range, grid=TRAJECTORY_GRID, n=200):
    """Fit one posterior_surface per grid step and cache the (cg, mg, pg)
    grids, so every figure for this band pair reuses the same fit instead of
    re-fitting the classifier per figure."""
    surfaces = {}
    for day in grid:
        suffix = _step_suffix(day)
        colour_key, magnitude_key = f'{pair_key}{suffix}', f'peakmag{suffix}'
        cg, mg, pg, _ = posterior_surface(
            signal, background, colour_key, magnitude_key,
            colour_range=colour_range, magnitude_range=magnitude_range, n=n,
            signal_weights=signal_weights, background_weights=background_weights)
        surfaces[day] = (cg, mg, pg)
    return surfaces


def plot_boundary_evolution(surfaces, pair_key, out_path, grid=TRAJECTORY_GRID):
    fig, axes = plt.subplots(1, len(grid), figsize=(4 * len(grid), 4.2), sharex=True, sharey=True)
    mesh = None
    for ax, day in zip(axes, grid):
        cg, mg, pg = surfaces[day]
        # cg/mg/pg are laid out (magnitude varying along rows, colour along
        # columns) by posterior_surface's np.meshgrid(colour, magnitude) --
        # transpose all three together to put magnitude on x, colour on y
        # (the lensing-literature convention) without touching their values.
        mesh = ax.contourf(mg.T, cg.T, pg.T, levels=HEATMAP_LEVELS, cmap='viridis', vmin=0, vmax=1)
        ax.contour(mg.T, cg.T, pg.T, levels=[0.5], colors='white', linewidths=1.2)
        ax.set_title(f'+{day:.0f}d')
        ax.set_xlabel('peakmag (brightest-observed)')
    axes[0].set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
    fig.suptitle(f'P(lensed Ia) vs {pair_key[0]}-{pair_key[1]} colour -- evolution since trigger')
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, 'rate-weighted; axis range is the 1st-99th weighted percentile, not full min/max',
            ha='center', fontsize=8, color='gray')
    fig.colorbar(mesh, ax=axes, shrink=0.85, label='P(lensed Ia)')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _weight_alpha(weights, alpha_range=(0.05, 0.9)):
    """Map rate weights -- often spanning >10 orders of magnitude -- to a
    per-point scatter alpha in alpha_range, via a 1st-99th percentile-clipped
    log10 scale (the same percentile-robustness convention as
    cmsne.contours.robust_range). A linear weight-to-alpha mapping would make
    everything but the single heaviest point invisible.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w
    wmax = np.nanmax(w) if np.isfinite(w).any() else 0.0
    if wmax <= 0:
        # Every weight in this slice is zero (or otherwise non-positive) --
        # none of these events are realised. Render them at the LOWEST
        # alpha, not the default degenerate-case high alpha below: these are
        # exactly the implausible-outlier events this function exists to
        # suppress, not a legitimately uniform-weight population.
        return np.full(w.shape, alpha_range[0])
    floor = wmax * 1e-20
    log_w = np.log10(np.clip(w, floor, None))
    lo, hi = np.percentile(log_w, [1, 99])
    if hi <= lo:
        return np.full(w.shape, alpha_range[1])
    frac = np.clip((log_w - lo) / (hi - lo), 0.0, 1.0)
    return alpha_range[0] + frac * (alpha_range[1] - alpha_range[0])


def _corner_panel(fig, outer_spec):
    """One corner-plot cell: a main axes plus a top and right marginal axes
    sharing its x/y, built from a single outer GridSpec cell."""
    inner = outer_spec.subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 3),
                                   wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(inner[1, 0])
    ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)
    ax_top.tick_params(labelbottom=False, labelleft=False, left=False)
    ax_right.tick_params(labelleft=False, labelbottom=False, bottom=False)
    return ax_main, ax_top, ax_right


def _draw_weighted_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                           colour_range, magnitude_range, colors,
                           emphasize_signal_edge=False, zorder=1):
    """Draw every class's weight-rated scatter onto ax_main, plus weighted,
    density-normalised marginal histograms onto ax_top (magnitude) and
    ax_right (colour)."""
    suffix = _step_suffix(day)
    colour_key, magnitude_key = f'{pair_key}{suffix}', f'peakmag{suffix}'
    mag_bins = np.linspace(*magnitude_range, N_MARGINAL_BINS + 1)
    colour_bins = np.linspace(*colour_range, N_MARGINAL_BINS + 1)

    for i, c in enumerate(CLASSES):
        events = pop[c]
        colours = np.array([e.get(colour_key, np.nan) for e in events])
        mags = np.array([e.get(magnitude_key, np.nan) for e in events])
        weights = np.array([e['weight'] for e in events])
        finite = np.isfinite(colours) & np.isfinite(mags)
        colours, mags, weights = colours[finite], mags[finite], weights[finite]
        if colours.size == 0:
            continue
        is_signal = c == 'lensed_Ia'
        alphas = _weight_alpha(weights, alpha_range=(0.08, 0.9) if is_signal else (0.02, 0.6))
        rgba = np.tile(colors[i], (len(alphas), 1))
        rgba[:, 3] = alphas
        edgecolors = 'black' if (is_signal and emphasize_signal_edge) else 'none'
        ax_main.scatter(mags, colours, s=16 if is_signal else 3,
                        marker='*' if is_signal else '.', c=rgba,
                        edgecolors=edgecolors, linewidths=0.3,
                        rasterized=True, zorder=zorder)
        if weights.sum() > 0:
            # density=True divides by the weight sum; an all-zero-weight
            # slice (the same degenerate case _weight_alpha guards above)
            # would otherwise silently draw a NaN-filled (invisible) step.
            lw = 1.6 if is_signal else 0.9
            ax_top.hist(mags, bins=mag_bins, weights=weights, density=True,
                       histtype='step', color=colors[i], linewidth=lw)
            ax_right.hist(colours, bins=colour_bins, weights=weights, density=True,
                         histtype='step', orientation='horizontal', color=colors[i], linewidth=lw)

    ax_main.set_xlim(*magnitude_range)
    ax_main.set_ylim(*colour_range)
    ax_main.set_xlabel('peakmag')
    ax_top.set_title(f'+{day:.0f}d')


def _class_legend_handles(colors):
    return [Line2D([0], [0], color=colors[i],
                   lw=1.6 if c == 'lensed_Ia' else 0.9,
                   marker='*' if c == 'lensed_Ia' else None,
                   markersize=8 if c == 'lensed_Ia' else 0)
           for i, c in enumerate(CLASSES)]


def plot_weighted_sources_corner(pop, pair_key, out_path, colour_range, magnitude_range,
                                 grid=TRAJECTORY_GRID):
    """Weight-rated corner-style companion to the boundary figure: for each
    grid step, a main colour-magnitude scatter (per-point alpha from each
    event's own rate weight, see _weight_alpha) plus weighted, density-
    normalised marginal histograms of every class above (magnitude) and to
    the right (colour).
    """
    fig = plt.figure(figsize=(4.2 * len(grid), 5.2))
    outer = fig.add_gridspec(1, len(grid), wspace=0.35)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))

    for col, day in enumerate(grid):
        ax_main, ax_top, ax_right = _corner_panel(fig, outer[0, col])
        _draw_weighted_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                               colour_range, magnitude_range, colors)
        if col == 0:
            ax_main.set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
        else:
            ax_main.tick_params(labelleft=False)

    fig.legend(_class_legend_handles(colors), CLASSES, fontsize=7,
              loc='upper left', bbox_to_anchor=(0.90, 0.88))
    fig.suptitle(f'Weight-rated discrete sources behind the {pair_key} boundary figure\n'
                f'(point alpha ~ log rate weight; top/right: weighted class histograms)',
                fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.89, top=0.84, bottom=0.08)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _draw_resampled_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                            colour_range, magnitude_range, colors, rng,
                            n_resample=N_RESAMPLE):
    """Draw each class as a weighted bootstrap resample rather than every raw
    event at weight-scaled alpha: `n_resample` points are drawn with
    replacement per class, with probability proportional to each event's own
    rate weight, so point *density* (not opacity) encodes the class's
    weighted distribution. Resampling is done per class independently (not
    pooled across classes), matching _draw_weighted_sources' per-class alpha
    scale -- lensed_Ia's total rate is many orders of magnitude below the
    background classes', so pooling globally would make it vanish from the
    plot entirely instead of showing where its (rare but real) weighted mass
    sits.
    """
    suffix = _step_suffix(day)
    colour_key, magnitude_key = f'{pair_key}{suffix}', f'peakmag{suffix}'
    mag_bins = np.linspace(*magnitude_range, N_MARGINAL_BINS + 1)
    colour_bins = np.linspace(*colour_range, N_MARGINAL_BINS + 1)

    for i, c in enumerate(CLASSES):
        events = pop[c]
        colours = np.array([e.get(colour_key, np.nan) for e in events])
        mags = np.array([e.get(magnitude_key, np.nan) for e in events])
        weights = np.array([e['weight'] for e in events])
        finite = np.isfinite(colours) & np.isfinite(mags)
        colours, mags, weights = colours[finite], mags[finite], weights[finite]
        wsum = weights.sum()
        if colours.size == 0 or wsum <= 0:
            # No finite events, or every one has zero weight -- nothing in
            # this slice is realised, so there is nothing to resample.
            continue
        idx = rng.choice(colours.size, size=n_resample, replace=True, p=weights / wsum)
        rcolours, rmags = colours[idx], mags[idx]
        is_signal = c == 'lensed_Ia'
        ax_main.scatter(rmags, rcolours, s=16 if is_signal else 3,
                       marker='*' if is_signal else '.', color=colors[i],
                       alpha=0.35 if is_signal else 0.15,
                       rasterized=True, zorder=1)
        lw = 1.6 if is_signal else 0.9
        ax_top.hist(rmags, bins=mag_bins, density=True, histtype='step',
                   color=colors[i], linewidth=lw)
        ax_right.hist(rcolours, bins=colour_bins, density=True, histtype='step',
                     orientation='horizontal', color=colors[i], linewidth=lw)

    ax_main.set_xlim(*magnitude_range)
    ax_main.set_ylim(*colour_range)
    ax_main.set_xlabel('peakmag')
    ax_top.set_title(f'+{day:.0f}d')


def plot_resampled_sources_corner(pop, pair_key, out_path, colour_range, magnitude_range,
                                  grid=TRAJECTORY_GRID, n_resample=N_RESAMPLE, seed=RESAMPLE_SEED):
    """Corner-style companion like plot_weighted_sources_corner, but drawing a
    weighted resample of each class (see _draw_resampled_sources) instead of
    every raw event at weight-scaled alpha -- point density directly
    reflects each class's weighted distribution rather than relying on
    opacity, which is easy to misjudge across >10 orders of magnitude of
    weight and prone to overplotting from the sheer number of raw events.
    """
    fig = plt.figure(figsize=(4.2 * len(grid), 5.2))
    outer = fig.add_gridspec(1, len(grid), wspace=0.35)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))
    rng = np.random.default_rng(seed)

    for col, day in enumerate(grid):
        ax_main, ax_top, ax_right = _corner_panel(fig, outer[0, col])
        _draw_resampled_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                                colour_range, magnitude_range, colors, rng,
                                n_resample=n_resample)
        if col == 0:
            ax_main.set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
        else:
            ax_main.tick_params(labelleft=False)

    fig.legend(_class_legend_handles(colors), CLASSES, fontsize=7,
              loc='upper left', bbox_to_anchor=(0.90, 0.88))
    fig.suptitle(f'Weighted resample ({n_resample}/class/step) of discrete sources behind '
                f'the {pair_key} boundary figure\n'
                f'(point density ~ rate weight, not alpha; top/right: resampled class histograms)',
                fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.89, top=0.84, bottom=0.08)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_corner_with_heatmap(pop, surfaces, pair_key, out_path, colour_range, magnitude_range,
                             grid=TRAJECTORY_GRID):
    """Same corner layout as plot_weighted_sources_corner, with the already-
    fitted P(lensed Ia) posterior surface (from compute_surfaces -- not
    re-fitted here) underlaid in each main panel beneath the weighted
    scatter, so the decision surface and the weighted sources it was fitted
    from read together in one panel.
    """
    fig = plt.figure(figsize=(4.2 * len(grid), 5.2))
    outer = fig.add_gridspec(1, len(grid), wspace=0.35)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))
    mesh = None

    for col, day in enumerate(grid):
        ax_main, ax_top, ax_right = _corner_panel(fig, outer[0, col])
        cg, mg, pg = surfaces[day]
        mesh = ax_main.contourf(mg.T, cg.T, pg.T, levels=HEATMAP_LEVELS, cmap='viridis', vmin=0, vmax=1, zorder=0)
        ax_main.contour(mg.T, cg.T, pg.T, levels=[0.5], colors='white', linewidths=1.0, zorder=1)
        _draw_weighted_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                               colour_range, magnitude_range, colors,
                               emphasize_signal_edge=True, zorder=2)
        if col == 0:
            ax_main.set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
        else:
            ax_main.tick_params(labelleft=False)

    cax = fig.add_axes((0.91, 0.55, 0.012, 0.28))
    fig.colorbar(mesh, cax=cax, label='P(lensed Ia)')
    fig.legend(_class_legend_handles(colors), CLASSES, fontsize=7,
              loc='upper left', bbox_to_anchor=(0.91, 0.42))
    fig.suptitle(f'P(lensed Ia) heatmap + weight-rated discrete sources -- '
                f'{pair_key[0]}-{pair_key[1]} colour, evolution since trigger', fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.89, top=0.84, bottom=0.08)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='directory of per-class <name>.pkl files')
    ap.add_argument('--out-dir', default='docs/figures')
    ap.add_argument('--n', type=int, default=200, help='mesh resolution per axis')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pop = {}
    for c in CLASSES:
        with open(os.path.join(args.data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']
    signal = pop['lensed_Ia']
    background = [e for c in CLASSES if c != 'lensed_Ia' for e in pop[c]]
    signal_weights = np.array([e['weight'] for e in signal])
    background_weights = np.array([e['weight'] for e in background])
    print(f'signal (lensed_Ia): {len(signal)}, background: {len(background)}')

    for pair_key in PAIR_KEYS:
        boundary_path = os.path.join(args.out_dir, f'boundary_{pair_key}.png')
        sources_path = os.path.join(args.out_dir, f'boundary_{pair_key}_sources.png')
        corner_path = os.path.join(args.out_dir, f'boundary_{pair_key}_corner.png')
        resampled_path = os.path.join(args.out_dir, f'boundary_{pair_key}_resampled.png')
        print(f'fitting {pair_key} ({len(TRAJECTORY_GRID)} grid steps)...')
        try:
            colour_range, magnitude_range = shared_ranges(
                signal, background, signal_weights, background_weights, pair_key)
        except ValueError as exc:
            print(f'skipping {pair_key}: {exc}')
            continue

        surfaces = compute_surfaces(signal, background, signal_weights, background_weights,
                                    pair_key, colour_range, magnitude_range, n=args.n)

        plot_boundary_evolution(surfaces, pair_key, boundary_path)
        print(f'wrote {boundary_path}')

        plot_weighted_sources_corner(pop, pair_key, sources_path, colour_range, magnitude_range)
        print(f'wrote {sources_path}')

        plot_corner_with_heatmap(pop, surfaces, pair_key, corner_path, colour_range, magnitude_range)
        print(f'wrote {corner_path}')

        plot_resampled_sources_corner(pop, pair_key, resampled_path, colour_range, magnitude_range)
        print(f'wrote {resampled_path}')


if __name__ == '__main__':
    main()
