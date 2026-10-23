"""Evolving colour-magnitude decision boundary: how P(lensed) moves across
the trajectory's Δt-since-trigger grid (issue #18).

`cmsne.contours` already computes this (`posterior_surface`/`evolving_contours`,
added in PR #14) -- it has just never been run on real data or plotted. This
script does both: loads a real dual-schema production population, fits one
2-feature classifier per (band pair, grid step) via `posterior_surface`, and
renders a 5-panel figure per band pair, plus a companion scatter of the
discrete per-class events behind each boundary figure.

`posterior_surface` is called directly rather than through `evolving_contours`
so a single colour/magnitude axis range can be shared across all 5 panels in
a figure -- otherwise each panel auto-scales to its own step's finite-value
range (which drifts slightly step to step), making the boundary's motion
hard to read side by side. The shared range is rate-weighted and percentile-
based (`cmsne.contours.robust_range`, issue #24) rather than plain min/max,
since a handful of near-zero-probability outliers (implausibly low-redshift,
near-maximally-magnified lensed Ia draws) otherwise stretch the range so far
that most of each panel shows empty extrapolation instead of real data
density -- which is also where the classifier's axis-aligned, piecewise-
constant decision surface looks most erratic.

Axes follow the lensing-literature convention: magnitude on x, colour on y
(not the stellar-CMD convention of colour on x, magnitude on y).

    python scripts/plot_evolving_boundaries.py --data-dir results/dual_58563115 --out-dir docs/figures
"""
import argparse, os, pickle, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.multicolour import ADJ_COLOURS
from cmsne.trajectory import TRAJECTORY_GRID, _step_suffix
from cmsne.contours import posterior_surface, robust_range

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']

# Adjacent-colour pair keys in trajectory feature-name order, e.g. ('lsstu','lsstg') -> 'ug'.
PAIR_KEYS = [f'{a[-1]}{b[-1]}' for a, b in ADJ_COLOURS]


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


def plot_boundary_evolution(signal, background, signal_weights, background_weights,
                            pair_key, out_path, grid=TRAJECTORY_GRID, n=200):
    colour_range, magnitude_range = shared_ranges(
        signal, background, signal_weights, background_weights, pair_key, grid)

    fig, axes = plt.subplots(1, len(grid), figsize=(4 * len(grid), 4.2), sharex=True, sharey=True)
    mesh = None
    for ax, day in zip(axes, grid):
        suffix = _step_suffix(day)
        colour_key, magnitude_key = f'{pair_key}{suffix}', f'peakmag{suffix}'
        cg, mg, pg, _ = posterior_surface(
            signal, background, colour_key, magnitude_key,
            colour_range=colour_range, magnitude_range=magnitude_range, n=n,
            signal_weights=signal_weights, background_weights=background_weights)
        # cg/mg/pg are laid out (magnitude varying along rows, colour along
        # columns) by posterior_surface's np.meshgrid(colour, magnitude) --
        # transpose all three together to put magnitude on x, colour on y
        # (the lensing-literature convention) without touching their values.
        mesh = ax.contourf(mg.T, cg.T, pg.T, levels=20, cmap='viridis', vmin=0, vmax=1)
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
    return colour_range, magnitude_range


def plot_discrete_sources_by_class(pop, pair_key, out_path, colour_range, magnitude_range,
                                   grid=TRAJECTORY_GRID):
    """Scatter every class's discrete events (colour-coded) behind one
    boundary figure's colour-magnitude axes.

    Zoomed to the same ``colour_range``/``magnitude_range`` the boundary
    figure used (from :func:`shared_ranges`), so the two are directly
    comparable panel for panel -- not each auto-scaled to its own full data
    extent, outliers included.
    """
    fig, axes = plt.subplots(1, len(grid), figsize=(4 * len(grid), 4.5), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))

    for ax, day in zip(axes, grid):
        suffix = _step_suffix(day)
        colour_key, magnitude_key = f'{pair_key}{suffix}', f'peakmag{suffix}'
        for i, c in enumerate(CLASSES):
            events = pop[c]
            colours = np.array([e.get(colour_key, np.nan) for e in events])
            mags = np.array([e.get(magnitude_key, np.nan) for e in events])
            finite = np.isfinite(colours) & np.isfinite(mags)
            is_signal = c == 'lensed_Ia'
            ax.scatter(mags[finite], colours[finite], s=18 if is_signal else 3,
                      marker='*' if is_signal else '.', color=colors[i],
                      alpha=0.5 if is_signal else 0.15, label=c, rasterized=True)
        ax.set_title(f'+{day:.0f}d')
        ax.set_xlabel('peakmag')
        ax.set_xlim(*magnitude_range)
        ax.set_ylim(*colour_range)
    axes[0].set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
    axes[-1].legend(fontsize=7, markerscale=3, loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.suptitle(f'Discrete per-class events behind the {pair_key} boundary figure')
    fig.subplots_adjust(right=0.85)
    fig.savefig(out_path, dpi=130)
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
        print(f'fitting {pair_key} ({len(TRAJECTORY_GRID)} grid steps)...')
        try:
            colour_range, magnitude_range = plot_boundary_evolution(
                signal, background, signal_weights, background_weights,
                pair_key, boundary_path, n=args.n)
        except ValueError as exc:
            print(f'skipping {pair_key}: {exc}')
            continue
        print(f'wrote {boundary_path}')
        plot_discrete_sources_by_class(pop, pair_key, sources_path, colour_range, magnitude_range)
        print(f'wrote {sources_path}')


if __name__ == '__main__':
    main()
