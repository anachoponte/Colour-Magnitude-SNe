"""Evolving colour-magnitude decision boundary: how P(lensed) moves across
the trajectory's Δt-since-trigger grid (issue #18).

`cmsne.contours` already computes this (`posterior_surface`/`evolving_contours`,
added in PR #14) -- it has just never been run on real data or plotted. This
script does both: loads a real dual-schema production population, fits one
2-feature classifier per (band pair, grid step) via `posterior_surface`, and
renders a 5-panel figure per band pair.

`posterior_surface` is called directly rather than through `evolving_contours`
so a single colour/magnitude axis range can be shared across all 5 panels in
a figure -- otherwise each panel auto-scales to its own step's finite-value
range (which drifts slightly step to step), making the boundary's motion
hard to read side by side.

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
from cmsne.contours import posterior_surface

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']

# Adjacent-colour pair keys in trajectory feature-name order, e.g. ('lsstu','lsstg') -> 'ug'.
PAIR_KEYS = [f'{a[-1]}{b[-1]}' for a, b in ADJ_COLOURS]


def load_population(data_dir):
    """Load every class's pkl, split into signal (lensed_Ia) vs background (rest)."""
    pop = {}
    for c in CLASSES:
        with open(os.path.join(data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']
    signal = pop['lensed_Ia']
    background = [e for c in CLASSES if c != 'lensed_Ia' for e in pop[c]]
    return signal, background


def shared_ranges(signal, background, pair_key, grid=TRAJECTORY_GRID):
    """Pooled colour/magnitude min-max across every grid step, for one shared
    axis range across all of a figure's panels."""
    colours, mags = [], []
    pooled = signal + background
    for day in grid:
        suffix = _step_suffix(day)
        colour_key, magnitude_key = f'{pair_key}{suffix}', f'peakmag{suffix}'
        colours.extend(e.get(colour_key, np.nan) for e in pooled)
        mags.extend(e.get(magnitude_key, np.nan) for e in pooled)
    colours, mags = np.asarray(colours, float), np.asarray(mags, float)
    return ((float(np.nanmin(colours)), float(np.nanmax(colours))),
            (float(np.nanmin(mags)), float(np.nanmax(mags))))


def plot_boundary_evolution(signal, background, pair_key, out_path, grid=TRAJECTORY_GRID, n=200):
    colour_range, magnitude_range = shared_ranges(signal, background, pair_key, grid)

    fig, axes = plt.subplots(1, len(grid), figsize=(4 * len(grid), 4.2), sharey=True)
    mesh = None
    for ax, day in zip(axes, grid):
        suffix = _step_suffix(day)
        colour_key, magnitude_key = f'{pair_key}{suffix}', f'peakmag{suffix}'
        cg, mg, pg, _ = posterior_surface(
            signal, background, colour_key, magnitude_key,
            colour_range=colour_range, magnitude_range=magnitude_range, n=n)
        mesh = ax.contourf(cg, mg, pg, levels=20, cmap='viridis', vmin=0, vmax=1)
        ax.contour(cg, mg, pg, levels=[0.5], colors='white', linewidths=1.2)
        ax.set_title(f'+{day:.0f}d')
        ax.set_xlabel(f'{pair_key[0]}-{pair_key[1]}')
        ax.invert_yaxis()
    axes[0].set_ylabel('peakmag (brightest-observed)')
    fig.suptitle(f'P(lensed Ia) vs {pair_key[0]}-{pair_key[1]} colour -- evolution since trigger')
    fig.colorbar(mesh, ax=axes, shrink=0.85, label='P(lensed Ia)')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='directory of per-class <name>.pkl files')
    ap.add_argument('--out-dir', default='docs/figures')
    ap.add_argument('--n', type=int, default=200, help='mesh resolution per axis')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    signal, background = load_population(args.data_dir)
    print(f'signal (lensed_Ia): {len(signal)}, background: {len(background)}')

    for pair_key in PAIR_KEYS:
        out_path = os.path.join(args.out_dir, f'boundary_{pair_key}.png')
        print(f'fitting {pair_key} ({len(TRAJECTORY_GRID)} grid steps)...')
        plot_boundary_evolution(signal, background, pair_key, out_path, n=args.n)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
