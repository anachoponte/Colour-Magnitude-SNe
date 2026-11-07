"""Compare visual encodings of event rate weight (Size, Colour, Transparency, and Size+Alpha)
on the colour-magnitude decision boundary plots.

Also demonstrates the cleaned lensed_Ia vs unlensed_* boundary (with other lensed subtypes
excluded from background contaminants).
"""
import argparse, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TRAJECTORY_GRID, _step_suffix, resolve_magnitude_key
from scripts.plot_evolving_boundaries import (
    CLASSES, UNLENSED_CLASSES, LENSED_CLASSES, HEATMAP_LEVELS,
    shared_ranges, compute_surfaces, _corner_panel, _draw_weighted_sources,
    _class_legend_handles, _binary_legend_handles,
    plot_boundary_evolution, plot_weighted_sources_corner, plot_corner_with_heatmap
)


def plot_encoding_comparison_grid(pop, surfaces, pair_key, out_path, colour_range, magnitude_range,
                                  signal_classes, signal_label, day=0.0, binary_colour=False,
                                  mag_mode='snapshot', mag_band='auto', mag_label='r(t)',
                                  n_sources=150, seed=0):
    """Render a 2x2 comparison figure at a single epoch (e.g. +0d or +14d) comparing:
      1. Uniform (Unweighted appearance - why raw scatter looks misleading)
      2. Transparency / Alpha encoding (opacity ~ log10 weight)
      3. Marker Size encoding (area ~ log10 weight)
      4. Marker Colour + Size encoding (colormap shade & size ~ log10 weight)
    """
    encodings = [
        ('alpha', '1. Transparency (Alpha ~ log10 weight)'),
        ('size', '2. Marker Size (Area ~ log10 weight)'),
        ('colour', '3. Colour Shade (Colormap ~ log10 weight: Blue=Lensed, Red=Unlensed)'),
        ('size_alpha', '4. Combined Size + Alpha (Area & Opacity ~ log10 weight)'),
    ]

    fig = plt.figure(figsize=(16, 13))
    outer = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.28)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))
    cg, mg, pg = surfaces[day]
    mesh = None

    for idx, (enc, title) in enumerate(encodings):
        r, c = divmod(idx, 2)
        ax_main, ax_top, ax_right = _corner_panel(fig, outer[r, c])
        mesh = ax_main.contourf(mg.T, cg.T, pg.T, levels=HEATMAP_LEVELS, cmap='viridis', vmin=0, vmax=1, zorder=0)
        ax_main.contour(mg.T, cg.T, pg.T, levels=[0.5], colors='white', linewidths=1.2, zorder=1)

        rng = np.random.default_rng(seed)
        _draw_weighted_sources(
            ax_main, ax_top, ax_right, pop, pair_key, day,
            colour_range, magnitude_range, colors, signal_classes,
            emphasize_signal_edge=True, zorder=2, binary_colour=binary_colour,
            mag_mode=mag_mode, mag_band=mag_band, mag_label=mag_label,
            n_sources=n_sources, rng=rng, weight_encoding=enc
        )
        ax_main.set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
        ax_top.set_title(title, fontsize=11, fontweight='bold', pad=8)

    cax = fig.add_axes((0.92, 0.55, 0.015, 0.30))
    fig.colorbar(mesh, cax=cax, label=f'P({signal_label})')

    if binary_colour:
        handles, labels = _binary_legend_handles(distinguishable=True)
    else:
        handles, labels = _class_legend_handles(colors, signal_classes, distinguishable=True)
    fig.legend(handles, labels, fontsize=8.5, loc='upper left', bbox_to_anchor=(0.91, 0.45), title='Classes')

    fig.suptitle(
        f'Visualizing Event Rate Weights on the {pair_key[0]}-{pair_key[1]} vs {mag_label} Decision Boundary (+{day:.0f}d)\n'
        f'Signal: {signal_label} vs Unlensed Background ({n_sources} events/class sampled)',
        fontsize=13, fontweight='bold'
    )
    fig.subplots_adjust(left=0.06, right=0.89, top=0.90, bottom=0.06)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', default='docs/figures/weight_encodings')
    ap.add_argument('--pair', default='gr')
    ap.add_argument('--mag-mode', default='snapshot', choices=['matched', 'peakmag', 'snapshot'])
    ap.add_argument('--n-sources', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pop = {}
    for c in CLASSES:
        with open(os.path.join(args.data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']

    background = [e for c in UNLENSED_CLASSES for e in pop[c]]
    background_weights = np.array([e['weight'] for e in background])

    for target, signal_classes, signal_label, suffix in [
        ('lensed_ia', ['lensed_Ia'], 'lensed Ia', '_lensed_ia'),
        ('all_lensed', LENSED_CLASSES, 'all lensed', '_alllensed'),
    ]:
        signal = [e for c in signal_classes for e in pop[c]]
        signal_weights = np.array([e['weight'] for e in signal])

        sample_event = signal[0]
        _, mag_label = resolve_magnitude_key(args.pair, 0.0, args.mag_mode, 'auto', sample_event)

        colour_range, magnitude_range = shared_ranges(
            signal, background, signal_weights, background_weights, args.pair,
            mag_mode=args.mag_mode, mag_band='auto'
        )
        surfaces = compute_surfaces(
            signal, background, signal_weights, background_weights,
            args.pair, colour_range, magnitude_range, n=200,
            mag_mode=args.mag_mode, mag_band='auto'
        )

        # 1. 2x2 Comparison grid at +0d and +14d
        for day in (0.0, 14.0):
            grid_path = os.path.join(args.out_dir, f'encoding_comparison_{args.pair}_day{int(day)}{suffix}.png')
            plot_encoding_comparison_grid(
                pop, surfaces, args.pair, grid_path, colour_range, magnitude_range,
                signal_classes, signal_label, day=day, binary_colour=(target == 'all_lensed'),
                mag_mode=args.mag_mode, mag_band='auto', mag_label=mag_label,
                n_sources=args.n_sources, seed=args.seed
            )

        # 2. Full 5-epoch evolution for each encoding mode (alpha, size, colour, size_alpha)
        for enc in ('alpha', 'size', 'colour', 'size_alpha'):
            corner_path = os.path.join(args.out_dir, f'boundary_{args.pair}_corner{suffix}_{enc}.png')
            plot_corner_with_heatmap(
                pop, surfaces, args.pair, corner_path, colour_range, magnitude_range,
                signal_classes, signal_label, binary_colour=False,
                mag_mode=args.mag_mode, mag_band='auto', mag_label=mag_label,
                n_sources=args.n_sources, seed=args.seed, weight_encoding=enc
            )
            print(f'Wrote {corner_path}')


if __name__ == '__main__':
    main()
