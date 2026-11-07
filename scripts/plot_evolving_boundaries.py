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
    size + alpha reflects its own rate weight, not just its class membership)
    plus weighted marginal histograms of every class above (magnitude) and to
    the right (colour).
  - `boundary_<pair>_corner.png` -- the same corner layout, with the
    posterior surface heatmap underlaid in each main panel.
  - `boundary_<pair>_resampled.png` -- the same corner layout again, but
    plotting a weighted *resample* of each class rather than every raw event
    at weight-scaled alpha: point density, not opacity, then encodes each
    class's weighted distribution (see `plot_resampled_sources_corner`).

When `--show-errors` is enabled (or `--also-plot-errors` generates `_err.png`
companion figures), per-event 1-sigma photometric measurement error bars
(`{colour}_err` and `{magnitude}_err`) are drawn on the scattered events,
with line opacity and thickness scaled to each event's rate weight so high-weight
physical events show crisp uncertainty crosses without cluttering low-weight specks.

Palette design: The posterior heatmap uses a cool, desaturated `GnBu` / `PuBu`
(or `cividis`) gradient, while scattered classes use a high-contrast palette:
vivid warm/magenta/gold hues for lensed signal stars (with crisp dark edges)
and distinct contrasting hues for unlensed circles, ensuring zero colour collision
against the background probability gradient.
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

from cmsne.multicolour import ADJ_COLOURS
from cmsne.trajectory import (TRAJECTORY_GRID, _step_suffix,
                              matching_band_for_pair, resolve_magnitude_key)
from cmsne.contours import posterior_surface, robust_range

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']
UNLENSED_CLASSES = [c for c in CLASSES if c.startswith('unlensed_')]
LENSED_CLASSES = [c for c in CLASSES if c.startswith('lensed_')]

# Adjacent-colour pair keys in trajectory feature-name order, e.g. ('lsstu','lsstg') -> 'ug'.
PAIR_KEYS = [f'{a[-1]}{b[-1]}' for a, b in ADJ_COLOURS]

N_MARGINAL_BINS = 30
HEATMAP_LEVELS = np.linspace(0, 1, 21)
# Soft, perceptual cool-slate/blue-teal heatmap (`PuBuGn` / `GnBu` or `Blues_r`/`YlGnBu_r`?
# Wait: `PuBu` goes from clean white/ice at P=0 to deep slate-navy at P=1!
# Against white-to-deep-navy (`PuBu`), warm/vivid markers (crimson, magenta, gold, coral,
# bright orange, lime-white, violet) pop out dramatically at EVERY probability level [0, 1],
# unlike `viridis` where purple/blue/teal/green/yellow swallows half the colour wheel!)
HEATMAP_CMAP = 'PuBu'

N_RESAMPLE = 2000
RESAMPLE_SEED = 0

# High-contrast 10-class palette specifically engineered to stand out against `PuBu` (white -> slate blue):
# Lensed classes (stars): vivid warm/neon-magenta/crimson/gold/violet hues
# Unlensed classes (circles/dots): distinct warm-earth / olive / coral / dark-charcoal / terracotta hues
HIGH_CONTRAST_CLASS_COLORS = {
    'lensed_Ia':      to_rgba('#e6194B'),  # Vivid Crimson Red
    'lensed_sn1bc':   to_rgba('#ff7f00'),  # Vivid Tangerine Orange
    'lensed_sn2l':    to_rgba('#f032e6'),  # Bright Magenta
    'lensed_sn2n':    to_rgba('#ffe119'),  # Bright Gold
    'lensed_sn2p':    to_rgba('#911eb4'),  # Rich Royal Purple
    'unlensed_Ia':    to_rgba('#2b2b2b'),  # Crisp Charcoal / Onyx
    'unlensed_sn1bc': to_rgba('#a65628'),  # Warm Sienna Brown
    'unlensed_sn2l':  to_rgba('#4daf4a'),  # Crisp Emerald Green
    'unlensed_sn2n':  to_rgba('#d95f02'),  # Rust / Burnt Orange
    'unlensed_sn2p':  to_rgba('#808000'),  # Deep Olive
}

# Binary palette: Vivid Crimson-Red Stars (Lensed) vs Crisp Gold/Amber Circles with dark rim (Unlensed)
BINARY_COLORS = {
    True:  to_rgba('#e6194B'),  # Vivid Crimson Red (Lensed)
    False: to_rgba('#f59e0b'),  # Warm Amber-Gold (Unlensed)
}


def _class_color_array():
    return [HIGH_CONTRAST_CLASS_COLORS[c] for c in CLASSES]


def _binary_legend_handles(distinguishable=False):
    if distinguishable:
        handles = [
            Line2D([0], [0], color=BINARY_COLORS[True], lw=1.8, marker='*', markersize=10,
                   markeredgecolor='black', markeredgewidth=0.6),
            Line2D([0], [0], color=BINARY_COLORS[False], lw=1.4, marker='o', markersize=6.5,
                   markeredgecolor='black', markeredgewidth=0.6),
        ]
    else:
        handles = [
            Line2D([0], [0], color=BINARY_COLORS[True], lw=1.8, marker='*', markersize=9,
                   markeredgecolor='black', markeredgewidth=0.4),
            Line2D([0], [0], color=BINARY_COLORS[False], lw=1.3, marker='o', markersize=5,
                   markeredgecolor='black', markeredgewidth=0.4),
        ]
    return handles, ['lensed', 'unlensed']


def shared_ranges(signal, background, signal_weights, background_weights,
                  pair_key, grid=TRAJECTORY_GRID, mag_mode='matched', mag_band='auto'):
    """Pooled, rate-weighted, outlier-robust colour/magnitude range across
    every grid step, for one shared axis range across all of a figure's
    panels (see :func:`cmsne.contours.robust_range`)."""
    colours, mags, weights = [], [], []
    pooled = signal + background
    sample_event = pooled[0] if pooled else None
    pooled_weights = np.r_[signal_weights, background_weights]
    for day in grid:
        suffix = _step_suffix(day)
        colour_key = f'{pair_key}{suffix}'
        magnitude_key, _ = resolve_magnitude_key(pair_key, day, mag_mode, mag_band, sample_event)
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
                     pair_key, colour_range, magnitude_range, grid=TRAJECTORY_GRID, n=200,
                     mag_mode='matched', mag_band='auto'):
    """Fit one posterior_surface per grid step and cache the (cg, mg, pg)
    grids, so every figure for this band pair reuses the same fit instead of
    re-fitting the classifier per figure."""
    surfaces = {}
    sample_event = (signal[0] if signal else (background[0] if background else None))
    for day in grid:
        suffix = _step_suffix(day)
        colour_key = f'{pair_key}{suffix}'
        magnitude_key, _ = resolve_magnitude_key(pair_key, day, mag_mode, mag_band, sample_event)
        cg, mg, pg, _ = posterior_surface(
            signal, background, colour_key, magnitude_key,
            colour_range=colour_range, magnitude_range=magnitude_range, n=n,
            signal_weights=signal_weights, background_weights=background_weights)
        surfaces[day] = (cg, mg, pg)
    return surfaces


def plot_boundary_evolution(surfaces, pair_key, out_path, signal_label, grid=TRAJECTORY_GRID,
                            mag_label='peakmag', cmap=HEATMAP_CMAP):
    fig, axes = plt.subplots(1, len(grid), figsize=(4 * len(grid), 4.2), sharex=True, sharey=True)
    mesh = None
    for ax, day in zip(axes, grid):
        cg, mg, pg = surfaces[day]
        mesh = ax.contourf(mg.T, cg.T, pg.T, levels=HEATMAP_LEVELS, cmap=cmap, vmin=0, vmax=1)
        ax.contour(mg.T, cg.T, pg.T, levels=[0.5], colors='#111111', linewidths=1.3, linestyles='--')
        ax.set_title(f'+{day:.0f}d')
        ax.set_xlabel(mag_label)
    axes[0].set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
    fig.suptitle(f'P({signal_label}) vs {pair_key[0]}-{pair_key[1]} colour -- evolution since trigger')
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, 'rate-weighted; axis range is the 1st-99th weighted percentile; dashed contour = P(0.5)',
            ha='center', fontsize=8, color='gray')
    fig.colorbar(mesh, ax=axes, shrink=0.85, label=f'P({signal_label})')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _weight_fraction(weights):
    """Map rate weights -- often spanning >10 orders of magnitude -- to a
    normalized [0, 1] fraction via a 1st-99th percentile-clipped log10 scale."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w
    wmax = np.nanmax(w) if np.isfinite(w).any() else 0.0
    if wmax <= 0:
        return np.zeros(w.shape, dtype=float)
    floor = wmax * 1e-20
    log_w = np.log10(np.clip(w, floor, None))
    lo, hi = np.percentile(log_w, [1, 99])
    if hi <= lo:
        return np.ones(w.shape, dtype=float)
    return np.clip((log_w - lo) / (hi - lo), 0.0, 1.0)


def _weight_alpha(weights, alpha_range=(0.05, 0.9)):
    frac = _weight_fraction(weights)
    return alpha_range[0] + frac * (alpha_range[1] - alpha_range[0])


def _weight_size(weights, size_range=(8.0, 160.0)):
    """Map rate weights to marker area in size_range via log10 percentile scaling."""
    frac = _weight_fraction(weights)
    return size_range[0] + (frac ** 1.6) * (size_range[1] - size_range[0])


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


def _active_classes(signal_classes):
    """Return only signal classes + unlensed background classes, excluding
    other lensed subtypes when plotting an individual lensed subtype."""
    return [c for c in CLASSES if (c in signal_classes or c in UNLENSED_CLASSES)]


def _draw_error_bars(ax, x, y, xerr, yerr, rgba, lw=0.75, zorder=2):
    """Efficiently render per-point 1-sigma measurement error bars using LineCollection,
    matching each event's own weight-scaled RGBA colour and opacity."""
    from matplotlib.collections import LineCollection
    segments = []
    seg_colors = []
    for xi, yi, xe, ye, col in zip(x, y, xerr, yerr, rgba):
        err_col = np.array(col, dtype=float)
        # Keep error bars slightly softer than the marker core so markers remain crisp
        err_col[3] = min(0.85, err_col[3] * 0.78)
        if np.isfinite(xe) and xe > 0:
            segments.append([(xi - xe, yi), (xi + xe, yi)])
            seg_colors.append(err_col)
        if np.isfinite(ye) and ye > 0:
            segments.append([(xi, yi - ye), (xi, yi + ye)])
            seg_colors.append(err_col)
    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=lw, zorder=zorder, rasterized=True)
        ax.add_collection(lc)


def _draw_weighted_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                           colour_range, magnitude_range, colors, signal_classes,
                           emphasize_signal_edge=False, zorder=1, binary_colour=False,
                           mag_mode='matched', mag_band='auto', mag_label='peakmag',
                           n_sources=None, rng=None, weight_encoding='size_alpha',
                           show_errors=False):
    """Draw every active class's weight-rated scatter onto ax_main, plus weighted,
    density-normalised marginal histograms onto ax_top (magnitude) and
    ax_right (colour).

    weight_encoding defaults to 'size_alpha' (both marker area and opacity encode
    log rate weight). If show_errors=True, draws 1-sigma photometric uncertainty
    crosses ({magnitude_key}_err, {colour_key}_err) scaled by event weight.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    is_sparse = (n_sources is not None and n_sources <= 500)

    suffix = _step_suffix(day)
    colour_key = f'{pair_key}{suffix}'
    colour_err_key = f'{colour_key}_err'
    first_ev = next((pop[c][0] for c in CLASSES if pop[c]), None)
    magnitude_key, _ = resolve_magnitude_key(pair_key, day, mag_mode, mag_band, first_ev)
    magnitude_err_key = f'{magnitude_key}_err'

    mag_bins = np.linspace(*magnitude_range, N_MARGINAL_BINS + 1)
    colour_bins = np.linspace(*colour_range, N_MARGINAL_BINS + 1)
    pooled = {True: {'mags': [], 'colours': [], 'weights': []},
             False: {'mags': [], 'colours': [], 'weights': []}}

    active = _active_classes(signal_classes)
    for i, c in enumerate(CLASSES):
        if c not in active:
            continue
        events = pop[c]
        colours = np.array([e.get(colour_key, np.nan) for e in events])
        mags = np.array([e.get(magnitude_key, np.nan) for e in events])
        c_errs = np.array([e.get(colour_err_key, np.nan) for e in events])
        m_errs = np.array([e.get(magnitude_err_key, np.nan) for e in events])
        weights = np.array([e['weight'] for e in events])

        finite = np.isfinite(colours) & np.isfinite(mags)
        colours, mags, weights = colours[finite], mags[finite], weights[finite]
        c_errs, m_errs = c_errs[finite], m_errs[finite]
        if colours.size == 0:
            continue

        is_signal = c in signal_classes
        point_color = BINARY_COLORS[is_signal] if binary_colour else colors[i]

        # Marginal histograms: always computed on the FULL weighted class population
        if binary_colour:
            pooled[is_signal]['mags'].append(mags)
            pooled[is_signal]['colours'].append(colours)
            pooled[is_signal]['weights'].append(weights)
        elif weights.sum() > 0:
            lw = 1.8 if is_signal else 1.0
            ax_top.hist(mags, bins=mag_bins, weights=weights, density=True,
                       histtype='step', color=point_color, linewidth=lw)
            ax_right.hist(colours, bins=colour_bins, weights=weights, density=True,
                         histtype='step', orientation='horizontal', color=point_color, linewidth=lw)

        # Scatter points: optionally subsampled to n_sources per class
        if n_sources is not None and n_sources < colours.size:
            wsum = weights.sum()
            if wsum > 0:
                p = weights / wsum
                n_pos = np.count_nonzero(weights > 0)
                if n_pos >= n_sources:
                    idx = rng.choice(colours.size, size=n_sources, replace=False, p=p)
                else:
                    idx = rng.choice(colours.size, size=n_sources, replace=True, p=p)
            else:
                idx = rng.choice(colours.size, size=min(n_sources, colours.size), replace=False)
            draw_colours, draw_mags, draw_weights = colours[idx], mags[idx], weights[idx]
            draw_cerrs, draw_merrs = c_errs[idx], m_errs[idx]
        else:
            draw_colours, draw_mags, draw_weights = colours, mags, weights
            draw_cerrs, draw_merrs = c_errs, m_errs

        # Sort by weight ascending so heavier events are drawn on top
        w_order = np.argsort(draw_weights)
        draw_colours = draw_colours[w_order]
        draw_mags = draw_mags[w_order]
        draw_weights = draw_weights[w_order]
        draw_cerrs = draw_cerrs[w_order]
        draw_merrs = draw_merrs[w_order]

        frac = _weight_fraction(draw_weights)

        if weight_encoding == 'size':
            s_range = (14, 190) if is_signal else (7, 95)
            if not is_sparse:
                s_range = (4, 65) if is_signal else (1.5, 28)
            s = _weight_size(draw_weights, size_range=s_range)
            alpha_val = 0.88 if is_signal else 0.72
            rgba = np.tile(point_color, (len(frac), 1))
            rgba[:, 3] = alpha_val
            marker = '*' if is_signal else 'o'
            edgecolors = '#111111' if (is_sparse or is_signal) else 'none'
            linewidths = 0.5 if is_sparse else 0.25
            zo = zorder + (3 if is_signal else 1)
        elif weight_encoding == 'colour':
            cmap = plt.cm.YlOrRd if is_signal else plt.cm.YlGn
            rgba = cmap(0.25 + 0.75 * frac)
            rgba[:, 3] = 0.90 if is_signal else 0.75
            marker = '*' if is_signal else 'o'
            s = (65 if is_signal else 26) if is_sparse else (18 if is_signal else 5)
            edgecolors = '#111111' if is_sparse else 'none'
            linewidths = 0.45 if is_sparse else 0.2
            zo = zorder + (3 if is_signal else 1)
        elif weight_encoding == 'size_alpha':
            if is_sparse:
                s = _weight_size(draw_weights, size_range=(15, 200) if is_signal else (8, 95))
                alphas = _weight_alpha(draw_weights, alpha_range=(0.28, 0.98) if is_signal else (0.18, 0.88))
                marker = '*' if is_signal else 'o'
                edgecolors = '#111111'
                linewidths = 0.55 if is_signal else 0.40
            else:
                s = _weight_size(draw_weights, size_range=(3.0, 58.0) if is_signal else (1.0, 22.0))
                alphas = _weight_alpha(draw_weights, alpha_range=(0.06, 0.92) if is_signal else (0.02, 0.65))
                marker = '*' if is_signal else 'o'
                edgecolors = '#111111' if (is_signal and emphasize_signal_edge) else 'none'
                linewidths = 0.25
            rgba = np.tile(point_color, (len(alphas), 1))
            rgba[:, 3] = alphas
            zo = zorder + (3 if is_signal else 1)
        else:  # 'alpha'
            if is_sparse:
                alphas = _weight_alpha(draw_weights, alpha_range=(0.22, 0.98) if is_signal else (0.16, 0.88))
                rgba = np.tile(point_color, (len(alphas), 1))
                rgba[:, 3] = alphas
                marker = '*' if is_signal else 'o'
                s = 55 if is_signal else 24
                edgecolors = '#111111'
                linewidths = 0.5
                zo = zorder + (3 if is_signal else 1)
            else:
                alphas = _weight_alpha(draw_weights, alpha_range=(0.08, 0.9) if is_signal else (0.02, 0.6))
                rgba = np.tile(point_color, (len(alphas), 1))
                rgba[:, 3] = alphas
                marker = '*' if is_signal else '.'
                s = 16 if is_signal else 3
                edgecolors = '#111111' if (is_signal and emphasize_signal_edge) else 'none'
                linewidths = 0.3
                zo = zorder + (2 if is_signal else 1)

        if show_errors:
            err_lw = (0.95 if is_signal else 0.65) if is_sparse else (0.45 if is_signal else 0.25)
            _draw_error_bars(ax_main, draw_mags, draw_colours, draw_merrs, draw_cerrs,
                             rgba, lw=err_lw, zorder=zo - 0.5)

        # Scale per-point edge opacity with marker alpha so low-weight pinpoints don't show dark rings
        if isinstance(edgecolors, str) and edgecolors != 'none':
            edge_rgba = np.tile(to_rgba(edgecolors), (len(rgba), 1))
            edge_rgba[:, 3] = rgba[:, 3]
        else:
            edge_rgba = edgecolors

        ax_main.scatter(draw_mags, draw_colours, s=s,
                        marker=marker, c=rgba,
                        edgecolors=edge_rgba, linewidths=linewidths,
                        rasterized=True, zorder=zo)

    if binary_colour:
        for is_signal in (True, False):
            group = pooled[is_signal]
            if not group['mags']:
                continue
            group_weights = np.concatenate(group['weights'])
            if group_weights.sum() <= 0:
                continue
            group_mags = np.concatenate(group['mags'])
            group_colours = np.concatenate(group['colours'])
            lw = 1.8 if is_signal else 1.1
            ax_top.hist(group_mags, bins=mag_bins, weights=group_weights, density=True,
                       histtype='step', color=BINARY_COLORS[is_signal], linewidth=lw)
            ax_right.hist(group_colours, bins=colour_bins, weights=group_weights, density=True,
                         histtype='step', orientation='horizontal', color=BINARY_COLORS[is_signal], linewidth=lw)

    ax_main.set_xlim(*magnitude_range)
    ax_main.set_ylim(*colour_range)
    ax_main.set_xlabel(mag_label)
    ax_top.set_title(f'+{day:.0f}d')


def _class_legend_handles(colors, signal_classes, distinguishable=False):
    active = _active_classes(signal_classes)
    handles, labels = [], []
    for i, c in enumerate(CLASSES):
        if c not in active:
            continue
        is_sig = c in signal_classes
        h = Line2D([0], [0], color=colors[i],
                   lw=1.8 if is_sig else 1.1,
                   marker='*' if is_sig else 'o',
                   markersize=(9.5 if is_sig else 6.0) if distinguishable else (8.5 if is_sig else 4.5),
                   markeredgecolor='#111111', markeredgewidth=0.5)
        handles.append(h)
        labels.append(c)
    return handles, labels


def plot_weighted_sources_corner(pop, pair_key, out_path, colour_range, magnitude_range,
                                 signal_classes, signal_label, grid=TRAJECTORY_GRID,
                                 binary_colour=False, mag_mode='matched', mag_band='auto',
                                 mag_label='peakmag', n_sources=None, seed=RESAMPLE_SEED,
                                 weight_encoding='size_alpha', show_errors=False):
    """Weight-rated corner-style companion to the boundary figure."""
    fig = plt.figure(figsize=(4.2 * len(grid), 5.2))
    outer = fig.add_gridspec(1, len(grid), wspace=0.35)
    colors = _class_color_array()
    rng = np.random.default_rng(seed)
    is_sparse = (n_sources is not None and n_sources <= 500)

    for col, day in enumerate(grid):
        ax_main, ax_top, ax_right = _corner_panel(fig, outer[0, col])
        ax_main.set_facecolor('#f8fafc')
        _draw_weighted_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                               colour_range, magnitude_range, colors, signal_classes,
                               binary_colour=binary_colour, mag_mode=mag_mode,
                               mag_band=mag_band, mag_label=mag_label,
                               n_sources=n_sources, rng=rng,
                               weight_encoding=weight_encoding,
                               show_errors=show_errors)
        if col == 0:
            ax_main.set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
        else:
            ax_main.tick_params(labelleft=False)

    if binary_colour:
        handles, labels = _binary_legend_handles(distinguishable=is_sparse)
    else:
        handles, labels = _class_legend_handles(colors, signal_classes, distinguishable=is_sparse)
    fig.legend(handles, labels, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.90, 0.88))
    binary_note = ' (binary: lensed vs unlensed)' if binary_colour else ''
    sample_note = f' ({n_sources}/class displayed)' if n_sources is not None else ''
    err_note = ' + 1σ photometric error bars' if show_errors else ''
    enc_desc = {
        'alpha': 'point alpha ~ log rate weight',
        'size': 'marker size ~ log rate weight',
        'colour': 'marker colour shade ~ log rate weight',
        'size_alpha': 'marker size + opacity ~ log rate weight',
    }.get(weight_encoding, 'marker size + opacity ~ log rate weight')
    fig.suptitle(f'Weight-rated discrete sources behind the {pair_key} boundary figure '
                f'(signal: {signal_label}){binary_note}{sample_note}{err_note}\n'
                f'({enc_desc}; top/right: weighted class histograms)',
                fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.89, top=0.84, bottom=0.08)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _draw_resampled_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                            colour_range, magnitude_range, colors, rng, signal_classes,
                            n_resample=N_RESAMPLE, binary_colour=False,
                            mag_mode='matched', mag_band='auto', mag_label='peakmag',
                            show_errors=False):
    """Draw each active class as a weighted bootstrap resample."""
    is_sparse = (n_resample <= 500)
    suffix = _step_suffix(day)
    colour_key = f'{pair_key}{suffix}'
    colour_err_key = f'{colour_key}_err'
    first_ev = next((pop[c][0] for c in CLASSES if pop[c]), None)
    magnitude_key, _ = resolve_magnitude_key(pair_key, day, mag_mode, mag_band, first_ev)
    magnitude_err_key = f'{magnitude_key}_err'

    mag_bins = np.linspace(*magnitude_range, N_MARGINAL_BINS + 1)
    colour_bins = np.linspace(*colour_range, N_MARGINAL_BINS + 1)
    pooled = {True: {'mags': [], 'colours': []}, False: {'mags': [], 'colours': []}}

    active = _active_classes(signal_classes)
    for i, c in enumerate(CLASSES):
        if c not in active:
            continue
        events = pop[c]
        colours = np.array([e.get(colour_key, np.nan) for e in events])
        mags = np.array([e.get(magnitude_key, np.nan) for e in events])
        c_errs = np.array([e.get(colour_err_key, np.nan) for e in events])
        m_errs = np.array([e.get(magnitude_err_key, np.nan) for e in events])
        weights = np.array([e['weight'] for e in events])
        finite = np.isfinite(colours) & np.isfinite(mags)
        colours, mags, weights = colours[finite], mags[finite], weights[finite]
        c_errs, m_errs = c_errs[finite], m_errs[finite]
        wsum = weights.sum()
        if colours.size == 0 or wsum <= 0:
            continue
        idx = rng.choice(colours.size, size=n_resample, replace=True, p=weights / wsum)
        rcolours, rmags = colours[idx], mags[idx]
        rcerrs, rmerrs = c_errs[idx], m_errs[idx]
        is_signal = c in signal_classes
        point_color = BINARY_COLORS[is_signal] if binary_colour else colors[i]

        if is_sparse:
            marker = '*' if is_signal else 'o'
            s = 55 if is_signal else 22
            alpha = 0.78 if is_signal else 0.58
            edgecolors = '#111111'
            linewidths = 0.5
            zo = 3 if is_signal else 1
        else:
            marker = '*' if is_signal else 'o'
            s = 18 if is_signal else 5
            alpha = 0.42 if is_signal else 0.22
            edgecolors = 'none'
            linewidths = 0.25
            zo = 2 if is_signal else 1

        if show_errors:
            rgba_arr = np.tile(point_color, (len(rmags), 1))
            rgba_arr[:, 3] = alpha
            _draw_error_bars(ax_main, rmags, rcolours, rmerrs, rcerrs, rgba_arr,
                             lw=0.75 if is_sparse else 0.3, zorder=zo - 0.5)

        ax_main.scatter(rmags, rcolours, s=s,
                       marker=marker, color=point_color,
                       edgecolors=edgecolors, linewidths=linewidths,
                       alpha=alpha, rasterized=True, zorder=zo)
        if binary_colour:
            pooled[is_signal]['mags'].append(rmags)
            pooled[is_signal]['colours'].append(rcolours)
        else:
            lw = 1.8 if is_signal else 1.0
            ax_top.hist(rmags, bins=mag_bins, density=True, histtype='step',
                       color=point_color, linewidth=lw)
            ax_right.hist(rcolours, bins=colour_bins, density=True, histtype='step',
                         orientation='horizontal', color=point_color, linewidth=lw)

    if binary_colour:
        for is_signal in (True, False):
            group = pooled[is_signal]
            if not group['mags']:
                continue
            lw = 1.8 if is_signal else 1.1
            ax_top.hist(np.concatenate(group['mags']), bins=mag_bins, density=True,
                       histtype='step', color=BINARY_COLORS[is_signal], linewidth=lw)
            ax_right.hist(np.concatenate(group['colours']), bins=colour_bins, density=True,
                         histtype='step', orientation='horizontal',
                         color=BINARY_COLORS[is_signal], linewidth=lw)

    ax_main.set_xlim(*magnitude_range)
    ax_main.set_ylim(*colour_range)
    ax_main.set_xlabel(mag_label)
    ax_top.set_title(f'+{day:.0f}d')


def plot_resampled_sources_corner(pop, pair_key, out_path, colour_range, magnitude_range,
                                  signal_classes, signal_label, grid=TRAJECTORY_GRID,
                                  n_resample=N_RESAMPLE, seed=RESAMPLE_SEED, binary_colour=False,
                                  mag_mode='matched', mag_band='auto', mag_label='peakmag',
                                  show_errors=False):
    fig = plt.figure(figsize=(4.2 * len(grid), 5.2))
    outer = fig.add_gridspec(1, len(grid), wspace=0.35)
    colors = _class_color_array()
    rng = np.random.default_rng(seed)
    is_sparse = (n_resample <= 500)

    for col, day in enumerate(grid):
        ax_main, ax_top, ax_right = _corner_panel(fig, outer[0, col])
        ax_main.set_facecolor('#f8fafc')
        _draw_resampled_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                                colour_range, magnitude_range, colors, rng, signal_classes,
                                n_resample=n_resample, binary_colour=binary_colour,
                                mag_mode=mag_mode, mag_band=mag_band, mag_label=mag_label,
                                show_errors=show_errors)
        if col == 0:
            ax_main.set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
        else:
            ax_main.tick_params(labelleft=False)

    if binary_colour:
        handles, labels = _binary_legend_handles(distinguishable=is_sparse)
    else:
        handles, labels = _class_legend_handles(colors, signal_classes, distinguishable=is_sparse)
    fig.legend(handles, labels, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.90, 0.88))
    binary_note = ' (binary: lensed vs unlensed)' if binary_colour else ''
    err_note = ' + 1σ error bars' if show_errors else ''
    fig.suptitle(f'Weighted resample ({n_resample}/class/step) of discrete sources behind '
                f'the {pair_key} boundary figure (signal: {signal_label}){binary_note}{err_note}\n'
                f'(point density ~ rate weight; top/right: resampled class histograms)',
                fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.89, top=0.84, bottom=0.08)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_corner_with_heatmap(pop, surfaces, pair_key, out_path, colour_range, magnitude_range,
                             signal_classes, signal_label, grid=TRAJECTORY_GRID, binary_colour=False,
                             mag_mode='matched', mag_band='auto', mag_label='peakmag',
                             n_sources=None, seed=RESAMPLE_SEED, weight_encoding='size_alpha',
                             show_errors=False, cmap=HEATMAP_CMAP):
    """Same corner layout as plot_weighted_sources_corner, with the already-
    fitted P(signal) posterior surface underlaid in each main panel using a
    cool, low-collision `PuBu` gradient so high-contrast warm/vivid points
    stand out cleanly across [0, 1]."""
    fig = plt.figure(figsize=(4.2 * len(grid), 5.2))
    outer = fig.add_gridspec(1, len(grid), wspace=0.35)
    colors = _class_color_array()
    rng = np.random.default_rng(seed)
    is_sparse = (n_sources is not None and n_sources <= 500)
    mesh = None

    for col, day in enumerate(grid):
        ax_main, ax_top, ax_right = _corner_panel(fig, outer[0, col])
        cg, mg, pg = surfaces[day]
        mesh = ax_main.contourf(mg.T, cg.T, pg.T, levels=HEATMAP_LEVELS, cmap=cmap, vmin=0, vmax=1, zorder=0)
        ax_main.contour(mg.T, cg.T, pg.T, levels=[0.5], colors='#111111', linewidths=1.2, linestyles='--', zorder=1)
        _draw_weighted_sources(ax_main, ax_top, ax_right, pop, pair_key, day,
                               colour_range, magnitude_range, colors, signal_classes,
                               emphasize_signal_edge=True, zorder=2, binary_colour=binary_colour,
                               mag_mode=mag_mode, mag_band=mag_band, mag_label=mag_label,
                               n_sources=n_sources, rng=rng,
                               weight_encoding=weight_encoding,
                               show_errors=show_errors)
        if col == 0:
            ax_main.set_ylabel(f'{pair_key[0]}-{pair_key[1]}')
        else:
            ax_main.tick_params(labelleft=False)

    cax = fig.add_axes((0.91, 0.55, 0.012, 0.28))
    fig.colorbar(mesh, cax=cax, label=f'P({signal_label})')
    if binary_colour:
        handles, labels = _binary_legend_handles(distinguishable=is_sparse)
    else:
        handles, labels = _class_legend_handles(colors, signal_classes, distinguishable=is_sparse)
    fig.legend(handles, labels, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.91, 0.42))
    binary_note = ' (binary: lensed vs unlensed)' if binary_colour else ''
    sample_note = f' ({n_sources}/class)' if n_sources is not None else ''
    err_note = ' + 1σ Photometric Error Bars' if show_errors else ''
    fig.suptitle(f'P({signal_label}) heatmap + size/opacity-weighted sources -- '
                f'{pair_key[0]}-{pair_key[1]} vs {mag_label}{binary_note}{sample_note}{err_note}', fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.89, top=0.84, bottom=0.08)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='directory of per-class <name>.pkl files')
    ap.add_argument('--out-dir', default='docs/figures')
    ap.add_argument('--n', type=int, default=200, help='mesh resolution per axis')
    ap.add_argument('--target', choices=['lensed_ia', 'all_lensed', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n', 'lensed_sn2p'],
                    default='lensed_ia',
                    help="signal definition: 'lensed_ia' (default, lensed_Ia only vs unlensed_*), "
                         "'all_lensed' (every lensed_* class vs unlensed_*), or a specific lensed subtype")
    ap.add_argument('--mag-mode', choices=['matched', 'peakmag', 'snapshot'], default='matched',
                    help="magnitude feature mode: 'matched' (default), 'peakmag', or 'snapshot'")
    ap.add_argument('--mag-band', default='auto',
                    help="override companion magnitude band (default: 'auto', redder band of pair)")
    ap.add_argument('--pairs', default=','.join(PAIR_KEYS),
                    help=f"comma-separated colour pairs to process (default: '{','.join(PAIR_KEYS)}')")
    ap.add_argument('--n-sources', type=int, default=None,
                    help="limit number of rate-weighted events displayed per class in sources/corner plots")
    ap.add_argument('--n-resample', type=int, default=N_RESAMPLE,
                    help=f"number of resampled events per class in resampled plots (default: {N_RESAMPLE})")
    ap.add_argument('--seed', type=int, default=RESAMPLE_SEED,
                    help=f"random seed for sampling (default: {RESAMPLE_SEED})")
    ap.add_argument('--weight-encoding', choices=['size_alpha', 'alpha', 'size', 'colour'], default='size_alpha',
                    help="visual encoding for event rate weights in sources/corner plots (default: 'size_alpha')")
    ap.add_argument('--show-errors', action='store_true',
                    help="render 1-sigma photometric error bars on scattered events")
    ap.add_argument('--also-plot-errors', action='store_true',
                    help="in addition to standard figures, write `*_errorbars.png` companion figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.target == 'all_lensed':
        signal_classes = LENSED_CLASSES
        signal_label, file_suffix = 'lensed', '_alllensed'
    elif args.target == 'lensed_ia':
        signal_classes = ['lensed_Ia']
        signal_label, file_suffix = 'lensed Ia', ''
    else:
        signal_classes = [args.target]
        signal_label, file_suffix = args.target.replace('_', ' '), f'_{args.target}'

    pop = {}
    for c in CLASSES:
        with open(os.path.join(args.data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']
    signal = [e for c in signal_classes for e in pop[c]]
    background = [e for c in UNLENSED_CLASSES for e in pop[c]]
    signal_weights = np.array([e['weight'] for e in signal])
    background_weights = np.array([e['weight'] for e in background])
    print(f'signal ({" + ".join(signal_classes)}): {len(signal)}, background (unlensed_* only): {len(background)}')

    selected_pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
    for pair_key in selected_pairs:
        sample_event = signal[0] if signal else (background[0] if background else None)
        _, mag_label = resolve_magnitude_key(pair_key, 0.0, args.mag_mode, args.mag_band, sample_event)

        boundary_path = os.path.join(args.out_dir, f'boundary_{pair_key}{file_suffix}.png')
        sources_path = os.path.join(args.out_dir, f'boundary_{pair_key}_sources{file_suffix}.png')
        corner_path = os.path.join(args.out_dir, f'boundary_{pair_key}_corner{file_suffix}.png')
        resampled_path = os.path.join(args.out_dir, f'boundary_{pair_key}_resampled{file_suffix}.png')
        print(f'fitting {pair_key} ({len(TRAJECTORY_GRID)} grid steps, {mag_label})...')
        try:
            colour_range, magnitude_range = shared_ranges(
                signal, background, signal_weights, background_weights, pair_key,
                mag_mode=args.mag_mode, mag_band=args.mag_band)
        except ValueError as exc:
            print(f'skipping {pair_key}: {exc}')
            continue

        surfaces = compute_surfaces(signal, background, signal_weights, background_weights,
                                    pair_key, colour_range, magnitude_range, n=args.n,
                                    mag_mode=args.mag_mode, mag_band=args.mag_band)

        plot_boundary_evolution(surfaces, pair_key, boundary_path, signal_label,
                                mag_label=mag_label)
        print(f'wrote {boundary_path}')

        plot_weighted_sources_corner(pop, pair_key, sources_path, colour_range, magnitude_range,
                                     signal_classes, signal_label,
                                     mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                     n_sources=args.n_sources, seed=args.seed,
                                     weight_encoding=args.weight_encoding,
                                     show_errors=args.show_errors)
        print(f'wrote {sources_path}')

        plot_corner_with_heatmap(pop, surfaces, pair_key, corner_path, colour_range, magnitude_range,
                                 signal_classes, signal_label,
                                 mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                 n_sources=args.n_sources, seed=args.seed,
                                 weight_encoding=args.weight_encoding,
                                 show_errors=args.show_errors)
        print(f'wrote {corner_path}')

        plot_resampled_sources_corner(pop, pair_key, resampled_path, colour_range, magnitude_range,
                                      signal_classes, signal_label,
                                      mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                      n_resample=args.n_resample, seed=args.seed,
                                      show_errors=args.show_errors)
        print(f'wrote {resampled_path}')

        if args.also_plot_errors and not args.show_errors:
            err_sources = args.n_sources if (args.n_sources is not None and args.n_sources <= 150) else 100
            corner_err_path = os.path.join(args.out_dir, f'boundary_{pair_key}_corner{file_suffix}_errorbars.png')
            sources_err_path = os.path.join(args.out_dir, f'boundary_{pair_key}_sources{file_suffix}_errorbars.png')
            plot_corner_with_heatmap(pop, surfaces, pair_key, corner_err_path, colour_range, magnitude_range,
                                     signal_classes, signal_label,
                                     mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                     n_sources=err_sources, seed=args.seed,
                                     weight_encoding=args.weight_encoding,
                                     show_errors=True)
            print(f'wrote {corner_err_path}')
            plot_weighted_sources_corner(pop, pair_key, sources_err_path, colour_range, magnitude_range,
                                         signal_classes, signal_label,
                                         mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                         n_sources=err_sources, seed=args.seed,
                                         weight_encoding=args.weight_encoding,
                                         show_errors=True)
            print(f'wrote {sources_err_path}')

        if args.target == 'all_lensed':
            sources_bin_path = os.path.join(args.out_dir, f'boundary_{pair_key}_sources{file_suffix}_binary.png')
            corner_bin_path = os.path.join(args.out_dir, f'boundary_{pair_key}_corner{file_suffix}_binary.png')
            resampled_bin_path = os.path.join(args.out_dir, f'boundary_{pair_key}_resampled{file_suffix}_binary.png')

            plot_weighted_sources_corner(pop, pair_key, sources_bin_path, colour_range, magnitude_range,
                                         signal_classes, signal_label, binary_colour=True,
                                         mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                         n_sources=args.n_sources, seed=args.seed,
                                         weight_encoding=args.weight_encoding,
                                         show_errors=args.show_errors)
            print(f'wrote {sources_bin_path}')

            plot_corner_with_heatmap(pop, surfaces, pair_key, corner_bin_path, colour_range, magnitude_range,
                                     signal_classes, signal_label, binary_colour=True,
                                     mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                     n_sources=args.n_sources, seed=args.seed,
                                     weight_encoding=args.weight_encoding,
                                     show_errors=args.show_errors)
            print(f'wrote {corner_bin_path}')

            plot_resampled_sources_corner(pop, pair_key, resampled_bin_path, colour_range, magnitude_range,
                                          signal_classes, signal_label, binary_colour=True,
                                          mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                          n_resample=args.n_resample, seed=args.seed,
                                          show_errors=args.show_errors)
            print(f'wrote {resampled_bin_path}')

            if args.also_plot_errors and not args.show_errors:
                err_sources = args.n_sources if (args.n_sources is not None and args.n_sources <= 150) else 100
                corner_bin_err_path = os.path.join(args.out_dir, f'boundary_{pair_key}_corner{file_suffix}_binary_errorbars.png')
                plot_corner_with_heatmap(pop, surfaces, pair_key, corner_bin_err_path, colour_range, magnitude_range,
                                         signal_classes, signal_label, binary_colour=True,
                                         mag_mode=args.mag_mode, mag_band=args.mag_band, mag_label=mag_label,
                                         n_sources=err_sources, seed=args.seed,
                                         weight_encoding=args.weight_encoding,
                                         show_errors=True)
                print(f'wrote {corner_bin_err_path}')


if __name__ == '__main__':
    main()
