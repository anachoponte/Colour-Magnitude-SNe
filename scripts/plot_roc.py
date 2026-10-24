"""ROC curves for the static vs trajectory classifiers, including a
time-evolving view as more of the trajectory is revealed (issue #17).

Loads a dual-schema production population (see ``run_dual_schema.py`` /
``cmsne.trajectory.DualSchemaGenerator`` -- one ``<class>.pkl`` per class,
each holding ``events`` + ``n_attempted``), fits both classifiers on a 70/30
held-out split (same split/seed convention as the real-data analyses done
for issue #16's investigation), and produces:

- ``roc_overall.png``   -- static vs trajectory ROC on the fully-revealed
                           feature vector.
- ``roc_by_step.png``   -- one ROC curve per Δt-since-trigger grid step for
                           the trajectory classifier, overlaid.
- ``roc_by_step.gif``   -- the same, animated: each frame adds the next
                           grid step's curve.

``--target`` chooses which classes count as signal: the default
``lensed_ia`` treats only ``lensed_Ia`` as signal (unchanged filenames).
``all_lensed`` pools all five ``lensed_*`` classes as signal and all five
``unlensed_*`` classes as background (see the same rationale in
``plot_evolving_boundaries.py``'s module docstring), writing to filenames
with an ``_alllensed`` suffix so both targets' outputs can coexist.

Under ``--target all_lensed`` only, three more evaluations run against the
SAME already-fit TrajectoryClassifier (pooled signal, boundary fit once --
nothing below refits it):

- ``roc_by_subtype_alllensed.png`` -- one ROC curve per lensed subtype
  (``lensed_Ia``, ``lensed_sn1bc``, ...), each subtype's own held-out test
  events scored as "signal" against the SAME unlensed background test set.
  Other lensed subtypes never enter a given subtype's curve, so they are
  never counted as false positives for it.
- ``roc_by_step_<subtype>_alllensed.png``/``.gif`` -- the same per-subtype
  idea, but swept over Δt-since-trigger grid steps like ``roc_by_step``/
  ``roc_by_step.gif``, one file pair per lensed subtype.
- ``rate_by_subtype_alllensed.png``/``.gif`` -- weighted recovered rate
  (lensed subtypes) and false-positive rate (unlensed classes) at the
  classifier's already-pinned ``target_fpr`` threshold, per class per grid
  step. This codebase's per-event ``weight`` is a relative/shape quantity,
  not calibrated to an absolute survey rate (see
  ``cmsne/sn_rates.py``/``cmsne/colour_magnitude.py``'s weight functions),
  so these are relative weighted-rate units -- Σ weight of flagged events --
  not literal events/year.

    python scripts/plot_roc.py --data-dir results/dual_58563115 --out-dir docs/figures
    python scripts/plot_roc.py --data-dir results/dual_58563115 --out-dir docs/figures --target all_lensed
"""
import argparse, io, os, pickle, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.classifier import ColourClassifier
from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID, reveal_up_to

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']

# Same tab10 indexing as plot_evolving_boundaries.py's identically-ordered
# CLASSES list, so a given class's colour matches across every figure.
CLASS_COLORS = dict(zip(CLASSES, plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))))


def load_population(data_dir, signal_classes):
    """Load every class's pkl, split into signal (signal_classes) vs background (rest)."""
    pop = {}
    for c in CLASSES:
        with open(os.path.join(data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']
    signal = [e for c in signal_classes for e in pop[c]]
    background = [e for c in CLASSES if c not in signal_classes for e in pop[c]]
    return signal, background


def split(events, rng, frac_train=0.7):
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    return [events[i] for i in idx[:n_train]], [events[i] for i in idx[n_train:]]


def auc(fpr, tpr):
    fpr, tpr = np.asarray(fpr, float), np.asarray(tpr, float)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def plot_overall_roc(static_roc, traj_roc, out_path, signal_label):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (fpr, tpr, _), label in [(static_roc, 'static'), (traj_roc, 'trajectory')]:
        ax.plot(fpr, tpr, label=f'{label} (AUC={auc(fpr, tpr):.3f})', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='chance')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate (recovery)')
    ax.set_title(f'Static vs trajectory classifier ROC (held-out, rate-weighted)\nsignal: {signal_label}')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _step_colour(i, n_steps):
    return cm.viridis(i / max(n_steps - 1, 1))


def plot_roc_by_step(by_step, out_path, signal_label):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, day in enumerate(TRAJECTORY_GRID):
        fpr, tpr, _ = by_step[day]
        ax.plot(fpr, tpr, color=_step_colour(i, len(TRAJECTORY_GRID)), lw=2,
                label=f'+{day:.0f}d (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='chance')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate (recovery)')
    ax.set_title(f'Trajectory classifier ROC vs time-since-trigger (held-out)\nsignal: {signal_label}')
    ax.legend(loc='lower right', title='Δt revealed')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_roc_gif(by_step, out_path, signal_label, duration_ms=900):
    """One frame per grid step, each adding that step's curve to the ones before."""
    frames = []
    n_steps = len(TRAJECTORY_GRID)
    for k, day in enumerate(TRAJECTORY_GRID):
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(k + 1):
            prior_day = TRAJECTORY_GRID[i]
            fpr, tpr, _ = by_step[prior_day]
            style = dict(color=_step_colour(i, n_steps), lw=3 if i == k else 1.5,
                        alpha=1.0 if i == k else 0.4)
            ax.plot(fpr, tpr, label=f'+{prior_day:.0f}d (AUC={auc(fpr, tpr):.3f})', **style)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate (recovery)')
        ax.set_title(f'Trajectory classifier ROC -- Δt since trigger: +{day:.0f}d\nsignal: {signal_label}')
        ax.legend(loc='lower right', title='Δt revealed')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE))

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)


def plot_roc_by_subtype(subtype_roc, signal_classes, out_path):
    """Trajectory classifier ROC (fully revealed), one curve per lensed
    subtype. Each subtype's own held-out test events are scored as "signal"
    against the SAME unlensed background test set, using the SAME already-
    fit all_lensed classifier -- the decision boundary is not refit per
    subtype, and other lensed subtypes never enter a given subtype's curve
    (so they are never counted as false positives for it).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for c in signal_classes:
        fpr, tpr, _ = subtype_roc[c]
        ax.plot(fpr, tpr, color=CLASS_COLORS[c], lw=2, label=f'{c} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='chance')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate (recovery)')
    ax.set_title('Trajectory classifier ROC per lensed subtype (held-out)\n'
                'signal: all lensed (boundary fit once, pooled)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _rate_ylim(rate_by_class, classes):
    """Shared y-limits across a set of classes' per-step rates, for a
    log-scale axis that stays fixed across plot_rate_by_subtype's two
    panels and every make_rate_gif frame -- otherwise each frame's y-axis
    would auto-rescale as more steps are revealed, which reads as the data
    changing rather than just more of it appearing.
    """
    vals = [v for c in classes for v in rate_by_class[c].values() if v > 0]
    if not vals:
        return 1e-10, 1.0
    return min(vals) * 0.5, max(vals) * 2.0


def _draw_rate_panel(ax, rate_by_class, classes, days, marker, title, ylabel, ylim):
    for c in classes:
        ys = [rate_by_class[c][d] for d in days]
        ax.plot(days, ys, marker=marker, ms=8, lw=1.6, color=CLASS_COLORS[c], label=c)
    ax.set_yscale('log')
    ax.set_ylim(*ylim)
    ax.set_xlabel('Δt since trigger (days)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)


def plot_rate_by_subtype(rate_by_class, signal_classes, out_path):
    """Weighted recovered/false-positive rate per class vs Δt-since-trigger,
    at the TrajectoryClassifier's already-pinned target_fpr threshold (not
    swept like an ROC curve). Relative units -- Σ rate-weight of flagged
    events -- not an absolute events/year calibration; see this module's
    docstring. Left panel: lensed subtypes (recovered rate, true
    positives). Right panel: unlensed classes (false-positive rate). A
    class/step with zero flagged weight is simply absent from its
    log-scale line (matplotlib drops non-positive points), not an error.
    """
    background_classes = [c for c in CLASSES if c not in signal_classes]
    days = list(TRAJECTORY_GRID)
    ylim_sig = _rate_ylim(rate_by_class, signal_classes)
    ylim_bg = _rate_ylim(rate_by_class, background_classes)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
    _draw_rate_panel(ax_l, rate_by_class, signal_classes, days, '*',
                     'Recovered rate -- lensed subtypes',
                     'weighted recovered rate (relative units)', ylim_sig)
    _draw_rate_panel(ax_r, rate_by_class, background_classes, days, '.',
                     'False-positive rate -- unlensed classes',
                     'weighted false-positive rate (relative units)', ylim_bg)
    fig.suptitle('Weighted recovered / false-positive rate vs time-since-trigger '
                '(all_lensed target, relative units)')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_rate_gif(rate_by_class, signal_classes, out_path, duration_ms=900):
    """Animated version of plot_rate_by_subtype: each frame reveals one more
    Δt-since-trigger grid step (x-truncated to the steps shown so far),
    mirroring make_roc_gif's progressive reveal. Y-limits are fixed across
    frames (see _rate_ylim) so only the revealed data changes, not the
    axis scale.
    """
    background_classes = [c for c in CLASSES if c not in signal_classes]
    days = list(TRAJECTORY_GRID)
    ylim_sig = _rate_ylim(rate_by_class, signal_classes)
    ylim_bg = _rate_ylim(rate_by_class, background_classes)
    frames = []

    for k in range(len(days)):
        shown = days[:k + 1]
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
        _draw_rate_panel(ax_l, rate_by_class, signal_classes, shown, '*',
                         'Recovered rate -- lensed subtypes',
                         'weighted recovered rate (relative units)', ylim_sig)
        _draw_rate_panel(ax_r, rate_by_class, background_classes, shown, '.',
                         'False-positive rate -- unlensed classes',
                         'weighted false-positive rate (relative units)', ylim_bg)
        ax_l.set_xlim(days[0], days[-1])
        ax_r.set_xlim(days[0], days[-1])
        fig.suptitle(f'Weighted recovered/false-positive rate -- revealed through +{days[k]:.0f}d')
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE))

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='directory of per-class <name>.pkl files')
    ap.add_argument('--out-dir', default='docs/figures')
    ap.add_argument('--target-fpr', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target', choices=['lensed_ia', 'all_lensed'], default='lensed_ia',
                    help="signal definition: 'lensed_ia' (default, lensed_Ia only -- "
                         "unchanged filenames) or 'all_lensed' (every lensed_* class; "
                         "output filenames get an _alllensed suffix)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.target == 'all_lensed':
        signal_classes = [c for c in CLASSES if c.startswith('lensed_')]
        signal_label, file_suffix = 'lensed', '_alllensed'
    else:
        signal_classes = ['lensed_Ia']
        signal_label, file_suffix = 'lensed Ia', ''

    print(f'signal classes: {"+".join(signal_classes)}')
    signal_all, background_all = load_population(args.data_dir, signal_classes)
    rng = np.random.default_rng(args.seed)
    sig_train, sig_test = split(signal_all, rng)
    bg_train, bg_test = split(background_all, rng)
    print(f'train: {len(sig_train)} signal, {len(bg_train)} background')
    print(f'test:  {len(sig_test)} signal, {len(bg_test)} background')

    sig_train_w = [e['weight'] for e in sig_train]
    bg_train_w = [e['weight'] for e in bg_train]
    sig_test_w = [e['weight'] for e in sig_test]
    bg_test_w = [e['weight'] for e in bg_test]

    print('fitting static ColourClassifier...')
    static_clf = ColourClassifier(target_fpr=args.target_fpr).fit(
        sig_train, bg_train, sig_train_w, bg_train_w)
    print('fitting TrajectoryClassifier...')
    traj_clf = TrajectoryClassifier(target_fpr=args.target_fpr).fit(
        sig_train, bg_train, sig_train_w, bg_train_w)

    static_roc = static_clf.roc_curve(sig_test, bg_test, sig_test_w, bg_test_w)
    traj_roc = traj_clf.roc_curve(sig_test, bg_test, sig_test_w, bg_test_w)
    by_step = traj_clf.roc_curve_by_step(sig_test, bg_test, sig_test_w, bg_test_w)

    if args.target == 'all_lensed':
        # Per-subtype evaluation against the SAME already-fit traj_clf (no
        # refit) and the SAME unlensed background test set -- see module
        # docstring. sig_test/bg_test events each carry their own 'class'
        # (set at generation time), so subtype subsets are a plain filter.
        subtype_roc, subtype_by_step = {}, {}
        for c in signal_classes:
            sub_test = [e for e in sig_test if e['class'] == c]
            sub_test_w = [e['weight'] for e in sub_test]
            subtype_roc[c] = traj_clf.roc_curve(sub_test, bg_test, sub_test_w, bg_test_w)
            subtype_by_step[c] = traj_clf.roc_curve_by_step(sub_test, bg_test, sub_test_w, bg_test_w)

        rate_by_class = {}
        for c in CLASSES:
            pool = sig_test if c in signal_classes else bg_test
            c_test = [e for e in pool if e['class'] == c]
            c_test_w = np.array([e['weight'] for e in c_test], dtype=float)
            per_step = {}
            for k, day in enumerate(TRAJECTORY_GRID):
                if len(c_test) == 0:
                    per_step[day] = 0.0
                    continue
                partial = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in c_test]
                flagged = traj_clf.predict(partial)
                per_step[day] = float(c_test_w[flagged].sum())
            rate_by_class[c] = per_step

    overall_path = os.path.join(args.out_dir, f'roc_overall{file_suffix}.png')
    by_step_path = os.path.join(args.out_dir, f'roc_by_step{file_suffix}.png')
    gif_path = os.path.join(args.out_dir, f'roc_by_step{file_suffix}.gif')

    plot_overall_roc(static_roc, traj_roc, overall_path, signal_label)
    plot_roc_by_step(by_step, by_step_path, signal_label)
    make_roc_gif(by_step, gif_path, signal_label)

    print(f'wrote {overall_path}')
    print(f'wrote {by_step_path}')
    print(f'wrote {gif_path}')

    if args.target == 'all_lensed':
        subtype_overall_path = os.path.join(args.out_dir, 'roc_by_subtype_alllensed.png')
        plot_roc_by_subtype(subtype_roc, signal_classes, subtype_overall_path)
        print(f'wrote {subtype_overall_path}')

        for c in signal_classes:
            step_path = os.path.join(args.out_dir, f'roc_by_step_{c}_alllensed.png')
            step_gif_path = os.path.join(args.out_dir, f'roc_by_step_{c}_alllensed.gif')
            plot_roc_by_step(subtype_by_step[c], step_path, f'{c} (vs unlensed background)')
            make_roc_gif(subtype_by_step[c], step_gif_path, f'{c} (vs unlensed background)')
            print(f'wrote {step_path}')
            print(f'wrote {step_gif_path}')

        rate_path = os.path.join(args.out_dir, 'rate_by_subtype_alllensed.png')
        rate_gif_path = os.path.join(args.out_dir, 'rate_by_subtype_alllensed.gif')
        plot_rate_by_subtype(rate_by_class, signal_classes, rate_path)
        make_rate_gif(rate_by_class, signal_classes, rate_gif_path)
        print(f'wrote {rate_path}')
        print(f'wrote {rate_gif_path}')


if __name__ == '__main__':
    main()
