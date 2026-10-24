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
from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']


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
    ax.set_title(f'Static vs trajectory classifier ROC (held-out, rate-weighted) -- signal: {signal_label}')
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
    ax.set_title(f'Trajectory classifier ROC vs time-since-trigger (held-out) -- signal: {signal_label}')
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
        ax.set_title(f'Trajectory classifier ROC -- Δt since trigger: +{day:.0f}d (signal: {signal_label})')
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

    overall_path = os.path.join(args.out_dir, f'roc_overall{file_suffix}.png')
    by_step_path = os.path.join(args.out_dir, f'roc_by_step{file_suffix}.png')
    gif_path = os.path.join(args.out_dir, f'roc_by_step{file_suffix}.gif')

    plot_overall_roc(static_roc, traj_roc, overall_path, signal_label)
    plot_roc_by_step(by_step, by_step_path, signal_label)
    make_roc_gif(by_step, gif_path, signal_label)

    print(f'wrote {overall_path}')
    print(f'wrote {by_step_path}')
    print(f'wrote {gif_path}')


if __name__ == '__main__':
    main()
