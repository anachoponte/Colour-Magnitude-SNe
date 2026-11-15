"""Does TrajectoryClassifier give confident ("pure"/low-FPR) scores to events
observed at only a handful of the 5 trajectory epochs?

Reuses the exact train/test split and TrajectoryClassifier fit from
plot_purity_detections_overlay.py (same seed=0), then for every test signal
event counts how many of the 5 TRAJECTORY_GRID epochs have ANY finite colour
feature, and cross-tabulates that against whether the event clears the
zero-FP "pure" threshold or the 10%-FPR operating point.
"""
import os, pickle, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID, TRAJECTORY_COLOUR_KEYS
from cmsne.multicolour import ADJ_COLOURS

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']
UNLENSED_CLASSES = [c for c in CLASSES if c.startswith('unlensed_')]


def _step_suffix(day):
    return '' if day == 0.0 else f'_d{int(day)}'


def load(data_dir):
    pop = {}
    for c in CLASSES:
        with open(os.path.join(data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']
    return pop


def split(events, rng, frac_train=0.7):
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    return [events[i] for i in idx[:n_train]], [events[i] for i in idx[n_train:]]


def n_finite_epochs(event, grid=TRAJECTORY_GRID):
    """How many of the 5 trajectory epochs have at least one finite colour."""
    n = 0
    for day in grid:
        suffix = _step_suffix(day)
        keys = [f'{a[-1]}{b[-1]}{suffix}' for a, b in ADJ_COLOURS]
        if any(np.isfinite(event.get(k, np.nan)) for k in keys):
            n += 1
    return n


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/dual_59297152'
    seed = 0
    pop = load(data_dir)
    signal = pop['lensed_Ia']
    background = [e for c in UNLENSED_CLASSES for e in pop[c]]

    rng = np.random.default_rng(seed)
    train_sig, test_sig = split(signal, rng)
    train_bg, test_bg = split(background, rng)
    ws_tr = np.array([e['weight'] for e in train_sig])
    wb_tr = np.array([e['weight'] for e in train_bg])
    ws_te = np.array([e['weight'] for e in test_sig])
    wb_te = np.array([e['weight'] for e in test_bg])

    print(f'Fitting TrajectoryClassifier on {len(train_sig)} signal, {len(train_bg)} background...')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(train_sig, train_bg, ws_tr, wb_tr)
    score_sig = clf.score(test_sig)
    score_bg = clf.score(test_bg)
    max_bg_score = np.max(score_bg)
    pure_mask = score_sig > max_bg_score

    fpr, tpr, thresholds = clf.roc_curve(test_sig, test_bg, ws_te, wb_te)
    order = np.argsort(fpr)
    fpr, thresholds = fpr[order], thresholds[order]
    thr_10fpr = thresholds[np.argmin(np.abs(fpr - 0.10))]
    op10_mask = score_sig > thr_10fpr

    n_epochs = np.array([n_finite_epochs(e) for e in test_sig])

    print()
    print('=== n_finite_epochs distribution: all test signal vs detected subsets ===')
    print(f'{"n_epochs":>9s} {"all_N":>7s} {"all_wtfrac":>11s} {"pure_N":>7s} {"pure_wtfrac":>12s} '
          f'{"op10_N":>7s} {"op10_wtfrac":>12s}')
    w_all_total = ws_te.sum()
    w_pure_total = ws_te[pure_mask].sum() if pure_mask.any() else 0.0
    w_op10_total = ws_te[op10_mask].sum() if op10_mask.any() else 0.0
    for k in range(1, 6):
        sel = n_epochs == k
        n_all = int(sel.sum())
        w_all = ws_te[sel].sum() / w_all_total if w_all_total > 0 else np.nan
        n_pure = int((sel & pure_mask).sum())
        w_pure = ws_te[sel & pure_mask].sum() / w_pure_total if w_pure_total > 0 else np.nan
        n_op10 = int((sel & op10_mask).sum())
        w_op10 = ws_te[sel & op10_mask].sum() / w_op10_total if w_op10_total > 0 else np.nan
        print(f'{k:9d} {n_all:7d} {w_all:11.2%} {n_pure:7d} {w_pure:12.2%} {n_op10:7d} {w_op10:12.2%}')

    print()
    print('=== Pure (zero-FP) detections: raw listing ===')
    for i in np.where(pure_mask)[0]:
        e = test_sig[i]
        print(f'  n_epochs={n_epochs[i]}  score={score_sig[i]:.4f}  weight={ws_te[i]:.4e}  '
              f'z={e.get("z"):.3f}  mu={e.get("magnification", 1.0):.2f}')

    print()
    print('=== 10%-FPR detections with only 1-2 finite epochs (if any) ===')
    sparse_op10 = op10_mask & (n_epochs <= 2)
    print(f'{sparse_op10.sum()} / {op10_mask.sum()} of the 10%-FPR detections have <=2 finite epochs')
    for i in np.where(sparse_op10)[0][:15]:
        e = test_sig[i]
        print(f'  n_epochs={n_epochs[i]}  score={score_sig[i]:.4f}  weight={ws_te[i]:.4e}  '
              f'z={e.get("z"):.3f}  mu={e.get("magnification", 1.0):.2f}  peakmag={e.get("peakmag"):.2f}')


if __name__ == '__main__':
    main()
