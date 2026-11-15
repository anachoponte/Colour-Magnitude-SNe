"""Follow-up to diag_pattern_ess_ceiling.py (issue #50): for the same real
n_epochs=1 overconfident example (z~1.327, mu~48.14, score()~0.9451), does
the EXISTING local-density blend in probability() -- which diag_sparse_
trajectory_detections.py and investigate_purity_tradeoff.py never call, both
using score() directly -- already discount it via LOCAL (not per-pattern)
ESS? If so, the fix for issue #50 may be architectural (route operational
evaluation through probability(), not score()) rather than a new statistical
mechanism.
"""
import os, pickle, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier
from cmsne.classifier import (colour_feature_matrix, _finite_pattern, _local_density_ratio,
                              _blend_weight)

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']
UNLENSED_CLASSES = [c for c in CLASSES if c.startswith('unlensed_')]


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

    print(f'Fitting TrajectoryClassifier on {len(train_sig)} signal, {len(train_bg)} background...')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(train_sig, train_bg, ws_tr, wb_tr)

    best_i, best_d = None, np.inf
    for i, e in enumerate(test_sig):
        d = abs(e.get('z', np.nan) - 1.327) + abs(e.get('magnification', 1.0) - 48.14)
        if d < best_d:
            best_d, best_i = d, i
    event = test_sig[best_i]
    print(f'event: z={event["z"]:.3f} mu={event.get("magnification",1.0):.2f}')
    print(f'score():       {clf.score([event])[0]:.4f}')
    print(f'probability(): {clf.probability([event])[0]:.4f}')

    X, _ = colour_feature_matrix([event], clf.keys, clf.extra)
    pattern = _finite_pattern(X[0])
    dims = np.where(np.asarray(pattern))[0]
    subset = clf._kde_subset_for_pattern(pattern, dims)
    p_kde, ess_sig, ess_bg = _local_density_ratio(
        X[0], subset.Xs, subset.ws, subset.Xb, subset.wb, clf._kde_bandwidths, dims,
        clf.kde_min_ess, ess_ceiling_sig=subset.ess_ceiling_sig, ess_ceiling_bg=subset.ess_ceiling_bg,
        event_id_s=subset.event_id_s, event_id_b=subset.event_id_b,
        n_orig_s=subset.n_orig_s, n_orig_b=subset.n_orig_b)
    alpha = _blend_weight(ess_sig, ess_bg, clf.ess_threshold, clf.log_width) if p_kde is not None else None
    print(f'p_kde={p_kde}  local_ess_sig={ess_sig:.3f}  local_ess_bg={ess_bg:.3f}  alpha={alpha}')

    # Also check a handful of OTHER n_epochs=1 events scoring high, for a
    # broader picture than one single event.
    print('\n=== broader check: 15 n_epochs=1 events with score() > 0.8 ===')
    from cmsne.multicolour import ADJ_COLOURS
    def n_fin(e, grid=(0.0, 3.0, 14.0, 30.0, 60.0)):
        n = 0
        for day in grid:
            suffix = '' if day == 0.0 else f'_d{int(day)}'
            keys = [f'{a[-1]}{b[-1]}{suffix}' for a, b in ADJ_COLOURS]
            if any(np.isfinite(e.get(k, np.nan)) for k in keys):
                n += 1
        return n
    scores_all = clf.score(test_sig)
    probs_all = clf.probability(test_sig)
    shown = 0
    for i, e in enumerate(test_sig):
        if n_fin(e) == 1 and scores_all[i] > 0.8:
            print(f'  score={scores_all[i]:.4f}  probability={probs_all[i]:.4f}  '
                  f'z={e["z"]:.3f}  mu={e.get("magnification",1.0):.2f}')
            shown += 1
            if shown >= 15:
                break


if __name__ == '__main__':
    main()
