"""One-off diagnostic (issue #50): what is the REAL per-pattern Kish ESS
ceiling for the sparse (n_epochs=1/2) missing-data patterns that
diag_sparse_trajectory_detections.py found scoring with near-full confidence
on the real production dataset?

Fits the exact same TrajectoryClassifier (same seed/split) as
diag_sparse_trajectory_detections.py, then for the lensed_Ia test-signal
population, groups events by their exact missing-data pattern and reports
each pattern's own uncapped effective_sample_size(ws)/effective_sample_size(wb)
(the same quantity _pattern_ess_ceiling would compute), broken out by
n_finite_epochs -- to find out whether the pattern-confidence discount
(score()) can actually have any effect on this real data, or whether these
sparse EPOCH patterns are themselves backed by plenty of distinct real
training events (large ESS) despite being sparse in FEATURE count.
"""
import os, pickle, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_COLOUR_KEYS, TRAJECTORY_MAGNITUDE_KEYS
from cmsne.classifier import colour_feature_matrix, _finite_pattern
from cmsne.importance_sampling import effective_sample_size
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


def n_finite_epochs(event, grid=(0.0, 3.0, 14.0, 30.0, 60.0)):
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

    print(f'Fitting TrajectoryClassifier on {len(train_sig)} signal, {len(train_bg)} background...')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(train_sig, train_bg, ws_tr, wb_tr)

    X, _ = colour_feature_matrix(test_sig, clf.keys, clf.extra)
    n_epochs = np.array([n_finite_epochs(e) for e in test_sig])

    by_pattern = {}
    for i, row in enumerate(X):
        pat = _finite_pattern(row)
        by_pattern.setdefault(pat, []).append(i)

    print(f'\n{len(by_pattern)} distinct missing-data patterns among {len(test_sig)} test signal events')
    print(f'{"n_epochs":>9s} {"pattern_rank_by_count":>22s} {"n_events_this_pattern":>22s} '
          f'{"ess_ceiling_sig":>16s} {"ess_ceiling_bg":>16s}')

    # Report the LARGEST few patterns per n_epochs bucket -- these are what
    # most individual sparse-epoch events actually fall into.
    for k in range(1, 6):
        idxs_k = np.where(n_epochs == k)[0]
        pats_k = {}
        for i in idxs_k:
            pat = _finite_pattern(X[i])
            pats_k.setdefault(pat, []).append(i)
        ranked = sorted(pats_k.items(), key=lambda kv: -len(kv[1]))
        for rank, (pat, members) in enumerate(ranked[:3]):
            dims = np.where(np.asarray(pat))[0]
            finite_s = np.all(np.isfinite(clf._kde_Xs[:, dims]), axis=1)
            finite_b = np.all(np.isfinite(clf._kde_Xb[:, dims]), axis=1)
            ess_sig = effective_sample_size(clf._kde_ws[finite_s])
            ess_bg = effective_sample_size(clf._kde_wb[finite_b])
            print(f'{k:9d} {rank:22d} {len(members):22d} {ess_sig:16.3f} {ess_bg:16.3f}')

    # Specifically re-examine ONE of the exact overconfident events issue #50
    # reported (n_epochs=1, score~0.9451) by closest match on z/mu, and print
    # its own pattern's ESS ceiling directly.
    print('\n=== closest match to the issue #50 example event (n_epochs=1, z~1.327, mu~48.14) ===')
    best_i, best_d = None, np.inf
    for i in np.where(n_epochs == 1)[0]:
        e = test_sig[i]
        d = abs(e.get('z', np.nan) - 1.327) + abs(e.get('magnification', 1.0) - 48.14)
        if d < best_d:
            best_d, best_i = d, i
    if best_i is not None:
        pat = _finite_pattern(X[best_i])
        dims = np.where(np.asarray(pat))[0]
        finite_s = np.all(np.isfinite(clf._kde_Xs[:, dims]), axis=1)
        finite_b = np.all(np.isfinite(clf._kde_Xb[:, dims]), axis=1)
        ess_sig = effective_sample_size(clf._kde_ws[finite_s])
        ess_bg = effective_sample_size(clf._kde_wb[finite_b])
        print(f'z={test_sig[best_i]["z"]:.3f} mu={test_sig[best_i].get("magnification",1.0):.2f} '
              f'score={clf.score([test_sig[best_i]])[0]:.4f}')
        print(f'this event\'s own pattern: {np.sum(finite_s)} signal rows / {np.sum(finite_b)} bg rows finite over dims, '
              f'ess_ceiling_sig={ess_sig:.3f} ess_ceiling_bg={ess_bg:.3f}')


if __name__ == '__main__':
    main()
