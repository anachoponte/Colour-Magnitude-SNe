"""Does a CHEAP per-pattern ESS (no subsampling, no noise-injection
replication -- just effective_sample_size of the raw finite-filtered
weights) correlate with where the expensive probability() blend disagrees
with score() on real production data? If yes, it's a viable fast proxy for
issue #50's discount; if not, the earlier pattern-ESS hypothesis is dead for
real reasons, not just the one example event already checked.

Reuses the same fit/split as diag_probability_vs_score_sparse.py.
"""
import os, pickle, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier
from cmsne.classifier import colour_feature_matrix, _finite_pattern
from cmsne.importance_sampling import effective_sample_size
from cmsne.multicolour import ADJ_COLOURS

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


def n_fin(e, grid=(0.0, 3.0, 14.0, 30.0, 60.0)):
    n = 0
    for day in grid:
        suffix = '' if day == 0.0 else f'_d{int(day)}'
        keys = [f'{a[-1]}{b[-1]}{suffix}' for a, b in ADJ_COLOURS]
        if any(np.isfinite(e.get(k, np.nan)) for k in keys):
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

    # Take all n_epochs<=2 events with score() > 0.7 (the overconfident
    # regime issue #50 flagged), compute score(), probability() (expensive,
    # existing), and the CHEAP raw per-pattern ESS (no replication/subsample
    # -- same formula as the earlier _pattern_ess_ceiling attempt).
    sparse_idx = [i for i, e in enumerate(test_sig) if n_fin(e) <= 2]
    sparse_events = [test_sig[i] for i in sparse_idx]
    scores = clf.score(sparse_events)
    high_mask = scores > 0.7
    high_idx = [i for i in range(len(sparse_events)) if high_mask[i]]
    print(f'{len(high_idx)} / {len(sparse_events)} sparse (n_epochs<=2) test events score > 0.7')
    # Cap to a manageable number for the expensive probability() call.
    sample_idx = high_idx[:300]
    sample_events = [sparse_events[i] for i in sample_idx]
    sample_scores = scores[sample_idx]

    print('Computing probability() (expensive, existing blend) on the sample...')
    sample_probs = clf.probability(sample_events)

    X, _ = colour_feature_matrix(sample_events, clf.keys, clf.extra)
    cheap_ess = np.zeros(len(sample_events))
    for i, row in enumerate(X):
        pattern = _finite_pattern(row)
        dims = np.where(np.asarray(pattern))[0]
        finite_s = np.all(np.isfinite(clf._kde_Xs[:, dims]), axis=1)
        finite_b = np.all(np.isfinite(clf._kde_Xb[:, dims]), axis=1)
        ess_s = effective_sample_size(clf._kde_ws[finite_s])
        ess_b = effective_sample_size(clf._kde_wb[finite_b])
        cheap_ess[i] = min(ess_s, ess_b)

    gap = sample_scores - sample_probs  # positive = score() overconfident relative to probability()

    print('\n=== correlation check: cheap per-pattern ESS vs (score - probability) gap ===')
    valid = np.isfinite(gap) & np.isfinite(cheap_ess)
    corr = np.corrcoef(np.log(cheap_ess[valid] + 1e-6), gap[valid])[0, 1]
    print(f'Pearson correlation of log(cheap_ess) vs gap: {corr:.3f} (n={valid.sum()})')

    print(f'\n{"cheap_ess":>12s} {"score":>8s} {"probability":>12s} {"gap":>8s}')
    order = np.argsort(-gap)
    for i in order[:20]:
        print(f'{cheap_ess[i]:12.2f} {sample_scores[i]:8.4f} {sample_probs[i]:12.4f} {gap[i]:8.4f}')
    print('...')
    for i in order[-10:]:
        print(f'{cheap_ess[i]:12.2f} {sample_scores[i]:8.4f} {sample_probs[i]:12.4f} {gap[i]:8.4f}')


if __name__ == '__main__':
    main()
