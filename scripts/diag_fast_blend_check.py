"""Does the EXPENSIVE noise-injection replication inside the KDE reference-
set build (_kde_subset_for_pattern's call to _replicate_with_uncertainty,
the dominant cost per profiling) actually matter for the local-density
blend's discrimination, or is most of the real signal already present in a
much CHEAPER un-replicated kernel evaluation over the same (capped,
unreplicated) reference rows?

Builds the local-density ratio directly from clf's own internals, bypassing
_kde_subset_for_pattern's replication step entirely (n_replicates=1
effectively, no SHA256 seeding, no per-row python loop -- just the capped
finite-filtered rows themselves), and compares the resulting "fast blended"
probability against the existing (expensive, replicated) probability() on
the same real overconfident-sparse-event sample used by
diag_cheap_ess_correlation.py.
"""
import os, pickle, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier
from cmsne.classifier import (colour_feature_matrix, _finite_pattern, _local_density_ratio,
                              _blend_weight, _weighted_subsample)
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


def fast_blended_probability(clf, x, p_tree_i):
    """Same local-density blend as _blend_with_local_density, but with NO
    noise-injection replication anywhere (not the query, not the reference
    set) -- just the raw capped, finite-filtered rows directly."""
    pattern = _finite_pattern(x)
    dims = np.where(np.asarray(pattern))[0]
    if len(dims) == 0:
        return p_tree_i
    finite_s = np.all(np.isfinite(clf._kde_Xs[:, dims]), axis=1)
    finite_b = np.all(np.isfinite(clf._kde_Xb[:, dims]), axis=1)
    Xs_cap, ws_cap, _ = _weighted_subsample(clf._kde_Xs[finite_s], clf._kde_ws[finite_s], clf.kde_max_events)
    Xb_cap, wb_cap, _ = _weighted_subsample(clf._kde_Xb[finite_b], clf._kde_wb[finite_b], clf.kde_max_events)
    ess_ceiling_sig = effective_sample_size(ws_cap)
    ess_ceiling_bg = effective_sample_size(wb_cap)
    p_kde, ess_sig, ess_bg = _local_density_ratio(
        x, Xs_cap, ws_cap, Xb_cap, wb_cap, clf._kde_bandwidths, dims, clf.kde_min_ess,
        ess_ceiling_sig=ess_ceiling_sig, ess_ceiling_bg=ess_ceiling_bg)
    if p_kde is None:
        return p_tree_i
    alpha = _blend_weight(ess_sig, ess_bg, clf.ess_threshold, clf.log_width)
    return alpha * p_tree_i + (1 - alpha) * p_kde


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

    sparse_idx = [i for i, e in enumerate(test_sig) if n_fin(e) <= 2]
    sparse_events = [test_sig[i] for i in sparse_idx]
    scores = clf.score(sparse_events)
    high_idx = [i for i in range(len(sparse_events)) if scores[i] > 0.7][:300]
    sample_events = [sparse_events[i] for i in high_idx]
    sample_scores = scores[high_idx]

    import time
    t0 = time.time()
    sample_probs_expensive = clf.probability(sample_events)
    t_expensive = time.time() - t0

    X, _ = colour_feature_matrix(sample_events, clf.keys, clf.extra)
    t0 = time.time()
    fast_probs = np.array([fast_blended_probability(clf, X[i], sample_scores[i]) for i in range(len(sample_events))])
    t_fast = time.time() - t0

    print(f'\nTiming: expensive probability() = {t_expensive:.2f}s, fast (no-replication) blend = {t_fast:.2f}s '
          f'({t_expensive / max(t_fast, 1e-9):.1f}x speedup)')

    gap_expensive = sample_scores - sample_probs_expensive
    gap_fast = sample_scores - fast_probs
    agree = np.corrcoef(gap_expensive, gap_fast)[0, 1]
    print(f'Correlation between expensive-blend gap and fast-blend gap: {agree:.3f} (n={len(sample_events)})')
    print(f'Mean |expensive_prob - fast_prob|: {np.mean(np.abs(sample_probs_expensive - fast_probs)):.4f}')
    print(f'Max  |expensive_prob - fast_prob|: {np.max(np.abs(sample_probs_expensive - fast_probs)):.4f}')

    print(f'\n{"score":>8s} {"prob_expensive":>15s} {"prob_fast":>10s} {"gap_exp":>8s} {"gap_fast":>8s}')
    order = np.argsort(-gap_expensive)
    for i in order[:15]:
        print(f'{sample_scores[i]:8.4f} {sample_probs_expensive[i]:15.4f} {fast_probs[i]:10.4f} '
              f'{gap_expensive[i]:8.4f} {gap_fast[i]:8.4f}')


if __name__ == '__main__':
    main()
