"""THROWAWAY SPIKE: does the (z, mu) ESS gain reach the score tail?

`p ∝ rate * sqrt(eps)` minimises the variance of `∫ rate * eps` -- a class's
TOTAL annual detection rate. But the thing blocking the 0.1% and 0.01% FPR
budgets is not the total: it is the extreme TAIL of the *background score*
distribution, in classifier-feature space rather than in (z, mu). A proposal
that concentrates attempts on the bright, easily-detected corner could raise
global ESS while thinning the rare tail that actually sets a strict threshold.

If that is what happens, `rate * sqrt(eps)` is optimising the wrong functional
and the detection-aware design does not serve its stated purpose.

The question reduces cleanly. Restricting the estimator to events above a score
threshold multiplies each bin's contribution by `f_b`, the fraction of that
bin's detections landing in the tail, so

    ESS_tail / N  =  (sum rate*eps*f*a)^2 / sum(rate^2*eps*f*a / p)

which is the global formula with `eps -> eps*f`. Two consequences:
  * the TAIL-optimal proposal is `p ∝ rate*sqrt(eps*f)`, not `rate*sqrt(eps)`;
  * whether the gain transfers is just the ratio of the two columns below.

Reports, per class and for the pooled background:
    gain_global = ESS(rate*sqrt(eps))  / ESS(uniform),  weighting g = 1
    gain_tail   = same, with g = f_b
    ceiling     = ESS(rate*sqrt(eps*f)) / ESS(uniform), the tail-aware proposal

If gain_tail is close to gain_global the design stands. If gain_tail collapses
toward 1, it does not.

Throwaway: no tests, not part of any production path.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID, reveal_up_to
from cmsne.importance_sampling import effective_sample_size  # noqa: F401  (parity check)

from rank_candidates import (  # reuse the production loader/split verbatim
    load_data_tagged, split3, TARGET_SIGNAL_CLASSES,
)

N_BINS = 24
TAIL_QUANTILES = (0.90, 0.99, 0.999)


def ess_per_attempt(rate_b, eps_b, g_b, area_b, p_b):
    """ESS per generation attempt for proposal density `p_b`.

    `g_b` restricts the estimator to a sub-population (1.0 = everything,
    `f_b` = the score tail).
    """
    live = (p_b > 0) & (eps_b > 0) & (g_b > 0)
    num = np.sum(rate_b * eps_b * g_b * area_b) ** 2
    if not np.any(live):
        return 0.0
    den = np.sum((rate_b[live] ** 2) * eps_b[live] * g_b[live]
                 * area_b[live] / p_b[live])
    return float(num / den) if den > 0 else 0.0


def normalise(x, area):
    """Turn a non-negative shape into a density over the binned domain."""
    x = np.clip(np.asarray(x, float), 0.0, None)
    tot = float(np.sum(x * area))
    if tot <= 0:
        return np.ones_like(x) / float(np.sum(area))
    return x / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--target', default='lensed_ia', choices=sorted(TARGET_SIGNAL_CLASSES))
    ap.add_argument('--epoch-index', type=int, default=len(TRAJECTORY_GRID) - 1,
                    help='decision epoch to score at (default: last, +60d)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='docs/figures/purity_investigation/spike_tail_transfer.json')
    args = ap.parse_args()

    k = args.epoch_index
    day = TRAJECTORY_GRID[k]
    print('=' * 92)
    print(f'SPIKE: tail transfer -- target={args.target}  epoch=+{day}d  bins={N_BINS}')
    print('=' * 92)

    signal, background, bg_subtype = load_data_tagged(args.data_dir, TARGET_SIGNAL_CLASSES[args.target])
    # load_data_tagged returns subtypes as a parallel array, which split3 would
    # decorrelate; attach it to each event so it survives the split.
    for e, tag in zip(background, bg_subtype):
        e['_subtype'] = str(tag)
    rng = np.random.default_rng(args.seed)
    train_sig, _calib_sig, val_sig = split3(signal, rng)
    train_bg, _calib_bg, val_bg = split3(background, rng)

    print(f'Fitting on train ({len(train_sig)} sig / {len(train_bg)} bg); '
          f'measuring on val ({len(val_sig)} / {len(val_bg)}).')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(
        train_sig, train_bg, [e['weight'] for e in train_sig],
        [e['weight'] for e in train_bg])

    val_bg_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in val_bg]
    bg_scores = np.asarray(clf.score(val_bg_k), dtype=float)
    bg_w = np.asarray([e['weight'] for e in val_bg], dtype=float)
    print(f'Scored {len(bg_scores)} val background events.')

    # Background is unlensed: mu == 1, so the proposal is 1-D in z.
    subt = np.asarray([e.get('_subtype', e.get('class', '?')) for e in val_bg])
    z_bg = np.asarray([e['z'] for e in val_bg], dtype=float)

    results = {'target': args.target, 'epoch_day': day, 'n_bins': N_BINS,
               'classes': {}, 'pooled_background': {}}

    for q in TAIL_QUANTILES:
        thr = weighted_quantile_simple(bg_scores, bg_w, q)
        in_tail = bg_scores >= thr
        print(f'\n--- background score tail: top {100*(1-q):g}% by weight '
              f'(threshold {thr:.6f}, {int(in_tail.sum())} raw events) ---')
        print(f'{"class":<18}{"gain_global":>13}{"gain_tail":>12}{"tail_ceiling":>14}'
              f'{"raw in tail":>13}')

        pooled = {'rate': [], 'eps': [], 'f': [], 'area': []}
        for name in sorted(set(subt.tolist())):
            m = (subt == name)
            if m.sum() < 50:
                continue
            row = per_class_row(z_bg[m], bg_w[m], in_tail[m])
            if row is None:
                continue
            rate_b, eps_b, f_b, area_b = row
            ones = np.ones_like(f_b)

            p_uni = normalise(np.ones_like(rate_b), area_b)
            p_mat = normalise(rate_b * np.sqrt(eps_b), area_b)
            p_tail = normalise(rate_b * np.sqrt(eps_b * f_b), area_b)

            g_glob = ess_per_attempt(rate_b, eps_b, ones, area_b, p_uni)
            m_glob = ess_per_attempt(rate_b, eps_b, ones, area_b, p_mat)
            g_tail = ess_per_attempt(rate_b, eps_b, f_b, area_b, p_uni)
            m_tail = ess_per_attempt(rate_b, eps_b, f_b, area_b, p_mat)
            c_tail = ess_per_attempt(rate_b, eps_b, f_b, area_b, p_tail)

            gg = m_glob / g_glob if g_glob > 0 else float('nan')
            gt = m_tail / g_tail if g_tail > 0 else float('nan')
            cc = c_tail / g_tail if g_tail > 0 else float('nan')
            print(f'{name:<18}{gg:>12.2f}x{gt:>11.2f}x{cc:>13.2f}x'
                  f'{int((m & in_tail).sum()):>13}')
            results['classes'].setdefault(f'{1-q:.4f}', {})[name] = {
                'gain_global': gg, 'gain_tail': gt, 'tail_ceiling': cc,
                'raw_in_tail': int((m & in_tail).sum())}

            pooled['rate'].append(rate_b); pooled['eps'].append(eps_b)
            pooled['f'].append(f_b); pooled['area'].append(area_b)

        if pooled['rate']:
            rate_b = np.concatenate(pooled['rate']); eps_b = np.concatenate(pooled['eps'])
            f_b = np.concatenate(pooled['f']); area_b = np.concatenate(pooled['area'])
            ones = np.ones_like(f_b)
            p_uni = normalise(np.ones_like(rate_b), area_b)
            p_mat = normalise(rate_b * np.sqrt(eps_b), area_b)
            p_tail = normalise(rate_b * np.sqrt(eps_b * f_b), area_b)
            g_glob = ess_per_attempt(rate_b, eps_b, ones, area_b, p_uni)
            m_glob = ess_per_attempt(rate_b, eps_b, ones, area_b, p_mat)
            g_tl = ess_per_attempt(rate_b, eps_b, f_b, area_b, p_uni)
            m_tl = ess_per_attempt(rate_b, eps_b, f_b, area_b, p_mat)
            c_tl = ess_per_attempt(rate_b, eps_b, f_b, area_b, p_tail)
            gg = m_glob / g_glob if g_glob > 0 else float('nan')
            gt = m_tl / g_tl if g_tl > 0 else float('nan')
            cc = c_tl / g_tl if g_tl > 0 else float('nan')
            print(f'{"POOLED BACKGROUND":<18}{gg:>12.2f}x{gt:>11.2f}x{cc:>13.2f}x'
                  f'{int(in_tail.sum()):>13}')
            results['pooled_background'][f'{1-q:.4f}'] = {
                'gain_global': gg, 'gain_tail': gt, 'tail_ceiling': cc,
                'raw_in_tail': int(in_tail.sum())}

    print('\nReading: gain_tail close to gain_global => the (z, mu) gain reaches the')
    print('tail and the detection-aware design serves its purpose. gain_tail near 1')
    print('=> it does not, and the proposal should target rate*sqrt(eps*f) instead.')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nWrote {args.out}')


def per_class_row(z, w, in_tail):
    """Per-z-bin (rate, eps, tail fraction, area) for one unlensed class."""
    if z.size == 0:
        return None
    lo, hi = float(z.min()), float(z.max())
    if not (hi > lo):
        return None
    edges = np.linspace(lo, hi, N_BINS + 1)
    idx = np.clip(np.searchsorted(edges, z, side='right') - 1, 0, N_BINS - 1)

    counts = np.bincount(idx, minlength=N_BINS).astype(float)
    tail_counts = np.bincount(idx, weights=in_tail.astype(float), minlength=N_BINS)
    # Each event's stored weight is (v_sample / n_attempted) * rate_density, so
    # within one class the mean weight in a bin is proportional to the rate.
    wsum = np.bincount(idx, weights=w, minlength=N_BINS)
    rate_b = np.divide(wsum, counts, out=np.zeros_like(wsum), where=counts > 0)

    area_b = np.diff(edges)
    # Attempts were uniform in z, so attempts per bin is constant; eps is then
    # proportional to counts per bin and the constant cancels in every ratio.
    eps_b = counts / max(counts.sum(), 1.0)
    f_b = np.divide(tail_counts, counts, out=np.zeros_like(tail_counts), where=counts > 0)
    return rate_b, eps_b, f_b, area_b


def weighted_quantile_simple(x, w, q):
    o = np.argsort(x)
    xs, ws = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(ws)
    if c[-1] <= 0:
        return float(np.quantile(x, q))
    return float(np.interp(q * c[-1], c, xs))


if __name__ == '__main__':
    main()
