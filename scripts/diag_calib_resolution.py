"""Scoping diagnostic: how large must a calib split be for each FPR budget to
be measurable at all?

Motivated by job 62174264, where the 0.01% ("Spectroscopic Candidate") budget
turned out to be below the resolution of the calibration set for every target,
and rank_candidates.py had been silently reporting the nearest achievable ROC
point as though it were that budget.

The naive bound is "a calib split of N background events cannot resolve an FPR
finer than 1/N". That is the UNWEIGHTED bound and it is optimistic here: this
project's ROC curves are rate-weighted, so what actually matters is the
effective sample size (Kish ESS) of the background weights, which is
substantially smaller than N whenever the weights are skewed -- as they are.

To quote an FPR of p you need enough effective background events ABOVE the
threshold for the estimate to mean anything; requiring at least
MIN_SAMPLE_FOR_STATS of them gives

    p_min  =  MIN_SAMPLE_FOR_STATS / ESS_bg

as the finest budget that split can support. Weights are per-event and do not
depend on the decision epoch, so this is one number per pool, not one per epoch.

Reads the pickles only -- no classifier fit, so it is cheap.
"""
import argparse, json, os, sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.importance_sampling import effective_sample_size
from investigate_contaminant_properties import (
    TARGET_SIGNAL_CLASSES, TARGET_LABELS, TABLE_POINTS, MIN_SAMPLE_FOR_STATS,
    load_data_tagged,
)

CALIB_FRACTION = 0.2  # split3's calib share


def pool_stats(weights, name):
    w = np.asarray(weights, dtype=float)
    n = int(w.size)
    ess = effective_sample_size(w)
    ratio = ess / n if n else 0.0
    order = np.sort(w)[::-1]
    tot = float(np.sum(w)) or 1.0
    return {
        'pool': name, 'n_raw': n, 'ess': ess, 'ess_per_event': ratio,
        'top1_weight_share_pct': 100.0 * float(order[0]) / tot if n else 0.0,
        'top10_weight_share_pct': 100.0 * float(np.sum(order[:10])) / tot if n else 0.0,
        'top1pct_weight_share_pct': (100.0 * float(np.sum(order[:max(1, n // 100)])) / tot
                                     if n else 0.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', default='docs/figures/purity_investigation')
    ap.add_argument('--target', choices=sorted(TARGET_SIGNAL_CLASSES), default='lensed_ia')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    signal_classes = TARGET_SIGNAL_CLASSES[args.target]
    signal_label, file_suffix = TARGET_LABELS[args.target]
    signal, background, _ = load_data_tagged(args.data_dir, signal_classes)

    w_sig = np.array([e['weight'] for e in signal])
    w_bg = np.array([e['weight'] for e in background])
    sig_stats = pool_stats(w_sig, f'signal ({signal_label})')
    bg_stats = pool_stats(w_bg, 'background (all unlensed)')

    print('=' * 96)
    print(f'CALIB-RESOLUTION SCOPING: {signal_label.upper()}')
    print('=' * 96)
    for s in (sig_stats, bg_stats):
        print(f'\n{s["pool"]}:')
        print(f'  raw events                 : {s["n_raw"]}')
        print(f'  effective sample size (ESS): {s["ess"]:.1f}  '
              f'({100 * s["ess_per_event"]:.1f}% of raw)')
        print(f'  weight share of top 1 event: {s["top1_weight_share_pct"]:.2f}%')
        print(f'  weight share of top 10     : {s["top10_weight_share_pct"]:.2f}%')
        print(f'  weight share of top 1%     : {s["top1pct_weight_share_pct"]:.2f}%')

    # What the CURRENT calib split can support, and what each budget would need.
    calib_n = int(bg_stats['n_raw'] * CALIB_FRACTION)
    calib_ess = bg_stats['ess'] * CALIB_FRACTION  # ESS scales ~linearly with a random subsample
    p_min = MIN_SAMPLE_FOR_STATS / calib_ess if calib_ess else float('inf')
    ratio = bg_stats['ess_per_event']

    print(f'\n--- Current calib split ({CALIB_FRACTION:.0%} of background) ---')
    print(f'  calib raw background events: {calib_n}')
    print(f'  calib background ESS       : {calib_ess:.1f}')
    print(f'  naive (unweighted) floor   : 1/{calib_n} = {100.0 / calib_n:.4f}% FPR')
    print(f'  ACTUAL floor (ESS-based)   : {MIN_SAMPLE_FOR_STATS}/{calib_ess:.0f} = '
          f'{100 * p_min:.4f}% FPR')

    print(f'\n--- Per budget: what it needs ---')
    rows = []
    for name, target_f in TABLE_POINTS:
        ess_needed = MIN_SAMPLE_FOR_STATS / target_f
        calib_n_needed = ess_needed / ratio if ratio else float('inf')
        pool_n_needed = calib_n_needed / CALIB_FRACTION
        ok = target_f >= p_min
        print(f'  {name:<38}: {"OK" if ok else "NOT RESOLVABLE":<15} '
              f'needs calib ESS {ess_needed:>9.0f} -> {calib_n_needed:>10.0f} calib events '
              f'-> {pool_n_needed:>10.0f} total background pool '
              f'({pool_n_needed / bg_stats["n_raw"]:.1f}x current)')
        rows.append({'budget': name, 'target_fpr': target_f, 'resolvable_now': bool(ok),
                     'calib_ess_needed': ess_needed, 'calib_raw_needed': calib_n_needed,
                     'pool_raw_needed': pool_n_needed,
                     'factor_vs_current': pool_n_needed / bg_stats['n_raw']})
    print('=' * 96)

    out = {'signal_label': signal_label, 'signal_pool': sig_stats,
           'background_pool': bg_stats, 'calib_fraction': CALIB_FRACTION,
           'calib_raw': calib_n, 'calib_ess': calib_ess,
           'finest_resolvable_fpr_now': p_min, 'budgets': rows}
    path = os.path.join(args.out_dir, f'calib_resolution{file_suffix}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
