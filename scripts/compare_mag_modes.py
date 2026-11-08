"""Compare lensed recovery between Snapshot Magnitude and Cumulative Best Peak Magnitude.

Evaluates 2-feature ColourClassifiers across trajectory grid steps (+0d, +7d, +14d, +30d, +60d)
using:
  1. Matched Mode: Companion band cumulative best (e.g. peakmag_r for gr)
  2. Snapshot Mode: Companion band observation magnitude at that epoch (e.g. r for gr)

Measures:
  - Rate-weighted true positive rate (TPR, lensed recovery) at 10% FPR
  - Rate-weighted TPR at 5% FPR
  - ROC AUC
"""
import argparse, json, os, pickle, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.classifier import ColourClassifier
from cmsne.trajectory import TRAJECTORY_GRID, _step_suffix, resolve_magnitude_key
from cmsne.multicolour import ADJ_COLOURS

PAIR_KEYS = [f'{a[-1]}{b[-1]}' for a, b in ADJ_COLOURS]
CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']
UNLENSED_CLASSES = [c for c in CLASSES if c.startswith('unlensed_')]


def split(events, rng, frac_train=0.7):
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    return [events[i] for i in idx[:n_train]], [events[i] for i in idx[n_train:]]


def auc(fpr, tpr):
    fpr, tpr = np.asarray(fpr, float), np.asarray(tpr, float)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def evaluate_mode(train_sig, train_bg, test_sig, test_bg, colour_key, magnitude_key):
    tr_s = [e for e in train_sig if np.isfinite(e.get(colour_key, np.nan)) and np.isfinite(e.get(magnitude_key, np.nan))]
    tr_b = [e for e in train_bg if np.isfinite(e.get(colour_key, np.nan)) and np.isfinite(e.get(magnitude_key, np.nan))]
    te_s = [e for e in test_sig if np.isfinite(e.get(colour_key, np.nan)) and np.isfinite(e.get(magnitude_key, np.nan))]
    te_b = [e for e in test_bg if np.isfinite(e.get(colour_key, np.nan)) and np.isfinite(e.get(magnitude_key, np.nan))]

    if len(tr_s) == 0 or len(tr_b) == 0 or len(te_s) == 0 or len(te_b) == 0:
        return {'tpr_at_fpr10': np.nan, 'tpr_at_fpr05': np.nan, 'auc': np.nan, 'n_test_sig': len(te_s), 'n_test_bg': len(te_b)}

    ws_tr = np.array([e['weight'] for e in tr_s])
    wb_tr = np.array([e['weight'] for e in tr_b])
    ws_te = np.array([e['weight'] for e in te_s])
    wb_te = np.array([e['weight'] for e in te_b])

    clf = ColourClassifier(keys=[colour_key], include_peakmag=False, target_fpr=0.10)
    clf.magnitude_feature = magnitude_key
    clf.extra = (magnitude_key,)
    clf.fit(tr_s, tr_b, ws_tr, wb_tr)

    fpr, tpr, _ = clf.roc_curve(te_s, te_b, ws_te, wb_te)

    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]

    tpr10 = float(np.interp(0.10, fpr_sorted, tpr_sorted))
    tpr05 = float(np.interp(0.05, fpr_sorted, tpr_sorted))
    roc_auc = auc(fpr, tpr)

    return {
        'tpr_at_fpr10': tpr10,
        'tpr_at_fpr05': tpr05,
        'auc': roc_auc,
        'n_test_sig': len(te_s),
        'n_test_bg': len(te_b)
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True, help='directory containing per-class .pkl files')
    ap.add_argument('--out-file', default=None, help='optional output path for json results')
    ap.add_argument('--pairs', default='gr,ri,iz', help='comma-separated colour pairs (default: gr,ri,iz)')
    ap.add_argument('--target', choices=['lensed_ia', 'all_lensed'], default='lensed_ia')
    ap.add_argument('--seed', type=int, default=0, help='random seed for train/test split')
    args = ap.parse_args()

    if args.target == 'all_lensed':
        signal_classes = [c for c in CLASSES if c.startswith('lensed_')]
        signal_label = 'all lensed'
    else:
        signal_classes = ['lensed_Ia']
        signal_label = 'lensed Ia only'

    print(f'Loading data from {args.data_dir}...')
    pop = {}
    for c in CLASSES:
        with open(os.path.join(args.data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']

    signal = [e for c in signal_classes for e in pop[c]]
    background = [e for c in UNLENSED_CLASSES for e in pop[c]]
    print(f'Signal ({signal_label}): {len(signal)} events, Background (unlensed_* only): {len(background)} events')

    rng = np.random.default_rng(args.seed)
    train_sig, test_sig = split(signal, rng, frac_train=0.7)
    train_bg, test_bg = split(background, rng, frac_train=0.7)
    print(f'Split: train ({len(train_sig)} sig, {len(train_bg)} bg), test ({len(test_sig)} sig, {len(test_bg)} bg)')

    pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
    results = {}

    sample_event = signal[0] if signal else background[0]

    for pair_key in pairs:
        results[pair_key] = {}
        print(f'\n======================================================')
        print(f'Evaluating Colour Pair: {pair_key} (Signal: {signal_label})')
        print(f'======================================================')
        print(f'{"Step":<8} | {"Metric":<14} | {"Cumulative-Best (peakmag)":<25} | {"Snapshot Magnitude":<20} | {"Diff (Snap - Peak)":<18}')
        print(f'-'*95)

        for day in TRAJECTORY_GRID:
            suffix = _step_suffix(day)
            colour_key = f'{pair_key}{suffix}'

            # 1. Matched cumulative-best
            mag_key_matched, label_matched = resolve_magnitude_key(pair_key, day, 'matched', 'auto', sample_event)
            res_matched = evaluate_mode(train_sig, train_bg, test_sig, test_bg, colour_key, mag_key_matched)

            # 2. Snapshot
            mag_key_snapshot, label_snapshot = resolve_magnitude_key(pair_key, day, 'snapshot', 'auto', sample_event)
            res_snapshot = evaluate_mode(train_sig, train_bg, test_sig, test_bg, colour_key, mag_key_snapshot)

            diff_tpr10 = res_snapshot['tpr_at_fpr10'] - res_matched['tpr_at_fpr10']
            diff_auc = res_snapshot['auc'] - res_matched['auc']

            step_name = f'+{day:.0f}d'
            print(f'{step_name:<8} | {"TPR @ 10% FPR":<14} | {res_matched["tpr_at_fpr10"]*100:6.2f}% ({mag_key_matched:<14}) | {res_snapshot["tpr_at_fpr10"]*100:6.2f}% ({mag_key_snapshot:<10}) | {diff_tpr10*100:+6.2f}%')
            print(f'{"":<8} | {"ROC AUC":<14} | {res_matched["auc"]:6.4f}{"":<18} | {res_snapshot["auc"]:6.4f}{"":<13} | {diff_auc:+6.4f}')
            print(f'-'*95)

            results[pair_key][day] = {
                'matched': res_matched,
                'snapshot': res_snapshot,
                'mag_key_matched': mag_key_matched,
                'mag_key_snapshot': mag_key_snapshot,
            }

    if args.out_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
        with open(args.out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nWrote detailed results to {args.out_file}')


if __name__ == '__main__':
    main()
