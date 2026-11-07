"""Investigate purity, TP/FP ratio, and zero-FPR sweet spots for LSST lensed SNe.

Evaluates the rate-weighted TrajectoryClassifier on the 100k production dataset,
sweeping thresholds from 0 to 1, and maps out:
  - TPR vs FPR (standard ROC)
  - TP / FP ratio across thresholds given realistic LSST survey base rates
  - Purity (Precision = TP / (TP + FP)) vs Recovery (Recall = TPR)
  - Analysis of the extreme low-FPR "sweet spot" where FPR -> 0 but TPR > 0
  - Physical properties of events in the pure sweet spot (z, magnification, colour)
"""
import argparse, json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.classifier import ColourClassifier
from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']
UNLENSED_CLASSES = [c for c in CLASSES if c.startswith('unlensed_')]

# Annual discovery rates in the LSST WFD survey footprint.
# Lensed SN Ia: ~10/yr, order-of-magnitude literature figure already cited
#   elsewhere in this project (Oguri et al. 2010; Wojtak et al. 2019; Goldstein
#   et al. 2019 -- see docs/methods_and_results.md Sec 1/3.6 and
#   docs/paper/paper.tex), not the previous 50/yr, which had no citation and
#   contradicted those.
# Unlensed: ~100,000/yr detected SNe, matching the "~10^5 detected SNe/yr"
#   figure in docs/methods_and_results.md Sec 3.6 (not the previous 1,000,000/yr).
# All lensed (Ia + CC): NOT independently literature-validated -- scaled from
#   the lensed Ia figure by the project's own calibrated LOCAL unlensed CC:Ia
#   rate ratio (CC_TO_IA_RATE=3.5 in cmsne/lsst.py) as an order-of-magnitude
#   placeholder (10 * (1 + 3.5) = 45/yr). This ignores that lensed CC
#   detectability and the lensing optical depth integral both differ from Ia's
#   -- see https://github.com/anachoponte/Colour-Magnitude-SNe/issues/43 for
#   the follow-up to derive it properly from cmsne.rates / summed event weights.
ANNUAL_RATES = {
    'all_lensed': {'N_sig': 45.0, 'N_bg': 100_000.0},
    'lensed_ia':  {'N_sig': 10.0, 'N_bg': 100_000.0},
}


def load_data(data_dir, signal_classes):
    pop = {}
    for c in CLASSES:
        with open(os.path.join(data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']
    signal = [e for c in signal_classes for e in pop[c]]
    # IMPORTANT: Only unlensed_* classes are treated as background contaminants!
    background = [e for c in UNLENSED_CLASSES for e in pop[c]]
    return signal, background, pop


def split(events, rng, frac_train=0.7):
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    return [events[i] for i in idx[:n_train]], [events[i] for i in idx[n_train:]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', default='docs/figures')
    ap.add_argument('--target', choices=['lensed_ia', 'all_lensed'], default='all_lensed')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.target == 'all_lensed':
        signal_classes = [c for c in CLASSES if c.startswith('lensed_')]
        signal_label = 'all lensed'
        file_suffix = '_alllensed'
    else:
        signal_classes = ['lensed_Ia']
        signal_label = 'lensed Ia'
        file_suffix = ''

    rates = ANNUAL_RATES[args.target]
    N_sig_ann = rates['N_sig']
    N_bg_ann = rates['N_bg']
    base_rate_ratio = N_sig_ann / N_bg_ann

    print(f'Loading data from {args.data_dir} for {signal_label}...')
    signal, background, pop = load_data(args.data_dir, signal_classes)

    rng = np.random.default_rng(args.seed)
    train_sig, test_sig = split(signal, rng)
    train_bg, test_bg = split(background, rng)

    ws_tr = [e['weight'] for e in train_sig]
    wb_tr = [e['weight'] for e in train_bg]
    ws_te = np.array([e['weight'] for e in test_sig])
    wb_te = np.array([e['weight'] for e in test_bg])

    print(f'Fitting TrajectoryClassifier on {len(train_sig)} signal, {len(train_bg)} background (unlensed_* only)...')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(train_sig, train_bg, ws_tr, wb_tr)

    print('Evaluating scores on held-out test set...')
    score_sig = clf.score(test_sig)
    score_bg = clf.score(test_bg)

    # Rate-weighted ROC curve
    fpr, tpr, thresholds = clf.roc_curve(test_sig, test_bg, ws_te, wb_te)

    # Sort in ascending FPR order
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    thresholds = thresholds[order]

    # Calculate Annual TP and FP
    annual_tp = tpr * N_sig_ann
    annual_fp = fpr * N_bg_ann

    # TP/FP ratio and Precision (Purity)
    with np.errstate(divide='ignore', invalid='ignore'):
        tp_fp_ratio = np.where(annual_fp > 0, annual_tp / annual_fp, np.where(annual_tp > 0, np.inf, np.nan))
        precision = np.where((annual_tp + annual_fp) > 0, annual_tp / (annual_tp + annual_fp), 1.0)

    # Find the "sweet spot": Maximum score achieved by ANY background event
    max_bg_score = np.max(score_bg)
    pure_mask_sig = score_sig > max_bg_score
    pure_count_sig = np.count_nonzero(pure_mask_sig)
    pure_w_sig = np.sum(ws_te[pure_mask_sig])
    total_w_sig = np.sum(ws_te)
    pure_tpr = pure_w_sig / total_w_sig if total_w_sig > 0 else 0.0
    pure_annual_tp = pure_tpr * N_sig_ann

    print('\n' + '='*75)
    print(f'INVESTIGATION SUMMARY: {signal_label.upper()} BASE-RATE & PURITY TRADE-OFF')
    print('='*75)
    print(f'Physical Survey Prior: {N_sig_ann:.0f} lensed / yr vs {N_bg_ann:,.0f} unlensed / yr')
    print(f'Base Rate Prior: 1 lensed SN per {1/base_rate_ratio:,.0f} unlensed SNe ({base_rate_ratio*100:.4f}%)')
    print('-'*75)
    print(f'Maximum Score achieved by background in test set: {max_bg_score:.5f}')
    print(f'Zero-Contamination Sweet Spot (Threshold > {max_bg_score:.5f}):')
    print(f'  - Empirical FPR on test set: EXACTLY 0.00%')
    print(f'  - Raw test signal events surviving: {pure_count_sig} / {len(test_sig)}')
    print(f'  - Rate-weighted recovery (TPR): {pure_tpr*100:.2f}%')
    print(f'  - Expected pure lensed SNe discovered per year: {pure_annual_tp:.1f} / yr')
    print(f'  - Purity: 100% (zero false alarms!)')
    print('-'*75)

    # Physical properties of the sweet spot events
    pure_events = [test_sig[i] for i in np.where(pure_mask_sig)[0]]
    if pure_events:
        pure_z = [e['z'] for e in pure_events]
        pure_mu = [e.get('magnification', 1.0) for e in pure_events]
        print(f'Properties of sweet-spot lensed events:')
        print(f'  - Redshift range: {np.min(pure_z):.2f} - {np.max(pure_z):.2f} (median: {np.median(pure_z):.2f})')
        print(f'  - Magnification range: {np.min(pure_mu):.1f} - {np.max(pure_mu):.1f} (median: {np.median(pure_mu):.1f})')
        # Check subtypes in sweet spot
        for c in signal_classes:
            sub_count = sum(1 for e in pure_events if e.get('class', '') == c or e.get('kind', '') == c)
            if sub_count > 0:
                print(f'    * {c}: {sub_count} events')
    print('='*75)

    # Sample standard operating points
    table_points = [
        ('10% FPR (Standard Broker)', 0.10),
        ('1% FPR (Low Contamination)', 0.01),
        ('0.1% FPR (Strict Filter)', 0.001),
        ('0.01% FPR (Spectroscopic Candidate)', 0.0001),
    ]

    print('\nOPERATING POINTS SUMMARY:')
    print(f'{"Operating Point":<30} | {"FPR":<8} | {"TPR":<8} | {"TP/yr":<8} | {"FP/yr":<10} | {"TP : FP":<10} | {"Purity":<8}')
    print('-'*95)
    for name, target_f in table_points:
        idx = np.argmin(np.abs(fpr - target_f))
        f_val = fpr[idx]
        t_val = tpr[idx]
        tp_val = t_val * N_sig_ann
        fp_val = f_val * N_bg_ann
        ratio_str = f'1 : {fp_val/tp_val:.1f}' if tp_val > 0 and fp_val > tp_val else f'{tp_val/fp_val:.1f} : 1'
        pur_val = tp_val / (tp_val + fp_val) * 100 if (tp_val + fp_val) > 0 else 0
        print(f'{name:<30} | {f_val*100:6.3f}% | {t_val*100:6.2f}% | {tp_val:6.1f}  | {fp_val:8.0f}   | {ratio_str:<10} | {pur_val:6.2f}%')
    print(f'{"Zero-FP Sweet Spot":<30} | 0.000%  | {pure_tpr*100:6.2f}% | {pure_annual_tp:6.1f}  | 0          | 1 : 0      | 100.00%')
    print('-'*95)

    # ----------------------------------------------------
    # Plot 1: Dual-panel ROC + TP/FP Ratio and Purity
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Standard ROC with Log-scale FPR inset or highlighted low-FPR region
    ax1.plot(fpr * 100, tpr * 100, color='tab:blue', lw=2.5, label='Trajectory Classifier ROC')
    ax1.plot([0, 100], [0, 100], 'k--', lw=1, alpha=0.4, label='Random Chance')
    ax1.scatter([0], [pure_tpr * 100], color='tab:green', s=120, zorder=5, marker='*',
                label=f'Zero-FP Sweet Spot ({pure_annual_tp:.1f} lensed/yr, 100% pure)')
    ax1.set_xlabel('False Positive Rate (FPR, %)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (TPR / Recovery, %)', fontsize=11)
    ax1.set_title(f'ROC Curve ({signal_label})\nFocus: Low-FPR Operating Regimes', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(-1, 101)
    ax1.set_ylim(0, 102)

    # Inset for ultra-low FPR [0, 1%]
    ax_ins = ax1.inset_axes([0.18, 0.40, 0.45, 0.45])
    ax_ins.plot(fpr * 100, tpr * 100, color='tab:blue', lw=2)
    ax_ins.scatter([0], [pure_tpr * 100], color='tab:green', s=140, zorder=5, marker='*')
    ax_ins.set_xlim(-0.02, 1.0)
    ax_ins.set_ylim(max(0, pure_tpr * 100 - 5), min(100, tpr[np.argmin(np.abs(fpr - 0.01))] * 100 + 10))
    ax_ins.set_title('Zoom: 0% to 1% FPR', fontsize=8)
    ax_ins.grid(True, linestyle=':', alpha=0.5)
    ax_ins.tick_params(labelsize=8)

    # Right: Purity (Precision) and TP/FP Ratio vs True Recovery
    valid = np.isfinite(precision) & (tpr > 0)
    ax2.plot(tpr[valid] * 100, precision[valid] * 100, color='tab:purple', lw=2.5, label='Sample Purity (Precision)')
    ax2.axhline(50, color='gray', linestyle=':', label='1:1 TP:FP (50% purity)')
    ax2.axvline(pure_tpr * 100, color='tab:green', linestyle='--', label=f'Sweet spot threshold ({pure_tpr*100:.1f}% recovery)')
    ax2.scatter([pure_tpr * 100], [100], color='tab:green', s=120, zorder=5, marker='*')

    ax2.set_xlabel('Lensed SNe Recovery (Recall / TPR, %)', fontsize=11)
    ax2.set_ylabel('Sample Purity: TP / (TP + FP) (%)', fontsize=11, color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    ax2.set_ylim(-2, 105)
    ax2.set_xlim(0, 100)
    ax2.set_title(f'Purity vs Recovery in LSST\n(Accounting for 1:{int(N_bg_ann/N_sig_ann):,} Base Rate)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Twin axis for TP/FP ratio
    ax2_r = ax2.twinx()
    ax2_r.semilogy(tpr[valid] * 100, tp_fp_ratio[valid], color='tab:red', lw=1.8, linestyle='-.', label='TP / FP Ratio')
    ax2_r.set_ylabel('Ratio of True Positives to False Positives (TP / FP)', fontsize=11, color='tab:red')
    ax2_r.tick_params(axis='y', labelcolor='tab:red')
    ax2_r.axhline(1.0, color='tab:red', linestyle=':', alpha=0.5)

    fig.tight_layout()
    purity_roc_path = os.path.join(args.out_dir, f'roc_purity_ratio{file_suffix}.png')
    fig.savefig(purity_roc_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {purity_roc_path}')

    # ----------------------------------------------------
    # Plot 2: Telescope Follow-up Trade-off Curve
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(annual_fp, annual_tp, color='navy', lw=2.5, label='Alert Operating Curve')
    ax.scatter([0], [pure_annual_tp], color='tab:green', s=140, marker='*', zorder=5,
               label=f'Zero-FP Sweet Spot: {pure_annual_tp:.1f} clean lensed/yr')

    for name, target_f in table_points:
        idx = np.argmin(np.abs(fpr - target_f))
        f_p = annual_fp[idx]
        t_p = annual_tp[idx]
        ax.scatter([f_p], [t_p], s=60, zorder=4)
        ax.annotate(f'{name}\n({t_p:.0f} lensed, {f_p:,.0f} false)',
                    xy=(f_p, t_p), xytext=(10, -15),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray'))

    ax.set_xscale('log')
    ax.set_xlim(0.8, N_bg_ann * 0.5)
    ax.set_ylim(0, N_sig_ann * 1.05)
    ax.set_xlabel('False Alarms per Year (Spectroscopic Waste, Log Scale)', fontsize=11)
    ax.set_ylabel('True Lensed SNe Recovered per Year', fontsize=11)
    ax.set_title(f'LSST Follow-Up Economy: Recovered Lensed SNe vs False Alarms\nSignal: {signal_label} ({N_sig_ann:.0f}/yr true, {N_bg_ann:,.0f}/yr unlensed)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)

    fig.tight_layout()
    yield_path = os.path.join(args.out_dir, f'telescope_followup_yield{file_suffix}.png')
    fig.savefig(yield_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {yield_path}')

    # Save summary json
    summary_data = {
        'signal_label': signal_label,
        'N_sig_annual': N_sig_ann,
        'N_bg_annual': N_bg_ann,
        'pure_sweet_spot': {
            'max_bg_score': float(max_bg_score),
            'pure_tpr': float(pure_tpr),
            'pure_annual_tp': float(pure_annual_tp),
            'pure_events_count': pure_count_sig,
        },
        'operating_points': [
            {
                'name': name,
                'target_fpr': target_f,
                'achieved_fpr': float(fpr[np.argmin(np.abs(fpr - target_f))]),
                'achieved_tpr': float(tpr[np.argmin(np.abs(fpr - target_f))]),
                'annual_tp': float(annual_tp[np.argmin(np.abs(fpr - target_f))]),
                'annual_fp': float(annual_fp[np.argmin(np.abs(fpr - target_f))]),
            }
            for name, target_f in table_points
        ]
    }
    json_path = os.path.join(args.out_dir, f'purity_investigation{file_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
