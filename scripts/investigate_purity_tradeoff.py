"""Investigate purity, TP/FP ratio, and zero-FPR sweet spots for LSST lensed SNe.

Evaluates the rate-weighted TrajectoryClassifier on the 100k production dataset
SEPARATELY AT EACH of the 5 trajectory decision epochs (+0d, +3d, +14d, +30d,
+60d), using `cmsne.trajectory.reveal_up_to` to mask out any feature that would
not yet exist at that decision point. For each epoch, maps out:
  - TPR vs FPR (standard ROC)
  - Purity (Precision = TP / (TP + FP)) vs Recovery (Recall = TPR) at
    standard operating points (10%, 1%, 0.1%, 0.01% FPR)
  - The extreme zero-FPR "pure" sweet spot (threshold = max background score)

A single TrajectoryClassifier is fit ONCE on the full (uncensored) training
set -- same convention as `TrajectoryClassifier.roc_curve_by_step`, already
used correctly in scripts/plot_roc.py and scripts/compare_trajectory_models.py
-- then re-scored per epoch via `reveal_up_to`. An earlier version of this
script scored every test event on its full, fully-realised light curve
regardless of how many epochs it actually had, which silently assumed a
broker could wait up to 60 days post-trigger before deciding on follow-up
(look-ahead bias; see issue #51). Every number below is now tied to a single,
causally-consistent decision epoch instead of mixing them into one curve.
"""
import argparse, json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID, reveal_up_to

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

TABLE_POINTS = [
    ('10% FPR (Standard Broker)', 0.10),
    ('1% FPR (Low Contamination)', 0.01),
    ('0.1% FPR (Strict Filter)', 0.001),
    ('0.01% FPR (Spectroscopic Candidate)', 0.0001),
]


def load_data(data_dir, signal_classes):
    pop = {}
    for c in CLASSES:
        with open(os.path.join(data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']
    signal = [e for c in signal_classes for e in pop[c]]
    # IMPORTANT: Only unlensed_* classes are treated as background contaminants!
    background = [e for c in UNLENSED_CLASSES for e in pop[c]]
    return signal, background


def split(events, rng, frac_train=0.7):
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    return [events[i] for i in idx[:n_train]], [events[i] for i in idx[n_train:]]


def analyze_epoch(clf, k, test_sig, test_bg, ws_te, wb_te, N_sig_ann, N_bg_ann):
    """Score test events using only information available as of decision epoch k."""
    sig_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in test_sig]
    bg_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in test_bg]
    score_sig = clf.score(sig_k)
    score_bg = clf.score(bg_k)

    fpr, tpr, thresholds = clf.roc_curve(sig_k, bg_k, ws_te, wb_te)
    order = np.argsort(fpr)
    fpr, tpr, thresholds = fpr[order], tpr[order], thresholds[order]
    annual_tp = tpr * N_sig_ann
    annual_fp = fpr * N_bg_ann

    max_bg_score = float(np.max(score_bg))
    pure_mask_sig = score_sig > max_bg_score
    pure_w_sig = np.sum(ws_te[pure_mask_sig])
    total_w_sig = np.sum(ws_te)
    pure_tpr = pure_w_sig / total_w_sig if total_w_sig > 0 else 0.0
    pure_annual_tp = pure_tpr * N_sig_ann

    operating_points = []
    for name, target_f in TABLE_POINTS:
        idx = int(np.argmin(np.abs(fpr - target_f)))
        f_val, t_val = float(fpr[idx]), float(tpr[idx])
        tp_val, fp_val = t_val * N_sig_ann, f_val * N_bg_ann
        pur_val = tp_val / (tp_val + fp_val) * 100 if (tp_val + fp_val) > 0 else 0.0
        operating_points.append({'name': name, 'target_fpr': target_f, 'achieved_fpr': f_val,
                                  'achieved_tpr': t_val, 'annual_tp': tp_val, 'annual_fp': fp_val,
                                  'purity_pct': pur_val})

    return {
        'fpr': fpr, 'tpr': tpr, 'annual_tp': annual_tp, 'annual_fp': annual_fp,
        'pure_tpr': pure_tpr, 'pure_annual_tp': pure_annual_tp, 'max_bg_score': max_bg_score,
        'operating_points': operating_points, 'pure_mask_sig': pure_mask_sig,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', default='docs/figures/purity_investigation')
    ap.add_argument('--target', choices=['lensed_ia', 'all_lensed'], default='all_lensed')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.target == 'all_lensed':
        signal_classes = [c for c in CLASSES if c.startswith('lensed_')]
        signal_label, file_suffix = 'all lensed', '_alllensed'
    else:
        signal_classes = ['lensed_Ia']
        signal_label, file_suffix = 'lensed Ia', ''

    rates = ANNUAL_RATES[args.target]
    N_sig_ann, N_bg_ann = rates['N_sig'], rates['N_bg']
    base_rate_ratio = N_sig_ann / N_bg_ann

    print(f'Loading data from {args.data_dir} for {signal_label}...')
    signal, background = load_data(args.data_dir, signal_classes)

    rng = np.random.default_rng(args.seed)
    train_sig, test_sig = split(signal, rng)
    train_bg, test_bg = split(background, rng)
    ws_tr = [e['weight'] for e in train_sig]
    wb_tr = [e['weight'] for e in train_bg]
    ws_te = np.array([e['weight'] for e in test_sig])
    wb_te = np.array([e['weight'] for e in test_bg])

    print(f'Fitting TrajectoryClassifier ONCE on full (uncensored) '
          f'{len(train_sig)} signal, {len(train_bg)} background (unlensed_* only)...')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(train_sig, train_bg, ws_tr, wb_tr)

    results = {}
    print('\n' + '=' * 100)
    print(f'CAUSAL (per-decision-epoch) PURITY & SWEET-SPOT INVESTIGATION: {signal_label.upper()}')
    print('=' * 100)
    print(f'Physical Survey Prior: {N_sig_ann:.0f} lensed / yr vs {N_bg_ann:,.0f} unlensed / yr '
          f'(1 lensed per {1 / base_rate_ratio:,.0f} unlensed, {base_rate_ratio * 100:.4f}%)')
    for k, day in enumerate(TRAJECTORY_GRID):
        res = analyze_epoch(clf, k, test_sig, test_bg, ws_te, wb_te, N_sig_ann, N_bg_ann)
        results[day] = res
        print(f'\n--- Decision epoch: trigger+{day:.0f}d (k={k}) ---')
        print(f'Zero-FP pure sweet spot: {res["pure_annual_tp"]:.2f} / yr '
              f'({res["pure_tpr"]*100:.2f}% recovery, 100% purity, '
              f'max background score = {res["max_bg_score"]:.5f})')
        print(f'{"Operating Point":<36} | {"FPR":<8} | {"TPR":<8} | {"TP/yr":<8} | {"FP/yr":<10} | {"Purity":<8}')
        for op in res['operating_points']:
            print(f'{op["name"]:<36} | {op["achieved_fpr"]*100:6.3f}% | {op["achieved_tpr"]*100:6.2f}% | '
                  f'{op["annual_tp"]:6.1f}  | {op["annual_fp"]:8.0f}   | {op["purity_pct"]:6.2f}%')

        pure_events = [test_sig[i] for i in np.where(res['pure_mask_sig'])[0]]
        if pure_events:
            pure_z = [e['z'] for e in pure_events]
            pure_mu = [e.get('magnification', 1.0) for e in pure_events]
            print(f'  Sweet-spot events: z = {np.min(pure_z):.2f}-{np.max(pure_z):.2f} '
                  f'(median {np.median(pure_z):.2f}), mu = {np.min(pure_mu):.1f}-{np.max(pure_mu):.1f} '
                  f'(median {np.median(pure_mu):.1f})')
    print('=' * 100)

    # ------------------------------------------------------------------
    # Plot 1: ROC curves at every decision epoch, overlaid
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(TRAJECTORY_GRID)))
    for (day, res), color in zip(results.items(), cmap):
        ax.plot(res['fpr'] * 100, res['tpr'] * 100, color=color, lw=2.2, label=f'+{day:.0f}d')
        ax.scatter([0], [res['pure_tpr'] * 100], color=color, s=80, marker='*', zorder=5)
    ax.plot([0, 100], [0, 100], 'k--', lw=1, alpha=0.3)
    ax.set_xlim(-1, 101); ax.set_ylim(0, 102)
    ax.set_xlabel('False Positive Rate (%)'); ax.set_ylabel('True Positive Rate (%)')
    ax.set_title(f'Causal ROC by decision epoch ({signal_label})\n'
                 f'stars = zero-FP pure sweet spot at that epoch')
    ax.legend(title='Decision epoch', loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    roc_path = os.path.join(args.out_dir, f'roc_by_decision_epoch{file_suffix}.png')
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {roc_path}')

    # ------------------------------------------------------------------
    # Plot 2: how much purity/recovery is lost by deciding early
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    days = list(results.keys())
    pure_tp = [results[d]['pure_annual_tp'] for d in days]
    op10_tp = [next(o for o in results[d]['operating_points'] if o['name'].startswith('10%'))['annual_tp']
               for d in days]
    op10_purity = [next(o for o in results[d]['operating_points'] if o['name'].startswith('10%'))['purity_pct']
                   for d in days]

    ax1.plot(days, pure_tp, 'o-', color='tab:green', lw=2, label='Zero-FP pure sweet spot')
    ax1.plot(days, op10_tp, 's-', color='tab:blue', lw=2, label='10% FPR operating point')
    ax1.set_xlabel('Decision epoch (days since trigger)')
    ax1.set_ylabel(f'Annual true positives ({signal_label}/yr)')
    ax1.set_title('Recovered signal vs. how long you wait to decide')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.plot(days, op10_purity, 's-', color='tab:purple', lw=2)
    ax2.set_xlabel('Decision epoch (days since trigger)')
    ax2.set_ylabel('Purity at 10% FPR operating point (%)')
    ax2.set_title('Purity vs. how long you wait to decide')
    ax2.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle(f'Cost of deciding early ({signal_label}) -- fixes issue #51', fontsize=12)
    fig.tight_layout()
    cost_path = os.path.join(args.out_dir, f'cost_of_early_decision{file_suffix}.png')
    fig.savefig(cost_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {cost_path}')

    # Save summary JSON
    summary = {
        'signal_label': signal_label, 'N_sig_annual': N_sig_ann, 'N_bg_annual': N_bg_ann,
        'by_decision_epoch': {
            f'{day:.0f}d': {
                'pure_tpr': res['pure_tpr'], 'pure_annual_tp': res['pure_annual_tp'],
                'max_bg_score': res['max_bg_score'], 'operating_points': res['operating_points'],
            }
            for day, res in results.items()
        }
    }
    json_path = os.path.join(args.out_dir, f'purity_investigation{file_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
