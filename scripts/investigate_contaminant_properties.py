"""Characterize the false-positive contaminants behind each FPR operating point,
and test whether a cheap secondary cut can clean them up further.

Context: Finding 2/4 of docs/session_writeup.md showed that, once issue #50's
local-density calibration discount is applied, none of the 7 architectures
tested in this project (the real deployed TrajectoryClassifier + 6 benchmark
models) achieve genuine zero-false-positive discrimination at LSST survey
scale. The classifier's real value is as a RANKED candidate list at a chosen
FPR budget (`rank()`/`probability()` -- already this project's own recommended
API over a hard `predict()` cut), not a zero-contamination filter. This script
asks the natural follow-up question: at a realistic operating point (e.g.
10%/1%/0.1% FPR), what IS leaking through as a false positive, and can a cheap
secondary cut remove more of it before committing scarce spectroscopic time?

"Host galaxy type" was the originally proposed secondary cut, but it does not
exist anywhere in this project's simulated dataset, and this project's own
explicit design constraint (docs/methods_and_results.md: "colour alone -- no
host photometric redshift, no spectroscopy") rules out using it IN THE TRIGGER
itself. Two fields *are* simulated for every event, signal and background
alike, and are deliberately excluded from the classifier's own feature set
for that same reason: redshift `z` and magnification `mu`. Using them as a
DOWNSTREAM triage cut -- after the photometric trigger has already fired, when
deciding which of the ranked candidates get follow-up time -- does not violate
that constraint, since it plays no role in the trigger decision itself. This
script measures (not merely projects) how much of a gain such a cut buys: it
fits the same TrajectoryClassifier as investigate_purity_tradeoff.py, and at
each FPR operating point and decision epoch, tags every surviving false
positive by its true unlensed subtype and compares its z distribution against
the true positives' -- then tests one concrete secondary filter (a z-window
calibrated on the true positives at that operating point) and reports exactly
how many false positives it removes vs. how many true positives it costs.
Magnification mu is reported descriptively for true positives only (the
simulator gives every unlensed background event mu=1.0 exactly, so there is no
background mu distribution to cut against).
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

# Same literature/placeholder rates as investigate_purity_tradeoff.py -- see
# that script's header comment for the citations and caveats behind these.
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

# Below this many true/false positives in the test set, per-operating-point
# subtype/z-cut statistics are too thin to report -- flagged, not silently
# dropped.
MIN_SAMPLE_FOR_STATS = 10
Z_WINDOW_PERCENTILES = (2.5, 97.5)


def load_data_tagged(data_dir, signal_classes):
    """Like investigate_purity_tradeoff.load_data, but keeps each background
    event's true unlensed subtype instead of discarding it into one pool."""
    pop = {}
    for c in CLASSES:
        with open(os.path.join(data_dir, f'{c}.pkl'), 'rb') as f:
            pop[c] = pickle.load(f)['events']
    signal = [e for c in signal_classes for e in pop[c]]
    background, bg_subtype = [], []
    for c in UNLENSED_CLASSES:
        background.extend(pop[c])
        bg_subtype.extend([c] * len(pop[c]))
    return signal, background, np.array(bg_subtype)


def split_tagged(events, tags, rng, frac_train=0.7):
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    return ([events[i] for i in tr_idx], [events[i] for i in te_idx],
            tags[tr_idx], tags[te_idx])


def split(events, rng, frac_train=0.7):
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    return [events[i] for i in idx[:n_train]], [events[i] for i in idx[n_train:]]


def _subtype_composition(tags):
    if len(tags) == 0:
        return {}
    subtypes, counts = np.unique(tags, return_counts=True)
    return {s: int(c) for s, c in zip(subtypes, counts)}


def analyze_epoch_contamination(clf, k, test_sig, test_bg, bg_subtype_te, ws_te, wb_te,
                                 N_sig_ann, N_bg_ann):
    """Score test events using only information available as of decision epoch
    k, then characterize the false positives at each standard FPR budget."""
    sig_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in test_sig]
    bg_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in test_bg]
    score_sig = clf.score(sig_k)
    score_bg = clf.score(bg_k)
    z_sig = np.array([e['z'] for e in test_sig])
    z_bg = np.array([e['z'] for e in test_bg])
    mu_sig = np.array([e.get('magnification', 1.0) for e in test_sig])

    fpr, tpr, thresholds = clf.roc_curve(sig_k, bg_k, ws_te, wb_te)
    order = np.argsort(fpr)
    fpr, tpr, thresholds = fpr[order], tpr[order], thresholds[order]

    max_bg_score = float(np.max(score_bg))
    pure_mask_sig = score_sig > max_bg_score
    pure_w_sig = np.sum(ws_te[pure_mask_sig])
    total_w_sig = np.sum(ws_te)
    pure_tpr = pure_w_sig / total_w_sig if total_w_sig > 0 else 0.0
    pure_annual_tp = pure_tpr * N_sig_ann

    operating_points = []
    for name, target_f in TABLE_POINTS:
        idx = int(np.argmin(np.abs(fpr - target_f)))
        thr = float(thresholds[idx])
        fp_mask = score_bg >= thr
        tp_mask = score_sig >= thr
        n_fp, n_tp = int(np.sum(fp_mask)), int(np.sum(tp_mask))

        f_val, t_val = float(fpr[idx]), float(tpr[idx])
        tp_ann, fp_ann = t_val * N_sig_ann, f_val * N_bg_ann
        purity = tp_ann / (tp_ann + fp_ann) * 100 if (tp_ann + fp_ann) > 0 else 0.0

        op = {
            'name': name, 'target_fpr': target_f, 'achieved_fpr': f_val, 'achieved_tpr': t_val,
            'annual_tp': tp_ann, 'annual_fp': fp_ann, 'purity_pct': purity, 'threshold': thr,
            'n_fp_test': n_fp, 'n_tp_test': n_tp,
            'thin_sample': n_fp < MIN_SAMPLE_FOR_STATS or n_tp < MIN_SAMPLE_FOR_STATS,
            'fp_subtype_composition': _subtype_composition(bg_subtype_te[fp_mask]),
            'tp_z_median': float(np.median(z_sig[tp_mask])) if n_tp > 0 else None,
            'tp_mu_median': float(np.median(mu_sig[tp_mask])) if n_tp > 0 else None,
            'z_secondary_cut': None,
        }

        if n_tp >= MIN_SAMPLE_FOR_STATS and n_fp >= MIN_SAMPLE_FOR_STATS:
            z_tp = z_sig[tp_mask]
            z_lo, z_hi = (float(x) for x in np.percentile(z_tp, Z_WINDOW_PERCENTILES))
            fp_z = z_bg[fp_mask]
            fp_in_window = (fp_z >= z_lo) & (fp_z <= z_hi)
            tp_in_window = (z_tp >= z_lo) & (z_tp <= z_hi)
            n_fp_after, n_tp_after = int(np.sum(fp_in_window)), int(np.sum(tp_in_window))
            tp_ann_after = tp_ann * (n_tp_after / n_tp)
            fp_ann_after = fp_ann * (n_fp_after / n_fp)
            purity_after = (tp_ann_after / (tp_ann_after + fp_ann_after) * 100
                             if (tp_ann_after + fp_ann_after) > 0 else 0.0)
            op['z_secondary_cut'] = {
                'z_window': [z_lo, z_hi],
                'fp_before': n_fp, 'fp_after': n_fp_after,
                'fp_removed_pct': 100.0 * (n_fp - n_fp_after) / n_fp,
                'tp_before': n_tp, 'tp_after': n_tp_after,
                'tp_retained_pct': 100.0 * n_tp_after / n_tp,
                'annual_tp_after': tp_ann_after, 'annual_fp_after': fp_ann_after,
                'purity_before_pct': purity, 'purity_after_pct': purity_after,
            }
        operating_points.append(op)

    return {
        'fpr': fpr, 'tpr': tpr, 'pure_tpr': pure_tpr, 'pure_annual_tp': pure_annual_tp,
        'max_bg_score': max_bg_score, 'operating_points': operating_points,
        'score_sig': score_sig, 'score_bg': score_bg, 'z_sig': z_sig, 'z_bg': z_bg,
        'bg_subtype_te': bg_subtype_te,
    }


def _op(results_epoch, prefix):
    return next(o for o in results_epoch['operating_points'] if o['name'].startswith(prefix))


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

    print(f'Loading data from {args.data_dir} for {signal_label}...')
    signal, background, bg_subtype = load_data_tagged(args.data_dir, signal_classes)

    rng = np.random.default_rng(args.seed)
    train_sig, test_sig = split(signal, rng)
    train_bg, test_bg, bg_subtype_tr, bg_subtype_te = split_tagged(background, bg_subtype, rng)
    ws_tr = [e['weight'] for e in train_sig]
    wb_tr = [e['weight'] for e in train_bg]
    ws_te = np.array([e['weight'] for e in test_sig])
    wb_te = np.array([e['weight'] for e in test_bg])

    print(f'Fitting TrajectoryClassifier ONCE on full (uncensored) '
          f'{len(train_sig)} signal, {len(train_bg)} background (unlensed_* only)...')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(train_sig, train_bg, ws_tr, wb_tr)

    results = {}
    print('\n' + '=' * 100)
    print(f'CONTAMINANT PROPERTIES & SECONDARY-CUT INVESTIGATION: {signal_label.upper()}')
    print('=' * 100)
    for k, day in enumerate(TRAJECTORY_GRID):
        res = analyze_epoch_contamination(clf, k, test_sig, test_bg, bg_subtype_te,
                                           ws_te, wb_te, N_sig_ann, N_bg_ann)
        results[day] = res
        print(f'\n--- Decision epoch: trigger+{day:.0f}d (k={k}) ---')
        print(f'Zero-FP pure sweet spot: {res["pure_annual_tp"]:.2f} / yr '
              f'({res["pure_tpr"]*100:.2f}% recovery) -- 0 false positives by construction, '
              f'nothing to characterize.')
        for op in res['operating_points']:
            thin = ' [THIN SAMPLE]' if op['thin_sample'] else ''
            print(f'\n  {op["name"]}{thin}: FPR={op["achieved_fpr"]*100:.3f}% '
                  f'TPR={op["achieved_tpr"]*100:.2f}% TP/yr={op["annual_tp"]:.1f} '
                  f'FP/yr={op["annual_fp"]:.0f} Purity={op["purity_pct"]:.2f}% '
                  f'(n_fp_test={op["n_fp_test"]}, n_tp_test={op["n_tp_test"]})')
            if op['fp_subtype_composition']:
                total_fp = sum(op['fp_subtype_composition'].values())
                comp_str = ', '.join(f'{s}={c} ({100*c/total_fp:.0f}%)'
                                      for s, c in sorted(op['fp_subtype_composition'].items(),
                                                          key=lambda kv: -kv[1]))
                print(f'    FP subtype composition: {comp_str}')
            if op['tp_z_median'] is not None:
                print(f'    TP median z={op["tp_z_median"]:.2f}, median mu={op["tp_mu_median"]:.1f}')
            zc = op['z_secondary_cut']
            if zc is not None:
                print(f'    Secondary z-window cut [{zc["z_window"][0]:.2f}, {zc["z_window"][1]:.2f}] '
                      f'(TP {Z_WINDOW_PERCENTILES[0]}-{Z_WINDOW_PERCENTILES[1]} pct range): '
                      f'removes {zc["fp_removed_pct"]:.1f}% of FPs, costs {100-zc["tp_retained_pct"]:.1f}% of TPs '
                      f'-- purity {zc["purity_before_pct"]:.2f}% -> {zc["purity_after_pct"]:.2f}%')
    print('=' * 100)

    # ------------------------------------------------------------------
    # Figure 1: ranked-follow-up summary -- TP/yr and purity vs FPR budget,
    # one line per decision epoch.
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(TRAJECTORY_GRID)))
    budget_names = [name.split(' ')[0] for name, _ in TABLE_POINTS]
    x = np.arange(len(TABLE_POINTS))
    for (day, res), color in zip(results.items(), cmap):
        tp_vals = [o['annual_tp'] for o in res['operating_points']]
        pur_vals = [o['purity_pct'] for o in res['operating_points']]
        ax1.plot(x, tp_vals, 'o-', color=color, lw=2, label=f'+{day:.0f}d')
        ax2.plot(x, pur_vals, 'o-', color=color, lw=2, label=f'+{day:.0f}d')
    for ax, ylabel, title in ((ax1, f'Annual true positives ({signal_label}/yr)',
                                'Ranked follow-up yield vs. FPR budget'),
                               (ax2, 'Purity (%)', 'Ranked follow-up purity vs. FPR budget')):
        ax.set_xticks(x); ax.set_xticklabels(budget_names)
        ax.set_xlabel('FPR budget'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(title='Decision epoch', fontsize=8); ax.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle(f'Recommended consumption model: ranked candidates at a chosen FPR budget '
                 f'({signal_label}), not a zero-contamination cut', fontsize=11)
    fig.tight_layout()
    summary_path = os.path.join(args.out_dir, f'ranked_followup_summary{file_suffix}.png')
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {summary_path}')

    # ------------------------------------------------------------------
    # Figure 2: FP subtype composition at each FPR budget, +60d epoch
    # (most mature decision point), vs. the overall unlensed test mix.
    # ------------------------------------------------------------------
    last_day = TRAJECTORY_GRID[-1]
    res60 = results[last_day]
    overall_comp = _subtype_composition(res60['bg_subtype_te'])
    overall_total = sum(overall_comp.values())
    subtypes_sorted = sorted(overall_comp, key=lambda s: -overall_comp[s])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Flag budgets whose FP count is too thin to trust the composition shown
    # (e.g. a single leaked-through event reading as "100% one subtype").
    labels = [f'{name}\n(n={op["n_fp_test"]}){"*" if op["thin_sample"] else ""}'
              for name, op in zip(budget_names, res60['operating_points'])] + ['Overall bg\nmix']
    bar_x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    cmap2 = plt.cm.tab10(np.linspace(0, 1, len(subtypes_sorted)))
    for subtype, color in zip(subtypes_sorted, cmap2):
        fracs = []
        for op in res60['operating_points']:
            total = sum(op['fp_subtype_composition'].values())
            fracs.append(100.0 * op['fp_subtype_composition'].get(subtype, 0) / total if total > 0 else 0.0)
        fracs.append(100.0 * overall_comp[subtype] / overall_total)
        ax.bar(bar_x, fracs, bottom=bottoms, color=color, label=subtype, width=0.6)
        bottoms += np.array(fracs)
    ax.set_xticks(bar_x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Share of false positives (%)')
    ax.set_title(f'False-positive subtype composition by FPR budget ({signal_label}, +60d)\n'
                 f'rightmost bar = overall unlensed test-set mix, for reference', fontsize=11)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    fig.text(0.01, 0.01, f'* = fewer than {MIN_SAMPLE_FOR_STATS} FPs in the test set at that budget -- composition is noisy',
              fontsize=7, style='italic')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    comp_path = os.path.join(args.out_dir, f'fp_subtype_composition{file_suffix}.png')
    fig.savefig(comp_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {comp_path}')

    # ------------------------------------------------------------------
    # Figure 3: TP vs FP redshift distributions at the 10% FPR operating
    # point (+60d), with the secondary z-window cut marked.
    # ------------------------------------------------------------------
    op10 = _op(res60, '10%')
    fig, ax = plt.subplots(figsize=(8, 5.5))
    score_bg, score_sig = res60['score_bg'], res60['score_sig']
    z_sig, z_bg = res60['z_sig'], res60['z_bg']
    thr_val = op10['threshold']
    tp_mask = score_sig >= thr_val
    fp_mask = score_bg >= thr_val
    bins = np.linspace(0, max(z_sig.max(), z_bg.max()), 40)
    ax.hist(z_sig[tp_mask], bins=bins, alpha=0.6, density=True, color='tab:green', label='True positives')
    ax.hist(z_bg[fp_mask], bins=bins, alpha=0.6, density=True, color='tab:red', label='False positives')
    if op10['z_secondary_cut'] is not None:
        z_lo, z_hi = op10['z_secondary_cut']['z_window']
        ax.axvline(z_lo, color='k', ls='--', lw=1.2)
        ax.axvline(z_hi, color='k', ls='--', lw=1.2,
                   label=f'Secondary z-window cut\n(removes {op10["z_secondary_cut"]["fp_removed_pct"]:.0f}% of FPs, '
                         f'costs {100-op10["z_secondary_cut"]["tp_retained_pct"]:.0f}% of TPs)')
    ax.set_xlabel('Redshift z'); ax.set_ylabel('Density')
    ax.set_title(f'Redshift distribution of TP vs FP at 10% FPR budget ({signal_label}, +60d)\n'
                 f'z is excluded from the classifier\'s own features by design')
    ax.legend(fontsize=8); ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    z_path = os.path.join(args.out_dir, f'contaminant_redshift_10pct{file_suffix}.png')
    fig.savefig(z_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {z_path}')

    # Save summary JSON (score/z arrays dropped -- large and reconstructable)
    summary = {
        'signal_label': signal_label, 'N_sig_annual': N_sig_ann, 'N_bg_annual': N_bg_ann,
        'by_decision_epoch': {
            f'{day:.0f}d': {
                'pure_tpr': res['pure_tpr'], 'pure_annual_tp': res['pure_annual_tp'],
                'operating_points': [
                    {k: v for k, v in op.items()} for op in res['operating_points']
                ],
            }
            for day, res in results.items()
        }
    }
    json_path = os.path.join(args.out_dir, f'contaminant_properties{file_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
