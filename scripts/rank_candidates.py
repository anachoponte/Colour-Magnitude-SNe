"""Operationalizes the "ranked candidate list at a chosen FPR budget" recommendation
(docs/session_writeup.md Executive Summary point 7, Finding 5) for the real deployed
TrajectoryClassifier, instead of leaving it as a documentation recommendation.

For each target, fits one TrajectoryClassifier (train split), calibrates the four
standard FPR-budget thresholds (10%/1%/0.1%/0.01%, same convention as
investigate_purity_tradeoff.py/investigate_contaminant_properties.py) on a separate
calib split, then RANKS a third, disjoint val split -- the "new candidates" -- using
only `rank()`/`score()`/`probability()`. This is what this project's own code already
recommends over a hard `predict()` cut (see their docstrings).

IMPORTANT -- the ranking/scoring/tier-assignment logic below never looks at the true
class label. Every candidate in `val_events` is scored purely from its own features,
exactly as it would be for a genuinely new, unconfirmed alert. The true label is
reattached ONLY for display (the printed table, the CSV export, and the two
validation figures), clearly after the ranking decision is already made, so this
demo can show how well the (label-blind) ranking actually worked -- never to inform
it. Every place this happens is marked "DISPLAY ONLY" below.

All reported candidate counts, purities and lift curves are RATE-WEIGHTED to
expected on-sky events/year via cmsne.importance_sampling conventions (see
`annual_shares` and PR #57): the simulated val split's lensed:unlensed ratio is a
generation artefact, so raw event counts would wildly overstate purity. Raw counts
are carried alongside only as a sample-size indicator, and tiers whose effective
sample size is too thin to support a purity number are flagged, not hidden.

Output per target (reference epoch +60d, the most mature decision point):
  - `ranked_candidates_<target>.csv`: the actual operational artifact -- every val
    candidate, ranked, with its score/probability/tier and its expected annual
    rate contribution, plus the DISPLAY-ONLY true label for inspection.
  - `rank_lift_<target>.png`: cumulative true vs. false candidates captured against
    expected candidates reviewed per year -- directly answers "how many real
    detections do I get for how much follow-up load."
  - `rank_distribution_<target>.png`: rate-weighted percentile-rank distribution of
    true vs. false candidates, overlaid -- shows how much further up the list true
    candidates cluster.
  - `rank_candidates_<target>.json`: tier counts and a down-sampled lift-curve table
    for all 5 decision epochs (the figures above use +60d only).

Does not estimate real follow-up time cost (exposure time, telescope scheduling) of
working down this list to a given tier -- see issue #61.
"""
import argparse, csv, json, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID, reveal_up_to
from cmsne.importance_sampling import effective_sample_size
from investigate_contaminant_properties import (
    TARGET_SIGNAL_CLASSES, TARGET_LABELS, ANNUAL_RATES, TABLE_POINTS, load_data_tagged,
    MIN_SAMPLE_FOR_STATS,
)
from validate_zcut_outofsample import split3

REFERENCE_DAY = TRAJECTORY_GRID[-1]  # +60d, the most mature decision point


def annual_shares(w_sig, w_bg, N_sig_ann, N_bg_ann):
    """Per-event expected on-sky events/year, for signal and background separately.

    The simulated val split's raw lensed:unlensed ratio is an artefact of how
    many of each class were generated (and of importance sampling), NOT the
    on-sky ratio. Every candidate-count, purity and lift number reported here
    must therefore be rate-weighted the same way the rest of scripts/ does
    (cmsne.importance_sampling convention; see PR #57) -- each class's weights
    are renormalised to sum to that class's literature annual rate, so one unit
    of "candidate" means the same thing on both sides.
    """
    tot_s, tot_b = float(np.sum(w_sig)), float(np.sum(w_bg))
    a_sig = w_sig / tot_s * N_sig_ann if tot_s > 0 else np.zeros_like(w_sig)
    a_bg = w_bg / tot_b * N_bg_ann if tot_b > 0 else np.zeros_like(w_bg)
    return a_sig, a_bg


def calibrate_thresholds(clf, calib_sig_k, calib_bg_k, w_calib_sig, w_calib_bg):
    """FPR-budget thresholds chosen on calib only -- see module docstring."""
    fpr_c, _tpr_c, thr_c = clf.roc_curve(calib_sig_k, calib_bg_k, w_calib_sig, w_calib_bg)
    order = np.argsort(fpr_c)
    fpr_c, thr_c = fpr_c[order], thr_c[order]
    thresholds = {}
    for name, target_f in TABLE_POINTS:
        idx = int(np.argmin(np.abs(fpr_c - target_f)))
        thresholds[name] = float(thr_c[idx])
    return thresholds


def assign_tier(score, thresholds):
    """Strictest FPR budget this score still clears, nested 0.01% < 0.1% < 1% < 10%.

    TABLE_POINTS is ordered loosest-first (10% -> 0.01%), but the strictest
    budget has the HIGHEST score threshold, so it must be checked first --
    otherwise everything clearing the loose 10% threshold (which is also
    cleared by anything meeting a stricter one) gets bucketed there and the
    stricter tiers always read zero.
    """
    for name, _ in reversed(TABLE_POINTS):
        if score >= thresholds[name]:
            return name
    return 'below 10% FPR budget'


def rank_epoch(clf, k, calib_sig, calib_bg, val_sig, val_bg, w_calib_sig, w_calib_bg,
               w_val_sig, w_val_bg, N_sig_ann, N_bg_ann):
    calib_sig_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in calib_sig]
    calib_bg_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in calib_bg]
    thresholds = calibrate_thresholds(clf, calib_sig_k, calib_bg_k, w_calib_sig, w_calib_bg)

    # ---- Everything from here to the marked line is what a real deployment would
    # do against genuinely new, unconfirmed candidates: no label is used. ----
    val_sig_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in val_sig]
    val_bg_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in val_bg]
    val_events = val_sig_k + val_bg_k
    order = clf.rank(val_events)              # indices into val_events, best-first
    scores = clf.score(val_events)
    probs = clf.probability(val_events)
    tiers = [assign_tier(float(scores[i]), thresholds) for i in range(len(val_events))]
    # ---- DISPLAY ONLY from here: true label reattached purely for inspection ----
    true_label = np.array([1] * len(val_sig_k) + [0] * len(val_bg_k))
    # Rate weights, in the same val_events order (signal block then background
    # block), renormalised to annual on-sky rates -- see annual_shares().
    a_sig, a_bg = annual_shares(w_val_sig, w_val_bg, N_sig_ann, N_bg_ann)
    annual = np.concatenate([a_sig, a_bg])
    raw_w = np.concatenate([w_val_sig, w_val_bg])

    ranked_rows = []
    cum_tp, cum_fp = 0, 0
    cum_tp_ann, cum_fp_ann = 0.0, 0.0
    blank = lambda: {'n_tp': 0, 'n_fp': 0, 'annual_tp': 0.0, 'annual_fp': 0.0,
                     'w_tp': [], 'w_fp': []}
    tier_stats = {name: blank() for name, _ in TABLE_POINTS}
    tier_stats['below 10% FPR budget'] = blank()
    for rank_pos, idx in enumerate(order, start=1):
        is_tp = bool(true_label[idx])  # DISPLAY ONLY
        cum_tp += int(is_tp)
        cum_fp += int(not is_tp)
        ts = tier_stats[tiers[idx]]
        if is_tp:
            cum_tp_ann += float(annual[idx])
            ts['n_tp'] += 1; ts['annual_tp'] += float(annual[idx]); ts['w_tp'].append(raw_w[idx])
        else:
            cum_fp_ann += float(annual[idx])
            ts['n_fp'] += 1; ts['annual_fp'] += float(annual[idx]); ts['w_fp'].append(raw_w[idx])
        ranked_rows.append({
            'rank': rank_pos, 'score': float(scores[idx]), 'probability': float(probs[idx]),
            'tier': tiers[idx],
            'expected_annual_events': float(annual[idx]),
            'cumulative_annual_tp_display_only': cum_tp_ann,
            'cumulative_annual_fp_display_only': cum_fp_ann,
            'cumulative_tp_raw_display_only': cum_tp, 'cumulative_fp_raw_display_only': cum_fp,
            'true_label_display_only': 'lensed (TP)' if is_tp else 'unlensed (FP)',
        })

    # Finalise each tier: rate-weighted purity, plus an ESS-based thin-sample
    # flag so a purity resting on a handful of heavily-weighted events is
    # visible rather than silently reported (same convention as
    # investigate_contaminant_properties.py).
    for ts in tier_stats.values():
        tp_ess = effective_sample_size(np.array(ts['w_tp']))
        fp_ess = effective_sample_size(np.array(ts['w_fp']))
        denom = ts['annual_tp'] + ts['annual_fp']
        ts['purity_pct'] = 100.0 * ts['annual_tp'] / denom if denom > 0 else 0.0
        ts['tp_ess'], ts['fp_ess'] = tp_ess, fp_ess
        ts['thin_sample'] = (ts['n_tp'] + ts['n_fp'] > 0 and
                             (tp_ess + fp_ess) < MIN_SAMPLE_FOR_STATS)
        del ts['w_tp'], ts['w_fp']

    return (ranked_rows, thresholds, tier_stats, int(true_label.sum()),
            int((1 - true_label).sum()), float(np.sum(a_sig)), float(np.sum(a_bg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', default='docs/figures/purity_investigation')
    ap.add_argument('--target', choices=sorted(TARGET_SIGNAL_CLASSES), default='lensed_ia')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    signal_classes = TARGET_SIGNAL_CLASSES[args.target]
    signal_label, file_suffix = TARGET_LABELS[args.target]

    print(f'Loading data from {args.data_dir} for {signal_label}...')
    signal, background, _bg_subtype = load_data_tagged(args.data_dir, signal_classes)

    rng = np.random.default_rng(args.seed)
    train_sig, calib_sig, val_sig = split3(signal, rng)
    train_bg, calib_bg, val_bg = split3(background, rng)
    ws_tr = [e['weight'] for e in train_sig]
    wb_tr = [e['weight'] for e in train_bg]
    w_calib_sig = np.array([e['weight'] for e in calib_sig])
    w_calib_bg = np.array([e['weight'] for e in calib_bg])
    w_val_sig = np.array([e['weight'] for e in val_sig])
    w_val_bg = np.array([e['weight'] for e in val_bg])
    rates = ANNUAL_RATES[args.target]
    N_sig_ann, N_bg_ann = rates['N_sig'], rates['N_bg']

    print(f'Fitting TrajectoryClassifier on train only ({len(train_sig)} signal, '
          f'{len(train_bg)} background) -- calib ({len(calib_sig)}/{len(calib_bg)}) '
          f'calibrates FPR-budget thresholds; val ({len(val_sig)}/{len(val_bg)}) is '
          f'"new candidates" to rank, scored blind to label.')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(train_sig, train_bg, ws_tr, wb_tr)

    print('\n' + '=' * 100)
    print(f'RANKED CANDIDATE LIST: {signal_label.upper()}')
    print('=' * 100)

    by_epoch_summary = {}
    ref_rows = None
    print('All candidate counts and purities below are RATE-WEIGHTED to expected '
          'on-sky events/year\n(signal normalised to N_sig=%.2f/yr, background to '
          'N_bg=%.0f/yr); raw simulated\ncounts are shown alongside only as a '
          'sample-size indicator.' % (N_sig_ann, N_bg_ann))
    for k, day in enumerate(TRAJECTORY_GRID):
        rows, thresholds, tier_stats, n_tp_total, n_fp_total, ann_sig, ann_bg = rank_epoch(
            clf, k, calib_sig, calib_bg, val_sig, val_bg, w_calib_sig, w_calib_bg,
            w_val_sig, w_val_bg, N_sig_ann, N_bg_ann)
        print(f'\n--- Decision epoch: trigger+{day:.0f}d (k={k}) '
              f'(val pool: {ann_sig:.2f} true + {ann_bg:.0f} false candidates/yr; '
              f'{n_tp_total}/{n_fp_total} raw simulated events) ---')
        for name, ts in tier_stats.items():
            n_ann = ts['annual_tp'] + ts['annual_fp']
            flag = '  [THIN SAMPLE]' if ts['thin_sample'] else ''
            print(f'  {name:<38}: {n_ann:9.1f} candidates/yr, {ts["annual_tp"]:7.2f} true / '
                  f'{ts["annual_fp"]:8.1f} false (purity {ts["purity_pct"]:.2f}%) '
                  f'[raw n={ts["n_tp"]}/{ts["n_fp"]}, ESS={ts["tp_ess"]:.1f}/{ts["fp_ess"]:.1f}]{flag}')
        # Down-sample the lift curve to ~200 points for the JSON summary.
        step = max(1, len(rows) // 200)
        by_epoch_summary[f'{day:.0f}d'] = {
            'thresholds': thresholds, 'tier_stats': tier_stats,
            'annual_tp_total': ann_sig, 'annual_fp_total': ann_bg,
            'n_tp_raw': n_tp_total, 'n_fp_raw': n_fp_total,
            'lift_curve_sampled': [
                {'rank': r['rank'],
                 'cumulative_annual_tp': r['cumulative_annual_tp_display_only'],
                 'cumulative_annual_fp': r['cumulative_annual_fp_display_only'],
                 'cumulative_tp_raw': r['cumulative_tp_raw_display_only'],
                 'cumulative_fp_raw': r['cumulative_fp_raw_display_only']}
                for r in rows[::step]
            ],
        }
        if day == REFERENCE_DAY:
            ref_rows = rows
    print('=' * 100)

    # ------------------------------------------------------------------
    # CSV: the actual ranked-list operational artifact, reference epoch only.
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.out_dir, f'ranked_candidates{file_suffix}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ref_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ref_rows)
    print(f'Wrote {csv_path}')

    # ------------------------------------------------------------------
    # Figure 1: cumulative gain / lift curve at the reference epoch.
    # ------------------------------------------------------------------
    # X axis is rate-weighted "candidates you would actually have to review in a
    # year", not raw simulated rank, so the curve answers the operational
    # question in on-sky units.
    reviewed = [r['cumulative_annual_tp_display_only'] + r['cumulative_annual_fp_display_only']
                for r in ref_rows]
    cum_tp = [r['cumulative_annual_tp_display_only'] for r in ref_rows]
    cum_fp = [r['cumulative_annual_fp_display_only'] for r in ref_rows]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(reviewed, cum_tp, color='tab:green', lw=2, label='Cumulative true candidates (lensed)')
    ax.plot(reviewed, cum_fp, color='tab:red', lw=2, label='Cumulative false candidates (unlensed)')
    ax.set_xscale('log')
    ax.set_xlabel('Expected candidates reviewed per year, working down the ranked list '
                  '(rate-weighted)')
    ax.set_ylabel('Cumulative expected events/yr\n(DISPLAY ONLY -- not used by the ranking itself)')
    ax.set_yscale('log')
    ax.set_title(f'Ranked follow-up: true vs. false candidates per N reviewed\n'
                 f'({signal_label}, +60d, rank()/probability() -- label-blind scoring)')
    ax.legend(fontsize=8); ax.grid(True, which='both', linestyle='--', alpha=0.4)
    fig.tight_layout()
    lift_path = os.path.join(args.out_dir, f'rank_lift{file_suffix}.png')
    fig.savefig(lift_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {lift_path}')

    # ------------------------------------------------------------------
    # Figure 2: percentile-rank distribution, TP vs FP overlaid.
    # ------------------------------------------------------------------
    # Percentile rank is itself defined on the rate-weighted list position
    # (what fraction of a year's candidate load you have worked through), and
    # each event enters its histogram with its own rate weight -- an unweighted
    # histogram of simulated events would show the simulation's sampling, not
    # the on-sky distribution.
    total_reviewed = reviewed[-1] if reviewed else 0.0
    pct_rank = [100.0 * x / total_reviewed if total_reviewed > 0 else 0.0 for x in reviewed]
    is_tp = [r['true_label_display_only'].startswith('lensed') for r in ref_rows]
    w_ann = [r['expected_annual_events'] for r in ref_rows]
    pct_rank_tp = [p for p, t in zip(pct_rank, is_tp) if t]
    pct_rank_fp = [p for p, t in zip(pct_rank, is_tp) if not t]
    w_tp = [w for w, t in zip(w_ann, is_tp) if t]
    w_fp = [w for w, t in zip(w_ann, is_tp) if not t]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(0, 100, 40)
    ax.hist(pct_rank_tp, bins=bins, weights=w_tp, alpha=0.6, density=True,
            color='tab:green', label='True candidates (lensed)')
    ax.hist(pct_rank_fp, bins=bins, weights=w_fp, alpha=0.6, density=True,
            color='tab:red', label='False candidates (unlensed)')
    ax.set_xlabel('Percentile of the year\'s candidate load reviewed (0% = top of ranked list)')
    ax.set_ylabel('Density')
    ax.set_title(f'How much further up true candidates rank, vs. false ones\n'
                 f'({signal_label}, +60d)')
    ax.legend(fontsize=8); ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    dist_path = os.path.join(args.out_dir, f'rank_distribution{file_suffix}.png')
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {dist_path}')

    json_path = os.path.join(args.out_dir, f'rank_candidates{file_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump({'signal_label': signal_label, 'by_decision_epoch': by_epoch_summary}, f, indent=2)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
