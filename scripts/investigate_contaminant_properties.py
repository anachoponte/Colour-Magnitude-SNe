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

IMPORTANT -- every per-event statistic below is RATE-WEIGHTED, not a raw
test-set count. This dataset is produced by pilot+boost importance sampling
(cmsne.importance_sampling), which deliberately over-samples rare, extreme
(z, magnification) regions for Monte Carlo statistics and corrects each such
event's own `weight` down accordingly. An earlier version of this script
computed subtype composition, z/trigger_day/SNR medians, and the z-cut's
removed/retained percentages from raw unweighted counts -- which silently
re-introduces exactly the sampling-density bias this project's own importance
weights exist to correct for, and produced a materially misleading picture
("surprising high-z non-lensed contaminants" that were mostly deliberately
oversampled, heavily-downweighted rare draws, not a real contamination level;
see docs/session_writeup.md Finding 5's correction). Composition uses
cmsne.importance_sampling.weighted_composition; all medians and the z-window
percentiles use weighted_quantile/weighted_median; both histogram figures
pass `weights=` to matplotlib. Any new diagnostic added to this script MUST
go through these same weighted helpers, never a plain np.median/np.percentile
or a raw np.unique count, on pain of repeating this exact mistake.
"""
import argparse, json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID, reveal_up_to
from cmsne.lsst import CC_FRACTIONS
from cmsne.importance_sampling import (weighted_composition, weighted_quantile,
                                        weighted_median, effective_sample_size)

CLASSES = ['lensed_Ia', 'unlensed_Ia', 'lensed_sn1bc', 'lensed_sn2l', 'lensed_sn2n',
           'lensed_sn2p', 'unlensed_sn1bc', 'unlensed_sn2l', 'unlensed_sn2n', 'unlensed_sn2p']
UNLENSED_CLASSES = [c for c in CLASSES if c.startswith('unlensed_')]

# Maps each individual lensed CC class to its cmsne.lsst.CC_FRACTIONS source key,
# so its own annual rate can be derived below instead of invented.
_CC_SOURCE_FOR_CLASS = {
    'lensed_sn1bc': 'nugent-sn1bc', 'lensed_sn2l': 'nugent-sn2l',
    'lensed_sn2n': 'nugent-sn2n', 'lensed_sn2p': 'nugent-sn2p',
}

# Literature/placeholder rates -- RETAINED FOR PROVENANCE AND FALLBACK ONLY.
# Prefer annual_rates_from_data() below, which derives the same quantities from
# the analysed run's own LTT per-event weights (cmsne.rates.absolute_rate_weight).
# See issue #63 for the measurement that retired these as the default, and the
# citation/derivation notes in investigate_purity_tradeoff.py plus issue #43.
#
# These are WRONG IN ORDERING, not merely uncertain in scale. The per-CC-subtype
# entries split the lensed-CC total by cmsne.lsst.CC_FRACTIONS (Li et al. 2011,
# Shivvers et al. 2017), which are VOLUMETRIC, UNLENSED subtype demographics.
# Applying them to a magnification- and detection-limited lensed sample discards
# the very selection effect the simulation exists to model: SNe IIn are
# intrinsically luminous and long-lived, so they stay above threshold out into
# the high-z region where LensingOpticalDepth.dtau_dmu_dz is largest. Measured on
# results/brute15_62368493, IIn are 90.6% of lensed CC by summed weight and the
# single dominant lensed class overall (272.2/yr of 449.3/yr); CC_FRACTIONS
# assigns them 6%, ranking them fourth of five.
LITERATURE_ANNUAL_RATES = {
    'all_lensed': {'N_sig': 45.0, 'N_bg': 100_000.0},
    'lensed_ia':  {'N_sig': 10.0, 'N_bg': 100_000.0},
}
_LENSED_CC_TOTAL_ANN = (LITERATURE_ANNUAL_RATES['all_lensed']['N_sig']
                        - LITERATURE_ANNUAL_RATES['lensed_ia']['N_sig'])
for _cls, _source in _CC_SOURCE_FOR_CLASS.items():
    LITERATURE_ANNUAL_RATES[_cls] = {
        'N_sig': _LENSED_CC_TOTAL_ANN * CC_FRACTIONS[_source], 'N_bg': 100_000.0}

TARGET_SIGNAL_CLASSES = {
    'all_lensed': [c for c in CLASSES if c.startswith('lensed_')],
    'lensed_ia': ['lensed_Ia'],
    'lensed_sn1bc': ['lensed_sn1bc'], 'lensed_sn2l': ['lensed_sn2l'],
    'lensed_sn2n': ['lensed_sn2n'], 'lensed_sn2p': ['lensed_sn2p'],
}
TARGET_LABELS = {
    'all_lensed': ('all lensed', '_alllensed'),
    'lensed_ia': ('lensed Ia', ''),
    'lensed_sn1bc': ('lensed SN Ibc', '_snibc'),
    'lensed_sn2l': ('lensed SN IIL', '_sn2l'),
    'lensed_sn2n': ('lensed SN IIn', '_sn2n'),
    'lensed_sn2p': ('lensed SN IIP', '_sn2p'),
}


def _summary_annual_rate(data_dir, cls):
    """events/yr in the LSST WFD footprint for one simulated class.

    Reads `annual_detection_rate` from that class's run summary: the weighted
    sum over its DETECTED events, which is exactly the quantity
    cmsne.rates.absolute_rate_weight() produces per event (the LTT
    absolute-rate framework; issue #43's "summed event weights").
    """
    with open(os.path.join(data_dir, f'{cls}.summary.json')) as fh:
        return float(json.load(fh)['annual_detection_rate'])


def annual_rates_from_data(data_dir, target, strict=False):
    """Annual signal/background rates for `target`, from the run's own weights.

    N_sig sums the target's lensed classes; N_bg sums all five unlensed classes.

    NO renormalisation to the resolvable, multiply-imaged lensed population
    (~45/yr in the literature) is applied, BY DESIGN: this colour method exists
    to pick out systems whose image multiplicity is NOT recognised, so the
    magnified population the simulation integrates (mu in [2, 50],
    min_detections=5) is the correct target population, and the LTT rates apply
    directly. See issue #63.

    Falls back to LITERATURE_ANNUAL_RATES (with a warning on stderr) for
    datasets that predate per-class summaries. Pass strict=True to raise
    instead -- appropriate when a result is going into a paper.
    """
    try:
        n_sig = sum(_summary_annual_rate(data_dir, c)
                    for c in TARGET_SIGNAL_CLASSES[target])
        n_bg = sum(_summary_annual_rate(data_dir, c) for c in UNLENSED_CLASSES)
    except (OSError, KeyError, ValueError, TypeError):
        if strict:
            raise
        print(f'WARNING: could not derive annual rates from {data_dir}; falling '
              f'back to the LITERATURE_ANNUAL_RATES placeholders for {target}. '
              f'Those get the lensed subtype ordering wrong (see issue #63) -- '
              f'treat any purity figure from this run as indicative only.',
              file=sys.stderr)
        return dict(LITERATURE_ANNUAL_RATES[target])
    return {'N_sig': n_sig, 'N_bg': n_bg}

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


def analyze_epoch_contamination(clf, k, test_sig, test_bg, bg_subtype_te, ws_te, wb_te,
                                 N_sig_ann, N_bg_ann):
    """Score test events using only information available as of decision epoch
    k, then characterize the false positives at each standard FPR budget.

    Every per-event statistic reported here is rate-weighted (see the module
    docstring) -- composition, medians, and the z-cut's removed/retained
    percentages all go through weighted_composition/weighted_quantile, never
    a plain np.median/np.unique count, on this dataset's own importance-
    sampled population."""
    sig_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in test_sig]
    bg_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in test_bg]
    score_sig = clf.score(sig_k)
    score_bg = clf.score(bg_k)
    z_sig = np.array([e['z'] for e in test_sig])
    z_bg = np.array([e['z'] for e in test_bg])
    mu_sig = np.array([e.get('magnification', 1.0) for e in test_sig])
    # Two more fields simulated for every event but never fed to the
    # classifier as a feature: the survey epoch of first detection, and the
    # final cumulative-best-magnitude measurement uncertainty (a photometric
    # SNR proxy) -- purely diagnostic here, not a proposed cut.
    trigday_sig = np.array([e.get('trigger_day', np.nan) for e in test_sig])
    trigday_bg = np.array([e.get('trigger_day', np.nan) for e in test_bg])
    err_sig = np.array([e.get('peakmag_d60_err', np.nan) for e in test_sig])
    err_bg = np.array([e.get('peakmag_d60_err', np.nan) for e in test_bg])

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
        w_fp, w_tp = wb_te[fp_mask], ws_te[tp_mask]
        fp_ess, tp_ess = effective_sample_size(w_fp), effective_sample_size(w_tp)

        f_val, t_val = float(fpr[idx]), float(tpr[idx])
        tp_ann, fp_ann = t_val * N_sig_ann, f_val * N_bg_ann
        purity = tp_ann / (tp_ann + fp_ann) * 100 if (tp_ann + fp_ann) > 0 else 0.0

        op = {
            'name': name, 'target_fpr': target_f, 'achieved_fpr': f_val, 'achieved_tpr': t_val,
            'annual_tp': tp_ann, 'annual_fp': fp_ann, 'purity_pct': purity, 'threshold': thr,
            'n_fp_test': n_fp, 'n_tp_test': n_tp, 'fp_ess': fp_ess, 'tp_ess': tp_ess,
            'thin_sample': (n_fp < MIN_SAMPLE_FOR_STATS or n_tp < MIN_SAMPLE_FOR_STATS or
                            fp_ess < MIN_SAMPLE_FOR_STATS or tp_ess < MIN_SAMPLE_FOR_STATS),
            'fp_subtype_composition': weighted_composition(bg_subtype_te[fp_mask], w_fp),
            'tp_z_median': weighted_median(z_sig[tp_mask], w_tp) if n_tp > 0 else None,
            'tp_mu_median': weighted_median(mu_sig[tp_mask], w_tp) if n_tp > 0 else None,
            'tp_trigger_day_median': weighted_median(trigday_sig[tp_mask], w_tp) if n_tp > 0 else None,
            'fp_trigger_day_median': weighted_median(trigday_bg[fp_mask], w_fp) if n_fp > 0 else None,
            'tp_peakmag_err_median': weighted_median(err_sig[tp_mask], w_tp) if n_tp > 0 else None,
            'fp_peakmag_err_median': weighted_median(err_bg[fp_mask], w_fp) if n_fp > 0 else None,
            'z_secondary_cut': None,
        }

        if (n_tp >= MIN_SAMPLE_FOR_STATS and n_fp >= MIN_SAMPLE_FOR_STATS and
                tp_ess >= MIN_SAMPLE_FOR_STATS and fp_ess >= MIN_SAMPLE_FOR_STATS):
            z_tp, fp_z = z_sig[tp_mask], z_bg[fp_mask]
            q_lo, q_hi = (p / 100.0 for p in Z_WINDOW_PERCENTILES)
            z_lo = weighted_quantile(z_tp, w_tp, q_lo)
            z_hi = weighted_quantile(z_tp, w_tp, q_hi)
            fp_in_window = (fp_z >= z_lo) & (fp_z <= z_hi)
            tp_in_window = (z_tp >= z_lo) & (z_tp <= z_hi)
            fp_w_before, fp_w_after = float(w_fp.sum()), float(w_fp[fp_in_window].sum())
            tp_w_before, tp_w_after = float(w_tp.sum()), float(w_tp[tp_in_window].sum())
            fp_removed_pct = float(np.clip(100.0 * (1.0 - fp_w_after / fp_w_before), 0.0, 100.0)) if fp_w_before > 0 else 0.0
            tp_retained_pct = float(np.clip(100.0 * tp_w_after / tp_w_before, 0.0, 100.0)) if tp_w_before > 0 else 0.0
            tp_ann_after = tp_ann * (tp_w_after / tp_w_before) if tp_w_before > 0 else 0.0
            fp_ann_after = fp_ann * (fp_w_after / fp_w_before) if fp_w_before > 0 else 0.0
            purity_after = (tp_ann_after / (tp_ann_after + fp_ann_after) * 100
                             if (tp_ann_after + fp_ann_after) > 0 else 0.0)
            op['z_secondary_cut'] = {
                'z_window': [z_lo, z_hi],
                'fp_before': n_fp, 'fp_after': int(np.sum(fp_in_window)),
                'fp_removed_pct': fp_removed_pct,
                'tp_before': n_tp, 'tp_after': int(np.sum(tp_in_window)),
                'tp_retained_pct': tp_retained_pct,
                'annual_tp_after': tp_ann_after, 'annual_fp_after': fp_ann_after,
                'purity_before_pct': purity, 'purity_after_pct': purity_after,
            }
        operating_points.append(op)

    return {
        'fpr': fpr, 'tpr': tpr, 'pure_tpr': pure_tpr, 'pure_annual_tp': pure_annual_tp,
        'max_bg_score': max_bg_score, 'operating_points': operating_points,
        'score_sig': score_sig, 'score_bg': score_bg, 'z_sig': z_sig, 'z_bg': z_bg,
        'trigday_sig': trigday_sig, 'trigday_bg': trigday_bg,
        'err_sig': err_sig, 'err_bg': err_bg,
        'bg_subtype_te': bg_subtype_te, 'ws_te': ws_te, 'wb_te': wb_te,
    }


def _op(results_epoch, prefix):
    return next(o for o in results_epoch['operating_points'] if o['name'].startswith(prefix))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', default='docs/figures/purity_investigation')
    ap.add_argument('--target', choices=sorted(TARGET_SIGNAL_CLASSES), default='all_lensed')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    signal_classes = TARGET_SIGNAL_CLASSES[args.target]
    signal_label, file_suffix = TARGET_LABELS[args.target]

    rates = annual_rates_from_data(args.data_dir, args.target)
    N_sig_ann, N_bg_ann = rates['N_sig'], rates['N_bg']
    print(f'Annual rates for {args.target}: N_sig={N_sig_ann:.4g}/yr, '
          f'N_bg={N_bg_ann:.4g}/yr (derived from {args.data_dir})')

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
                  f'(n_fp_test={op["n_fp_test"]}, n_tp_test={op["n_tp_test"]}, '
                  f'fp_ess={op["fp_ess"]:.1f}, tp_ess={op["tp_ess"]:.1f})')
            if op['fp_subtype_composition']:
                # Weight fractions (rate-weighted, NOT raw counts -- see module docstring).
                comp_str = ', '.join(f'{s}={100*frac:.0f}%'
                                      for s, frac in sorted(op['fp_subtype_composition'].items(),
                                                             key=lambda kv: -kv[1]))
                print(f'    FP subtype composition (rate-weighted): {comp_str}')
            if op['tp_z_median'] is not None:
                print(f'    TP median z={op["tp_z_median"]:.2f}, median mu={op["tp_mu_median"]:.1f}, '
                      f'median trigger_day={op["tp_trigger_day_median"]:.1f}, '
                      f'median peakmag_d60_err={op["tp_peakmag_err_median"]:.3f}')
            if op['fp_trigger_day_median'] is not None:
                print(f'    FP median trigger_day={op["fp_trigger_day_median"]:.1f}, '
                      f'median peakmag_d60_err={op["fp_peakmag_err_median"]:.3f}')
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
    # Rate-weighted, like every other composition figure here (see module
    # docstring) -- NOT the raw count mix of this importance-sampled test set.
    overall_comp = weighted_composition(res60['bg_subtype_te'], res60['wb_te'])
    subtypes_sorted = sorted(overall_comp, key=lambda s: -overall_comp[s])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Flag budgets whose FP count/ESS is too thin to trust the composition
    # shown (e.g. a single leaked-through event reading as "100% one subtype").
    labels = [f'{name}\n(n={op["n_fp_test"]}, ESS={op["fp_ess"]:.0f}){"*" if op["thin_sample"] else ""}'
              for name, op in zip(budget_names, res60['operating_points'])] + ['Overall bg\nmix']
    bar_x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    cmap2 = plt.cm.tab10(np.linspace(0, 1, len(subtypes_sorted)))
    for subtype, color in zip(subtypes_sorted, cmap2):
        fracs = [100.0 * op['fp_subtype_composition'].get(subtype, 0.0) for op in res60['operating_points']]
        fracs.append(100.0 * overall_comp[subtype])
        ax.bar(bar_x, fracs, bottom=bottoms, color=color, label=subtype, width=0.6)
        bottoms += np.array(fracs)
    ax.set_xticks(bar_x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Share of false positives (%, rate-weighted)')
    ax.set_title(f'False-positive subtype composition by FPR budget ({signal_label}, +60d)\n'
                 f'rightmost bar = overall unlensed test-set mix, for reference', fontsize=11)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    fig.text(0.01, 0.01,
              f'Rate-weighted composition, not raw counts. * = fewer than {MIN_SAMPLE_FOR_STATS} FPs or '
              f'ESS < {MIN_SAMPLE_FOR_STATS} at that budget -- composition is unreliable',
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
    ws_te, wb_te = res60['ws_te'], res60['wb_te']
    thr_val = op10['threshold']
    tp_mask = score_sig >= thr_val
    fp_mask = score_bg >= thr_val
    w_tp_10, w_fp_10 = ws_te[tp_mask], wb_te[fp_mask]
    bins = np.linspace(0, max(z_sig.max(), z_bg.max()), 40)
    # weights= makes these rate-weighted densities, not raw-count histograms
    # -- see module docstring on why that distinction matters here.
    ax.hist(z_sig[tp_mask], bins=bins, weights=w_tp_10, alpha=0.6, density=True,
            color='tab:green', label='True positives')
    ax.hist(z_bg[fp_mask], bins=bins, weights=w_fp_10, alpha=0.6, density=True,
            color='tab:red', label='False positives')
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

    # ------------------------------------------------------------------
    # Figures 4-5: TP vs FP distributions of two more fields simulated for
    # every event but never used as classifier features -- survey epoch of
    # first detection, and final photometric uncertainty (SNR proxy) -- at
    # the same 10% FPR / +60d reference point as Figure 3. Diagnostic only:
    # no secondary cut is proposed on these this round.
    # ------------------------------------------------------------------
    for field, sig_arr, bg_arr, xlabel, fname in (
        ('trigger_day', res60['trigday_sig'], res60['trigday_bg'],
         'Survey day of first detection (trigger_day)', 'contaminant_trigday_10pct'),
        ('peakmag_d60_err', res60['err_sig'], res60['err_bg'],
         'Final cumulative-best-magnitude uncertainty (mag)', 'contaminant_snr_10pct'),
    ):
        tp_fin, fp_fin = np.isfinite(sig_arr[tp_mask]), np.isfinite(bg_arr[fp_mask])
        tp_vals, fp_vals = sig_arr[tp_mask][tp_fin], bg_arr[fp_mask][fp_fin]
        w_tp_vals, w_fp_vals = w_tp_10[tp_fin], w_fp_10[fp_fin]
        if len(tp_vals) == 0 or len(fp_vals) == 0:
            print(f'Skipping {field} figure -- no finite values at this operating point.')
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        lo, hi = min(tp_vals.min(), fp_vals.min()), max(tp_vals.max(), fp_vals.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(tp_vals, bins=bins, weights=w_tp_vals, alpha=0.6, density=True,
                color='tab:green', label='True positives')
        ax.hist(fp_vals, bins=bins, weights=w_fp_vals, alpha=0.6, density=True,
                color='tab:red', label='False positives')
        ax.set_xlabel(xlabel); ax.set_ylabel('Density')
        ax.set_title(f'{xlabel} -- TP vs FP at 10% FPR budget ({signal_label}, +60d)\n'
                     f'diagnostic only -- not used as a classifier feature or a proposed cut')
        ax.legend(fontsize=8); ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f'{fname}{file_suffix}.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Wrote {out_path}')

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
