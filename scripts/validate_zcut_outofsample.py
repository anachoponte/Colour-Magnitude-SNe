"""Out-of-sample validation of Finding 5's secondary z-window cut.

`investigate_contaminant_properties.py` calibrates the z-window (its
2.5-97.5 percentile bounds) and evaluates it (removed/retained weight
percentages) on the *same* held-out test split -- flagged there as an
"in-sample cut calibration" caveat, since the reported gains are an
optimistic estimate of what a cut tuned once and applied to genuinely
unseen data would achieve. This script closes that gap with a proper
three-way split:

  - ``train``: fits the TrajectoryClassifier (as before).
  - ``calib``: picks the FPR-budget threshold AND the z-window bounds,
    exactly as the single test split did before.
  - ``val``: held out from both of the above. The already-chosen threshold
    and window are applied here, fresh, to measure the cut's real
    out-of-sample removed/retained percentages.

It also stress-tests a deployment detail raised when discussing how this
cut would actually be applied: the ``z`` used throughout Finding 5 is the
event's own true redshift, but in practice you cannot know that without
either a resolved host-galaxy photometric redshift (the realistic path --
it never touches the SN's own trigger photometry, so it doesn't reopen the
"colour alone" constraint) or spectroscopy (which defeats the purpose).
Host photo-z has real measurement uncertainty, typically
sigma_z/(1+z) ~ 0.02-0.05 for broad-band photometry. The window itself is
calibrated on CLEAN z (the realistic scenario: a historical sample of
already spectroscopically-confirmed lensed Ia, whose z is known precisely
from the confirmation process itself), but applied to the validation set's
z after adding Gaussian photo-z-like noise -- the realistic scenario for a
new, unconfirmed candidate. Both the clean- and noisy-z out-of-sample
results are reported, so the gap between them shows how much of the cut's
value survives realistic photo-z error.
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.trajectory import TrajectoryClassifier, TRAJECTORY_GRID, reveal_up_to
from cmsne.importance_sampling import weighted_quantile, effective_sample_size
from investigate_contaminant_properties import (
    TARGET_SIGNAL_CLASSES, TARGET_LABELS, TABLE_POINTS,
    MIN_SAMPLE_FOR_STATS, Z_WINDOW_PERCENTILES, load_data_tagged,
    annual_rates_from_data,
)

PHOTOZ_SIGMA_FRAC = 0.03  # sigma_z = PHOTOZ_SIGMA_FRAC * (1 + z); mid-range for broad-band photo-z


def split3(events, rng, frac_train=0.6, frac_calib=0.2):
    """Three-way split: train (classifier fit), calib (pick threshold/window),
    val (evaluate, never seen by either of the above)."""
    idx = rng.permutation(len(events))
    n_train = int(len(events) * frac_train)
    n_calib = int(len(events) * frac_calib)
    tr, ca, va = idx[:n_train], idx[n_train:n_train + n_calib], idx[n_train + n_calib:]
    return [events[i] for i in tr], [events[i] for i in ca], [events[i] for i in va]


def _weights_z(events):
    return (np.array([e['weight'] for e in events]),
            np.array([e['z'] for e in events]))


def evaluate_epoch(clf, k, calib_sig, calib_bg, val_sig, val_bg, N_sig_ann, N_bg_ann, rng):
    calib_sig_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in calib_sig]
    calib_bg_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in calib_bg]
    val_sig_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in val_sig]
    val_bg_k = [reveal_up_to(e, k, TRAJECTORY_GRID) for e in val_bg]

    w_calib_sig, z_calib_sig = _weights_z(calib_sig)
    w_calib_bg, _ = _weights_z(calib_bg)
    w_val_sig, z_val_sig = _weights_z(val_sig)
    w_val_bg, z_val_bg = _weights_z(val_bg)

    score_calib_sig = clf.score(calib_sig_k)
    score_calib_bg = clf.score(calib_bg_k)
    score_val_sig = clf.score(val_sig_k)
    score_val_bg = clf.score(val_bg_k)

    fpr_c, _tpr_c, thr_c = clf.roc_curve(calib_sig_k, calib_bg_k, w_calib_sig, w_calib_bg)
    order = np.argsort(fpr_c)
    fpr_c, thr_c = fpr_c[order], thr_c[order]

    # Realistic photo-z noise applied ONLY to the validation query z's (see
    # module docstring): the window is calibrated on clean z, as a historical
    # confirmed sample would give, and applied to a noisy z, as a new
    # candidate's host photo-z would give.
    sigma_val = PHOTOZ_SIGMA_FRAC * (1.0 + z_val_sig)
    z_val_sig_noisy = z_val_sig + rng.normal(0.0, sigma_val)
    sigma_bg = PHOTOZ_SIGMA_FRAC * (1.0 + z_val_bg)
    z_val_bg_noisy = z_val_bg + rng.normal(0.0, sigma_bg)

    out = []
    for name, target_f in TABLE_POINTS:
        idx = int(np.argmin(np.abs(fpr_c - target_f)))
        thr = float(thr_c[idx])

        calib_fp_mask = score_calib_bg >= thr
        calib_tp_mask = score_calib_sig >= thr
        n_calib_fp, n_calib_tp = int(np.sum(calib_fp_mask)), int(np.sum(calib_tp_mask))
        w_calib_fp_sub, w_calib_tp_sub = w_calib_bg[calib_fp_mask], w_calib_sig[calib_tp_mask]
        calib_fp_ess = effective_sample_size(w_calib_fp_sub)
        calib_tp_ess = effective_sample_size(w_calib_tp_sub)

        val_fp_mask = score_val_bg >= thr
        val_tp_mask = score_val_sig >= thr
        n_val_fp, n_val_tp = int(np.sum(val_fp_mask)), int(np.sum(val_tp_mask))
        w_val_fp_sub, w_val_tp_sub = w_val_bg[val_fp_mask], w_val_sig[val_tp_mask]
        val_fp_ess, val_tp_ess = effective_sample_size(w_val_fp_sub), effective_sample_size(w_val_tp_sub)

        # val's own empirical tpr/fpr at this fixed (calib-chosen) threshold
        # is what actually matters out-of-sample.
        val_fpr_emp = float(np.sum(w_val_fp_sub) / np.sum(w_val_bg)) if np.sum(w_val_bg) > 0 else 0.0
        val_tpr_emp = float(np.sum(w_val_tp_sub) / np.sum(w_val_sig)) if np.sum(w_val_sig) > 0 else 0.0
        tp_ann = val_tpr_emp * N_sig_ann
        fp_ann = val_fpr_emp * N_bg_ann
        purity = tp_ann / (tp_ann + fp_ann) * 100 if (tp_ann + fp_ann) > 0 else 0.0

        row = {
            'name': name, 'threshold': thr,
            'n_calib_fp': n_calib_fp, 'n_calib_tp': n_calib_tp,
            'calib_fp_ess': calib_fp_ess, 'calib_tp_ess': calib_tp_ess,
            'n_val_fp': n_val_fp, 'n_val_tp': n_val_tp,
            'val_fp_ess': val_fp_ess, 'val_tp_ess': val_tp_ess,
            'val_annual_tp': tp_ann, 'val_annual_fp': fp_ann, 'val_purity_pct': purity,
            'thin_sample': (n_calib_fp < MIN_SAMPLE_FOR_STATS or n_calib_tp < MIN_SAMPLE_FOR_STATS or
                            n_val_fp < MIN_SAMPLE_FOR_STATS or n_val_tp < MIN_SAMPLE_FOR_STATS or
                            calib_fp_ess < MIN_SAMPLE_FOR_STATS or calib_tp_ess < MIN_SAMPLE_FOR_STATS or
                            val_fp_ess < MIN_SAMPLE_FOR_STATS or val_tp_ess < MIN_SAMPLE_FOR_STATS),
            'clean': None, 'noisy': None,
        }

        if not row['thin_sample']:
            z_calib_tp = z_calib_sig[calib_tp_mask]
            q_lo, q_hi = (p / 100.0 for p in Z_WINDOW_PERCENTILES)
            z_lo = weighted_quantile(z_calib_tp, w_calib_tp_sub, q_lo)
            z_hi = weighted_quantile(z_calib_tp, w_calib_tp_sub, q_hi)
            row['z_window'] = [z_lo, z_hi]

            for label, z_sig_vals, z_bg_vals in (
                ('clean', z_val_sig[val_tp_mask], z_val_bg[val_fp_mask]),
                ('noisy', z_val_sig_noisy[val_tp_mask], z_val_bg_noisy[val_fp_mask]),
            ):
                tp_in = (z_sig_vals >= z_lo) & (z_sig_vals <= z_hi)
                fp_in = (z_bg_vals >= z_lo) & (z_bg_vals <= z_hi)
                w_tp_before, w_tp_after = float(w_val_tp_sub.sum()), float(w_val_tp_sub[tp_in].sum())
                w_fp_before, w_fp_after = float(w_val_fp_sub.sum()), float(w_val_fp_sub[fp_in].sum())
                fp_removed = float(np.clip(100.0 * (1.0 - w_fp_after / w_fp_before), 0.0, 100.0)) \
                    if w_fp_before > 0 else 0.0
                tp_retained = float(np.clip(100.0 * w_tp_after / w_tp_before, 0.0, 100.0)) \
                    if w_tp_before > 0 else 0.0
                tp_ann_after = tp_ann * (w_tp_after / w_tp_before) if w_tp_before > 0 else 0.0
                fp_ann_after = fp_ann * (w_fp_after / w_fp_before) if w_fp_before > 0 else 0.0
                purity_after = (tp_ann_after / (tp_ann_after + fp_ann_after) * 100
                                 if (tp_ann_after + fp_ann_after) > 0 else 0.0)
                row[label] = {
                    'fp_removed_pct': fp_removed, 'tp_retained_pct': tp_retained,
                    'purity_before_pct': purity, 'purity_after_pct': purity_after,
                }
        out.append(row)
    return out


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
    rates = annual_rates_from_data(args.data_dir, args.target)
    N_sig_ann, N_bg_ann = rates['N_sig'], rates['N_bg']
    print(f'Annual rates for {args.target}: N_sig={N_sig_ann:.4g}/yr, '
          f'N_bg={N_bg_ann:.4g}/yr (derived from {args.data_dir})')

    print(f'Loading data from {args.data_dir} for {signal_label}...')
    signal, background, bg_subtype = load_data_tagged(args.data_dir, signal_classes)

    rng = np.random.default_rng(args.seed)
    train_sig, calib_sig, val_sig = split3(signal, rng)
    train_bg, calib_bg, val_bg = split3(background, rng)
    ws_tr = [e['weight'] for e in train_sig]
    wb_tr = [e['weight'] for e in train_bg]

    print(f'Fitting TrajectoryClassifier on train only '
          f'({len(train_sig)} signal, {len(train_bg)} background) -- '
          f'calib ({len(calib_sig)}/{len(calib_bg)}) and val ({len(val_sig)}/{len(val_bg)}) '
          f'both held out.')
    clf = TrajectoryClassifier(target_fpr=0.10).fit(train_sig, train_bg, ws_tr, wb_tr)

    results = {}
    print('\n' + '=' * 100)
    print(f'OUT-OF-SAMPLE Z-CUT VALIDATION: {signal_label.upper()} '
          f'(photo-z noise sigma = {PHOTOZ_SIGMA_FRAC} * (1+z))')
    print('=' * 100)
    for k, day in enumerate(TRAJECTORY_GRID):
        rows = evaluate_epoch(clf, k, calib_sig, calib_bg, val_sig, val_bg, N_sig_ann, N_bg_ann, rng)
        results[day] = rows
        print(f'\n--- Decision epoch: trigger+{day:.0f}d (k={k}) ---')
        for row in rows:
            thin = ' [THIN SAMPLE]' if row['thin_sample'] else ''
            print(f'  {row["name"]}{thin}: val purity={row["val_purity_pct"]:.3f}% '
                  f'(n_val_fp={row["n_val_fp"]}, n_val_tp={row["n_val_tp"]}, '
                  f'val_fp_ess={row["val_fp_ess"]:.1f}, val_tp_ess={row["val_tp_ess"]:.1f})')
            if row['clean'] is not None:
                c, n = row['clean'], row['noisy']
                print(f'    z-window {row["z_window"][0]:.2f}-{row["z_window"][1]:.2f} (from calib set)')
                print(f'    out-of-sample, clean z:  removes {c["fp_removed_pct"]:.1f}% of FPs, '
                      f'costs {100-c["tp_retained_pct"]:.1f}% of TPs -- purity '
                      f'{c["purity_before_pct"]:.3f}% -> {c["purity_after_pct"]:.3f}%')
                print(f'    out-of-sample, noisy z:  removes {n["fp_removed_pct"]:.1f}% of FPs, '
                      f'costs {100-n["tp_retained_pct"]:.1f}% of TPs -- purity '
                      f'{n["purity_before_pct"]:.3f}% -> {n["purity_after_pct"]:.3f}%')
    print('=' * 100)

    # Figure: out-of-sample gain (clean vs noisy z) at the 10% FPR budget,
    # one line per decision epoch, vs. the in-sample figure this validates.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    days_list = list(results.keys())
    clean_removed = [next(r for r in results[d] if r['name'].startswith('10%'))['clean']['fp_removed_pct']
                      if next(r for r in results[d] if r['name'].startswith('10%'))['clean'] else np.nan
                      for d in days_list]
    noisy_removed = [next(r for r in results[d] if r['name'].startswith('10%'))['noisy']['fp_removed_pct']
                      if next(r for r in results[d] if r['name'].startswith('10%'))['noisy'] else np.nan
                      for d in days_list]
    ax1.plot(days_list, clean_removed, 'o-', color='tab:green', label='Out-of-sample, clean z')
    ax1.plot(days_list, noisy_removed, 's--', color='tab:orange', label='Out-of-sample, noisy photo-z')
    ax1.set_xlabel('Decision epoch (days since trigger)')
    ax1.set_ylabel('% of false positives removed (10% FPR budget)')
    ax1.set_title('z-cut gain: out-of-sample, clean vs. noisy z')
    ax1.legend(fontsize=8); ax1.grid(True, linestyle='--', alpha=0.4)

    clean_purity_after = [next(r for r in results[d] if r['name'].startswith('10%'))['clean']['purity_after_pct']
                           if next(r for r in results[d] if r['name'].startswith('10%'))['clean'] else np.nan
                           for d in days_list]
    purity_before = [next(r for r in results[d] if r['name'].startswith('10%'))['val_purity_pct'] for d in days_list]
    ax2.plot(days_list, purity_before, 'o-', color='tab:gray', label='Before cut (val)')
    ax2.plot(days_list, clean_purity_after, 'o-', color='tab:green', label='After cut, clean z (val)')
    ax2.set_xlabel('Decision epoch (days since trigger)')
    ax2.set_ylabel('Purity (%)')
    ax2.set_title('Out-of-sample purity before/after the z-cut')
    ax2.legend(fontsize=8); ax2.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle(f'Out-of-sample validation of the secondary z-window cut ({signal_label}, 10% FPR budget)',
                 fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, f'zcut_outofsample_validation{file_suffix}.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    summary = {
        'signal_label': signal_label, 'N_sig_annual': N_sig_ann, 'N_bg_annual': N_bg_ann,
        'photoz_sigma_frac': PHOTOZ_SIGMA_FRAC,
        'by_decision_epoch': {f'{day:.0f}d': rows for day, rows in results.items()},
    }
    json_path = os.path.join(args.out_dir, f'zcut_outofsample_validation{file_suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
