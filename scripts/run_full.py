"""Full colour-magnitude production run.

Reproduces the notebook-04 recipe for every LSST band pair and saves the COMPLETE
populations to disk, so the colour-magnitude diagrams, the band-pair separation
ranking, and the (new) multi-lens time-delay / lead-time analysis can all be
rebuilt offline without re-running the ~hours-long cadence loop.

Unlike the notebook, every cadence event is stored with ALL of its fields
(the per-lens delays, ``time_of_SN_rel_peak`` / ``time_of_SN_rel_first``,
``first_alert_day``, ``n_alerts_before_id``, ...), not just the legacy scalars.

Run on BlueBEAR through ``scripts/run_full.sh`` (SLURM). Point ``CMSNE_OPSIM_DB``
at a ``baseline_v5.3.5_10yrs.db``. For a quick check of the non-cadence path on a
machine without the database, use ``--skip-cadence``.

    python scripts/run_full.py                       # full run, all pairs
    python scripts/run_full.py --pairs g-r,r-i       # a subset
    python scripts/run_full.py --skip-cadence --n-background 40 --n-draw 20 --pairs g-r
"""
import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.lsst import band_pairs, contaminant_info
from cmsne.supernovae import Supernovae, Supernovae2, Nugent, redshift_limits
from cmsne.colour_magnitude import (full_creation, combine_populations,
                                    exponential_regression, success_rate)


def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def _label(pair):
    """Short 'g-r' label from a ('lsstg', 'lsstr') pair."""
    return f'{pair[0][-1]}-{pair[1][-1]}'


def run_band_pair(pair, gens, args):
    """Generate every population for one band pair and return a result dict."""
    blue_band, red_band = pair
    band1, band2 = red_band, blue_band          # band1 = magnitude axis, band2 = colour's blue side
    gen_cadence, gen_bg, gen_cc = gens
    z_lo, z_hi = redshift_limits('salt3', band1, band2)

    # --- model-only populations (fast): SN Ia background + core-collapse contaminants ---
    SNe_Ia = gen_bg.generate_many(args.n_background, band1, band2)
    contaminants = [
        gen_cc.generate_many(args.n_contaminant, band1, band2, abs_mag, sigma, source)
        for abs_mag, sigma, source in contaminant_info
    ]

    # --- decision boundary from the drawn, rate-weighted populations ---
    all_colour_l, all_colour_ul, all_peak_mag_l, all_peak_mag_ul = combine_populations(
        [SNe_Ia] + contaminants, args.n_draw)
    x_range, y_range, m_, b_ = exponential_regression(
        all_peak_mag_ul, all_peak_mag_l, all_colour_ul, all_colour_l)

    drawn = {'Ia': full_creation(SNe_Ia, args.n_draw)}
    for (_, _, source), pop in zip(contaminant_info, contaminants):
        drawn[source] = full_creation(pop, args.n_draw)

    # --- lensed events through the real OpSim cadence (the slow, scarce one) ---
    cadence_events, n_attempted = [], 0
    if not args.skip_cadence:
        cadence_events = gen_cadence.generate_many(args.n_cadence, band1, band2)
        n_attempted = args.n_cadence

    l_peak_mags = [e['band_1_mag_l'] for e in cadence_events]
    l_colors = [e['lensed_colour'] for e in cadence_events]
    rate = success_rate(l_peak_mags, l_colors, m_, b_) if cadence_events else float('nan')

    return {
        'band_pair': pair,
        'label': _label(pair),
        'band1': band1, 'band2': band2,
        'z_range': (float(z_lo), float(z_hi)),
        'boundary': {'m': float(m_), 'b': float(b_),
                     'x_range': np.asarray(x_range), 'y_range': np.asarray(y_range)},
        'drawn': drawn,
        'combined': {'colour_l': all_colour_l, 'colour_ul': all_colour_ul,
                     'peak_mag_l': all_peak_mag_l, 'peak_mag_ul': all_peak_mag_ul},
        'cadence_events': cadence_events,           # FULL event dicts (all fields)
        'n_cadence_kept': len(cadence_events),
        'n_cadence_attempted': n_attempted,
        'success_rate': float(rate),
    }


def _select_pairs(spec):
    if not spec:
        return list(band_pairs)
    want = {s.strip() for s in spec.split(',')}
    pairs = [p for p in band_pairs
             if _label(p) in want or f'{p[0]}-{p[1]}' in want]
    if not pairs:
        sys.exit(f'no band pairs matched {spec!r}; known: '
                 + ', '.join(_label(p) for p in band_pairs))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-cadence', type=int, default=200000,
                    help='lensed events attempted through the cadence per pair (default 200000)')
    ap.add_argument('--n-background', type=int, default=10000)
    ap.add_argument('--n-contaminant', type=int, default=8000)
    ap.add_argument('--n-draw', type=int, default=600)
    ap.add_argument('--out', default=None, help='output dir (default results/run_<UTC>)')
    ap.add_argument('--pairs', default=None,
                    help='comma-separated subset like "g-r,r-i" (default: all 10)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--skip-cadence', action='store_true',
                    help='skip OpSim cadence generation (smoke test without a database)')
    args = ap.parse_args()

    pairs = _select_pairs(args.pairs)

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = args.out or os.path.join(_ROOT, 'results', f'run_{stamp}')
    os.makedirs(out, exist_ok=True)

    manifest = {
        'started_utc': stamp,
        'git_commit': _git_commit(),
        'python': platform.python_version(),
        'host': platform.node(),
        'opsim_db': os.environ.get('CMSNE_OPSIM_DB', '(config default)'),
        'params': {k: getattr(args, k) for k in
                   ('n_cadence', 'n_background', 'n_contaminant', 'n_draw', 'seed', 'skip_cadence')},
        'pairs': [_label(p) for p in pairs],
    }
    with open(os.path.join(out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'output dir : {out}', flush=True)
    print(f'commit     : {manifest["git_commit"]}', flush=True)
    print(f'opsim db   : {manifest["opsim_db"]}\n', flush=True)

    gens = (Supernovae(), Supernovae2(), Nugent())
    run_start = time.time()
    rows = []
    for i, pair in enumerate(pairs, 1):
        np.random.seed(args.seed + i)          # reproducible per pair, array-safe
        label = _label(pair)
        t0 = time.time()
        print(f'[{i}/{len(pairs)}] {label} ...', flush=True)
        try:
            res = run_band_pair(pair, gens, args)
        except Exception as e:                 # keep going; one bad pair shouldn't lose the rest
            print(f'  FAILED {label}: {e!r}', flush=True)
            row = {'label': label, 'error': repr(e)}
            with open(os.path.join(out, f'{label}.summary.json'), 'w') as f:
                json.dump(row, f, indent=2)
            rows.append(row)
            continue

        with open(os.path.join(out, f'{label}.pkl'), 'wb') as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        row = {'label': label,
               'n_cadence_kept': res['n_cadence_kept'],
               'n_cadence_attempted': res['n_cadence_attempted'],
               'success_rate': res['success_rate'],
               'minutes': round((time.time() - t0) / 60, 2)}
        # per-pair summary file: safe when many array tasks share one output dir
        with open(os.path.join(out, f'{label}.summary.json'), 'w') as f:
            json.dump(row, f, indent=2)
        rows.append(row)
        pct = (100 * row['n_cadence_kept'] / row['n_cadence_attempted']
               if row['n_cadence_attempted'] else float('nan'))
        print(f'  kept {row["n_cadence_kept"]}/{row["n_cadence_attempted"]} ({pct:.1f}%), '
              f'success_rate={row["success_rate"]:.3f}  [{row["minutes"]:.1f} min]', flush=True)

    # aggregate summary only when this process handled the whole set (avoids array races)
    if len(pairs) > 1:
        with open(os.path.join(out, 'summary.json'), 'w') as f:
            json.dump(rows, f, indent=2)

    print(f'\ntotal {(time.time() - run_start) / 3600:.2f} h  ->  {out}', flush=True)


if __name__ == '__main__':
    main()
