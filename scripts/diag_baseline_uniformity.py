"""Did results/dual_59297152 actually use a uniform proposal?

scripts/spike_proposal_ceiling.py assumes every generation attempt was drawn
uniformly in (z, mu), so that attempts-per-bin is constant and
eps_b = counts_b / (n_attempted / n_bins).  If boosting was enabled
(pilot_frac < 1.0) that assumption is false: the boost stage adds extra
attempts in selected regions, inflating eps exactly where it boosted.

This reads the pickles only.  No fitting, no generation.
"""
import argparse, glob, os, pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', default='results/dual_59297152')
args = ap.parse_args()

print('=' * 78)
print('BASELINE UNIFORMITY CHECK:', args.data_dir)
print('=' * 78)

for path in sorted(glob.glob(os.path.join(args.data_dir, '*.pkl'))):
    name = os.path.basename(path).replace('.pkl', '')
    with open(path, 'rb') as fh:
        blob = pickle.load(fh)
    if not isinstance(blob, dict):
        print(f'{name:<18} (not a dict blob: {type(blob).__name__})')
        continue
    n_att = blob.get('n_attempted')
    events = blob.get('events', [])
    bi = blob.get('boost_info')

    print(f'\n--- {name} ---')
    print(f'  n_attempted (stored) : {n_att}')
    print(f'  kept events          : {len(events)}')
    if bi is None:
        print('  boost_info           : ABSENT -> cannot tell; likely old format')
        continue
    if not isinstance(bi, dict):
        print(f'  boost_info           : {type(bi).__name__} -> {str(bi)[:120]}')
        continue
    print(f'  boost_info keys      : {sorted(bi.keys())}')
    for k in ('n_pilot', 'n_boost', 'total_attempted', 'pilot_frac',
              'n_cells_flagged', 'n_flagged', 'boost_counts'):
        if k in bi:
            v = bi[k]
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v)
                print(f'  {k:<20} : array len={v.size} sum={v.sum():.0f} '
                      f'max={v.max() if v.size else 0:.0f}')
            else:
                print(f'  {k:<20} : {v}')
    tot = bi.get('total_attempted')
    if tot is not None and n_att:
        print(f'  >> total_attempted / n_attempted = {tot / n_att:.4f}'
              f'   ({"UNIFORM-ONLY" if abs(tot/n_att - 1) < 1e-9 else "BOOSTED -- spike assumption VIOLATED"})')
print('\nDone.')
