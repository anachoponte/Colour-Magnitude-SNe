"""THROWAWAY SPIKE v2 -- what IS the best achievable proposal, given that the
detection cut throws away most of the rate density?

Spike v1 (spike_matched_proposal_gain.py) showed a purely rate-matched
proposal LOSES for 8 of 10 classes: it concentrates draws where the rate
density is large, which is exactly where events are too faint to survive
min_detections=5. The uniform proposal was accidentally doing better by
oversampling the detectable corner.

So the right object is a proposal that knows about detection. For a weighted
estimator whose kept events carry w = rate/p, ESS per ATTEMPT is

    ESS/N = (INT rate*eps)^2 / INT (rate^2 * eps / p)

Maximising over p subject to INT p = 1 gives, by Cauchy-Schwarz,

    p_opt  ~  rate * sqrt(eps)

-- NOT rate (v1's proposal, which ignores eps) and NOT rate*eps (which
over-corrects). This script evaluates all four candidates on the existing
populations, with no new generation.

eps(z, mu) is recoverable directly from the baseline because attempts were
UNIFORM: binning kept events over (z, mu) and dividing by the (equal) number
of attempts per bin gives eps per bin with no density estimation needed.

CAVEATS: eps is noisy in thinly-populated bins, so the rate*sqrt(eps) and
rate*eps ceilings are optimistic (they partly fit that noise). The uniform
column is computed the same way and should reproduce spike v1's measured
ESS/attempt -- that agreement is the sanity check on the whole calculation.
"""
import argparse, glob, json, os, pickle, sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.rates import absolute_rate_weight
from cmsne.supernovae import redshift_limits

MU_RANGE = (2.0, 50.0)


def class_spec(name):
    lensed = name.startswith('lensed_')
    short = name.split('_', 1)[1]
    if short == 'Ia':
        return ('sig' if lensed else 'uIa'), 'salt3', lensed
    return ('lCC' if lensed else 'uCC'), f'nugent-{short}', lensed


def ess_per_attempt(rate_b, eps_b, area_b, p_b):
    """ESS/N_attempt for proposal density p_b. Bins with p_b == 0 are never
    drawn from, so they contribute nothing; bins with eps_b == 0 contribute
    nothing either."""
    num = np.sum(rate_b * eps_b * area_b) ** 2
    live = (p_b > 0) & (eps_b > 0)
    den = np.sum((rate_b[live] ** 2) * eps_b[live] * area_b[live] / p_b[live])
    return float(num / den) if den > 0 else 0.0


def normalise(f_b, area_b):
    tot = np.sum(f_b * area_b)
    return f_b / tot if tot > 0 else f_b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--nbins', type=int, default=20, help='bins per axis (lensed: squared)')
    ap.add_argument('--out', default='docs/figures/purity_investigation/spike_proposal_ceiling.json')
    args = ap.parse_args()

    rows = []
    print('=' * 112)
    print('SPIKE v2: ESS per ATTEMPT under four proposals (higher is better)')
    print('=' * 112)
    print(f'{"class":<16}{"uniform":>12}{"rate":>12}{"rate*sqrt(e)":>15}{"rate*eps":>12}'
          f'{"best gain":>12}{"best proposal":>18}')
    print('-' * 112)
    for path in sorted(glob.glob(os.path.join(args.data_dir, '*.pkl'))):
        name = os.path.basename(path)[:-4]
        with open(path, 'rb') as f:
            blob = pickle.load(f)
        events = blob['events'] if isinstance(blob, dict) and 'events' in blob else blob
        n_att = blob.get('n_attempted', 100000) if isinstance(blob, dict) else 100000
        if not events:
            continue
        rate, source, lensed = class_spec(name)
        z_range = redshift_limits(source, 'lsstr', 'lsstr')
        z_i = np.array([e['z'] for e in events])

        if lensed:
            mu_i = np.array([e.get('magnification', 1.0) for e in events])
            nb = args.nbins
            z_edges = np.linspace(z_range[0], z_range[1], nb + 1)
            mu_edges = np.linspace(MU_RANGE[0], MU_RANGE[1], nb + 1)
            counts, _, _ = np.histogram2d(z_i, mu_i, bins=[z_edges, mu_edges])
            zc = 0.5 * (z_edges[:-1] + z_edges[1:])
            muc = 0.5 * (mu_edges[:-1] + mu_edges[1:])
            ZC, MC = np.meshgrid(zc, muc, indexing='ij')
            area_b = np.full(counts.size, (z_edges[1] - z_edges[0]) * (mu_edges[1] - mu_edges[0]))
            counts = counts.ravel()
            rate_b = np.array([absolute_rate_weight(rate, z, magnification=m, source=source)
                               for z, m in zip(ZC.ravel(), MC.ravel())])
        else:
            nb = args.nbins * args.nbins
            z_edges = np.linspace(z_range[0], z_range[1], nb + 1)
            counts, _ = np.histogram(z_i, bins=z_edges)
            zc = 0.5 * (z_edges[:-1] + z_edges[1:])
            area_b = np.full(nb, z_edges[1] - z_edges[0])
            rate_b = np.array([absolute_rate_weight(rate, z, source=source) for z in zc])

        # Attempts were uniform, so attempts per bin is the same everywhere.
        attempts_per_bin = n_att / counts.size
        eps_b = counts / attempts_per_bin

        total_area = np.sum(area_b)
        cands = {
            'uniform': np.full(counts.size, 1.0 / total_area),
            'rate': normalise(rate_b, area_b),
            'rate*sqrt(eps)': normalise(rate_b * np.sqrt(eps_b), area_b),
            'rate*eps': normalise(rate_b * eps_b, area_b),
        }
        vals = {k: ess_per_attempt(rate_b, eps_b, area_b, p) for k, p in cands.items()}
        base = vals['uniform']
        best_k = max(vals, key=vals.get)
        gain = vals[best_k] / base if base > 0 else float('inf')
        print(f'{name:<16}{vals["uniform"]:>12.5f}{vals["rate"]:>12.5f}'
              f'{vals["rate*sqrt(eps)"]:>15.5f}{vals["rate*eps"]:>12.5f}'
              f'{gain:>11.1f}x{best_k:>18}')
        rows.append({'class': name, 'ess_per_attempt': vals,
                     'best': best_k, 'best_gain_vs_uniform': gain})
    print('=' * 112)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
