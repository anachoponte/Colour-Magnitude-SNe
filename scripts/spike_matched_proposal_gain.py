"""THROWAWAY SPIKE -- predicts the ESS-per-attempt gain of a rate-matched
importance-sampling proposal, WITHOUT generating anything.

Design doc: docs/superpowers/specs/2026-11-20-matched-importance-proposal-design.md

The question this answers: matched sampling makes every weight equal, so
ESS/raw goes to ~100% -- but it also draws lower magnification (mu^-3 has
median ~2.8 vs uniform's mean 26) and higher redshift (f(z) rises with
comoving volume), both of which are DIMMER and so survive `min_detections=5`
less often. Compute is spent per ATTEMPT, so the real question is whether
ESS-per-attempt goes up at all.

Method (no new simulation needed):

Attempts were uniform over D = z_range x mu_range, area A. Kept events are
therefore a sample from a density proportional to eps(z,mu), the detection
efficiency, since the uniform proposal density is constant. Writing q for the
normalised density of kept events and eps_bar = N_kept/N_attempted for the
mean efficiency under uniform sampling,

    eps(z,mu) = q(z,mu) * A * eps_bar

The matched proposal's efficiency is then

    eps_matched = INT p_m * eps  =  eps_bar * E_q[ p_m / p_u ]

which is a plain average of the density ratio over the events we already
have. Under the matched proposal every kept event carries an equal weight, so
ESS = N_kept exactly and

    ESS_per_attempt(matched) = eps_matched
    gain = eps_matched * N_attempted / ESS_uniform

CAVEAT: ~15% of each baseline population came from the boost stage
(pilot_frac=0.85), not from the uniform pilot, which skews the eps estimate
slightly toward the sparse corners boosting targets. This is a go/no-go
estimate, not a precision prediction.
"""
import argparse, glob, json, os, pickle, sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.importance_sampling import effective_sample_size
from cmsne.rates import absolute_rate_weight
from cmsne.supernovae import redshift_limits

MU_RANGE = (2.0, 50.0)
N_GRID = 400

# name -> (rate_kind, source, lensed)
def class_spec(name):
    lensed = name.startswith('lensed_')
    short = name.split('_', 1)[1]
    if short == 'Ia':
        source, rate = 'salt3', ('sig' if lensed else 'uIa')
    else:
        source, rate = f'nugent-{short}', ('lCC' if lensed else 'uCC')
    return rate, source, lensed


def normalising_constant(rate, source, lensed, z_range):
    """Z = INT rate_weight over the sampling domain, by trapezoid on a grid."""
    zs = np.linspace(z_range[0], z_range[1], N_GRID)
    if not lensed:
        g = np.array([absolute_rate_weight(rate, z, source=source) for z in zs])
        return float(np.trapz(g, zs)), zs, None
    mus = np.linspace(MU_RANGE[0], MU_RANGE[1], N_GRID)
    # Separable: rate_weight(z, mu) = g(z) * mu^-3 up to a constant, but
    # integrate the real function on the grid rather than assume it.
    gz = np.array([absolute_rate_weight(rate, z, magnification=1.0, source=source) for z in zs])
    hmu = mus ** -3.0
    return float(np.trapz(gz, zs) * np.trapz(hmu, mus)), zs, mus


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', default='docs/figures/purity_investigation/spike_matched_proposal_gain.json')
    args = ap.parse_args()

    rows = []
    print('=' * 108)
    print('SPIKE: predicted ESS-per-attempt gain from a rate-matched proposal')
    print('=' * 108)
    hdr = (f'{"class":<16}{"N_att":>8}{"N_kept":>8}{"eff_u":>8}'
           f'{"ESS_u":>9}{"ESS/att_u":>11}{"eff_m":>9}{"ESS/att_m":>11}{"GAIN":>8}')
    print(hdr)
    print('-' * 108)
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
        A = (z_range[1] - z_range[0]) * ((MU_RANGE[1] - MU_RANGE[0]) if lensed else 1.0)
        Z, _, _ = normalising_constant(rate, source, lensed, z_range)

        z_i = np.array([e['z'] for e in events])
        mu_i = np.array([e.get('magnification', 1.0) for e in events]) if lensed else np.ones(len(events))
        # Bare uniform-proposal MC weight, recomputed (NOT the stored weight,
        # which carries the boost-mixture correction).
        rw = np.array([absolute_rate_weight(rate, z, magnification=m, source=source,
                                            z_range=z_range, magnification_range=MU_RANGE,
                                            n_attempted=n_att)
                       for z, m in zip(z_i, mu_i)])
        ess_u = effective_sample_size(rw)
        n_kept = len(events)
        eff_u = n_kept / n_att

        # p_m / p_u = A * rate_weight_density / Z. rw carries the (A/n_att)
        # factor already, so undo it to recover the bare density.
        density = rw * n_att / A
        ratio = A * density / Z
        eff_m = eff_u * float(np.mean(ratio))

        ess_att_u = ess_u / n_att
        gain = eff_m / ess_att_u if ess_att_u > 0 else float('inf')
        print(f'{name:<16}{n_att:>8}{n_kept:>8}{eff_u:>8.3f}{ess_u:>9.1f}'
              f'{ess_att_u:>11.5f}{eff_m:>9.5f}{eff_m:>11.5f}{gain:>8.2f}x')
        rows.append({'class': name, 'n_attempted': n_att, 'n_kept': n_kept,
                     'eff_uniform': eff_u, 'ess_uniform': ess_u,
                     'ess_per_attempt_uniform': ess_att_u,
                     'eff_matched_predicted': eff_m,
                     'ess_per_attempt_matched': eff_m, 'gain_factor': gain})
    print('=' * 108)
    print('GAIN < 1 means the matched proposal would COST more compute per unit ESS.')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
