"""Dual-schema production run for the sequential CMD classifier design.

Runs each transient class through a real OpSim cadence and records BOTH the
static two-point colour vector (cmsne.multicolour.event_multicolour) and the
full N-point trajectory (cmsne.trajectory.event_trajectory) for every event,
via cmsne.trajectory.DualSchemaGenerator -- see
docs/superpowers/specs/2026-10-22-sequential-cmd-classifier-design.md.

This is the production data source cmsne.comparison.compare_classifiers needs
to run on real, simulated-at-scale data instead of only its synthetic test
fixture: every event carries both schemas, so a fitted static ColourClassifier
and a fitted TrajectoryClassifier can score the same events directly.

Classes (10): lensed SN Ia (signal); unlensed SN Ia; and every core-collapse
template, lensed and unlensed -- identical class table to
scripts/run_multicolour.py and scripts/run_trajectory.py, so results from all
three productions are directly comparable.

    python scripts/run_dual_schema.py --n 100000              # all classes, serial
    python scripts/run_dual_schema.py --class-index 0 --n 100000 --out results/dual
"""
import argparse, json, os, pickle, platform, subprocess, sys, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.lsst import contaminant_info
from cmsne.supernovae import redshift_limits
from cmsne.trajectory import DualSchemaGenerator
from cmsne.multicolour import rate_weight
from cmsne.importance_sampling import generate_with_boosting

# Class table: (name, sncosmo source, 'lensed'/'unlensed', rate-kind, abs_mag, sigma)
# Identical to scripts/run_multicolour.py's and scripts/run_trajectory.py's
# CLASSES, so all three productions cover the same population and are
# directly comparable.
CLASSES = [
    dict(name='lensed_Ia',   source='salt3', kind='lensed',   rate='sig', am=-19.23, sd=0.10),
    dict(name='unlensed_Ia', source='salt3', kind='unlensed', rate='uIa', am=-19.23, sd=0.10),
]
for _am, _sd, _src in contaminant_info:
    _short = _src.replace('nugent-', '')
    CLASSES.append(dict(name=f'unlensed_{_short}', source=_src, kind='unlensed', rate='uCC', am=_am, sd=_sd))
    CLASSES.append(dict(name=f'lensed_{_short}',   source=_src, kind='lensed',   rate='lCC', am=_am, sd=_sd))


def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def run_class(cls, n, seed, pilot_frac=0.85):
    np.random.seed(seed)
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = DualSchemaGenerator(cls['source'])
    pop, boost_info = generate_with_boosting(
        gen, n, cls['kind'], z_range, cls['am'], cls['sd'],
        rate_weight_fn=lambda z, mu: rate_weight(
            cls['rate'], z, mu, source=cls['source'],
            z_range=z_range, magnification_range=(2.0, 50.0),
            n_attempted=n
        ),
        pilot_frac=pilot_frac)
    for e in pop:
        e['class'] = cls['name']
    return pop, z_range, boost_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=100000, help='attempts per class')
    ap.add_argument('--class-index', type=int, default=None, help='run one class (array mode)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--no-boost', action='store_true', help='disable sparse-corner boosting (pilot_frac=1.0)')
    args = ap.parse_args()

    which = [args.class_index] if args.class_index is not None else list(range(len(CLASSES)))

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = args.out or os.path.join(_ROOT, 'results', f'dual_{stamp}')
    os.makedirs(out, exist_ok=True)
    if args.class_index is None or args.class_index == 0:
        json.dump({'started_utc': stamp, 'git_commit': _git_commit(),
                   'host': platform.node(), 'python': platform.python_version(),
                   'opsim_db': os.environ.get('CMSNE_OPSIM_DB', '(config default)'),
                   'n_per_class': args.n, 'classes': [c['name'] for c in CLASSES]},
                  open(os.path.join(out, 'manifest.json'), 'w'), indent=2)
    print(f'output dir : {out}\ncommit     : {_git_commit()}\n', flush=True)

    run_start = time.time()
    for i in which:
        cls = CLASSES[i]
        t0 = time.time()
        print(f'[{i}] {cls["name"]} ...', flush=True)
        try:
            pop, zr, boost_info = run_class(cls, args.n, args.seed + i,
                                            pilot_frac=1.0 if args.no_boost else 0.85)
        except Exception as e:
            print(f'  FAILED {cls["name"]}: {e!r}', flush=True)
            continue
        with open(os.path.join(out, f'{cls["name"]}.pkl'), 'wb') as f:
            pickle.dump({'class': cls['name'], 'rate': cls['rate'], 'z_range': zr,
                         'n_attempted': args.n, 'events': pop, 'boost_info': boost_info},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        tot_rate = float(np.sum([e['weight'] for e in pop])) if pop else 0.0
        row = {'class': cls['name'], 'kept': len(pop), 'n_attempted': args.n,
               'annual_detection_rate': tot_rate,
               'minutes': round((time.time() - t0) / 60, 2)}
        json.dump(row, open(os.path.join(out, f'{cls["name"]}.summary.json'), 'w'), indent=2)
        print(f'  kept {row["kept"]}/{args.n}  annual_rate={tot_rate:.4e}/yr  [{row["minutes"]:.1f} min]', flush=True)

    print(f'\ntotal {(time.time() - run_start) / 3600:.2f} h  ->  {out}', flush=True)


if __name__ == '__main__':
    main()
