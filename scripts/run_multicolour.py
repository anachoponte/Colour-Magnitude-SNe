"""All-band (ugrizy) photometry production run for the colour-only identifier.

Runs each transient class through a real OpSim cadence and records the full colour
vector at the trigger (first-detection) epoch plus 15 observer-days later -- so the
multi-colour separation of lensed SN Ia from the contaminant background can be
measured with realistic band availability, not the idealised "every band at peak".

Classes (10): lensed SN Ia (signal); unlensed SN Ia; and every core-collapse
template, lensed and unlensed. Each event carries its rate weight.

    python scripts/run_multicolour.py --n 100000              # all classes, serial
    python scripts/run_multicolour.py --class-index 0 --n 100000 --out results/mc
"""
import argparse, json, os, pickle, platform, subprocess, sys, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from cmsne.lsst import contaminant_info
from cmsne.supernovae import redshift_limits
from cmsne.multicolour import MultiColourGenerator, rate_weight

# Class table: (name, sncosmo source, 'lensed'/'unlensed', rate-kind, abs_mag, sigma)
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


def run_class(cls, n, seed):
    np.random.seed(seed)
    # Sample z where the source covers lsstr (the normalisation band).
    z_range = redshift_limits(cls['source'], 'lsstr', 'lsstr')
    gen = MultiColourGenerator(cls['source'])
    pop = gen.generate_many(n, cls['kind'], z_range, cls['am'], cls['sd'])
    for e in pop:
        e['weight'] = float(rate_weight(cls['rate'], e['z'], e['magnification']))
        e['class'] = cls['name']
    return pop, z_range


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=100000, help='attempts per class')
    ap.add_argument('--class-index', type=int, default=None, help='run one class (array mode)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    which = [args.class_index] if args.class_index is not None else list(range(len(CLASSES)))

    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = args.out or os.path.join(_ROOT, 'results', f'mc_{stamp}')
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
            pop, zr = run_class(cls, args.n, args.seed + i)
        except Exception as e:
            print(f'  FAILED {cls["name"]}: {e!r}', flush=True)
            continue
        with open(os.path.join(out, f'{cls["name"]}.pkl'), 'wb') as f:
            pickle.dump({'class': cls['name'], 'rate': cls['rate'], 'z_range': zr,
                         'n_attempted': args.n, 'events': pop}, f, protocol=pickle.HIGHEST_PROTOCOL)
        nb = np.array([e['n_bands'] for e in pop]) if pop else np.array([0])
        row = {'class': cls['name'], 'kept': len(pop), 'n_attempted': args.n,
               'median_n_bands': float(np.median(nb)), 'minutes': round((time.time() - t0) / 60, 2)}
        json.dump(row, open(os.path.join(out, f'{cls["name"]}.summary.json'), 'w'), indent=2)
        print(f'  kept {row["kept"]}/{args.n}, median bands {row["median_n_bands"]:.0f}'
              f'  [{row["minutes"]:.1f} min]', flush=True)

    print(f'\ntotal {(time.time() - run_start) / 3600:.2f} h  ->  {out}', flush=True)


if __name__ == '__main__':
    main()
