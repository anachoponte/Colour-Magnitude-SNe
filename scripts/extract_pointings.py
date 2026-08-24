"""Extract the per-visit pointing table from an LSST OpSim database to CSV.

The output holds everything needed to draw a map of the LSST pointings: the
boresight coordinates (``fieldRA``, ``fieldDec``), the camera orientation
(``rotSkyPos``, ``rotTelPos``), the band, the epoch, and the per-visit
observing conditions usually used to colour such a map (depth, seeing, sky
brightness, airmass).

The v5 baseline dithers every visit, so there is no fixed field grid to
aggregate onto: 1.85M visits sit at ~1.0M distinct centres. The table is
therefore written one row per visit. Use ``--nights`` or ``--every`` to thin it
down for quick plots.

Examples
--------
    # Full 10-year survey (~1.85M rows)
    python scripts/extract_pointings.py --out data/lsst_pointings.csv

    # One year of WFD visits in a sky box
    python scripts/extract_pointings.py --wfd --year 1 \
        --ra-range 220 320 --dec-range -60 0 --out data/wfd_box_year1.csv

    # Every 20th visit, for a fast all-sky scatter plot
    python scripts/extract_pointings.py --every 20 --out data/thinned.csv
"""

import argparse
import os
import sqlite3
import sys

import pandas as pd

# The package is not installed, so make the repo root importable when this
# script is run directly (notebooks already run from the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmsne.config import MY_OPSIM_DB, survey_dates

# Columns pulled from the OpSim ``observations`` table. fieldRA/fieldDec and
# rotSkyPos are the map itself; the rest are the parameters normally used to
# select or colour-code the pointings.
POINTING_COLUMNS = [
    'observationId',
    'fieldRA',              # boresight right ascension [deg]
    'fieldDec',             # boresight declination [deg]
    'rotSkyPos',            # camera rotation w.r.t. north on sky [deg]
    'rotTelPos',            # camera rotation w.r.t. the telescope [deg]
    'band',                 # ugrizy
    'filter',               # physical filter id (e.g. i_39)
    'observationStartMJD',
    'night',                # survey night, 1 = first night
    'visitExposureTime',
    'numExposures',
    'airmass',
    'altitude',
    'azimuth',
    'seeingFwhmEff',
    'skyBrightness',
    'fiveSigmaDepth',
    'target_name',          # footprint region(s), e.g. lowdust, ddf_cosmos
    'scheduler_note',       # what the scheduler was doing, e.g. pair_33, iz, bs 52, a
    'observation_reason',
]

# The v5 baseline has no survey-type column: every row is science_program
# BLOCK-419. The WFD is identified by its footprint label instead. 'lowdust' is
# the low-dust extragalactic WFD footprint; 'bulgy' and 'dusty_plane' are the
# separate galactic-plane mini-surveys. Visits whose footprint label also names
# a DDF, and deep-drilling or target-of-opportunity sequences, are dropped so
# the result is WFD cadence rather than anything that merely lands in the WFD
# footprint.
WFD_CLAUSES = [
    "target_name LIKE '%lowdust%'",
    "target_name NOT LIKE '%ddf%'",
    "scheduler_note NOT LIKE 'DD%'",
    "scheduler_note NOT LIKE 'ToO%'",
]


def build_query(nights=None, bands=None, every=None, ra_range=None,
                dec_range=None, year=None, wfd=False):
    """Build the SELECT statement and its bound parameters."""
    sql = f"SELECT {', '.join(POINTING_COLUMNS)} FROM observations"
    where, params = [], []

    if ra_range is not None:
        # Ranges are taken as given; a range that crosses RA=0 would need
        # splitting into two OR'd clauses.
        where.append('fieldRA BETWEEN ? AND ?')
        params.extend(sorted(ra_range))
    if dec_range is not None:
        where.append('fieldDec BETWEEN ? AND ?')
        params.extend(sorted(dec_range))
    if year is not None:
        # Survey-year boundaries come from cmsne.config.survey_dates so the
        # definition matches the rest of the project.
        where.append('observationStartMJD >= ? AND observationStartMJD < ?')
        params.extend([survey_dates[year - 1], survey_dates[year]])
    if wfd:
        where.extend(WFD_CLAUSES)
    if nights is not None:
        where.append('night BETWEEN ? AND ?')
        params.extend(nights)
    if bands:
        where.append(f"band IN ({', '.join('?' * len(bands))})")
        params.extend(bands)
    if every is not None and every > 1:
        # observationId is a dense 0-based counter ordered by time, so this
        # takes an even sample across the whole survey.
        where.append('observationId % ? = 0')
        params.append(every)

    if where:
        sql += ' WHERE ' + ' AND '.join(where)
    return sql + ' ORDER BY observationStartMJD', params


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default=MY_OPSIM_DB,
                        help='path to the OpSim .db file (default: %(default)s)')
    parser.add_argument('--out', default='data/lsst_pointings.csv',
                        help='output CSV path (default: %(default)s)')
    parser.add_argument('--ra-range', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                        help='keep only visits with LOW <= fieldRA <= HIGH [deg]')
    parser.add_argument('--dec-range', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                        help='keep only visits with LOW <= fieldDec <= HIGH [deg]')
    parser.add_argument('--year', type=int, choices=range(1, 11), metavar='N',
                        help='keep only survey year N (1-10), per config.survey_dates')
    parser.add_argument('--wfd', action='store_true',
                        help='keep only WFD visits (low-dust footprint, no DDF/ToO)')
    parser.add_argument('--nights', nargs=2, type=int, metavar=('LOW', 'HIGH'),
                        help='keep only visits with LOW <= night <= HIGH')
    parser.add_argument('--bands', nargs='+', metavar='BAND',
                        help='keep only these bands, e.g. --bands g r i')
    parser.add_argument('--every', type=int,
                        help='keep only every Nth visit (thins the table)')
    parser.add_argument('--chunksize', type=int, default=200_000,
                        help='rows per read/write chunk (default: %(default)s)')
    args = parser.parse_args()

    db_path = os.path.expanduser(args.db)
    if not os.path.exists(db_path):
        raise SystemExit(f'OpSim database not found: {db_path}\n'
                         'Set CMSNE_OPSIM_DB or pass --db.')

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    sql, params = build_query(args.nights, args.bands, args.every, args.ra_range,
                              args.dec_range, args.year, args.wfd)
    print(f'Reading {db_path}')

    n_rows = 0
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as con:
        chunks = pd.read_sql_query(sql, con, params=params, chunksize=args.chunksize)
        for i, chunk in enumerate(chunks):
            # %.6f keeps sub-arcsecond coordinates and ~0.1 s timing while
            # roughly halving the file size against full float repr.
            chunk.to_csv(out_path, mode='w' if i == 0 else 'a',
                         header=(i == 0), index=False, float_format='%.6f')
            n_rows += len(chunk)
            print(f'  wrote {n_rows:,} rows', end='\r', flush=True)

    size_mb = os.path.getsize(out_path) / 1024 ** 2
    print(f'\nWrote {n_rows:,} visits to {out_path} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
