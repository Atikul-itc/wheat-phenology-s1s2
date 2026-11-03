"""Daily Sentinel-2 field series from the Earth Engine exports (Section 2.4).

One observation per field and date is kept: when overlapping tiles give several
scenes on the same day, the scene with the most clear pixels is used. Output, in
the layout read by ``phenology_pipeline``:

    <s2_daily>/<year>/<feature>_wheat<yy>.csv   columns XID, Date, <feature>, Stations_i, Rank

Usage:
    python -m extraction.s2_build_daily --config config.yaml --exports <folder with S2_wheat*.csv>
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import base_parser, load_config
from extraction.s1_build_daily import KEYS, WINDOW_DAYS, field_list
from extraction.s2_extract_gee import FEATURES


def build_year(cfg, exports: Path, year: int) -> dict:
    yy = f'{year % 100:02d}'
    files = sorted(exports.glob(f'S2_wheat{yy}_*.csv'))
    if not files:
        raise FileNotFoundError(f'no S2 exports for {year} in {exports}')
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).rename(columns={'date': 'Date'})
    day = (raw.sort_values('clear_count', ascending=False)
              .drop_duplicates(KEYS + ['Date'], keep='first'))

    fields = field_list(cfg, year, day)
    dates = pd.date_range(pd.Timestamp(year - 1, 9, 1), periods=WINDOW_DAYS, freq='D').strftime('%Y-%m-%d')
    full = fields.merge(pd.DataFrame({'Date': dates}), how='cross').merge(
        day[KEYS + ['Date'] + FEATURES], on=KEYS + ['Date'], how='left')

    out = cfg.path('s2_daily') / str(year)
    out.mkdir(parents=True, exist_ok=True)
    for f in FEATURES:
        full[['XID', 'Date', f, 'Stations_i', 'Rank']].to_csv(out / f'{f}_wheat{yy}.csv', index=False, na_rep='NA')
    rate = full['blue'].notna().mean()
    return {'year': year, 'fields': fields['XID'].nunique(), 'daily_obs_rate': round(rate, 3),
            'clear_obs_per_field': round(full['blue'].notna().groupby(full['XID']).sum().median(), 1)}


def main():
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument('--exports', required=True, help='folder with the downloaded S2_wheat*.csv files')
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows = [build_year(cfg, Path(args.exports), y) for y in cfg.years]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
