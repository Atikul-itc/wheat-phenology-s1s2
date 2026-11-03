"""Daily Sentinel-1 field series from the Earth Engine exports (Section 2.5).

Scene-field rows are kept when at least 90% of the field pixels are valid and at
least 20 pixels remain. Same-day scenes are averaged with pixel-count weights in
linear power and converted to dB; CR = VH / VV of the field means (VH - VV in dB).
The output has one row per field and day of the 426-day window (missing = NA),
in the layout read by ``phenology_pipeline``:

    <s1_daily>/<year>/{VV,VH,CR}_wheat<yy>.csv   columns XID, Date, <var>, Stations_i, Rank
    <s1_daily>/<year>/S1meta_wheat<yy>.csv       angle, scenes, orbits and valid share per field-day

Usage:
    python -m extraction.s1_build_daily --config config.yaml
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import base_parser, load_config

MIN_VALID_SHARE = 0.9
MIN_PIXELS = 20
WINDOW_DAYS = 426
KEYS = ['XID', 'Stations_i', 'Rank']


def field_list(cfg, year: int, day: pd.DataFrame) -> pd.DataFrame:
    """All fields of the year (from the GeoPackage), so fields without any S1 value keep their rows."""
    fp = Path(cfg.pattern('fields').format(year=year))
    if fp.exists():
        import geopandas as gpd
        return pd.DataFrame(gpd.read_file(fp, ignore_geometry=True)[KEYS]).astype('int64').drop_duplicates()
    print(f'  {fp.name} not found: using the fields present in the exports')
    return day[KEYS].drop_duplicates()


def build_year(cfg, year: int) -> dict:
    yy = f'{year % 100:02d}'
    files = sorted(cfg.path('s1_exports').glob(f'S1v2_wheat{yy}_*.csv'))
    if not files:
        raise FileNotFoundError(f'no exports for {year} in {cfg.path("s1_exports")}')
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    use = raw[(raw['valid_mean'] >= MIN_VALID_SHARE) & (raw['VVg_count'] >= MIN_PIXELS)]
    use = use.dropna(subset=['VVg_mean', 'VHg_mean']).copy()
    w = use['VVg_count']
    use['wVV'], use['wVH'], use['wang'] = use['VVg_mean'] * w, use['VHg_mean'] * w, use['angle_mean'] * w

    g = use.groupby(KEYS + ['date'])
    day = g[['wVV', 'wVH', 'wang', 'VVg_count']].sum()
    day['VV'] = 10 * np.log10(day['wVV'] / day['VVg_count'])
    day['VH'] = 10 * np.log10(day['wVH'] / day['VVg_count'])
    day['CR'] = day['VH'] - day['VV']
    day['angle'] = day['wang'] / day['VVg_count']
    day['n_scenes'] = g.size()
    day['rel_orbits'] = g['rel_orbit'].agg(lambda s: ';'.join(str(int(v)) for v in sorted(set(s))))
    day['passes'] = g['pass'].agg(lambda s: ';'.join(sorted(set(s))))
    day['valid_share'] = g['valid_mean'].mean()
    day = day.reset_index().rename(columns={'date': 'Date'})

    fields = field_list(cfg, year, day)
    dates = pd.date_range(pd.Timestamp(year - 1, 9, 1), periods=WINDOW_DAYS, freq='D').strftime('%Y-%m-%d')
    full = fields.merge(pd.DataFrame({'Date': dates}), how='cross').merge(day, on=KEYS + ['Date'], how='left')

    out = cfg.path('s1_daily') / str(year)
    out.mkdir(parents=True, exist_ok=True)
    for var in ['VV', 'VH', 'CR']:
        full[['XID', 'Date', var, 'Stations_i', 'Rank']].to_csv(out / f'{var}_wheat{yy}.csv', index=False, na_rep='NA')
    meta = ['XID', 'Date', 'Stations_i', 'Rank', 'angle', 'n_scenes', 'rel_orbits', 'passes', 'valid_share']
    full.loc[full['VV'].notna(), meta].to_csv(out / f'S1meta_wheat{yy}.csv', index=False)

    obs = full.dropna(subset=['VV'])
    return {'year': year, 'export_files': len(files), 'scene_field_rows': len(raw),
            'usable_rows_pct': round(100 * len(use) / len(raw), 1), 'fields': fields['XID'].nunique(),
            'fields_without_S1': int(fields['XID'].nunique() - obs['XID'].nunique()),
            'daily_obs_rate': float(round(full['VV'].notna().mean(), 3)),
            'days_between_obs': float(round(1 / max(full['VV'].notna().mean(), 1e-9), 2))}, obs


def main():
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = load_config(args.config)
    rows, obs = [], []
    for y in cfg.years:
        r, o = build_year(cfg, y)
        rows.append(r); obs.append(o)
        print(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.path('s1_daily') / 'build_summary.csv', index=False)

    allobs = pd.concat(obs)
    pct = allobs[['VV', 'VH', 'CR']].quantile([0.01, 0.5, 0.99]).round(1)
    pct.index = ['p01', 'median', 'p99']
    pct.to_csv(cfg.path('s1_daily') / 'value_ranges_dB.csv')
    print('\nField-level value ranges (dB), all years (Section 2.5):')
    print(pct.to_string())


if __name__ == '__main__':
    main()
