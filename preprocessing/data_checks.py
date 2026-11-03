"""Dataset summaries reported in the paper.

  counts : stations, field-years and field-years with all seven stages per year
           (Table 1) and stations per stage and year (Supplementary Table S2),
           computed from the model tensors.
  shape  : field area and shape metrics per year from the field GeoPackages
           (Section 2.3: minimum area-to-perimeter ratio of the final field set).

Usage:
    python -m preprocessing.data_checks --config config.yaml counts
    python -m preprocessing.data_checks --config config.yaml shape
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import STAGE_NAMES, base_parser, load_config


def counts(cfg) -> None:
    rows, stations = [], set()
    for y in cfg.years:
        z = np.load(cfg.path('tensors') / f'phenology_{y}.npz', allow_pickle=False)
        lm, bag = z['label_mask'], z['bag_id']
        stations |= set(bag.tolist())
        r = {'year': y, 'stations': len(np.unique(bag)), 'field_years': len(z['xid']),
             'field_years_all_7_stages': int(lm.all(axis=1).sum())}
        for j, p in enumerate(z['phase_ids']):
            r[STAGE_NAMES[int(p)]] = len(np.unique(bag[lm[:, j]]))
        rows.append(r)
    t = pd.DataFrame(rows).set_index('year')
    t.loc['Total'] = t.sum()
    t.loc['Total', 'stations'] = len(stations)          # distinct stations over all years
    out = cfg.path('tables', mkdir=True) / 'table1_counts.csv'
    t.to_csv(out)
    print(t.to_string())
    print('Saved', out)


def shape(cfg) -> None:
    import geopandas as gpd
    rows = []
    for y in cfg.years:
        g = gpd.read_file(cfg.pattern('fields').format(year=y)).to_crs(32632)
        a, p = g.area, g.length
        rows.append({'year': y, 'fields': len(g), 'area_ha_min': (a / 1e4).min(), 'area_ha_max': (a / 1e4).max(),
                     'area_to_perimeter_m_min': (a / p).min(), 'area_to_perimeter_m_median': (a / p).median(),
                     'polsby_popper_min': (4 * np.pi * a / p ** 2).min()})
    t = pd.DataFrame(rows).set_index('year').round(3)
    t.to_csv(cfg.path('tables', mkdir=True) / 'field_shape_metrics.csv')
    print(t.to_string())


def main():
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument('check', choices=['counts', 'shape'])
    args = ap.parse_args()
    cfg = load_config(args.config)
    {'counts': counts, 'shape': shape}[args.check](cfg)


if __name__ == '__main__':
    main()
