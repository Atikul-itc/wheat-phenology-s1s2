"""Build the daily model tensors per harvest year (Section 3.1).

Reads the daily S1, S2 and climate files and the DWD reference dates through
``phenology_pipeline`` and writes <tensors>/phenology_<year>.npz with
  X (records, 426 days, 19 features), mask, Y (soft Gaussian labels, sigma 5 d),
  label_mask, bag_id (station), rank, xid, harvest_year, feature_names, phase_ids, T.

Usage:
    python -m preprocessing.build_tensors --config config.yaml
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import base_parser, load_config
import phenology_pipeline.config as PC


def main():
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = load_config(args.config)
    PC.configure(BASE_DIR=cfg.root, S1_DIR=cfg.path('s1_daily'), S2_DIR=cfg.path('s2_daily'),
                 CLIM_DIR=cfg.path('climate'), META_DIR=cfg.path('metadata'),
                 PHENO_MASTER=cfg.path('phenology_master'), CACHE_DIR=cfg.path('tensors').parent)
    from phenology_pipeline.build_dataset import build_all_years

    out = cfg.path('tensors', mkdir=True)
    bundles = build_all_years(harvest_years=cfg.years, output_dir=out)

    rows = []
    for y, b in sorted(bundles.items()):
        f = [str(s) for s in b['feature_names']]
        obs = lambda name: float(b['mask'][:, :, f.index(name)].mean())
        rows.append({'year': y, 'records': len(b['xid']), 'stations': len(np.unique(b['bag_id'])),
                     'all_7_stages': int(b['label_mask'].all(axis=1).sum()), 'T': int(b['T']),
                     'S1_obs_rate': round(obs('VV'), 3), 'S2_obs_rate': round(obs('blue'), 3),
                     'met_obs_rate': round(obs('mean_temp_celsius'), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
