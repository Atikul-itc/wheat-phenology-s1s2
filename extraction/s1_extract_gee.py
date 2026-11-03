"""Field-level Sentinel-1 backscatter from Google Earth Engine (Section 2.5).

For every IW scene with VV and VH (COPERNICUS/S1_GRD_FLOAT, both passes) and every
winter wheat field, the script exports per scene and field:
  * mean VV and VH as gamma0 = sigma0 / cos(theta) in linear power,
  * pixel counts and the share of valid pixels,
  * platform, relative orbit, pass and mean incidence angle.

Border noise (IPF < 2.90) is removed with a pixel mask: incidence angle within
[31, 45] degrees, VV > -30 dB and VH > -35 dB. Scenes are aggregated to daily
values by ``extraction.s1_build_daily``.

Usage (from the repository root):
    python -m extraction.s1_extract_gee --config config.yaml              # dry run
    python -m extraction.s1_extract_gee --config config.yaml --submit     # start exports
    python -m extraction.s1_extract_gee --config config.yaml --status     # monitor
"""
from __future__ import annotations

import json
import math
import time

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

from common import base_parser, load_config

ANGLE_MIN, ANGLE_MAX = 31.0, 45.0
VV_NOISE_DB, VH_NOISE_DB = -30.0, -35.0
WINDOW_DAYS = 426            # 1 September (harvest year - 1) to 31 October (harvest year)
SIMPLIFY_M = 1.0             # vertex simplification of field polygons (m)
KEYS = ['XID', 'Stations_i', 'Rank']
OUT_COLS = KEYS + ['date', 'scene', 'platform', 'rel_orbit', 'pass',
                   'VVg_mean', 'VHg_mean', 'angle_mean', 'VVg_count', 'valid_count', 'valid_mean']


def init_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def load_fields(pattern: str, year: int) -> gpd.GeoDataFrame:
    """Fields of one harvest year in EPSG:4326, slightly simplified."""
    gdf = gpd.read_file(pattern.format(year=year))
    missing = [c for c in KEYS if c not in gdf.columns]
    if missing:
        raise KeyError(f'columns {missing} not found in {pattern.format(year=year)}')
    gdf = gdf[KEYS + ['geometry']].to_crs(32632)
    gdf['geometry'] = gdf.simplify(SIMPLIFY_M, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid].copy()
    gdf[KEYS] = gdf[KEYS].astype('int64')
    return gdf.to_crs(4326)


def s1_collection(start, end, region=None):
    col = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
           .filterDate(start, end)
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    return col.filterBounds(region) if region is not None else col


def prepare(img):
    """Mask border noise, normalise to gamma0 and add a 0/1 validity band."""
    vv, vh, ang = img.select('VV'), img.select('VH'), img.select('angle')
    footprint = ang.mask()
    valid = (ang.gte(ANGLE_MIN).And(ang.lte(ANGLE_MAX))
             .And(vv.gt(10 ** (VV_NOISE_DB / 10)))
             .And(vh.gt(10 ** (VH_NOISE_DB / 10))))
    cos_t = ang.multiply(math.pi / 180).cos()
    vvg = vv.divide(cos_t).updateMask(valid).rename('VVg')
    vhg = vh.divide(cos_t).updateMask(valid).rename('VHg')
    valid01 = valid.unmask(0).updateMask(footprint).rename('valid')
    return vvg.addBands(vhg).addBands(ang.updateMask(valid).rename('angle')).addBands(valid01)


def scene_to_rows(img, fields):
    """Field statistics for one scene; fields outside the footprint are dropped."""
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    meta = {'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
            'scene': img.get('system:index'),
            'platform': img.get('platform_number'),
            'rel_orbit': img.get('relativeOrbitNumber_start'),
            'pass': img.get('orbitProperties_pass')}
    fc = prepare(img).reduceRegions(collection=fields, reducer=reducer, scale=10, tileScale=4)
    return fc.filter(ee.Filter.gt('valid_count', 0)).map(lambda f: f.set(meta))


def gdf_to_fc(gdf):
    return ee.FeatureCollection([ee.Feature(ee.Geometry(mapping(g)), {k: int(r[k]) for k in KEYS})
                                 for g, (_, r) in zip(gdf.geometry, gdf.iterrows())])


def chunk_fields(gdf, max_mb):
    """Split fields so that each request carries at most max_mb of geometry."""
    sizes = np.array([len(json.dumps(mapping(g))) for g in gdf.geometry]) / 1e6
    chunks, start, acc = [], 0, 0.0
    for i, s in enumerate(sizes):
        if acc + s > max_mb and i > start:
            chunks.append(gdf.iloc[start:i]); start, acc = i, 0.0
        acc += s
    chunks.append(gdf.iloc[start:])
    return chunks


def submit(cfg, dry_run: bool) -> None:
    folder = cfg['gee'].get('drive_folder_s1', 'S1_v2')
    max_mb = float(cfg['gee'].get('max_chunk_mb', 4.0))
    n = 0
    for year in cfg.years:
        yy = f'{year % 100:02d}'
        gdf = load_fields(cfg.pattern('fields'), year)
        chunks = chunk_fields(gdf, max_mb)
        print(f'{year}: {len(gdf)} fields in {len(chunks)} export(s)')
        if dry_run:
            continue
        start = ee.Date.fromYMD(year - 1, 9, 1)
        end = start.advance(WINDOW_DAYS, 'day')
        for k, part in enumerate(chunks, 1):
            name = f'S1v2_wheat{yy}_part{k:02d}'
            region = ee.Geometry.Rectangle(list(part.total_bounds))
            fields = gdf_to_fc(part)
            rows = s1_collection(start, end, region).map(lambda im: scene_to_rows(im, fields)).flatten()
            ee.batch.Export.table.toDrive(collection=rows, description=name, folder=folder,
                                          fileNamePrefix=name, fileFormat='CSV', selectors=OUT_COLS).start()
            n += 1
            print(f'  started {name} ({len(part)} fields)')
            time.sleep(1)
    print('Dry run: nothing submitted. Add --submit to start the exports.' if dry_run else f'{n} exports started.')


def status() -> None:
    recs = [t.status() for t in ee.batch.Task.list()]
    df = pd.DataFrame([{'name': s.get('description'), 'state': s.get('state'), 'error': s.get('error_message', '')}
                       for s in recs if str(s.get('description', '')).startswith('S1v2_')])
    if df.empty:
        print('No S1v2 exports found.')
        return
    print(df['state'].value_counts().to_string())
    failed = df[df['state'].isin(['FAILED', 'CANCELLED'])]
    if len(failed):
        print(failed.to_string(index=False))


def main():
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument('--submit', action='store_true', help='start the Drive exports (default: dry run)')
    ap.add_argument('--status', action='store_true', help='list the state of submitted exports')
    args = ap.parse_args()
    cfg = load_config(args.config)
    init_ee(cfg['gee']['project'])
    if args.status:
        status()
    else:
        submit(cfg, dry_run=not args.submit)


if __name__ == '__main__':
    main()
