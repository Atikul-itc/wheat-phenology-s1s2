"""Field-level Sentinel-2 Level-1C reflectance and indices from Google Earth Engine (Section 2.4).

Processing per scene (COPERNICUS/S2_HARMONIZED, top-of-atmosphere reflectance,
no atmospheric correction), as in the extraction used for the paper:
  * scenes with CLOUDY_PIXEL_PERCENTAGE >= 95 are skipped (gee.s2_max_cloud_percent);
  * clouds: cloud score from the blue, coastal/cirrus, visible and NDMI tests
    (threshold 20), 1-pixel erosion and 5-pixel dilation;
  * cloud shadows: cloud mask projected along the solar azimuth for cloud heights
    of 200 to 10,000 m (500 m steps), kept where NIR + SWIR1 + SWIR2 < 0.35;
  * indices (EVI, NDRE, CIrededge, MTCI) computed per pixel after masking;
  * field means of the clear pixels for the 9 bands and 4 indices, plus the
    number of clear pixels, for every scene (tile) that covers the field.

Fields that are fully masked in a scene are not exported (the original script wrote
-9999 for them). Same-day values from overlapping tiles are resolved in
``extraction.s2_build_daily``.

Exports go to Google Drive as one CSV per harvest year and chunk of fields; build
the daily files with ``extraction.s2_build_daily``.

Usage:
    python -m extraction.s2_extract_gee --config config.yaml              # dry run
    python -m extraction.s2_extract_gee --config config.yaml --submit
    python -m extraction.s2_extract_gee --config config.yaml --status
"""
from __future__ import annotations

import math
import time

import ee
import pandas as pd

from common import base_parser, load_config
from extraction.s1_extract_gee import KEYS, chunk_fields, gdf_to_fc, init_ee, load_fields

MAX_CLOUD_PERCENT = 95       # default of the original extraction; override with gee.s2_max_cloud_percent
CLOUD_THRESH = 20
CLOUD_HEIGHTS = list(range(200, 10001, 500))
IR_SUM_THRESH = 0.35
DILATE_PIXELS = 5
CONTRACT_PIXELS = 1
WINDOW_DAYS = 426

BANDS_IN = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
BANDS = ['cb', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 'nir2', 'waterVapor', 'cirrus', 'swir1', 'swir2']
FEATURES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 're1', 're2', 're3', 'EVI', 'MTCI', 'NDRE', 'CIrededge']
OUT_COLS = KEYS + ['date', 'scene'] + FEATURES + ['clear_count']


def _rescale(img, exp, lo, hi):
    return img.expression(exp, {'img': img}).subtract(lo).divide(hi - lo)


def cloud_score(img):
    score = ee.Image(1)
    score = score.min(_rescale(img, 'img.blue', 0.1, 0.5))
    score = score.min(_rescale(img, 'img.cb', 0.1, 0.3))
    score = score.min(_rescale(img, 'img.cb + img.cirrus', 0.15, 0.2))
    score = score.min(_rescale(img, 'img.red + img.green + img.blue', 0.2, 0.8))
    ndmi = img.normalizedDifference(['nir', 'swir1'])
    score = score.min(_rescale(ndmi, 'img', -0.1, 0.1))
    return img.addBands(score.multiply(100).byte().rename('cloudScore'))


def mask_clouds_and_shadows(img):
    img = cloud_score(img)
    cloud = img.select('cloudScore').gt(CLOUD_THRESH).focal_min(CONTRACT_PIXELS).focal_max(DILATE_PIXELS)
    dark = (img.select(['nir', 'swir1', 'swir2']).reduce(ee.Reducer.sum()).lt(IR_SUM_THRESH)
            .focal_min(CONTRACT_PIXELS).focal_max(DILATE_PIXELS))
    az = ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')).multiply(math.pi / 180).add(0.5 * math.pi)
    zen = ee.Number(0.5 * math.pi).subtract(ee.Number(img.get('MEAN_SOLAR_ZENITH_ANGLE')).multiply(math.pi / 180))
    scale = cloud.projection().nominalScale()

    def shift(h):
        dist = zen.tan().multiply(ee.Number(h))
        x = az.cos().multiply(dist).divide(scale).round()
        y = az.sin().multiply(dist).divide(scale).round()
        return cloud.changeProj(cloud.projection(), cloud.projection().translate(x, y))

    shadow = ee.ImageCollection.fromImages([shift(h) for h in CLOUD_HEIGHTS]).max()
    shadow = shadow.And(cloud.Not()).And(dark)
    return img.updateMask(shadow.Or(cloud).Not())


def add_indices(img):
    evi = img.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                         {'NIR': img.select('nir'), 'RED': img.select('red'), 'BLUE': img.select('blue')})
    ndre = img.normalizedDifference(['nir', 're1'])
    cire = img.expression('(NIR / RE1) - 1', {'NIR': img.select('nir'), 'RE1': img.select('re1')})
    mtci = img.expression('(RE2 - RE1) / (RE1 - RED)',
                          {'RE2': img.select('re2'), 'RE1': img.select('re1'), 'RED': img.select('red')})
    return img.addBands(ee.Image.cat([evi.rename('EVI'), ndre.rename('NDRE'), cire.rename('CIrededge'),
                                      mtci.rename('MTCI')]).float())


def s2_collection(start, end, region, max_cloud=MAX_CLOUD_PERCENT):
    def prep(img):
        t = img.select(BANDS_IN, BANDS).divide(10000)
        return t.copyProperties(img, ['system:time_start', 'system:index',
                                      'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE'])
    return (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
            .filterDate(start, end).filterBounds(region)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
            .map(prep).map(mask_clouds_and_shadows).map(add_indices))


def scene_to_rows(img, fields):
    stats = img.select(FEATURES).reduceRegions(collection=fields, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
    counts = img.select('blue').rename('clear_count').reduceRegions(
        collection=fields, reducer=ee.Reducer.count().setOutputs(['clear_count']), scale=10, tileScale=4)
    joined = ee.Join.inner().apply(stats, counts, ee.Filter.equals(leftField='XID', rightField='XID'))
    meta = {'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'), 'scene': img.get('system:index')}

    def merge(pair):
        a, b = ee.Feature(pair.get('primary')), ee.Feature(pair.get('secondary'))
        return a.set('clear_count', b.get('clear_count')).set(meta)

    return joined.map(merge).filter(ee.Filter.gt('clear_count', 0))


def submit(cfg, dry_run: bool) -> None:
    folder = cfg['gee'].get('drive_folder_s2', 'S2')
    max_mb = float(cfg['gee'].get('max_chunk_mb', 4.0))
    max_cloud = float(cfg['gee'].get('s2_max_cloud_percent', MAX_CLOUD_PERCENT))
    n = 0
    for year in cfg.years:
        yy = f'{year % 100:02d}'
        gdf = load_fields(cfg.pattern('fields'), year)
        chunks = chunk_fields(gdf, max_mb)
        print(f'{year}: {len(gdf)} fields in {len(chunks)} export(s), scenes with < {max_cloud:g}% cloud')
        if dry_run:
            continue
        start = ee.Date.fromYMD(year - 1, 9, 1)
        end = start.advance(WINDOW_DAYS, 'day')
        for k, part in enumerate(chunks, 1):
            name = f'S2_wheat{yy}_part{k:02d}'
            region = ee.Geometry.Rectangle(list(part.total_bounds))
            fields = gdf_to_fc(part)
            rows = s2_collection(start, end, region, max_cloud).map(lambda im: scene_to_rows(im, fields)).flatten()
            ee.batch.Export.table.toDrive(collection=rows, description=name, folder=folder,
                                          fileNamePrefix=name, fileFormat='CSV', selectors=OUT_COLS).start()
            n += 1
            print(f'  started {name} ({len(part)} fields)')
            time.sleep(1)
    print('Dry run: nothing submitted. Add --submit to start the exports.' if dry_run else f'{n} exports started.')


def status() -> None:
    recs = [t.status() for t in ee.batch.Task.list()]
    df = pd.DataFrame([{'name': s.get('description'), 'state': s.get('state'), 'error': s.get('error_message', '')}
                       for s in recs if str(s.get('description', '')).startswith('S2_wheat')])
    print('No S2 exports found.' if df.empty else df['state'].value_counts().to_string())


def main():
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument('--submit', action='store_true', help='start the Drive exports (default: dry run)')
    ap.add_argument('--status', action='store_true', help='list the state of submitted exports')
    args = ap.parse_args()
    cfg = load_config(args.config)
    init_ee(cfg['gee']['project'])
    status() if args.status else submit(cfg, dry_run=not args.submit)


if __name__ == '__main__':
    main()
