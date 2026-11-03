"""Daily temperature, precipitation and growing degree days per phenology station (Section 2.6).

Steps:
  1. Download DWD Climate Data Center daily station data (kl: TMK, RSK;
     more_precip: RS) for all stations operating in the study period (cached).
  2. For each phenology station and day, interpolate by inverse distance weighting
     from the ten nearest stations with valid data that day (power 0.5).
     Temperatures are moved to the elevation of the phenology station with a
     lapse rate of -0.0065 degC per m before interpolation.
  3. Accumulate growing degree days (base 0 degC) from 1 September of the year
     before harvest. Missing temperature contributes zero.

Output: <climate>/climate_with_cgdd_<year>.csv and station_distance_summary.csv.

Usage:
    python -m preprocessing.dwd_meteorology --config config.yaml [--offline]
"""
from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from common import base_parser, load_config

N_NEAREST = 10
IDW_POWER = 0.5          # as in Lobert et al. (2023)
LAPSE_RATE = -0.0065     # degC per m
GDD_BASE = 0.0
MAX_CANDIDATES = 40      # nearest stations screened per target before picking the ten valid ones
N_THREADS = 8

CDC = 'https://opendata.dwd.de/climate_environment/CDC'
URLS = {'kl': f'{CDC}/observations_germany/climate/daily/kl/historical/',
        'rr': f'{CDC}/observations_germany/climate/daily/more_precip/historical/',
        'pheno_stations': f'{CDC}/help/PH_Beschreibung_Phaenologie_Stationen_Jahresmelder.txt'}
STATION_LISTS = {'kl': 'KL_Tageswerte_Beschreibung_Stationen.txt', 'rr': 'RR_Tageswerte_Beschreibung_Stationen.txt'}


class Downloader:
    def __init__(self, raw_dir: Path, offline: bool):
        self.raw_dir, self.offline, self.session = raw_dir, offline, requests.Session()

    def fetch(self, url: str, dest: Path | None = None, timeout: int = 120):
        """Return bytes (dest None) or the cached file path."""
        if dest is not None and dest.exists() and dest.stat().st_size > 0:
            return dest
        if self.offline:
            raise FileNotFoundError(f'offline and not cached: {dest or url}')
        r = self.session.get(url, timeout=timeout)
        r.raise_for_status()
        if dest is None:
            return r.content
        dest.write_bytes(r.content)
        return dest


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def parse_station_list(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines()[2:]:
        tok = line.split()
        if len(tok) < 6 or not tok[0].isdigit():
            continue
        rows.append({'station_id': int(tok[0]), 'from': pd.to_datetime(tok[1], format='%Y%m%d'),
                     'to': pd.to_datetime(tok[2], format='%Y%m%d'), 'elev_m': float(tok[3]),
                     'lat': float(tok[4]), 'lon': float(tok[5])})
    return pd.DataFrame(rows)


def phenology_stations(dl: Downloader, used: list[int]) -> pd.DataFrame:
    path = dl.fetch(URLS['pheno_stations'], dl.raw_dir / Path(URLS['pheno_stations']).name)
    df = pd.read_csv(io.StringIO(Path(path).read_bytes().decode('latin-1')), sep=';', skipinitialspace=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    col = lambda key: next(c for c in df.columns if key.lower() in c.lower())
    ph = pd.DataFrame({'Stations_i': pd.to_numeric(df[col('Stations_id')], errors='coerce'),
                       'lat': pd.to_numeric(df[col('Breite')], errors='coerce'),
                       'lon': pd.to_numeric(df[col('Laenge')], errors='coerce'),
                       'elev_m': pd.to_numeric(df[col('hoehe')], errors='coerce')}).dropna()
    ph['Stations_i'] = ph['Stations_i'].astype(int)
    out = ph[ph['Stations_i'].isin(used)].drop_duplicates('Stations_i').reset_index(drop=True)
    missing = sorted(set(used) - set(out['Stations_i']))
    print(f'{len(out)} phenology stations with coordinates; {len(missing)} without')
    return out


def download_kind(dl: Downloader, kind: str, start: str, end: str) -> pd.DataFrame:
    dest = dl.raw_dir / STATION_LISTS[kind]
    dl.fetch(URLS[kind] + STATION_LISTS[kind], dest)
    st = parse_station_list(dest.read_bytes().decode('latin-1'))
    st = st[(st['to'] >= start) & (st['from'] <= end)].reset_index(drop=True)
    (dl.raw_dir / kind).mkdir(parents=True, exist_ok=True)
    if dl.offline:
        files = [p.name for p in (dl.raw_dir / kind).glob('*.zip')]
    else:
        html = dl.fetch(URLS[kind]).decode('latin-1')
        files = sorted(set(re.findall(r'href="([^"]+\.zip)"', html)))
    wanted = set(st['station_id'])
    jobs = [(URLS[kind] + f, dl.raw_dir / kind / f) for f in files
            if (m := re.search(r'_(\d{5})_', f)) and int(m.group(1)) in wanted]
    failed = 0
    with ThreadPoolExecutor(N_THREADS) as ex:
        for fut in as_completed([ex.submit(dl.fetch, u, d) for u, d in jobs]):
            try:
                fut.result()
            except Exception:
                failed += 1
    print(f'{kind}: {len(wanted)} stations in the period, {len(jobs)} files, {failed} failed downloads')
    return st


def read_kind(raw_dir: Path, kind: str, cols: list[str], start: str, end: str) -> pd.DataFrame:
    parts = []
    for p in sorted((raw_dir / kind).glob('*.zip')):
        try:
            with zipfile.ZipFile(p) as z:
                name = next(n for n in z.namelist() if n.startswith('produkt_'))
                df = pd.read_csv(z.open(name), sep=';', na_values=['-999', -999], skipinitialspace=True,
                                 encoding='latin-1')
        except Exception as e:
            print('skip', p.name, e)
            continue
        df.columns = [c.strip() for c in df.columns]
        df['date'] = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d')
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        parts.append(df[['STATIONS_ID', 'date'] + cols].rename(columns={'STATIONS_ID': 'station_id'}))
    return pd.concat(parts).drop_duplicates(['station_id', 'date'])


def idw(t_lat, t_lon, t_elev, st: pd.DataFrame, mat: np.ndarray, lapse: float | None = None):
    """IDW estimate per day from the N_NEAREST stations with valid data that day."""
    d = haversine_km(t_lat, t_lon, st['lat'].to_numpy(), st['lon'].to_numpy())
    cand = np.argsort(d)[:MAX_CANDIDATES]
    dist = np.maximum(d[cand], 0.1)
    vals = mat[cand]
    if lapse is not None:
        vals = vals + lapse * (t_elev - st['elev_m'].to_numpy()[cand])[:, None]
    valid = np.isfinite(vals)
    use = valid & (np.cumsum(valid, axis=0) <= N_NEAREST)
    w = np.where(use, 1.0 / dist[:, None] ** IDW_POWER, 0.0)
    est = (np.where(use, vals, 0.0) * w).sum(0) / np.where(w.sum(0) > 0, w.sum(0), np.nan)
    n_used = use.sum(0)
    mean_dist = np.where(use, dist[:, None], 0).sum(0) / np.maximum(n_used, 1)
    return est, n_used, dist[0], mean_dist


def main():
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument('--offline', action='store_true', help='use only files already in the download cache')
    args = ap.parse_args()
    cfg = load_config(args.config)
    years = cfg.years
    raw_dir, out_dir = cfg.path('climate_raw', mkdir=True), cfg.path('climate', mkdir=True)
    start, end = f'{min(years) - 1}-09-01', f'{max(years)}-10-31'
    dl = Downloader(raw_dir, args.offline)

    used = sorted(set().union(*[set(pd.read_csv(cfg.path('metadata') / f'metadata_S2_{y}.csv',
                                                usecols=['Stations_i'])['Stations_i']) for y in years]))
    targets = phenology_stations(dl, used)

    kl_st = download_kind(dl, 'kl', start, end)
    rr_st = download_kind(dl, 'rr', start, end)
    kl = read_kind(raw_dir, 'kl', ['TMK', 'RSK'], start, end)
    rr = read_kind(raw_dir, 'rr', ['RS'], start, end)

    days = pd.date_range(start, end, freq='D')
    to_mat = lambda df, col, ids: (df.pivot_table(index='station_id', columns='date', values=col)
                                   .reindex(index=ids, columns=days).to_numpy(dtype=np.float64))
    t_st = kl_st[kl_st['station_id'].isin(kl['station_id'])].reset_index(drop=True)
    T_MAT = to_mat(kl, 'TMK', t_st['station_id'])
    kl_p = kl.rename(columns={'RSK': 'RS'})[['station_id', 'date', 'RS']]
    p_all = pd.concat([rr, kl_p[~kl_p['station_id'].isin(rr['station_id'])]])
    p_st = pd.concat([rr_st, kl_st]).drop_duplicates('station_id')
    p_st = p_st[p_st['station_id'].isin(p_all['station_id'])].reset_index(drop=True)
    P_MAT = to_mat(p_all, 'RS', p_st['station_id'])

    series, dist_rows = {}, []
    for r in targets.itertuples(index=False):
        t, nt, d1t, mdt = idw(r.lat, r.lon, r.elev_m, t_st, T_MAT, lapse=LAPSE_RATE)
        p, npn, d1p, mdp = idw(r.lat, r.lon, r.elev_m, p_st, P_MAT)
        series[r.Stations_i] = pd.DataFrame({'date': days, 'mean_temp_celsius': t, 'precipitation_mm': p,
                                             'n_temp_stations': nt, 'n_precip_stations': npn})
        dist_rows.append({'Stations_i': r.Stations_i, 'nearest_temp_km': d1t, 'mean_temp_km': np.nanmean(mdt),
                          'nearest_precip_km': d1p, 'mean_precip_km': np.nanmean(mdp)})
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out_dir / 'station_distance_summary.csv', index=False)
    print('Distances from phenology stations to climate stations (km):')
    print(dist.drop(columns='Stations_i').describe().round(1).to_string())

    for y in years:
        lo, hi = pd.Timestamp(y - 1, 9, 1), pd.Timestamp(y, 10, 31)
        parts = []
        for sid, s in series.items():
            s = s[(s['date'] >= lo) & (s['date'] <= hi)].copy()
            s['daily_gdd'] = (s['mean_temp_celsius'] - GDD_BASE).clip(lower=0).fillna(0.0)
            s['cumulative_gdd'] = s['daily_gdd'].cumsum()
            s.insert(0, 'Stations_i', sid)
            parts.append(s)
        out = pd.concat(parts)
        out['date'] = out['date'].dt.strftime('%m/%d/%Y')        # format read by phenology_pipeline
        out.to_csv(out_dir / f'climate_with_cgdd_{y}.csv', index=False)
        print(f'{y}: {out.Stations_i.nunique()} stations, missing temperature '
              f'{100 * out["mean_temp_celsius"].isna().mean():.2f}%')


if __name__ == '__main__':
    main()
