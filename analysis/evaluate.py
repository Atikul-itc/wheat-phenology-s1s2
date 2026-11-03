"""Accuracy tables, paired comparisons and baselines (Sections 3.3, 3.6, 3.7; Results).

Reads the stored predictions of model.train, adds the baselines and writes:
  tableA_accuracy.csv           MAE, 95% CI, seed SD, RMSE, share within +/-5 d, bias, MAE per stage (Table 2, Fig. 5, 7)
  tableA_dlog_four_stages.csv   double logistic baseline on the four stages it estimates
  tableB_paired.csv             paired differences with station-clustered bootstrap (Fig. 6, Table S3)
  tableC_policy.csv             label assignment experiments (Table 4)
  tableD_per_year.csv           macro MAE per test year (Table 3)
  tableE_by_rank.csv            MAE by distance rank of the field
  tableF_cloud_<measure>.csv    added value of Sentinel-1 by Sentinel-2 availability (Fig. 8)

Baselines: the mean-date baseline (mean reference day of each stage in the four
training years), histogram gradient boosting on seasonal summary statistics, and a
double logistic fit to the Sentinel-2 EVI series (raw and bias-corrected).

Usage:
    python -m analysis.evaluate --config config.yaml [--skip-hgb] [--skip-dlog]
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from common import STAGE_NAMES, base_parser, load_config

warnings.filterwarnings('ignore', category=RuntimeWarning)       # all-NaN slices for stages without references

N_BOOT = 2000
EQ_MARGIN = 1.0          # days, practical equivalence margin
CLOUD_HALF = 30          # days, half-width of the stage-specific Sentinel-2 window
RNG_SEED = 2026
STAGES = [10, 12, 15, 18, 19, 21, 24]
P = len(STAGES)
CFG_COLS = ['stream', 'policy', 'encoder', 'k']
MEAN_DATE = ('MeanDate', 'shared', 'MEAN', 1)


def G(stream, policy='shared', encoder='GRU', k=1):
    return (stream, policy, encoder, k)


PAIRS = [
    ('S1+S2 vs MeanDate', G('S1+S2'), MEAN_DATE),
    ('S1+S2 vs S2', G('S1+S2'), G('S2')),
    ('S1+S2 vs S1', G('S1+S2'), G('S1')),
    ('S1 vs S2', G('S1'), G('S2')),
    ('S2+Met vs S2', G('S2+Met'), G('S2')),
    ('S1+Met vs S1', G('S1+Met'), G('S1')),
    ('All vs S1+S2', G('All'), G('S1+S2')),
    ('Met vs MeanDate', G('Met'), MEAN_DATE),
    ('S2 bands vs S2', G('S2bands'), G('S2')),
    ('S1+S2 bands vs S1+S2', G('S1+S2bands'), G('S1+S2')),
    ('S1+S2 5-day vs daily', G('S1+S2', k=5), G('S1+S2')),
    ('S1+S2 10-day vs daily', G('S1+S2', k=10), G('S1+S2')),
    ('All 5-day vs daily', G('All', k=5), G('All')),
    ('LSTM vs GRU (S1+S2)', G('S1+S2', encoder='LSTM'), G('S1+S2')),
    ('LSTM vs GRU (All)', G('All', encoder='LSTM'), G('All')),
    ('GRU vs HGB (S1+S2)', G('S1+S2'), ('S1+S2', 'shared', 'HGB', 1)),
    ('GRU vs HGB (All)', G('All'), ('All', 'shared', 'HGB', 1)),
]


class Evaluation:
    def __init__(self, cfg):
        self.cfg, self.years = cfg, cfg.years
        self.grid = cfg.path('runs')
        self.tables = cfg.path('tables', mkdir=True)
        self.runs = pd.read_csv(self.grid / 'runs.csv').drop_duplicates('run_id', keep='last')
        self.ERR, self.PRED = {}, {}
        self._bundles = None
        self._load_predictions()
        self._setup_bootstrap()

    # ------------------------------------------------------------------ inputs
    def _load_predictions(self):
        self.REC = {}
        for y in self.years:
            rid = self.runs[self.runs['test_year'] == y]['run_id'].iloc[0]
            z = np.load(self.grid / 'preds' / f'{rid}.npz')
            self.REC[y] = pd.DataFrame({'xid': z['xid'], 'bag': z['bag'], 'rank': z['rank']})
            self.REC[y]['truth'] = list(z['truth'])
        self.TRUTH = {y: np.stack(self.REC[y]['truth'].to_numpy()) for y in self.years}
        for cfg_key, g in self.runs.groupby(CFG_COLS):
            for y, gy in g.groupby('test_year'):
                stack = []
                for rid in gy.sort_values('seed')['run_id']:
                    z = np.load(self.grid / 'preds' / f'{rid}.npz')
                    assert np.array_equal(z['xid'], self.REC[y]['xid'].to_numpy()), f'record order differs in {rid}'
                    stack.append(z['pred'].astype(float))
                self.add(cfg_key, int(y), np.stack(stack))
        for y in self.years:                                       # mean-date baseline
            train = np.concatenate([self.TRUTH[t] for t in self.years if t != y])
            mean_day = np.array([train[train[:, j] >= 0, j].mean() for j in range(P)])
            self.add(MEAN_DATE, y, np.rint(np.tile(mean_day, (len(self.REC[y]), 1)))[None])
        print(f'{len(self.ERR)} configurations loaded')

    def add(self, cfg_key, year, pred_stack):
        t = self.TRUTH[year]
        ae = np.where((t[None] >= 0) & np.isfinite(pred_stack), np.abs(pred_stack - t[None]), np.nan)
        self.ERR.setdefault(cfg_key, {})[year] = ae
        self.PRED.setdefault(cfg_key, {})[year] = pred_stack

    def bundles(self):
        if self._bundles is None:
            from baselines.dataset import load_year_bundle
            self._bundles = {y: load_year_bundle(self.cfg.path('tensors') / f'phenology_{y}.npz') for y in self.years}
            for y in self.years:
                assert np.array_equal(self._bundles[y]['xid'], self.REC[y]['xid'].to_numpy()), f'tensor order differs {y}'
        return self._bundles

    def add_hgb(self):
        from baselines.gradient_boosting import predict_models, train_tabular
        from baselines.engineered_features import compute_for_bundle
        s2 = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 're1', 're2', 're3', 'EVI', 'MTCI', 'NDRE', 'CIrededge']
        s1, met = ['VV', 'VH', 'CR'], ['mean_temp_celsius', 'precipitation_mm', 'cumulative_gdd']
        streams = {'S1': s1, 'S2': s2, 'Met': met, 'S1+S2': s1 + s2, 'All': s1 + s2 + met}
        cache_dir = self.grid / 'baselines'; cache_dir.mkdir(exist_ok=True)
        for st, feats in streams.items():
            cache = cache_dir / f'HGB_{st}.npz'
            if not cache.exists():
                B = self.bundles()
                F = {y: compute_for_bundle(B[y], restrict_features=feats)[0] for y in self.years}
                out = {}
                for y in self.years:
                    tr = [t for t in self.years if t != y]
                    models = train_tabular(np.concatenate([F[t] for t in tr]), np.concatenate([self.TRUTH[t] for t in tr]))
                    out[str(y)] = predict_models(models, F[y], n_phases=P)
                    print(f'  HGB {st} {y} done', flush=True)
                np.savez_compressed(cache, **out)
            z = np.load(cache)
            for y in self.years:
                self.add((st, 'shared', 'HGB', 1), y, np.rint(z[str(y)])[None])

    def add_double_logistic(self):
        from baselines.double_logistic import predict_for_bundle
        cache = self.grid / 'baselines' / 'DLOG.npz'
        cache.parent.mkdir(exist_ok=True)
        if not cache.exists():
            B = self.bundles()
            np.savez_compressed(cache, **{str(y): predict_for_bundle(B[y]) for y in self.years})
        z = np.load(cache)
        raw = {y: z[str(y)].astype(float) for y in self.years}
        for y in self.years:
            self.add(('S2 (EVI)', 'raw', 'DLOG', 1), y, np.rint(raw[y])[None])
            tr = [t for t in self.years if t != y]               # bias correction from the training years
            bias = np.array([np.nanmean(np.concatenate(
                [np.where(self.TRUTH[t][:, j] >= 0, raw[t][:, j] - self.TRUTH[t][:, j], np.nan) for t in tr]))
                for j in range(P)])
            self.add(('S2 (EVI)', 'calibrated', 'DLOG', 1), y, np.rint(raw[y] - bias[None])[None])

    # ------------------------------------------------------------------ statistics
    def _setup_bootstrap(self):
        cl = pd.concat([self.REC[y][['bag']].assign(year=y) for y in self.years], ignore_index=True)
        cl['cid'] = cl.groupby(['year', 'bag']).ngroup()
        self.CID = cl['cid'].to_numpy(); self.NC = self.CID.max() + 1
        cl_year = cl.groupby('cid')['year'].first().to_numpy()
        rng = np.random.default_rng(RNG_SEED)
        self.W = np.zeros((N_BOOT, self.NC))
        for y in self.years:                                       # station-years resampled within each test year
            ids = np.where(cl_year == y)[0]
            draws = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
            for b in range(N_BOOT):
                np.add.at(self.W[b], ids[draws[b]], 1)
        self.RANKS = np.concatenate([self.REC[y]['rank'].to_numpy() for y in self.years])
        print(f'{self.NC} station-year clusters, {len(self.CID):,} records')

    def seed_avg(self, c, mask=None):
        a = np.concatenate([np.nanmean(self.ERR[c][y], axis=0) for y in self.years])
        return a if mask is None else np.where(mask[:, None], a, np.nan)

    def cluster_sums(self, a):
        ok = np.isfinite(a)
        s = np.zeros((self.NC, a.shape[1])); n = np.zeros((self.NC, a.shape[1]))
        np.add.at(s, self.CID, np.where(ok, a, 0)); np.add.at(n, self.CID, ok)
        return s, n

    @staticmethod
    def macro(s, n, stages=None):
        j = slice(None) if stages is None else stages
        return np.nanmean(s[..., j] / np.where(n[..., j] > 0, n[..., j], np.nan), axis=-1)

    def point_macro(self, a, stages=None):
        s, n = self.cluster_sums(a)
        return self.macro(s.sum(0), n.sum(0), stages)

    def boot_macro(self, a, stages=None):
        s, n = self.cluster_sums(a)
        return self.macro(self.W @ s, self.W @ n, stages)

    def seed_macros(self, c, stages=None, mask=None):
        S = min(self.ERR[c][y].shape[0] for y in self.years)
        out = []
        for k in range(S):
            a = np.concatenate([self.ERR[c][y][k] for y in self.years])
            out.append(self.point_macro(a if mask is None else np.where(mask[:, None], a, np.nan), stages))
        return np.array(out)

    # ------------------------------------------------------------------ tables
    def describe(self, c, stages=None):
        a = self.seed_avg(c)
        s, n = self.cluster_sums(a)
        bs = self.boot_macro(a, stages)
        pred = np.concatenate([np.nanmean(self.PRED[c][y], 0) for y in self.years])
        tru = np.concatenate([self.TRUTH[y] for y in self.years]).astype(float); tru[tru < 0] = np.nan
        d = pred - tru
        j = slice(None) if stages is None else stages
        sm = self.seed_macros(c, stages)
        row = {**dict(zip(CFG_COLS, c)), 'MAE': self.macro(s.sum(0), n.sum(0), stages),
               'CI_low': np.percentile(bs, 2.5), 'CI_high': np.percentile(bs, 97.5),
               'seed_sd': sm.std(ddof=1) if len(sm) > 1 else np.nan,
               'RMSE': np.nanmean(np.sqrt(np.nanmean(d[:, j] ** 2, 0))),
               'within_5d': np.nanmean(np.nanmean(np.where(np.isfinite(d[:, j]), np.abs(d[:, j]) <= 5, np.nan), 0)),
               'bias': np.nanmean(np.nanmean(d[:, j], 0))}
        for jj, p in enumerate(STAGES):
            row[f'MAE_{STAGE_NAMES[p]}'] = s[:, jj].sum() / n[:, jj].sum() if n[:, jj].sum() else np.nan
        return row

    def paired(self, name, A, B, stages=None, mask=None):
        if A not in self.ERR or B not in self.ERR:
            return None
        a, b = self.seed_avg(A, mask), self.seed_avg(B, mask)
        both = np.isfinite(a) & np.isfinite(b)
        a, b = np.where(both, a, np.nan), np.where(both, b, np.nan)
        sa, na = self.cluster_sums(a); sb, nb = self.cluster_sums(b)
        d = self.macro(sa.sum(0), na.sum(0), stages) - self.macro(sb.sum(0), nb.sum(0), stages)
        bd = self.boot_macro(a, stages) - self.boot_macro(b, stages)
        row = {'comparison': name, 'delta': d, 'CI95_low': np.percentile(bd, 2.5), 'CI95_high': np.percentile(bd, 97.5),
               'CI90_low': np.percentile(bd, 5), 'CI90_high': np.percentile(bd, 95)}
        row['excludes_zero'] = row['CI95_low'] > 0 or row['CI95_high'] < 0
        row['equivalent'] = (row['CI90_low'] > -EQ_MARGIN) and (row['CI90_high'] < EQ_MARGIN)
        try:
            ds = self.seed_macros(A, stages, mask)[:5] - self.seed_macros(B, stages, mask)[:5]
            row['seeds_same_sign'] = f'{int((np.sign(ds) == np.sign(d)).sum())}/{len(ds)}'
        except ValueError:
            row['seeds_same_sign'] = 'n/a'
        for jj, p in enumerate(STAGES):
            row[f'd_{STAGE_NAMES[p]}'] = np.nansum(sa[:, jj]) / np.nansum(na[:, jj]) - np.nansum(sb[:, jj]) / np.nansum(nb[:, jj])
        return row

    def table_a(self):
        main = [c for c in self.ERR if c[1] == 'shared' and c[3] == 1 and c[2] in ('GRU', 'LSTM', 'HGB', 'MEAN')]
        t = pd.DataFrame([self.describe(c) for c in main]).sort_values(['encoder', 'MAE'])
        t.to_csv(self.tables / 'tableA_accuracy.csv', index=False)
        if any(c[2] == 'DLOG' for c in self.ERR):
            four = [STAGES.index(p) for p in (15, 18, 21, 24)]
            dl = [c for c in self.ERR if c[2] == 'DLOG'] + [c for c in main if c[2] in ('GRU', 'MEAN')]
            pd.DataFrame([self.describe(c, four) for c in dl]).to_csv(self.tables / 'tableA_dlog_four_stages.csv', index=False)
        return t

    def table_b(self):
        t = pd.DataFrame([r for r in (self.paired(*p) for p in PAIRS) if r is not None])
        t.to_csv(self.tables / 'tableB_paired.csv', index=False)
        return t

    def table_c(self):
        rank1 = self.RANKS == 1
        rows = []
        for st in ['S2', 'All']:
            for pol in ['shared', 'closest', 'random1', 'closest_steps']:
                c = G(st, pol)
                if c not in self.ERR:
                    continue
                s, n = self.cluster_sums(self.seed_avg(c)); s1, n1 = self.cluster_sums(self.seed_avg(c, rank1))
                r = {'stream': st, 'policy': pol, 'MAE_all_fields': self.macro(s.sum(0), n.sum(0)),
                     'MAE_rank1_fields': self.macro(s1.sum(0), n1.sum(0))}
                # Station level as in Table 4: median field prediction per station, one model per seed
                per_seed = self.runs[(self.runs['stream'] == st) & (self.runs['policy'] == pol) &
                                     (self.runs['k'] == 1) & (self.runs['encoder'] == 'GRU')]
                r['MAE_station'] = per_seed['mae_station'].mean()
                if pol != 'shared':
                    pr = self.paired(f'{st} {pol} vs shared', c, G(st))
                    r.update({'delta_vs_shared': pr['delta'], 'CI95_low': pr['CI95_low'],
                              'CI95_high': pr['CI95_high'], 'equivalent': pr['equivalent']})
                r['train_records'] = int(per_seed['n_train'].mean()); r['gradient_steps'] = int(per_seed['steps'].mean())
                rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(self.tables / 'tableC_policy.csv', index=False)
        return t

    def table_d(self):
        rows = []
        for st in ['S1', 'S2', 'Met', 'S1+S2', 'S2+Met', 'S1+Met', 'All']:
            c = G(st)
            if c not in self.ERR:
                continue
            r = {'stream': st}
            for y in self.years:
                e = self.ERR[c][y]
                per_seed = [np.nanmean(np.nanmean(e[k], 0)) for k in range(e.shape[0])]
                r[str(y)] = np.mean(per_seed)
            vals = [r[str(y)] for y in self.years]
            r['mean'], r['range'] = np.mean(vals), np.ptp(vals)
            rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(self.tables / 'tableD_per_year.csv', index=False)
        return t

    def table_e(self):
        rows = []
        for st in ['S1+S2', 'S2', 'All']:
            if G(st) not in self.ERR:
                continue
            a = self.seed_avg(G(st))
            for rk in sorted(np.unique(self.RANKS)):
                m = self.RANKS == rk
                rows.append({'stream': st, 'rank': int(rk), 'n': int(m.sum()), 'MAE': np.nanmean(np.nanmean(a[m], 0))})
        t = pd.DataFrame(rows).pivot(index='rank', columns='stream', values='MAE')
        t.to_csv(self.tables / 'tableE_by_rank.csv')
        return t

    def table_f(self):
        B = self.bundles()
        s2i = [str(f) for f in B[self.years[0]]['feature_names']].index('blue')
        avail = {'obs_share': {}, 'clear_share': {}}
        for y in self.years:
            M = B[y]['mask'][:, :, s2i]
            bag = self.REC[y]['bag'].to_numpy()
            tr = np.concatenate([self.TRUTH[t] for t in self.years if t != y])
            obs, clr = np.zeros((len(bag), P)), np.zeros((len(bag), P))
            for j in range(P):
                c = int(np.median(tr[tr[:, j] >= 0, j]))
                w = M[:, max(0, c - CLOUD_HALF):min(M.shape[1], c + CLOUD_HALF + 1)]
                obs[:, j] = w.mean(1)
                union = pd.DataFrame(w).groupby(bag).transform('max').to_numpy()   # dates seen at any field of the station
                clr[:, j] = w.sum(1) / np.maximum(union.sum(1), 1)
            avail['obs_share'][y], avail['clear_share'][y] = obs, clr

        delta = self.seed_avg(G('S2')) - self.seed_avg(G('S1+S2'))
        yr = np.concatenate([np.full(len(self.REC[y]), y) for y in self.years])
        out = {}
        for measure, av in avail.items():
            strata = np.concatenate([np.stack([pd.qcut(pd.Series(av[y][:, j]).rank(method='first'), 3, labels=False)
                                               .to_numpy() for j in range(P)], 1) for y in self.years])   # 0 = most cloudy
            rows = []
            for j, p in enumerate(STAGES):
                d = delta[:, j]
                r = {'stage': STAGE_NAMES[p]}
                for k, lab in enumerate(['most_cloudy', 'medium', 'least_cloudy']):
                    r[lab] = np.nanmean(d[strata[:, j] == k])

                def stratum(k):
                    m = (strata[:, j] == k) & np.isfinite(d)
                    s = np.zeros(self.NC); n = np.zeros(self.NC)
                    np.add.at(s, self.CID[m], d[m]); np.add.at(n, self.CID[m], 1)
                    return s, n
                s0, n0 = stratum(0); s2, n2 = stratum(2)
                bd = (self.W @ s0) / (self.W @ n0) - (self.W @ s2) / (self.W @ n2)
                r['cloudy_minus_clear'] = r['most_cloudy'] - r['least_cloudy']
                r['CI95_low'], r['CI95_high'] = np.nanpercentile(bd, 2.5), np.nanpercentile(bd, 97.5)
                yrs = [np.nanmean(d[(yr == y) & (strata[:, j] == 0)]) - np.nanmean(d[(yr == y) & (strata[:, j] == 2)])
                       for y in self.years]
                r['years_positive'] = f'{int((np.array(yrs) > 0).sum())}/{len(self.years)}'
                rows.append(r)
            out[measure] = pd.DataFrame(rows)
            out[measure].to_csv(self.tables / f'tableF_cloud_{measure}.csv', index=False)
        return out


def main():
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument('--skip-hgb', action='store_true', help='do not train the gradient boosting baseline')
    ap.add_argument('--skip-dlog', action='store_true', help='do not run the double logistic baseline')
    args = ap.parse_args()
    ev = Evaluation(load_config(args.config))
    if not args.skip_hgb:
        ev.add_hgb()
    if not args.skip_dlog:
        ev.add_double_logistic()

    a = ev.table_a()
    b = ev.table_b()
    ev.table_c(); ev.table_d(); ev.table_e(); ev.table_f()
    pd.set_option('display.width', 200)
    print('\nAccuracy (field level):')
    print(a[['stream', 'encoder', 'MAE', 'CI_low', 'CI_high', 'seed_sd', 'RMSE', 'within_5d', 'bias']].round(2).to_string(index=False))
    print('\nPaired differences (A minus B):')
    print(b[['comparison', 'delta', 'CI95_low', 'CI95_high', 'seeds_same_sign', 'equivalent']].round(2).to_string(index=False))
    print('\nTables written to', ev.tables)


if __name__ == '__main__':
    main()
