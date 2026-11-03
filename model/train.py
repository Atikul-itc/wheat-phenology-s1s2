"""Train the experiment grid with leave-one-year-out cross-validation (Sections 3.4, 3.5).

Experiments (five seeds x five test years each):
  E1_streams   seven combinations of Sentinel-1, Sentinel-2 and meteorology
  E2_policy    station date assigned to all fields, the nearest field, a random field,
               or the nearest field with matched gradient steps (S2 and all sources)
  E3_indices   Sentinel-2 bands with and without the four vegetation indices
  E4_temporal  5-day and 10-day composites instead of daily input
  E5_lstm      LSTM instead of GRU encoders

Every run appends one row to <runs>/runs.csv and stores its test predictions in
<runs>/preds/<run_id>.npz. Finished runs are skipped, so the script can be restarted.

Usage:
    python -m model.train --config config.yaml                    # full grid
    python -m model.train --config config.yaml --exp E1_streams   # one experiment
    python -m model.train --config config.yaml --smoke            # 1 seed, 2020, 2 epochs
"""
from __future__ import annotations

import gc
import math
import random
import time

import numpy as np
import pandas as pd
import torch

from common import base_parser, load_config
from model.network import PhenoNet, soft_label_loss, time_features

HIDDEN, BATCH_SIZE, EPOCHS = 32, 32, 15
LR, WEIGHT_DECAY, DROPOUT = 1e-3, 1e-5, 0.1
SIGMA_DAYS = 5.0
SEEDS = [0, 1, 2, 3, 4]
EXPERIMENTS = ['E1_streams', 'E2_policy', 'E3_indices', 'E4_temporal', 'E5_lstm']

S2_INDICES = ['EVI', 'MTCI', 'NDRE', 'CIrededge']
S2_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 're1', 're2', 're3']
GROUP_FEATS = {'S1': ['VV', 'VH', 'CR'], 'S2': S2_BANDS + S2_INDICES, 'S2bands': S2_BANDS,
               'Met': ['mean_temp_celsius', 'precipitation_mm', 'cumulative_gdd']}
STREAMS = {'S1': ['S1'], 'S2': ['S2'], 'Met': ['Met'], 'S1+S2': ['S1', 'S2'], 'S2+Met': ['S2', 'Met'],
           'S1+Met': ['S1', 'Met'], 'All': ['S1', 'S2', 'Met'],
           'S2bands': ['S2bands'], 'S1+S2bands': ['S1', 'S2bands']}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_mae(pred, truth):
    ae = np.where((truth >= 0) & (pred >= 0), np.abs(pred - truth), np.nan)
    return float(np.nanmean(np.nanmean(ae, axis=0)))


class Grid:
    def __init__(self, cfg, epochs: int = EPOCHS):
        self.cfg, self.years, self.epochs = cfg, cfg.years, epochs
        self.out = cfg.path('runs', mkdir=True)
        (self.out / 'preds').mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp = self.device.type == 'cuda'
        self._stats = {}
        self._load()

    # ------------------------------------------------------------------ data
    def _load(self):
        parts = []
        for y in self.years:
            z = np.load(self.cfg.path('tensors') / f'phenology_{y}.npz', allow_pickle=False)
            parts.append({k: z[k] for k in ['X', 'mask', 'Y', 'label_mask', 'bag_id', 'rank', 'harvest_year', 'xid']})
            self.features = [str(s) for s in z['feature_names']]
            self.stages = [int(p) for p in z['phase_ids']]
            self.T = int(z['T'])
        D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.X = torch.from_numpy(np.nan_to_num(D['X'].astype(np.float32)))
        self.M = torch.from_numpy(D['mask'].astype(bool))
        self.LM = torch.from_numpy(D['label_mask'].astype(bool))
        self.truth = D['Y'].argmax(axis=1).astype(np.int32)
        self.truth[~D['label_mask']] = -1
        self.Y = torch.from_numpy(D['Y'].astype(np.float32))
        self.meta = pd.DataFrame({'xid': D['xid'], 'bag': D['bag_id'], 'rank': D['rank'], 'year': D['harvest_year']})
        self.fidx = {f: i for i, f in enumerate(self.features)}
        print(f'N={len(self.meta):,}  T={self.T}  features={len(self.features)}  stages={len(self.stages)}  '
              f'device={self.device}')

    def train_index(self, train_years, policy, seed):
        sel = self.meta['year'].isin(train_years).to_numpy()
        if policy in ('closest', 'closest_steps'):
            sel = sel & (self.meta['rank'] == 1).to_numpy()
        idx = np.where(sel)[0]
        if policy == 'random1':
            rng = np.random.default_rng(seed)
            sub = self.meta.iloc[idx].assign(i=idx)
            idx = sub.groupby(['year', 'bag'])['i'].apply(lambda s: rng.choice(s.to_numpy())).to_numpy()
        return np.sort(idx)

    def standardiser(self, test_year, feat_ix):
        """Mean and SD of the observed values in the training years."""
        if test_year not in self._stats:
            idx = np.where(self.meta['year'] != test_year)[0]
            F = self.X.shape[2]; s = np.zeros(F); s2 = np.zeros(F); n = np.zeros(F)
            for b in range(0, len(idx), 2000):
                x = self.X[idx[b:b + 2000]].double(); m = self.M[idx[b:b + 2000]].double()
                s += (x * m).sum((0, 1)).numpy(); s2 += (x * x * m).sum((0, 1)).numpy(); n += m.sum((0, 1)).numpy()
            mu = s / np.maximum(n, 1); sd = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mu ** 2, 0))
            sd[sd < 1e-6] = 1.0
            self._stats[test_year] = (mu, sd)
        mu, sd = self._stats[test_year]
        return torch.tensor(mu[feat_ix], dtype=torch.float32), torch.tensor(sd[feat_ix], dtype=torch.float32)

    @staticmethod
    def composite(x, m, k):
        """Mean of observed values in k-day bins: (B, T, F) -> (B, ceil(T/k), F)."""
        B, Tn, F = x.shape
        nb = math.ceil(Tn / k); pad = nb * k - Tn
        if pad:
            x = torch.cat([x, torch.zeros(B, pad, F, device=x.device)], 1)
            m = torch.cat([m, torch.zeros(B, pad, F, device=m.device)], 1)
        xs = (x * m).view(B, nb, k, F).sum(2); ms = m.view(B, nb, k, F).sum(2)
        return torch.where(ms > 0, xs / ms.clamp(min=1), torch.zeros_like(xs)), (ms > 0).float()

    def bin_labels(self, truth, k):
        """Gaussian soft labels over bin centres: (B, P) -> (B, nb, P)."""
        nb = math.ceil(self.T / k)
        centres = torch.arange(nb, dtype=torch.float32) * k + (k - 1) / 2
        return torch.exp(-0.5 * ((centres.view(1, -1, 1) - truth.float().unsqueeze(1)) / SIGMA_DAYS) ** 2)

    def batch(self, idx, feat_ix, mu, sd, k):
        dev = self.device
        x = self.X[idx][:, :, feat_ix].to(dev, non_blocking=True)
        m = self.M[idx][:, :, feat_ix].to(dev, non_blocking=True).float()
        x = torch.where(m > 0, (x - mu.to(dev)) / sd.to(dev), torch.zeros_like(x))
        if k > 1:
            x, m = self.composite(x, m, k)
            y = self.bin_labels(torch.from_numpy(self.truth[idx]).clamp(min=0), k).to(dev)
        else:
            y = self.Y[idx].to(dev, non_blocking=True)
        return x, m, y, self.LM[idx].to(dev)

    def station_level(self, pred, truth, meta):
        """Median field prediction per station and year against the station date."""
        p = pd.DataFrame(pred, columns=self.stages).assign(year=meta['year'].values, bag=meta['bag'].values)
        t = pd.DataFrame(np.where(truth >= 0, truth, np.nan), columns=self.stages).assign(
            year=meta['year'].values, bag=meta['bag'].values)
        p = p.groupby(['year', 'bag']).median(); t = t.groupby(['year', 'bag']).median()
        return macro_mae(p.to_numpy(), np.nan_to_num(t.to_numpy(), nan=-1))

    # ------------------------------------------------------------------ one run
    def run(self, spec):
        set_seed(spec['seed'])
        k = spec['k']
        test_idx = np.where(self.meta['year'] == spec['test_year'])[0]
        train_years = [y for y in self.years if y != spec['test_year']]
        tr_idx = self.train_index(train_years, spec['policy'], spec['seed'])

        groups_abs = {g: [self.fidx[f] for f in GROUP_FEATS[g]] for g in STREAMS[spec['stream']]}
        feat_ix = sorted({i for ix in groups_abs.values() for i in ix})
        pos = {f: j for j, f in enumerate(feat_ix)}
        groups = {g: [pos[i] for i in ix] for g, ix in groups_abs.items()}
        mu, sd = self.standardiser(spec['test_year'], feat_ix)

        nb_t = math.ceil(self.T / k)
        tpos = time_features(np.arange(nb_t) * k + (k - 1) / 2, self.T).to(self.device)
        model = PhenoNet(groups, HIDDEN, len(self.stages), spec['encoder'], True, DROPOUT).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler('cuda', enabled=self.amp)

        if spec['policy'] == 'closest_steps':       # as many updates as the shared-label model
            n_shared = len(self.train_index(train_years, 'shared', spec['seed']))
            total_steps = self.epochs * math.ceil(n_shared / BATCH_SIZE)
        else:
            total_steps = self.epochs * math.ceil(len(tr_idx) / BATCH_SIZE)

        gen = torch.Generator().manual_seed(spec['seed'])
        step = 0
        model.train()
        while step < total_steps:
            perm = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
            for b in range(0, len(perm), BATCH_SIZE):
                if step >= total_steps:
                    break
                x, m, y, lm = self.batch(perm[b:b + BATCH_SIZE], feat_ix, mu, sd, k)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=self.amp):
                    loss = soft_label_loss(model(x, m, tpos), y, lm)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                step += 1

        model.eval(); preds = []
        with torch.no_grad():
            for b in range(0, len(test_idx), 256):
                x, m, _, _ = self.batch(test_idx[b:b + 256], feat_ix, mu, sd, k)
                with torch.amp.autocast('cuda', enabled=self.amp):
                    preds.append(model(x, m, tpos).float().argmax(dim=1).cpu().numpy())
        pred = np.rint(np.concatenate(preds) * k + (k - 1) / 2).astype(np.int32).clip(0, self.T - 1)
        truth = self.truth[test_idx]
        meta = self.meta.iloc[test_idx].reset_index(drop=True)
        r1 = (meta['rank'] == 1).to_numpy()
        out = {**spec, 'n_train': int(len(tr_idx)), 'steps': int(total_steps),
               'mae_all': macro_mae(pred, truth), 'mae_rank1': macro_mae(pred[r1], truth[r1]),
               'mae_station': self.station_level(pred, truth, meta)}
        for j, p in enumerate(self.stages):
            ok = truth[:, j] >= 0
            out[f'mae_p{p}'] = float(np.abs(pred[ok, j] - truth[ok, j]).mean())
        np.savez_compressed(self.out / 'preds' / f"{spec['run_id']}.npz", pred=pred, truth=truth,
                            xid=meta['xid'].values, bag=meta['bag'].values,
                            rank=meta['rank'].values, year=meta['year'].values)
        del model, opt; gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return out


def spec(exp, stream, policy='shared', encoder='GRU', k=1, seed=0, test_year=2020):
    rid = f'{encoder}_{stream}_{policy}_k{k}_s{seed}_{test_year}'
    return {'run_id': rid, 'exp': exp, 'stream': stream, 'policy': policy,
            'encoder': encoder, 'k': k, 'seed': seed, 'test_year': test_year}


def registry(experiments, seeds, test_years):
    reg = {}
    add = lambda **kw: reg.setdefault(spec(**kw)['run_id'], spec(**kw))
    for seed in seeds:
        for ty in test_years:
            c = dict(seed=seed, test_year=ty)
            if 'E1_streams' in experiments:
                for st in ['S1', 'S2', 'Met', 'S1+S2', 'S2+Met', 'S1+Met', 'All']:
                    add(exp='E1_streams', stream=st, **c)
            if 'E2_policy' in experiments:
                for st in ['S2', 'All']:
                    for pol in ['shared', 'closest', 'random1', 'closest_steps']:
                        add(exp='E2_policy', stream=st, policy=pol, **c)
            if 'E3_indices' in experiments:
                for st in ['S2bands', 'S1+S2bands', 'S2', 'S1+S2']:
                    add(exp='E3_indices', stream=st, **c)
            if 'E4_temporal' in experiments:
                for k in [5, 10]:
                    for st in ['S1+S2', 'All']:
                        add(exp='E4_temporal', stream=st, k=k, **c)
            if 'E5_lstm' in experiments:
                for st in ['S1+S2', 'All']:
                    add(exp='E5_lstm', stream=st, encoder='LSTM', **c)
    return reg


def main():
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument('--exp', nargs='+', default=EXPERIMENTS, choices=EXPERIMENTS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--smoke', action='store_true', help='quick test: seed 0, test year 2020, 2 epochs')
    ap.add_argument('--epochs', type=int, default=EPOCHS, help=f'training epochs (default {EPOCHS})')
    args = ap.parse_args()
    cfg = load_config(args.config)

    seeds, test_years, epochs = args.seeds, cfg.years, args.epochs
    if args.smoke:
        seeds, test_years, epochs = [0], [2020], 2
    grid = Grid(cfg, epochs)
    reg = registry(args.exp, seeds, test_years)

    runs_csv = grid.out / 'runs.csv'
    done = set(pd.read_csv(runs_csv)['run_id']) if runs_csv.exists() else set()
    todo = [s for rid, s in reg.items() if rid not in done]
    print(f'{len(reg)} runs registered, {len(reg) - len(todo)} done, {len(todo)} to run')

    for i, s in enumerate(todo, 1):
        t0 = time.time()
        try:
            row = grid.run(s)
        except torch.cuda.OutOfMemoryError:
            print('out of memory in', s['run_id']); torch.cuda.empty_cache(); continue
        row['secs'] = round(time.time() - t0, 1)
        pd.DataFrame([row]).to_csv(runs_csv, mode='a', header=not runs_csv.exists(), index=False)
        print(f"[{i}/{len(todo)}] {s['run_id']}: MAE {row['mae_all']:.2f} d ({row['secs']} s)", flush=True)

    runs = pd.read_csv(runs_csv)
    print(runs.groupby(['exp', 'stream', 'policy', 'encoder', 'k'])[['mae_all', 'mae_rank1', 'mae_station']]
              .agg(['mean', 'std']).round(2).to_string())


if __name__ == '__main__':
    main()
