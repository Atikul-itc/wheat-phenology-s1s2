"""Results figures from the tables written by analysis.evaluate.

  fig05_stage_mae          MAE per stage: mean-date baseline vs S1, S2 and S1+S2 (Figure 5)
  fig06_paired_differences paired differences in macro MAE with 95% intervals (Figure 6)
  fig07_stage_heatmap      MAE per stage for all seven input configurations (Figure 7)
  fig08_cloud_conditional  added value of Sentinel-1 by Sentinel-2 availability (Figure 8)

Figures 1 to 4 (study area, reference dates, example series, pipeline) are not
produced here.

Usage:
    python -m figures.make_figures --config config.yaml [--only fig05 fig07]
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import STAGE_NAMES, STAGES, base_parser, load_config
from figures.style import STREAM_COLORS, apply_style, save

NAMES = [STAGE_NAMES[p] for p in STAGES]
COL = {'S1': STREAM_COLORS['S1_only'], 'S2': STREAM_COLORS['S2_only'], 'S1+S2': STREAM_COLORS['S1+S2'],
       'Met': STREAM_COLORS['Met_only']}
LABEL = {'MeanDate': 'Mean date', 'S1': 'Sentinel-1', 'S2': 'Sentinel-2', 'S1+S2': 'Sentinel-1 + Sentinel-2',
         'Met': 'Meteorology'}
W = 7.5


def stage_mae(tables, stream, encoder):
    t = pd.read_csv(tables / 'tableA_accuracy.csv')
    t = t[(t['policy'] == 'shared') & (t['k'] == 1) & (t['stream'] == stream) & (t['encoder'] == encoder)]
    return t[[f'MAE_{n}' for n in NAMES]].to_numpy().ravel()


def fig05(tables, out, include_met=False):
    models = ['S1', 'S2', 'S1+S2'] + (['Met'] if include_met else [])
    base = stage_mae(tables, 'MeanDate', 'MEAN')
    fig, ax = plt.subplots(figsize=(W, 3.6))
    y = np.arange(len(STAGES))[::-1]
    offs = dict(zip(models, np.linspace(0.27, -0.27, len(models)) if len(models) > 3 else [0.18, 0.0, -0.18]))
    marker = {'S1': 'o', 'S2': 's', 'S1+S2': 'D', 'Met': '^'}
    for yy in y:
        ax.axhline(yy, color='#e6e6e6', lw=0.6, zorder=0)
    ax.scatter(base, y, marker='|', s=420, lw=2.2, color='#8a8a8a', zorder=2, label=LABEL['MeanDate'])
    for m in models:
        v = stage_mae(tables, m, 'GRU')
        for yi, c, e in zip(y + offs[m], base, v):
            ax.plot([c, e], [yi, yi], color=COL[m], lw=0.7, alpha=0.5, zorder=1)
        ax.scatter(v, y + offs[m], marker=marker[m], s=26, color=COL[m], edgecolor='white', lw=0.5, zorder=3,
                   label=LABEL[m])
    ax.set_yticks(y); ax.set_yticklabels(NAMES)
    ax.set_xlim(5, 11); ax.set_xticks(range(5, 12)); ax.set_xlabel('Mean absolute error (days)')
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(models) + 1, frameon=False,
              handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(); save(fig, out, 'fig05_stage_mae'); plt.close(fig)


GROUPS = [
    ('Sensors', [('S1+S2 vs MeanDate', 'S1 + S2 vs mean date'), ('S1 vs S2', 'S1 vs S2'),
                 ('S1+S2 vs S2', 'S1 + S2 vs S2'), ('S1+S2 vs S1', 'S1 + S2 vs S1')]),
    ('Meteorology', [('Met vs MeanDate', 'Met vs mean date'), ('S2+Met vs S2', 'S2 + Met vs S2'),
                     ('S1+Met vs S1', 'S1 + Met vs S1'), ('All vs S1+S2', 'All vs S1 + S2')]),
    ('Input design', [('S2 bands vs S2', 'S2 without indices vs S2'),
                      ('S1+S2 bands vs S1+S2', 'S1 + S2 without indices vs S1 + S2'),
                      ('S1+S2 5-day vs daily', 'S1 + S2, 5-day vs daily'),
                      ('S1+S2 10-day vs daily', 'S1 + S2, 10-day vs daily'),
                      ('All 5-day vs daily', 'All, 5-day vs daily')]),
    ('Model class', [('LSTM vs GRU (S1+S2)', 'LSTM vs GRU (S1 + S2)'), ('LSTM vs GRU (All)', 'LSTM vs GRU (All)'),
                     ('GRU vs HGB (S1+S2)', 'GRU vs HGB (S1 + S2)'), ('GRU vs HGB (All)', 'GRU vs HGB (All)')]),
]


def fig06(tables, out):
    t = pd.read_csv(tables / 'tableB_paired.csv').set_index('comparison')
    rows, heads, ticks, labels, y = [], [], [], [], 0
    for g, items in GROUPS:
        present = [(k, lab) for k, lab in items if k in t.index]
        if not present:
            continue
        heads.append((y, g)); y += 1
        for k, lab in present:
            rows.append((y, t.loc[k])); ticks.append(y); labels.append(lab); y += 1
        y += 0.4
    fig, ax = plt.subplots(figsize=(W, 0.24 * y + 0.8))
    ax.axvspan(-1, 1, color='#eeeeee', zorder=0, lw=0)
    ax.axvline(0, color='#222222', lw=0.8, zorder=1)
    for yy, r in rows:
        ax.errorbar(r['delta'], yy, xerr=[[r['delta'] - r['CI95_low']], [r['CI95_high'] - r['delta']]],
                    fmt='o', ms=4, color='#222222', elinewidth=1, capsize=2, zorder=3)
    ax.set_yticks(ticks + [h for h, _ in heads]); ax.set_yticklabels(labels + [g for _, g in heads])
    for lab in ax.get_yticklabels()[len(ticks):]:
        lab.set_fontweight('bold')
    ax.set_ylim(y - 0.4, -0.8); ax.set_xlim(-1.1, 1.1); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlabel('Difference in macro MAE (days), A minus B\n← A more accurate      B more accurate →')
    ax.text(1.0, -0.8, '±1 d equivalence margin', ha='right', va='bottom', fontsize=7, color='#666666')
    fig.tight_layout(); save(fig, out, 'fig06_paired_differences'); plt.close(fig)


def fig07(tables, out):
    t = pd.read_csv(tables / 'tableA_accuracy.csv')
    t = t[(t['policy'] == 'shared') & (t['k'] == 1) & t['encoder'].isin(['GRU', 'MEAN'])].set_index('stream')
    order = ['MeanDate', 'S1', 'S2', 'Met', 'S1+S2', 'S2+Met', 'S1+Met', 'All']
    hm = t.reindex(order)[[f'MAE_{n}' for n in NAMES]]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    im = ax.imshow(hm.to_numpy(), cmap='Reds', aspect='auto')
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=30, ha='right')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(['Mean date'] + order[1:])
    ax.axhline(0.5, color='white', lw=2)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f'{hm.iloc[i, j]:.1f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='MAE (days)')
    fig.tight_layout(); save(fig, out, 'fig07_stage_heatmap'); plt.close(fig)


def fig08(tables, out):
    t = pd.read_csv(tables / 'tableF_cloud_clear_share.csv')
    fig, ax = plt.subplots(figsize=(W, 2.8))
    x = np.arange(len(t)); w = 0.27
    for i, (lab, col, c) in enumerate([('most cloudy', 'most_cloudy', '#4477AA'), ('medium', 'medium', '#88AACC'),
                                       ('least cloudy', 'least_cloudy', '#CCDDEE')]):
        ax.bar(x + (i - 1) * w, t[col], w, label=lab, color=c)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(t['stage'], rotation=30, ha='right')
    ax.set_ylabel('MAE(S2) − MAE(S1+S2) (days)'); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, out, 'fig08_cloud_conditional'); plt.close(fig)


def main():
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument('--only', nargs='+', choices=['fig05', 'fig06', 'fig07', 'fig08'])
    ap.add_argument('--fig05-met', action='store_true', help='add meteorology alone to Figure 5')
    args = ap.parse_args()
    cfg = load_config(args.config)
    apply_style()
    tables, out = cfg.path('tables'), cfg.path('figures', mkdir=True)
    todo = args.only or ['fig05', 'fig06', 'fig07', 'fig08']
    if 'fig05' in todo: fig05(tables, out, args.fig05_met)
    if 'fig06' in todo: fig06(tables, out)
    if 'fig07' in todo: fig07(tables, out)
    if 'fig08' in todo: fig08(tables, out)
    print('Figures written to', out)


if __name__ == '__main__':
    main()
