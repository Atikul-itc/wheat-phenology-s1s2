"""Multi-source bidirectional recurrent network (Section 3.4).

Each input source has its own single-layer bidirectional GRU (or LSTM) encoder
that receives the standardised values times their observation mask together
with the mask. Encoder outputs are concatenated with three day-of-season
features and mapped by a two-layer head to one logit per stage and day.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_in: int, hidden: int, kind: str = 'GRU'):
        super().__init__()
        rnn = nn.GRU if kind == 'GRU' else nn.LSTM
        self.rnn = rnn(input_size=2 * n_in, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward(self, x, m):
        h, _ = self.rnn(torch.cat([x * m, m], dim=-1))
        return h


class PhenoNet(nn.Module):
    """groups: {source name: [feature indices]}; only these encoders are built."""

    def __init__(self, groups: dict, hidden: int, n_stages: int, kind: str = 'GRU',
                 time_encoding: bool = True, dropout: float = 0.1):
        super().__init__()
        self.groups = groups
        self.enc = nn.ModuleDict({g: Encoder(len(ix), hidden, kind) for g, ix in groups.items()})
        self.time_encoding = time_encoding
        d = sum(e.out_dim for e in self.enc.values()) + (3 if time_encoding else 0)
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_stages))

    def forward(self, x, m, tpos):
        hs = [self.enc[g](x[:, :, ix], m[:, :, ix]) for g, ix in self.groups.items()]
        if self.time_encoding:
            hs.append(tpos.expand(x.shape[0], -1, -1))
        return self.head(torch.cat(hs, dim=-1))


def soft_label_loss(logits, target, label_mask):
    """Binary cross-entropy against soft Gaussian labels, averaged over stages with a reference date."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    w = label_mask.unsqueeze(1).float()
    return (bce * w).sum() / (w.sum() * logits.shape[1] + 1e-8)


def time_features(centres, T: int):
    """Day-of-season features (t/T, sin 2 pi t/T, cos 2 pi t/T), shape (1, steps, 3)."""
    t = torch.as_tensor(centres, dtype=torch.float32) / (T - 1)
    return torch.stack([t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=-1).unsqueeze(0)
