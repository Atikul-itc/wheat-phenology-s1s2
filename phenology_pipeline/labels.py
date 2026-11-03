"""
Construct soft Gaussian labels for per-day per-phase classification.

Inputs:
  - reference dates (one per phase) for a (station, harvest_year)
  - window start, n_days
Outputs:
  - labels: (n_days, n_phases) float array, in [0, 1]
  - label_mask: (n_phases,) bool — True if that phase has a valid reference date
"""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd

from . import config as C


def gaussian_label_curve(n_days: int, center_idx: int, sigma: float,
                          truncation_sigma: float = 4.0) -> np.ndarray:
    """1D Gaussian centered at center_idx, support [0, n_days)."""
    out = np.zeros(n_days, dtype=np.float32)
    if center_idx < 0 or center_idx >= n_days:
        return out
    lo = max(0, int(center_idx - truncation_sigma * sigma))
    hi = min(n_days, int(center_idx + truncation_sigma * sigma) + 1)
    x = np.arange(lo, hi)
    out[lo:hi] = np.exp(-0.5 * ((x - center_idx) / sigma) ** 2).astype(np.float32)
    return out


def build_labels_for_record(phase_dates: dict,
                              window_start: date,
                              n_days: int,
                              sigma: float = None,
                              phases=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
      phase_dates: {phase_id: date or None}
      window_start: first day of the window
      n_days: window length
      sigma: Gaussian width in days (uniform across phases)
      phases: ordered list of phase ids; defaults to C.ALL_PHASES
    Returns:
      labels: (n_days, n_phases) float
      label_mask: (n_phases,) bool — True if that phase had a non-null reference
    """
    sigma = sigma if sigma is not None else C.GAUSSIAN_SIGMA_DAYS
    phases = phases if phases is not None else C.ALL_PHASES
    P = len(phases)
    labels = np.zeros((n_days, P), dtype=np.float32)
    label_mask = np.zeros(P, dtype=bool)
    for j, phase in enumerate(phases):
        d = phase_dates.get(phase)
        if d is None or (hasattr(d, "value") and pd.isna(d)) or (isinstance(d, float) and np.isnan(d)):
            continue
        # NaT check via pandas
        if pd.isna(d):
            continue
        if hasattr(d, "to_pydatetime"):
            d = d.to_pydatetime().date()
        elif hasattr(d, "date") and callable(d.date):
            d = d.date()
        idx = (d - window_start).days
        if 0 <= idx < n_days:
            labels[:, j] = gaussian_label_curve(n_days, idx, sigma,
                                                  C.GAUSSIAN_TRUNCATION_SIGMA)
            label_mask[j] = True
        # else: phase date falls outside window — treat as missing
    return labels, label_mask
