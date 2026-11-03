"""Loading of the per-year tensor bundles written by preprocessing.build_tensors."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_year_bundle(npz_path: Path) -> dict:
    """Load a per-year .npz bundle produced by phenology_pipeline.build_dataset."""
    z = np.load(npz_path, allow_pickle=False)
    return {
        "X":            z["X"],                # (N, T, F) float32
        "mask":         z["mask"],             # (N, T, F) bool
        "Y":            z["Y"],                # (N, T, P) float32 — soft Gaussian labels
        "label_mask":   z["label_mask"],       # (N, P)    bool
        "bag_id":       z["bag_id"],           # (N,) int64 (Stations_i)
        "rank":         z["rank"],             # (N,) int32 (1..10)
        "harvest_year": z["harvest_year"],     # (N,) int32
        "xid":          z["xid"],              # (N,) int64
        "feature_names": [str(s) for s in z["feature_names"]],
        "phase_ids":    z["phase_ids"].astype(int).tolist(),
        "T":            int(z["T"]),
    }
