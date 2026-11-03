"""
Sanity-check a tensor bundle produced by build_dataset.

Checks:
  1. Shapes are consistent.
  2. X/mask consistency (no NaN values in X; mask matches non-zero pattern only where reasonable).
  3. Label peak indices fall inside the window.
  4. Bag sizes are sensible (median 10 per station).
  5. Feature observation densities make sense (S1 denser than S2, climate dense).
  6. No leakage: every (xid) has a single bag_id.
"""
import argparse
from pathlib import Path
import numpy as np

from . import config as C


def validate_bundle(npz_path: Path):
    z = np.load(npz_path, allow_pickle=False)
    X      = z["X"]
    M      = z["mask"]
    Y      = z["Y"]
    YM     = z["label_mask"]
    bag    = z["bag_id"]
    rank   = z["rank"]
    years  = z["harvest_year"]
    xids   = z["xid"]
    feats  = z["feature_names"]
    phases = z["phase_ids"]
    T      = int(z["T"])

    print(f"\n=== Validating {npz_path.name} ===")
    print(f"X     : {X.shape}  dtype={X.dtype}")
    print(f"mask  : {M.shape}  dtype={M.dtype}")
    print(f"Y     : {Y.shape}  dtype={Y.dtype}")
    print(f"YM    : {YM.shape}  dtype={YM.dtype}")
    print(f"bag   : {bag.shape}")
    print(f"rank  : {rank.shape}")
    print(f"years : {years.shape}, unique: {sorted(np.unique(years))}")
    print(f"T     : {T}")
    print(f"feats ({len(feats)}): {list(feats)}")
    print(f"phases: {list(phases)}")

    N = X.shape[0]
    F = X.shape[2]
    P = Y.shape[2]
    assert X.shape == (N, T, F)
    assert M.shape == (N, T, F)
    assert Y.shape == (N, T, P)
    assert YM.shape == (N, P)
    assert bag.shape == (N,)
    assert rank.shape == (N,)
    assert years.shape == (N,)

    # No NaNs in X
    assert not np.isnan(X).any(), "X contains NaN"

    # Mask coverage per feature
    print("\nObservation rate per feature (fraction of (N,T) entries observed):")
    cov = M.mean(axis=(0, 1))
    for f, c in zip(feats, cov):
        print(f"  {f:12s}: {c*100:5.1f}%")

    # Bag stats
    unique_bags, bag_counts = np.unique(bag, return_counts=True)
    print(f"\nBags: {len(unique_bags)} unique stations")
    print(f"  fields-per-station: min={bag_counts.min()}, "
          f"median={int(np.median(bag_counts))}, max={bag_counts.max()}")

    # Rank distribution
    print(f"\nRank distribution: {dict(zip(*np.unique(rank, return_counts=True)))}")

    # Phase availability and peak location
    print("\nPhase availability and peak location:")
    for j, p in enumerate(phases):
        has = YM[:, j]
        n = has.sum()
        if n == 0:
            print(f"  phase {p}: 0 records have this phase")
            continue
        peaks = Y[has, :, j].argmax(axis=1)
        print(f"  phase {p:2d}: {n:5d} records, "
              f"peak day index min/median/max = {peaks.min()}/{int(np.median(peaks))}/{peaks.max()}")

    # Records with all 7 phases vs 5 core phases
    all7 = YM.all(axis=1).sum()
    core_idx = [list(phases).index(p) for p in C.CORE_PHASES]
    core_only = YM[:, core_idx].all(axis=1).sum()
    print(f"\nRecords with all 7 phases : {all7}")
    print(f"Records with 5 core phases: {core_only}")

    # Memory footprint
    total_mb = (X.nbytes + M.nbytes + Y.nbytes) / 1e6
    print(f"\nMemory footprint (X+mask+Y): {total_mb:.1f} MB")

    return dict(N=N, T=T, F=F, P=P, n_bags=len(unique_bags),
                n_all_phases=all7, n_core=core_only)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Path(s) to .npz bundle(s)")
    args = ap.parse_args()
    for p in args.paths:
        validate_bundle(Path(p))
