"""
Top-level dataset builder.

Per harvest_year:
  1. Load phenology (station-level phase dates from master file).
  2. Load XID↔Station mapping.
  3. Join: each XID inherits the phases of its station.
  4. Load S1/S2 wide table (XID × Date × features).
  5. Load climate (Stations_i × Date × climate features) and broadcast to XIDs.
  6. Build per-(XID, harvest_year) tensors:
        X      : (T, F)          float, NaN replaced with 0
        mask   : (T, F)          bool,  True where observed
        labels : (T, n_phases)   float Gaussian curves
        label_mask : (n_phases,) bool, which phases had reference dates
        bag_id : Stations_i      int
        rank   : within-station int
  7. Save to a single .npz per year.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C
from .utils import log, timer, harvest_year_window
from .data.phenology import load_phenology_all_years
from .data.mapping import load_xid_mapping
from .data.timeseries import build_rs_wide_for_year
from .data.climate import load_climate_for_year
from .labels import build_labels_for_record


# Feature order is fixed and recorded so the model knows which column is which
def feature_order(s2_features=None, s1_features=None, clim_features=None) -> list[str]:
    s2_features = s2_features or C.S2_FEATURES
    s1_features = s1_features or C.S1_FEATURES
    clim_features = clim_features or C.CLIM_FEATURES
    return list(s2_features) + list(s1_features) + list(clim_features)


def build_year(harvest_year: int,
               pheno: pd.DataFrame,
               s2_features=None,
               s1_features=None,
               clim_features=None,
               require_all_features: bool = False) -> dict:
    """Build the tensor bundle for one harvest year."""
    s2_features = s2_features or C.S2_FEATURES
    s1_features = s1_features or C.S1_FEATURES
    clim_features = clim_features or C.CLIM_FEATURES
    feats = feature_order(s2_features, s1_features, clim_features)
    F = len(feats)

    win_start, _, T_natural = harvest_year_window(harvest_year,
                                            C.WINDOW_START_MONTH_DAY,
                                            C.WINDOW_END_MONTH_DAY)
    # Trim to uniform sequence length across all years (leap years are 1 day longer).
    T = min(T_natural, C.TARGET_T)
    if T != T_natural:
        log.info(f"Trimming {harvest_year} from T={T_natural} to T={T} for uniform length")

    with timer(f"  load mapping {harvest_year}"):
        mapping = load_xid_mapping(harvest_year)  # XID, Stations_i, Rank, harvest_year

    # Restrict mapping to stations that have at least one phase date this year
    pheno_yr = pheno[pheno["harvest_year"] == harvest_year]
    valid_stations = pheno_yr["Stations_i"].unique()
    mapping = mapping[mapping["Stations_i"].isin(valid_stations)].copy()
    log.info(f"After station filter: {len(mapping)} XIDs across "
             f"{mapping['Stations_i'].nunique()} stations")

    with timer(f"  load RS wide {harvest_year}"):
        rs = build_rs_wide_for_year(harvest_year, s2_features, s1_features,
                                      require_all_features=require_all_features)
    # Keep only XIDs that are in our mapping
    rs = rs[rs["XID"].isin(mapping["XID"])].copy()
    log.info(f"After XID filter: {rs['XID'].nunique()} XIDs in RS table")

    with timer(f"  load climate {harvest_year}"):
        clim = load_climate_for_year(harvest_year)

    # Now build tensors per XID
    # Strategy:
    #  - Build a fast lookup: for each (XID, Date) we have S1+S2 feature values
    #  - For climate, we look up by (Stations_i, Date)
    #  - Then for each XID we slice T days of features
    
    # Sort once for stable per-XID slicing
    rs_sorted = rs.sort_values(["XID", "Date"]).reset_index(drop=True)
    clim_sorted = clim.sort_values(["Stations_i", "Date"]).reset_index(drop=True)

    rs_feature_cols = list(s2_features) + list(s1_features)
    # Ensure all expected columns exist (might be missing if a file was absent)
    for c in rs_feature_cols:
        if c not in rs_sorted.columns:
            rs_sorted[c] = np.nan

    # Group once after column completion
    rs_grouped = {xid: g for xid, g in rs_sorted.groupby("XID", sort=False)}
    clim_grouped = {sid: g for sid, g in clim_sorted.groupby("Stations_i", sort=False)}

    # Build phase lookup: Stations_i -> {phase_id: date}
    phase_cols_full = [f"phase_{p}_date" for p in C.ALL_PHASES]
    pheno_lookup = {}
    for _, row in pheno_yr.iterrows():
        pheno_lookup[row["Stations_i"]] = {
            p: row[f"phase_{p}_date"] for p in C.ALL_PHASES
        }

    xids = mapping["XID"].values
    stations_of = dict(zip(mapping["XID"], mapping["Stations_i"]))
    ranks_of    = dict(zip(mapping["XID"], mapping["Rank"]))

    n_records = len(xids)
    P = len(C.ALL_PHASES)
    X      = np.zeros((n_records, T, F), dtype=np.float32)
    MASK   = np.zeros((n_records, T, F), dtype=bool)
    Y      = np.zeros((n_records, T, P), dtype=np.float32)
    YMASK  = np.zeros((n_records, P), dtype=bool)
    BAG    = np.zeros(n_records, dtype=np.int64)
    RANK   = np.zeros(n_records, dtype=np.int32)

    s2_idx_end = len(s2_features)
    s1_idx_end = s2_idx_end + len(s1_features)

    with timer(f"  assemble {n_records} records for {harvest_year}"):
        for i, xid in enumerate(xids):
            sid = stations_of[xid]
            BAG[i] = sid
            RANK[i] = ranks_of[xid]

            # S1 + S2 features (slice first T rows to handle leap-year trim)
            g = rs_grouped.get(xid)
            if g is not None and len(g) >= T:
                vals = g[rs_feature_cols].values[:T]  # (T, n_rs_features)
                m = ~np.isnan(vals)
                X[i, :, :s1_idx_end] = np.where(m, vals, 0.0)
                MASK[i, :, :s1_idx_end] = m

            # Climate features
            cg = clim_grouped.get(sid)
            if cg is not None and len(cg) >= T:
                cvals = cg[list(clim_features)].values[:T]
                cm = ~np.isnan(cvals)
                X[i, :, s1_idx_end:] = np.where(cm, cvals, 0.0)
                MASK[i, :, s1_idx_end:] = cm

            # Labels (Gaussian curves over T days starting at win_start)
            phase_dates = pheno_lookup.get(sid, {})
            lbl, lmask = build_labels_for_record(
                phase_dates, win_start, T,
                sigma=C.GAUSSIAN_SIGMA_DAYS, phases=C.ALL_PHASES
            )
            Y[i] = lbl
            YMASK[i] = lmask

    return dict(
        X=X, mask=MASK, Y=Y, label_mask=YMASK,
        bag_id=BAG, rank=RANK,
        harvest_year=np.full(n_records, harvest_year, dtype=np.int32),
        xid=xids.astype(np.int64),
        feature_names=np.array(feats),
        phase_ids=np.array(C.ALL_PHASES, dtype=np.int32),
        window_start=np.datetime64(win_start),
        T=T,
    )


def build_all_years(harvest_years=range(2017, 2022),
                     output_dir: Path | None = None) -> dict[int, dict]:
    """Build tensors for all years; optionally save .npz files."""
    output_dir = output_dir or C.CACHE_DIR / "tensors"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading phenology master across all years...")
    pheno = load_phenology_all_years(harvest_years=harvest_years)

    bundles = {}
    for y in harvest_years:
        log.info(f"\n=== Building tensors for harvest year {y} ===")
        try:
            bundle = build_year(y, pheno)
        except FileNotFoundError as e:
            log.warning(f"Skipping {y}: {e}")
            continue
        bundles[y] = bundle
        # Save
        out_path = output_dir / f"phenology_{y}.npz"
        savable = {k: v for k, v in bundle.items() if not isinstance(v, int)}
        # window_start is a numpy.datetime64 - convert to string for npz
        savable["window_start"] = np.array(str(bundle["window_start"]))
        savable["T"] = np.array(bundle["T"])
        np.savez_compressed(out_path, **savable)
        log.info(f"Saved -> {out_path}")
    return bundles


if __name__ == "__main__":
    # Quick test on whatever data is available locally
    import sys
    years_arg = sys.argv[1:] if len(sys.argv) > 1 else ["2020"]
    years = [int(y) for y in years_arg]
    bundles = build_all_years(harvest_years=years)
    for y, b in bundles.items():
        print(f"\nYear {y}: X={b['X'].shape}, Y={b['Y'].shape}, "
              f"n_with_all_phases={b['label_mask'].all(axis=1).sum()}")
