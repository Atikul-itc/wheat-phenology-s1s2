"""
Configuration for the phenology pipeline.

Edit BASE_DIR (and CACHE_DIR) when deploying to a new environment.
Everything else should be stable across local / HPC.
"""
from pathlib import Path

# ---------- PATHS ----------
# Defaults point to a local "data" folder. Scripts call configure(...) with the
# paths from config.yaml before any data are loaded.
BASE_DIR = Path("data")
S1_DIR   = BASE_DIR / "Timeseries" / "S1_v2"
S2_DIR   = BASE_DIR / "Timeseries" / "S2"
CLIM_DIR = BASE_DIR / "Climate_v2"
META_DIR = BASE_DIR / "Metadata"
PHENO_MASTER = BASE_DIR / "winter_wheat_phenology_2017_2021.csv"
CACHE_DIR = BASE_DIR / "cache"

# S1 and S2 CSVs are stored in one subfolder per harvest year (FLAT_LAYOUT = False).
FLAT_LAYOUT = False


def configure(**paths):
    """Override path settings, e.g. configure(S1_DIR=Path(...), CLIM_DIR=Path(...))."""
    g = globals()
    for k, v in paths.items():
        if k not in g:
            raise KeyError(f"Unknown setting: {k}")
        g[k] = Path(v) if k.endswith(("_DIR", "_MASTER")) else v


# ---------- FEATURE DEFINITIONS ----------
S2_FEATURES = ["blue", "green", "red", "nir", "swir1", "swir2",
               "re1", "re2", "re3", "EVI", "MTCI", "NDRE", "CIrededge"]
S1_FEATURES = ["VV", "VH", "CR"]
CLIM_FEATURES = ["mean_temp_celsius", "precipitation_mm", "cumulative_gdd"]

# Phases. Core phases are well-supported by RS; secondary are early-season (more missing).
CORE_PHASES = [15, 18, 19, 21, 24]
SECONDARY_PHASES = [10, 12]
ALL_PHASES = SECONDARY_PHASES + CORE_PHASES   # ordered: 10,12,15,18,19,21,24
PHASE_NAMES = {10: "Seeding", 12: "LeafDev", 15: "StemElong", 18: "Heading",
               19: "MilkRipe", 21: "YellowRipe", 24: "Harvest"}

# Reference: phases 10,12 are in autumn of (harvest_year - 1); rest in (harvest_year)
EARLY_PHASES = {10, 12}

# ---------- TEMPORAL WINDOW ----------
# For harvest year Y: time series spans (Y-1)-09-01 to Y-10-31.
# Non-leap years -> 426 days; leap years (e.g., 2020) -> 427 days.
# To keep T uniform across years, we trim to TARGET_T = 426 by dropping the last day
# when the natural window is longer.
WINDOW_START_MONTH_DAY = (9, 1)    # Sep 1 of (Y-1)
WINDOW_END_MONTH_DAY   = (10, 31)  # Oct 31 of Y
TARGET_T               = 426       # uniform sequence length across all years

# ---------- GDD RECOMPUTATION ----------
# We recompute cumulative GDD from mean_temp_celsius for reproducibility and to
# handle the rdwd cumulative-NaN-propagation artifact (a few stations lose all
# cumulative_gdd when a single early temp value is NaN).
GDD_BASE_TEMP_C      = 0.0   # winter wheat base temperature
# Accumulation starts on 1 September of (harvest_year - 1), the start of the
# season window, so that the autumn stages also receive thermal time.
GDD_RESET_MONTH_DAY  = (9, 1)
# When mean_temp_celsius is NaN on a day, that day contributes 0 to the sum
# (rather than propagating NaN). This keeps the cumulative signal intact.

# ---------- QUALITY FILTERS ----------
QB_KEEP = {1, 2}    # Eintrittsdatum_QB values to keep

# ---------- LABEL CONSTRUCTION ----------
GAUSSIAN_SIGMA_DAYS = 5    # uniform across phases for now
GAUSSIAN_TRUNCATION_SIGMA = 4   # zero out beyond 4*sigma for sparsity

# ---------- DATA LAYOUT HELPERS ----------
def s1_csv(feature: str, harvest_year: int) -> Path:
    """Locate an S1 CSV. Two-digit year suffix (e.g., 20 for 2020)."""
    yy = f"{harvest_year % 100:02d}"
    fname = f"{feature}_wheat{yy}.csv"
    if FLAT_LAYOUT:
        return S1_DIR / fname
    return S1_DIR / str(harvest_year) / fname

def s2_csv(feature: str, harvest_year: int) -> Path:
    yy = f"{harvest_year % 100:02d}"
    fname = f"{feature}_wheat{yy}.csv"
    if FLAT_LAYOUT:
        return S2_DIR / fname
    return S2_DIR / str(harvest_year) / fname

def climate_csv(harvest_year: int) -> Path:
    return CLIM_DIR / f"climate_with_cgdd_{harvest_year}.csv"

def metadata_csv(harvest_year: int) -> Path:
    return META_DIR / f"metadata_S2_{harvest_year}.csv"
