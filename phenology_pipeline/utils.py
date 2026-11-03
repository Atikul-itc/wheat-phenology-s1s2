"""Logging, caching, and date helpers."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from datetime import date, timedelta
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- LOGGING ----------
def get_logger(name: str = "pheno") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log

log = get_logger()

@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    log.info(f"{label} took {time.time()-t0:.2f}s")

# ---------- CACHING ----------
def parquet_cache(cache_dir: Path, key: str):
    """Decorator: cache the returned DataFrame as parquet under cache_dir/key.parquet."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / f"{key}.parquet"
            if path.exists() and not kwargs.pop("rebuild", False):
                log.info(f"Cache hit: {path}")
                return pd.read_parquet(path)
            log.info(f"Cache miss: computing {key}")
            df = fn(*args, **kwargs)
            df.to_parquet(path, index=False)
            log.info(f"Cached -> {path} ({len(df)} rows)")
            return df
        return wrapper
    return deco

# ---------- DATE HELPERS ----------
def harvest_year_window(harvest_year: int,
                        start_md=(9, 1), end_md=(10, 31)):
    """Return (start_date, end_date, n_days) for a harvest year."""
    start = date(harvest_year - 1, *start_md)
    end   = date(harvest_year, *end_md)
    n = (end - start).days + 1
    return start, end, n

def date_to_index(d, window_start: date) -> int:
    """Days since window_start (0-indexed). Accepts pd.Timestamp, date, or NaT."""
    if d is None or (isinstance(d, float) and np.isnan(d)) or pd.isna(d):
        return -1
    if isinstance(d, pd.Timestamp):
        d = d.date()
    return (d - window_start).days

def build_daily_grid(window_start: date, n_days: int) -> pd.DatetimeIndex:
    return pd.date_range(window_start, periods=n_days, freq="D")
