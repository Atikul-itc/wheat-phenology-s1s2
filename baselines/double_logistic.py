"""Curve-fitting baseline: double logistic fit to the Sentinel-2 EVI series (Section 3.3).

For each field and harvest year, a double logistic function (Beck et al., 2006) is
fitted to the observed EVI values between days 180 and 365 of the season window
(about March to August). Its dates are mapped heuristically to four stages:
  stem elongation = start of season (green-up inflection),
  heading         = midpoint of start of season and peak,
  yellow ripeness = midpoint of peak and end of season,
  harvest         = end of season (senescence inflection).
The other stages are not estimated (NaN). A bias-corrected version is computed
in ``analysis.evaluate``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

# Window for fitting: focus on growing season of the harvest year
# T=426 starts Sep 1 of (Y-1). Day 180 ≈ Feb 28 of Y; day 365 ≈ Aug 31 of Y.
FIT_LO, FIT_HI = 180, 365


def double_logistic(t, m1, m2, m3, m4, m5, m6):
    """Double-logistic seasonal model (Beck et al., 2006-style)."""
    s1 = 1.0 / (1.0 + np.exp(-m3 * (t - m4)))
    s2 = 1.0 / (1.0 + np.exp(-m5 * (t - m6)))
    return m1 + m2 * (s1 - s2)


def fit_one(t: np.ndarray, y: np.ndarray) -> dict | None:
    """Fit double-logistic to a single field's EVI time series.

    Returns a dict of fitted params + derived dates, or None on failure.
    """
    if len(t) < 8:
        return None
    # Robust initial guesses
    y_low = np.percentile(y, 10)
    y_high = np.percentile(y, 90)
    t_mid = (t.min() + t.max()) / 2
    p0 = [y_low,                # m1
          max(y_high - y_low, 0.05),  # m2
          0.15,                  # m3 (greenup steepness)
          t_mid - 30,            # m4 (greenup midpoint)
          0.15,                  # m5
          t_mid + 30]            # m6 (senescence midpoint)
    bounds = ([-1, 0, 0.001, t.min()-30, 0.001, t.min()-30],
              [ 2, 2, 2.0,    t.max()+30, 2.0,    t.max()+30])

    try:
        res = least_squares(
            lambda p: double_logistic(t, *p) - y,
            x0=p0, bounds=bounds, method="trf", max_nfev=500
        )
    except Exception:
        return None
    if not res.success:
        return None

    m1, m2, m3, m4, m5, m6 = res.x
    # Sanity: m4 (green-up) should precede m6 (senescence)
    if m4 >= m6:
        return None
    # Peak: argmax of the fitted curve on a dense grid within the fitted window
    tt = np.linspace(t.min(), t.max(), 400)
    yy = double_logistic(tt, *res.x)
    peak = float(tt[np.argmax(yy)])
    return {
        "m": res.x.tolist(),
        "SOS": float(m4),
        "EOS": float(m6),
        "PEAK": peak,
        "residual_rms": float(np.sqrt(np.mean(res.fun ** 2))),
    }


def predict_for_bundle(bundle: dict) -> np.ndarray:
    """Predict day-of-window for each phase using curve-fit dates.

    Returns (N, P) float predictions; NaN for unestimated phases.
    """
    feats = bundle["feature_names"]
    if "EVI" not in feats:
        raise KeyError("Bundle lacks EVI feature")
    evi_idx = feats.index("EVI")

    X = bundle["X"][:, :, evi_idx]   # (N, T)
    M = bundle["mask"][:, :, evi_idx]
    phase_ids = bundle["phase_ids"]
    P = len(phase_ids)
    N, T = X.shape

    pred = np.full((N, P), np.nan, dtype=np.float32)
    n_fit_ok = 0
    n_attempt = 0

    # Phase index map
    phase_idx = {p: i for i, p in enumerate(phase_ids)}

    # Limit to the fit window
    t_all = np.arange(T, dtype=np.float32)
    for i in range(N):
        m_i = M[i, FIT_LO:FIT_HI]
        if m_i.sum() < 8:
            continue
        t = t_all[FIT_LO:FIT_HI][m_i]
        y = X[i, FIT_LO:FIT_HI][m_i]
        n_attempt += 1
        fit = fit_one(t, y)
        if fit is None:
            continue
        n_fit_ok += 1
        sos, eos, peak = fit["SOS"], fit["EOS"], fit["PEAK"]
        # Heuristic mapping:
        if 15 in phase_idx:
            pred[i, phase_idx[15]] = sos             # SOS ≈ stem elongation start
        if 18 in phase_idx:
            pred[i, phase_idx[18]] = 0.5 * (sos + peak)  # midpoint
        if 21 in phase_idx:
            pred[i, phase_idx[21]] = 0.5 * (peak + eos)
        if 24 in phase_idx:
            pred[i, phase_idx[24]] = eos             # EOS ≈ harvest

    print(f"  curve-fit: {n_fit_ok}/{n_attempt} succeeded "
          f"({100*n_fit_ok/max(n_attempt,1):.1f}%)")
    return pred
