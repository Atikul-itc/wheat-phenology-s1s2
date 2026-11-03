"""Tabular baseline: histogram gradient boosting on seasonal summary statistics (Section 3.3).

Features come from ``baselines.engineered_features.compute_for_bundle``.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def train_tabular(X_train: np.ndarray, Y_train: np.ndarray,
                  n_iter: int = 200, random_state: int = 0):
    """Train one histogram gradient boosting regressor per stage (200 iterations,
    learning rate 0.1, at most 63 leaf nodes). Y_train holds reference days, -1 = missing."""
    Y = Y_train.astype(np.float32).copy()
    Y[Y < 0] = np.nan

    P = Y.shape[1]
    models = []
    for j in range(P):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 50:
            models.append(None)
            continue
        reg = HistGradientBoostingRegressor(max_iter=n_iter, learning_rate=0.1, max_depth=None,
                                            max_leaf_nodes=63, random_state=random_state)
        reg.fit(X_train[m], Y[m, j])
        models.append(reg)
    return models


def predict_models(models: list, X: np.ndarray, n_phases: int) -> np.ndarray:
    """Run each phase-specific model. Returns (N, P) float predictions; NaN for unfit."""
    N = X.shape[0]
    out = np.full((N, n_phases), np.nan, dtype=np.float32)
    for j, m in enumerate(models):
        if m is None:
            continue
        out[:, j] = m.predict(X)
    return out
