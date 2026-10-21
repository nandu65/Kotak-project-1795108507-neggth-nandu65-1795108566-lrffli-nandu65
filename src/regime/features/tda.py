"""
Topological Data Analysis (TDA) Feature Engineering for Regime Detection.
Computes persistent-homology summaries of rolling return windows via ripser / giotto-tda.
Obeys strict zero look-ahead bias: at time t, only returns up to t are observed.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ripser import ripser

logger = logging.getLogger(__name__)


def compute_persistence_entropy(lifetimes: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute persistence entropy of a diagram:
        E = - sum (p_i * log(p_i)) where p_i = l_i / sum(l_j)
    """
    if len(lifetimes) == 0:
        return 0.0
    tot = np.sum(lifetimes) + eps
    p = lifetimes / tot
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + eps)))


def compute_rolling_tda_features(
    return_df: pd.DataFrame,
    window: int = 63,
    stride: int = 1,
) -> pd.DataFrame:
    r"""
    Compute persistent homology summaries of rolling return point clouds.

    For each rolling causal window of length $W$ up to date $t$:
    1. Form multi-asset point cloud $X_t \in \mathbb{R}^{W \times D}$ from return series.
    2. Compute Vietoris-Rips filtration up to homology dimension 1 ($H_0$ and $H_1$).
    3. Extract persistence summaries:
       - `tda_h0_entropy`: dispersion / clustering entropy of connected components.
       - `tda_h0_max_lifetime`: dominant persistence span of 0-cycles.
       - `tda_h1_count`: count of persistent 1-dimensional cycles (loops).
       - `tda_h1_total_persistence`: sum of lifetimes of 1-cycles (topological complexity).
       - `tda_h1_max_persistence`: max lifetime of 1-cycles (structural market voids).

    Strict Zero Look-Ahead Discipline:
        Observation at index $i$ strictly uses rows $[i - W + 1 : i + 1]$.
    """
    T, D = return_df.shape
    dates = return_df.index
    arr = return_df.values

    cols = [
        "tda_h0_entropy",
        "tda_h0_max_lifetime",
        "tda_h1_count",
        "tda_h1_total_persistence",
        "tda_h1_max_persistence",
    ]
    records = []

    for i in range(T):
        if i < window - 1:
            records.append({c: np.nan for c in cols})
            continue

        window_data = arr[i - window + 1 : i + 1]
        # Standardize within window to scale invariance
        w_std = np.std(window_data, axis=0, keepdims=True) + 1e-8
        w_norm = (window_data - np.mean(window_data, axis=0, keepdims=True)) / w_std

        try:
            # Subsample or takens delay if 1D
            if D == 1:
                # Takens delay embedding (dim=3, tau=2)
                tau = 2
                emb_len = window - 2 * tau
                if emb_len > 10:
                    pts = np.column_stack([
                        w_norm[k : k + emb_len, 0] for k in range(0, 3 * tau, tau)
                    ])
                else:
                    pts = w_norm
            else:
                pts = w_norm

            res = ripser(pts, maxdim=1)
            dgm0 = res["dgms"][0]
            dgm1 = res["dgms"][1]

            # H0 lifetimes (ignore infinite death on last bar)
            h0_finite = dgm0[np.isfinite(dgm0[:, 1])]
            h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0] if len(h0_finite) > 0 else np.array([0.0])
            h0_ent = compute_persistence_entropy(h0_lifetimes)
            h0_max = float(np.max(h0_lifetimes)) if len(h0_lifetimes) > 0 else 0.0

            # H1 lifetimes
            if len(dgm1) > 0:
                h1_lifetimes = dgm1[:, 1] - dgm1[:, 0]
                h1_cnt = float(len(dgm1))
                h1_tot = float(np.sum(h1_lifetimes))
                h1_max = float(np.max(h1_lifetimes))
            else:
                h1_cnt = 0.0
                h1_tot = 0.0
                h1_max = 0.0

            records.append({
                "tda_h0_entropy": h0_ent,
                "tda_h0_max_lifetime": h0_max,
                "tda_h1_count": h1_cnt,
                "tda_h1_total_persistence": h1_tot,
                "tda_h1_max_persistence": h1_max,
            })
        except Exception as e:
            records.append({
                "tda_h0_entropy": 0.0,
                "tda_h0_max_lifetime": 0.0,
                "tda_h1_count": 0.0,
                "tda_h1_total_persistence": 0.0,
                "tda_h1_max_persistence": 0.0,
            })

    return pd.DataFrame(records, index=dates)
