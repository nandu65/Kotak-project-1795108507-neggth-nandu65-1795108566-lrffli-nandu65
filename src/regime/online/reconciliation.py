"""
Online/Batch Reconciliation and Retraining Triggers (PDF Section A9.3 & A18.3).
Monitors divergence between nightly batch reference posteriors and intraday
fast online particle filters, signaling model drift and retraining triggers.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_total_variation_gap(p_batch: np.ndarray, p_online: np.ndarray) -> float:
    r"""
    Total Variation (TV) distance between batch and online probability distributions:
        $$\text{Gap} = \frac{1}{2} \sum_{k=0}^{K-1} |p_k^{\text{batch}} - p_k^{\text{online}}|$$
    Bounded strictly in $[0, 1]$. A gap $> 0.15$ indicates severe intraday drift.
    """
    pb = np.clip(p_batch, 0.0, 1.0)
    po = np.clip(p_online, 0.0, 1.0)
    pb = pb / np.sum(pb)
    po = po / np.sum(po)
    return float(0.5 * np.sum(np.abs(pb - po)))


class OnlineBatchReconciler:
    r"""
    Monitors model health in production two-speed serving layout.
    """

    def __init__(
        self,
        gap_threshold: float = 0.15,
        cp_threshold: float = 0.25,
        coverage_target: float = 0.90,
    ) -> None:
        self.gap_threshold = gap_threshold
        self.cp_threshold = cp_threshold
        self.coverage_target = coverage_target

    def evaluate_health(
        self,
        p_batch: np.ndarray,
        p_online: np.ndarray,
        changepoint_prob: float,
        rolling_coverage: float = 0.90,
        max_feature_psi: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluate health metrics and check for retraining triggers (PDF Section A18.3).

        Returns:
            Dictionary with health status and trigger recommendations.
        """
        tv_gap = compute_total_variation_gap(p_batch, p_online)
        triggers: List[str] = []

        if tv_gap > self.gap_threshold:
            triggers.append("online_batch_divergence")
        if changepoint_prob > self.cp_threshold:
            triggers.append("regime_changepoint")
        if rolling_coverage < (self.coverage_target - 0.05):
            triggers.append("conformal_coverage_shortfall")
        if max_feature_psi > 0.25:
            triggers.append("feature_distribution_drift")

        is_healthy = len(triggers) == 0
        return {
            "is_healthy": is_healthy,
            "tv_gap": float(np.round(tv_gap, 4)),
            "changepoint_prob": float(np.round(changepoint_prob, 4)),
            "rolling_coverage": float(np.round(rolling_coverage, 4)),
            "max_feature_psi": float(np.round(max_feature_psi, 4)),
            "retraining_triggers": triggers,
            "retrain_recommended": not is_healthy,
        }


def run_heldout_month_reconciliation(
    df: pd.DataFrame,
    heldout_start: str = "2024-05-01",
    heldout_end: str = "2024-05-31",
) -> Dict[str, Any]:
    r"""
    Reconciles streaming Sequential Monte Carlo path against nightly batch engine
    over an authentic held-out test month (Deliverable 4 requirement).

    Computes daily Total Variation (TV) distances:
        $$TV(p_{\text{batch}}, p_{\text{online}}) = \frac{1}{2} \sum_{k=0}^{K-1} |p_k^{\text{batch}} - p_k^{\text{online}}|$$
    Reports maximum TV gap and average drift.
    """
    from ..models.hmm_model import GaussianRegimeHMM
    from .particle_filter import RegimeParticleFilter

    dt_idx = pd.to_datetime(df.index)
    train_mask = dt_idx < heldout_start
    test_mask = (dt_idx >= heldout_start) & (dt_idx <= heldout_end)

    train_data = df.loc[train_mask]
    test_data = df.loc[test_mask]

    if len(test_data) == 0:
        # Fallback to the last 22 trading sessions (~1 month)
        test_data = df.iloc[-22:]
        train_data = df.iloc[:-22]

    ret_col = "nifty_return" if "nifty_return" in df.columns else df.columns[0]
    train_rets = train_data[ret_col].dropna().values.reshape(-1, 1)
    test_rets = test_data[ret_col].dropna().values

    # 1. Batch path: fit Gaussian HMM on historical train data
    batch_hmm = GaussianRegimeHMM(n_regimes=5, n_iter=60, seed=42)
    batch_hmm.fit(train_rets)
    batch_probs = batch_hmm.predict_proba(test_rets.reshape(-1, 1))

    vols = np.sqrt(np.maximum(batch_hmm.covars[:, 0, 0], 1e-6)) if batch_hmm.covars.ndim == 3 else np.array([0.01] * 5)
    pf = RegimeParticleFilter(
        P=batch_hmm.A,
        mu=batch_hmm.means[:, 0],
        sigma=vols,
        n_particles=2000,
        seed=42,
    )

    streaming_probs = []
    tv_gaps = []

    for t in range(len(test_rets)):
        # Streaming update on new day observation
        p_stream = pf.update(test_rets[t])
        streaming_probs.append(p_stream)

        # Total variation distance against batch prediction
        gap = compute_total_variation_gap(batch_probs[t], p_stream)
        tv_gaps.append(gap)

    max_gap = float(np.max(tv_gaps))
    mean_gap = float(np.mean(tv_gaps))
    max_gap_idx = int(np.argmax(tv_gaps))
    max_gap_date = str(test_data.index[max_gap_idx])[:10]

    return {
        "heldout_sessions_evaluated": len(test_rets),
        "heldout_start_date": str(test_data.index[0])[:10],
        "heldout_end_date": str(test_data.index[-1])[:10],
        "max_total_variation_gap": float(np.round(max_gap, 4)),
        "mean_total_variation_gap": float(np.round(mean_gap, 4)),
        "max_gap_date": max_gap_date,
        "is_within_tolerance": bool(max_gap <= 0.15),
        "daily_tv_gaps": [float(np.round(g, 4)) for g in tv_gaps],
    }
