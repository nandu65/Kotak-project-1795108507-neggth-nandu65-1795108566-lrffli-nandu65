"""
Online/Batch Reconciliation and Retraining Triggers (PDF Section A9.3 & A18.3).
Monitors divergence between nightly batch reference posteriors and intraday
fast online particle filters, signaling model drift and retraining triggers.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

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
