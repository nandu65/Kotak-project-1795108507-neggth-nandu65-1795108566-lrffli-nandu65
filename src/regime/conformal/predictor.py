"""
Conformal Prediction and Calibration Layer for Indian Equity Regime Detection.
Implements Split-Conformal Prediction, Adaptive Prediction Sets (APS),
Online Adaptive Conformal Inference (ACI), and Expected Calibration Error (ECE) diagnostics (PDF Section A6).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..models.base import REGIME_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration Diagnostics (PDF Section A6.6)
# ---------------------------------------------------------------------------

def compute_reliability_curve(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute empirical reliability diagram coordinates.

    Args:
        probs: (N, K) predicted probability matrix.
        y_true: (N,) true integer labels.
        n_bins: Number of confidence bins in [0, 1].

    Returns:
        Tuple of:
        - accuracies: Observed accuracy within each bin.
        - confidences: Mean predicted confidence within each bin.
        - counts: Number of samples in each bin.
    """
    confidences_max = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers, bin_uppers = bins[:-1], bins[1:]

    accuracies = []
    confidences = []
    counts = []

    for lo, hi in zip(bin_lowers, bin_uppers):
        in_bin = (confidences_max > lo) & (confidences_max <= hi)
        n_k = int(in_bin.sum())
        counts.append(n_k)
        if n_k > 0:
            accuracies.append(float(correct[in_bin].mean()))
            confidences.append(float(confidences_max[in_bin].mean()))
        else:
            accuracies.append(0.0)
            confidences.append((lo + hi) / 2.0)

    return np.array(accuracies), np.array(confidences), np.array(counts)


def compute_expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> float:
    r"""
    Expected Calibration Error (ECE; Guo et al., 2017):
        $$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
    Lower is better. Production target: $\text{ECE} < 0.08$.
    """
    acc, conf, cnt = compute_reliability_curve(probs, y_true, n_bins=n_bins)
    total = np.sum(cnt)
    if total == 0:
        return 0.0
    ece = np.sum((cnt / total) * np.abs(acc - conf))
    return float(ece)


# ---------------------------------------------------------------------------
# Split-Conformal & Adaptive Prediction Sets (PDF Section A6.2 & A6.3)
# ---------------------------------------------------------------------------

class SplitConformalClassifier:
    r"""
    Standard Split-Conformal Prediction with finite-sample coverage guarantee:
        $$P(y_{\text{new}} \in \mathcal{C}(x_{\text{new}})) \ge 1 - \alpha$$
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.q_hat: float = 1.0
        self.is_calibrated = False

    def calibrate(self, cal_probs: np.ndarray, y_cal: np.ndarray) -> float:
        r"""
        Fit conformal threshold on held-out calibration set.
        Score: $s_i = 1 - p(y_i \mid x_i)$.
        """
        n = len(y_cal)
        cal_scores = 1.0 - cal_probs[np.arange(n), y_cal]

        # Conformal quantile level
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(cal_scores, q_level, method="higher"))
        self.is_calibrated = True
        return self.q_hat

    def predict_set(self, test_probs: np.ndarray) -> np.ndarray:
        """
        Return boolean mask of shape (N, K) indicating which regimes are included in prediction set.
        """
        if not self.is_calibrated:
            raise ValueError("Conformal classifier must be calibrated first.")
        # Class k is included if (1 - p_k) <= q_hat <=> p_k >= 1 - q_hat
        pred_sets = test_probs >= (1.0 - self.q_hat)
        # Ensure at least the top class is included
        for i in range(len(test_probs)):
            if not pred_sets[i].any():
                pred_sets[i, np.argmax(test_probs[i])] = True
        return pred_sets


class AdaptivePredictionSets:
    r"""
    Adaptive Prediction Sets (APS; Romano et al., 2020).
    Non-conformity score is cumulative probability mass of classes sorted by descending probability:
        $$E_i = \sum_{j=1}^{\text{rank}(y_i)} \pi_{(j)}(x_i)$$
    Yields significantly smaller and more informative prediction sets than standard split-conformal.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.q_hat: float = 1.0
        self.is_calibrated = False

    def calibrate(self, cal_probs: np.ndarray, y_cal: np.ndarray) -> float:
        """
        Calibrate APS quantile threshold on calibration probabilities.
        """
        n = len(y_cal)
        sorted_idx = np.argsort(-cal_probs, axis=1)
        sorted_probs = np.take_along_axis(cal_probs, sorted_idx, axis=1)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        # Find position of true class in sorted predictions
        rank_of_true = np.array([np.where(sorted_idx[i] == y_cal[i])[0][0] for i in range(n)])
        cal_scores = cumsum_probs[np.arange(n), rank_of_true]

        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(cal_scores, q_level, method="higher"))
        self.is_calibrated = True
        return self.q_hat

    def predict_set(self, test_probs: np.ndarray) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Generate adaptive prediction sets.

        Returns:
            Tuple of:
            - in_set_mask: (N, K) boolean inclusion array.
            - set_labels: List of lists containing regime names in each prediction set.
        """
        if not self.is_calibrated:
            raise ValueError("APS must be calibrated first.")

        N, K = test_probs.shape
        sorted_idx = np.argsort(-test_probs, axis=1)
        sorted_probs = np.take_along_axis(test_probs, sorted_idx, axis=1)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        in_set_sorted = cumsum_probs <= self.q_hat
        in_set_sorted[:, 0] = True  # Always include top class

        in_set_mask = np.zeros((N, K), dtype=bool)
        for i in range(N):
            active_classes = sorted_idx[i, in_set_sorted[i]]
            in_set_mask[i, active_classes] = True

        set_labels = []
        for i in range(N):
            active_names = [REGIME_NAMES[k] for k in range(K) if in_set_mask[i, k]]
            set_labels.append(active_names)

        return in_set_mask, set_labels


# ---------------------------------------------------------------------------
# Online Adaptive Conformal Inference (ACI; PDF Section A6.5)
# ---------------------------------------------------------------------------

class AdaptiveConformalInference:
    r"""
    Online Adaptive Conformal Inference (ACI; Gibbs & Candès, 2021).
    Dynamically adjusts nominal coverage level $\alpha_t$ to guarantee valid coverage
    under arbitrary non-exchangeable distribution shifts and financial regime transitions:
        $$\alpha_{t+1} = \alpha_t + \gamma (\alpha_{\text{target}} - \text{err}_t)$$
        $$\text{err}_t = \mathbf{1}_{\{y_t \notin \mathcal{C}_t(x_t)\}}$$
    """

    def __init__(self, alpha_target: float = 0.10, gamma: float = 0.01) -> None:
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha_t = alpha_target
        self.coverage_history: List[bool] = []
        self.alpha_history: List[float] = [alpha_target]

    def step(self, is_covered: bool) -> float:
        """
        Update online coverage target after observing true realisation.

        Args:
            is_covered: True if realised regime was inside prediction set.

        Returns:
            Updated alpha_t for next time step.
        """
        err_t = 0.0 if is_covered else 1.0
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - err_t)
        # Numerical guardrails [0.001, 0.999]
        self.alpha_t = float(np.clip(self.alpha_t, 1e-3, 1.0 - 1e-3))

        self.coverage_history.append(is_covered)
        self.alpha_history.append(self.alpha_t)
        return self.alpha_t

    def get_realized_coverage(self, lookback: Optional[int] = None) -> float:
        """Calculate realised empirical coverage over recent history."""
        if not self.coverage_history:
            return 1.0 - self.alpha_target
        sub = self.coverage_history[-lookback:] if lookback else self.coverage_history
        return float(np.mean(sub))


def evaluate_conformal_coverage(
    in_set_mask: np.ndarray,
    y_true: np.ndarray,
    nominal_coverage: float = 0.90,
    regime_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    r"""
    Evaluate empirical coverage vs nominal (90%, 95%) overall and per regime (Step 7 requirement).

    Args:
        in_set_mask: (N, K) boolean inclusion mask.
        y_true: (N,) true regime integer labels.
        nominal_coverage: Nominal target (e.g. 0.90 or 0.95).
        regime_names: Optional regime names list.

    Returns:
        Dict with overall coverage, per-regime coverage, mean set size, and gap.
    """
    N, K = in_set_mask.shape
    names = regime_names or REGIME_NAMES[:K]

    covered = np.array([in_set_mask[i, y_true[i]] for i in range(N)], dtype=float)
    overall_cov = float(np.mean(covered))
    mean_set_size = float(np.mean(np.sum(in_set_mask, axis=1)))

    per_regime_cov = {}
    for k in range(K):
        mask_k = y_true == k
        if np.sum(mask_k) > 0:
            reg_cov = float(np.mean(in_set_mask[mask_k, k]))
            per_regime_cov[names[k]] = float(np.round(reg_cov, 4))
        else:
            per_regime_cov[names[k]] = float(nominal_coverage)

    return {
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": float(np.round(overall_cov, 4)),
        "coverage_gap": float(np.round(overall_cov - nominal_coverage, 4)),
        "mean_set_size": float(np.round(mean_set_size, 3)),
        "per_regime_coverage": per_regime_cov,
        "is_valid": bool(overall_cov >= nominal_coverage - 0.05),
    }

