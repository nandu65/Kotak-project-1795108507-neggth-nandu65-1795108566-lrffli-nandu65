"""
Ensemble Layer for Regime Detection: Bayesian Model Averaging, Constrained Stacking,
Proper Scoring Rules (RPS, Brier, Log Loss), and Uncertainty Decomposition (PDF Section A10 & A14).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..models.base import REGIME_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strictly Proper Scoring Rules (PDF Section A14)
# ---------------------------------------------------------------------------

def compute_log_loss(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    r"""
    Strictly proper log loss:
        $$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \ln (p_i(y_i))$$
    """
    N = len(y_true)
    if N == 0:
        return 0.0
    p = np.clip(probs[np.arange(N), y_true], eps, 1.0)
    return float(-np.mean(np.log(p)))


def compute_brier_score(probs: np.ndarray, y_true: np.ndarray, K: int = 5) -> float:
    r"""
    Multi-class Brier score (strictly proper, bounded [0, 2]):
        $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=0}^{K-1} (p_{ik} - y_{ik})^2$$
    """
    N = len(y_true)
    if N == 0:
        return 0.0
    onehot = np.eye(K)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def compute_ranked_probability_score(probs: np.ndarray, y_true: np.ndarray, K: int = 5) -> float:
    r"""
    Ranked Probability Score (RPS) for ordered regimes (PDF Section A14.2):
        $$\text{RPS} = \frac{1}{N(K-1)} \sum_{i=1}^N \sum_{k=0}^{K-1} \left( \sum_{j=0}^k p_{ij} - \sum_{j=0}^k y_{ij} \right)^2$$
    Respects ordinal regime distance: predicting Late-Cycle when truth is Risk-On is
    penalised much less than predicting Risk-Off.
    """
    N = len(y_true)
    if N == 0:
        return 0.0
    onehot = np.eye(K)[y_true]
    cdf_pred = np.cumsum(probs, axis=1)
    cdf_obs = np.cumsum(onehot, axis=1)
    diff_sq = (cdf_pred - cdf_obs) ** 2
    return float(np.mean(np.sum(diff_sq, axis=1)) / max(K - 1, 1))


def compute_brier_decomposition(probs_pos: np.ndarray, y_binary: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    r"""
    Murphy Brier score decomposition (Murphy, 1973; PDF Section A14.3):
        $$\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$
    - Reliability: calibration error (lower is better, 0 is perfectly calibrated).
    - Resolution: discrimination sharpness (higher is better).
    - Uncertainty: baseline entropy $p(1 - p)$.
    """
    N = len(y_binary)
    if N == 0:
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0, "brier": 0.0}

    base = float(np.mean(y_binary))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rel, res = 0.0, 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs_pos > lo) & (probs_pos <= hi)
        nk = int(m.sum())
        if nk == 0:
            continue
        pk = float(np.mean(probs_pos[m]))
        ok = float(np.mean(y_binary[m]))
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - base) ** 2

    rel /= N
    res /= N
    unc = base * (1.0 - base)
    return {
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "brier": float(rel - res + unc),
    }


def compute_skill_score(score_model: float, score_reference: float) -> float:
    r"""
    Skill score relative to a benchmark forecaster (PDF Section A14.4):
        $$\text{Skill} = 1 - \frac{\text{Score}_{\text{model}}}{\text{Score}_{\text{reference}}}$$
    Positive score signifies outperformance over naive baseline.
    """
    if score_reference <= 1e-12:
        return 0.0
    return float(1.0 - (score_model / score_reference))


# ---------------------------------------------------------------------------
# Ensembling & Stacking Algorithms (PDF Section A10)
# ---------------------------------------------------------------------------

class BayesianModelAverager:
    r"""
    Bayesian Model Averaging (BMA; PDF Section A10.1).
    Weights each candidate model by its out-of-sample log predictive evidence:
        $$w_m = \frac{\exp(\mathcal{L}_m - \max_j \mathcal{L}_j)}{\sum_{k=1}^M \exp(\mathcal{L}_k - \max_j \mathcal{L}_j)}$$
        $$\bar{p}_t = \sum_{m=1}^M w_m p_t^{(m)}$$
    """

    def __init__(self) -> None:
        self.weights: Optional[np.ndarray] = None

    def fit(self, log_predictive_likelihoods: np.ndarray) -> np.ndarray:
        """
        Compute BMA weights from out-of-sample log predictive scores.
        """
        l_arr = np.asarray(log_predictive_likelihoods, dtype=float)
        # Shift by max for numerical stability
        w = np.exp(l_arr - np.max(l_arr))
        self.weights = w / np.sum(w)
        return self.weights

    def combine(self, model_probs: np.ndarray) -> np.ndarray:
        """
        Combine predictions: model_probs of shape (M, N, K) or (M, K).
        """
        if self.weights is None:
            raise ValueError("BMA is not fitted.")
        if model_probs.ndim == 3:
            # (M, N, K) -> (N, K)
            return np.tensordot(self.weights, model_probs, axes=(0, 0))
        elif model_probs.ndim == 2:
            # (M, K) -> (K,)
            return self.weights @ model_probs
        else:
            raise ValueError(f"Invalid model_probs dimension: {model_probs.ndim}")


class ConstrainedStacker:
    r"""
    Time-Aware Constrained Stacking on out-of-fold log score (PDF Section A10.2).
    Finds convex combination weights $\mathbf{w} \in \Delta^{M-1}$ minimizing cross-entropy:
        $$\min_{\mathbf{w}} -\frac{1}{N} \sum_{i=1}^N \sum_{k=0}^{K-1} y_{ik} \ln \left( \sum_{m=1}^M w_m p_{ik}^{(m)} \right)$$
        $$\text{s.t.} \quad w_m \ge 0, \quad \sum_{m=1}^M w_m = 1$$
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.weights: Optional[np.ndarray] = None

    def fit(self, base_probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Fit simplex stacking weights via constrained optimization.

        Args:
            base_probs: Out-of-fold predictions of shape (M, N, K).
            y_true: True integer regime labels of shape (N,).

        Returns:
            Optimal simplex weight vector (M,).
        """
        M, N, K = base_probs.shape
        onehot = np.eye(K)[y_true]

        def neg_loglik(w):
            w_norm = np.clip(w, 0, None)
            s = np.sum(w_norm)
            if s > 0:
                w_norm = w_norm / s
            combined = np.tensordot(w_norm, base_probs, axes=(0, 0))  # (N, K)
            combined = np.clip(combined, 1e-9, 1.0)
            loss = -np.mean(np.sum(onehot * np.log(combined), axis=1))
            return float(loss)

        w0 = np.full(M, 1.0 / M)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * M

        res = minimize(neg_loglik, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        w_opt = np.clip(res.x, 0.0, 1.0)
        self.weights = w_opt / np.sum(w_opt)
        return self.weights

    def predict_proba(self, base_probs: np.ndarray) -> np.ndarray:
        """
        Combine base probabilities of shape (M, N, K) into (N, K).
        """
        if self.weights is None:
            raise ValueError("Stacker is not fitted.")
        combined = np.tensordot(self.weights, base_probs, axes=(0, 0))
        return combined / np.sum(combined, axis=-1, keepdims=True)


class EnsembleRegimeEngine:
    r"""
    Unified Regime Ensemble Engine (PDF Section A10.4 & A11.2).
    Coordinates base models, computes stacking/BMA mixtures, decomposes
    uncertainty into Epistemic (model disagreement) and Aleatoric (data noise),
    and emits the Daily Combined Regime Output Contract.
    """

    def __init__(self, model_names: List[str], seed: int = 42) -> None:
        self.model_names = model_names
        self.seed = seed
        self.stacker = ConstrainedStacker(seed=seed)
        self.bma = BayesianModelAverager()
        self.weights: Dict[str, float] = {m: 1.0 / len(model_names) for m in model_names}

    def fit_ensemble(
        self,
        base_probs: np.ndarray,
        y_true: np.ndarray,
        method: str = "stacking",
    ) -> Dict[str, float]:
        """
        Fit ensemble combination weights.

        Args:
            base_probs: (M, N, K) matrix of base model predictions.
            y_true: (N,) true regime labels.
            method: 'stacking' or 'bma'.

        Returns:
            Dict mapping model names to fitted weights.
        """
        M, N, K = base_probs.shape
        if method == "stacking":
            w = self.stacker.fit(base_probs, y_true)
        else:
            # BMA based on negative log-loss per model
            log_liks = []
            for m in range(M):
                ll = -compute_log_loss(base_probs[m], y_true)
                log_liks.append(ll)
            w = self.bma.fit(np.array(log_liks))

        self.weights = {self.model_names[i]: float(np.round(w[i], 4)) for i in range(M)}
        return self.weights

    def predict_contract(
        self,
        model_probs: np.ndarray,
        date_str: str = "2026-10-19",
        dominant_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        r"""
        Emit the Unified Daily Regime Output Contract (PDF Section A10.4).

        Returns:
            Contract dict containing:
            - date: Date string
            - regime_probs: (5,) ensembled posterior probability vector
            - dominant_regime: argmax regime name
            - dominant_model: Highest-weight ensemble member
            - ensemble_weights: Model weights dictionary
            - epistemic_uncertainty: Inter-model standard deviation
            - aleatoric_uncertainty: Intra-model average entropy
        """
        # model_probs: (M, K) for single day or (M, 1, K)
        if model_probs.ndim == 3 and model_probs.shape[1] == 1:
            p_day = model_probs[:, 0, :]
        else:
            p_day = model_probs

        M, K = p_day.shape
        w_vec = np.array([self.weights[m] for m in self.model_names])
        w_vec = w_vec / np.sum(w_vec)

        # Ensembled posterior probabilities
        ensembled_p = w_vec @ p_day
        ensembled_p = ensembled_p / np.sum(ensembled_p)

        dominant_idx = int(np.argmax(ensembled_p))
        dominant_regime = REGIME_NAMES[dominant_idx]

        # Epistemic: standard deviation across model predictions for the dominant regime
        epistemic = float(np.std(p_day[:, dominant_idx]))

        # Aleatoric: average per-member Bernoulli dispersion
        aleatoric = float(np.mean(p_day[:, dominant_idx] * (1.0 - p_day[:, dominant_idx])))

        # Find dominant model (highest weight)
        dom_model = max(self.weights.items(), key=lambda x: x[1])[0]

        contract = {
            "date": date_str,
            "regime_probs": {REGIME_NAMES[k]: float(np.round(ensembled_p[k], 5)) for k in range(K)},
            "dominant_regime": dominant_regime,
            "dominant_model": dom_model,
            "ensemble_weights": self.weights,
            "epistemic_uncertainty": float(np.round(epistemic, 5)),
            "aleatoric_uncertainty": float(np.round(aleatoric, 5)),
            "source": "ENSEMBLE_v1.0",
        }
        return contract
