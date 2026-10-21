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


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    r"""
    Expected Calibration Error (ECE; Guo et al. 2017; PDF Section A14):
        $$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
    where confidence is $\max_k p_{ik}$ and accuracy is $\mathbb{I}(\hat{y}_i = y_i)$.
    """
    N = len(y_true)
    if N == 0:
        return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for b in range(n_bins):
        in_bin = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(acc_in_bin - conf_in_bin)

    return float(ece)


def decompose_uncertainty_entropy(
    model_probs: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    r"""
    Information-Theoretic Uncertainty Decomposition into Epistemic and Aleatoric entropy (PDF Section A10.3):
        Ensemble Mixture: $$\bar{p}_k = \sum_{m=1}^M w_m p_{mk}$$
        Total Uncertainty (Total Entropy):
            $$H(\bar{p}) = -\sum_{k=0}^{K-1} \bar{p}_k \ln \bar{p}_k$$
        Aleatoric Uncertainty (Expected Entropy):
            $$\mathbb{E}[H] = \sum_{m=1}^M w_m H(p_m) = -\sum_{m=1}^M w_m \sum_{k=0}^{K-1} p_{mk} \ln p_{mk}$$
        Epistemic Uncertainty (Mutual Information):
            $$I(Y; \Theta) = H(\bar{p}) - \mathbb{E}[H]$$
    Satisfies exact identity: $$H = \mathbb{E}[H] + I$$
    """
    # model_probs: (M, K) or (M, N, K)
    if model_probs.ndim == 2:
        M, K = model_probs.shape
        w = np.ones(M) / M if weights is None else np.asarray(weights, dtype=float)
        w = w / np.sum(w)

        # Mixture (K,)
        p_bar = np.clip(w @ model_probs, eps, 1.0)
        p_bar = p_bar / np.sum(p_bar)
        total_entropy = float(-np.sum(p_bar * np.log(p_bar)))

        # Member entropies
        p_clipped = np.clip(model_probs, eps, 1.0)
        member_entropies = -np.sum(p_clipped * np.log(p_clipped), axis=1)  # (M,)
        aleatoric_entropy = float(np.sum(w * member_entropies))

        # Mutual information (Epistemic)
        epistemic_mi = float(max(0.0, total_entropy - aleatoric_entropy))

        return {
            "total_entropy": float(np.round(total_entropy, 6)),
            "aleatoric_entropy": float(np.round(aleatoric_entropy, 6)),
            "epistemic_mutual_info": float(np.round(epistemic_mi, 6)),
            "identity_verified": bool(abs(total_entropy - (aleatoric_entropy + epistemic_mi)) < 1e-5),
        }
    elif model_probs.ndim == 3:
        M, N, K = model_probs.shape
        w = np.ones(M) / M if weights is None else np.asarray(weights, dtype=float)
        w = w / np.sum(w)

        # Mixture (N, K)
        p_bar = np.clip(np.tensordot(w, model_probs, axes=(0, 0)), eps, 1.0)
        p_bar = p_bar / np.sum(p_bar, axis=1, keepdims=True)
        total_entropy = -np.sum(p_bar * np.log(p_bar), axis=1)  # (N,)

        p_clipped = np.clip(model_probs, eps, 1.0)
        member_h = -np.sum(p_clipped * np.log(p_clipped), axis=-1)  # (M, N)
        aleatoric_entropy = np.tensordot(w, member_h, axes=(0, 0))  # (N,)
        epistemic_mi = np.maximum(0.0, total_entropy - aleatoric_entropy)  # (N,)

        return {
            "total_entropy": total_entropy,
            "aleatoric_entropy": aleatoric_entropy,
            "epistemic_mutual_info": epistemic_mi,
            "mean_total_entropy": float(np.mean(total_entropy)),
            "mean_aleatoric_entropy": float(np.mean(aleatoric_entropy)),
            "mean_epistemic_mutual_info": float(np.mean(epistemic_mi)),
        }
    else:
        raise ValueError(f"Invalid model_probs dimension: {model_probs.ndim}")


def generate_reliability_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    save_path: str = "reports/figures/reliability_curve.png",
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Generate reliability diagram and save figure."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs, bin_accs, bin_counts = [], [], []

    for b in range(n_bins):
        in_bin = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
        if in_bin.sum() > 0:
            bin_confs.append(float(np.mean(confidences[in_bin])))
            bin_accs.append(float(np.mean(accuracies[in_bin])))
            bin_counts.append(int(in_bin.sum()))

    ece = compute_ece(probs, y_true, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    if bin_confs:
        ax.plot(bin_confs, bin_accs, "s-", color="#1f77b4", label=f"Ensemble (ECE={ece:.3f})")
        ax.bar(bin_confs, bin_accs, width=0.05, alpha=0.2, color="#1f77b4")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (5-Regime Stacking Ensemble)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    return {"ece": ece, "bin_confs": bin_confs, "bin_accs": bin_accs, "figure_path": save_path}


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
        assert np.isclose(np.sum(self.weights), 1.0, atol=1e-4), "Stacking weights must sum to 1 within 1e-4"
        assert (self.weights >= -1e-6).all(), "Stacking weights must be non-negative"
        return self.weights

    def predict_proba(self, base_probs: np.ndarray) -> np.ndarray:
        """
        Combine base probabilities of shape (M, N, K) into (N, K).
        """
        if self.weights is None:
            raise ValueError("Stacker is not fitted.")
        combined = np.tensordot(self.weights, base_probs, axes=(0, 0))
        return combined / np.sum(combined, axis=-1, keepdims=True)


class ClimatologyBenchmark:
    r"""
    Climatology Benchmark (PDF Section A14.4):
    Predicts unconditional historical empirical frequencies $\hat{\pi}_k = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i = k)$.
    """
    def __init__(self, K: int = 5) -> None:
        self.K = K
        self.probs_: np.ndarray = np.ones(K) / K

    def fit(self, y_train: np.ndarray) -> ClimatologyBenchmark:
        counts = np.bincount(y_train, minlength=self.K)
        self.probs_ = counts / np.maximum(np.sum(counts), 1)
        return self

    def predict_proba(self, N: int) -> np.ndarray:
        return np.tile(self.probs_, (N, 1))


class PersistenceBenchmark:
    r"""
    Persistence Benchmark (PDF Section A14.4):
    Predicts tomorrow's regime equals today's observed regime ($s_t = s_{t-1}$).
    """
    def __init__(self, K: int = 5, eps: float = 0.01) -> None:
        self.K = K
        self.eps = eps

    def predict_proba(self, y_prev: np.ndarray) -> np.ndarray:
        N = len(y_prev)
        probs = np.full((N, self.K), self.eps / (self.K - 1))
        for i in range(N):
            probs[i, y_prev[i]] = 1.0 - self.eps
        return probs / np.sum(probs, axis=1, keepdims=True)


class VolThresholdBenchmark:
    r"""
    20-day Realised Volatility Threshold Benchmark:
    Maps 20-day rolling realised volatility quantiles into 5 regimes.
    """
    def __init__(self, K: int = 5) -> None:
        self.K = K
        self.quantiles: Optional[np.ndarray] = None

    def fit(self, vol_20d: np.ndarray) -> VolThresholdBenchmark:
        v = vol_20d[~np.isnan(vol_20d)]
        self.quantiles = np.percentile(v, [25, 50, 75, 90])
        return self

    def predict_proba(self, vol_20d: np.ndarray) -> np.ndarray:
        N = len(vol_20d)
        probs = np.full((N, self.K), 0.05 / (self.K - 1))
        for i, val in enumerate(vol_20d):
            if np.isnan(val) or self.quantiles is None:
                probs[i] = 1.0 / self.K
            elif val <= self.quantiles[0]:
                probs[i, 0] = 0.95  # Risk-On (low vol)
            elif val <= self.quantiles[1]:
                probs[i, 1] = 0.95  # Late-Cycle
            elif val <= self.quantiles[2]:
                probs[i, 2] = 0.95  # Transitional
            elif val <= self.quantiles[3]:
                probs[i, 3] = 0.95  # Post-Shock
            else:
                probs[i, 4] = 0.95  # Risk-Off (crisis vol spike)
        return probs / np.sum(probs, axis=1, keepdims=True)


def log_ensemble_to_mlflow(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    run_name: str = "ensemble_evaluation",
) -> None:
    """Log scoring rules and ensemble weights to MLflow with offline fallback."""
    try:
        import mlflow
        with mlflow.start_run(run_name=run_name, nested=True):
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            if weights:
                for m, w in weights.items():
                    mlflow.log_metric(f"weight_{m}", float(w))
    except Exception as e:
        logger.info("MLflow logging skipped (offline/local fallback): %s", e)


class EnsembleRegimeEngine:
    r"""
    Unified Regime Ensemble Engine (PDF Section A10.4 & A11.2).
    Coordinates base models, computes stacking/BMA mixtures, decomposes
    uncertainty into Epistemic (mutual info) and Aleatoric (expected entropy),
    and emits the Daily Combined Regime Output Contract.
    """

    def __init__(self, model_names: List[str], seed: int = 42) -> None:
        self.model_names = model_names
        self.seed = seed
        self.stacker = ConstrainedStacker(seed=seed)
        self.bma = BayesianModelAverager()
        self.weights: Dict[str, float] = {m: 1.0 / len(model_names) for m in model_names}
        self.admitted_models: List[str] = list(model_names)
        self.member_calibration_records: Dict[str, Dict[str, float]] = {}

    def admit_members_and_log_mlflow(
        self,
        base_probs: np.ndarray,
        y_true: np.ndarray,
        waic_scores: Optional[Dict[str, float]] = None,
        max_allowed_ece: float = 0.20,
    ) -> List[str]:
        r"""
        Audit and log calibration metrics (ECE, Brier, RPS, Log Loss, WAIC)
        to MLflow for every candidate member model BEFORE admission to the ensemble.
        """
        M, N, K = base_probs.shape
        admitted = []

        for m_idx, name in enumerate(self.model_names):
            p_m = base_probs[m_idx]
            ece = compute_ece(p_m, y_true, n_bins=10)
            brier = compute_brier_score(p_m, y_true, K=K)
            rps = compute_ranked_probability_score(p_m, y_true, K=K)
            ll = compute_log_loss(p_m, y_true)
            waic = waic_scores.get(name, 0.0) if waic_scores else 0.0

            cal_metrics = {
                "ece": float(np.round(ece, 4)),
                "brier": float(np.round(brier, 4)),
                "rps": float(np.round(rps, 4)),
                "log_loss": float(np.round(ll, 4)),
                "waic": float(np.round(waic, 2)),
                "admitted": float(1.0 if ece <= max_allowed_ece else 0.0),
            }
            self.member_calibration_records[name] = cal_metrics

            # Log member model to MLflow BEFORE admission
            log_ensemble_to_mlflow(cal_metrics, run_name=f"member_audit_{name}")
            logger.info("Member audit '%s': ECE=%.4f, Brier=%.4f, RPS=%.4f, Admitted=%s",
                        name, ece, brier, rps, ece <= max_allowed_ece)

            if ece <= max_allowed_ece:
                admitted.append(name)

        self.admitted_models = admitted if admitted else list(self.model_names)
        return self.admitted_models

    def fit_ensemble(
        self,
        base_probs: np.ndarray,
        y_true: np.ndarray,
        method: str = "stacking",
        waic_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Fit ensemble weights using WAIC/LOO-guided BMA or constrained stacking.
        """
        # Audit and log members to MLflow before admission
        self.admit_members_and_log_mlflow(base_probs, y_true, waic_scores=waic_scores)

        M, N, K = base_probs.shape
        if method == "stacking":
            w = self.stacker.fit(base_probs, y_true)
        else:
            log_liks = []
            for m in range(M):
                name = self.model_names[m]
                if waic_scores and name in waic_scores:
                    # WAIC-guided prior evidence: -0.5 * WAIC
                    ll = -0.5 * float(waic_scores[name])
                else:
                    ll = -compute_log_loss(base_probs[m], y_true) * len(y_true)
                log_liks.append(ll)
            w = self.bma.fit(np.array(log_liks))

        self.weights = {self.model_names[i]: float(np.round(w[i], 4)) for i in range(M)}
        log_ensemble_to_mlflow(self.weights, run_name="ensemble_weights_admitted")
        return self.weights

    def predict_contract(
        self,
        model_probs: np.ndarray,
        date_str: str = "2026-10-19",
        dominant_override: Optional[str] = None,
    ) -> Dict[str, Any]:
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

        # Information-theoretic uncertainty decomposition: H = E[H] + I
        decomp = decompose_uncertainty_entropy(p_day, weights=w_vec)

        # Find dominant model (highest weight)
        dom_model = max(self.weights.items(), key=lambda x: x[1])[0]

        contract = {
            "date": date_str,
            "regime_probs": {REGIME_NAMES[k]: float(np.round(ensembled_p[k], 5)) for k in range(K)},
            "dominant_regime": dominant_regime,
            "dominant_model": dom_model,
            "ensemble_weights": self.weights,
            "epistemic_uncertainty": decomp["epistemic_mutual_info"],
            "aleatoric_uncertainty": decomp["aleatoric_entropy"],
            "total_entropy": decomp["total_entropy"],
            "uncertainty_budget": {
                "total_entropy": decomp["total_entropy"],
                "aleatoric_entropy": decomp["aleatoric_entropy"],
                "epistemic_mutual_info": decomp["epistemic_mutual_info"],
                "identity_verified": decomp["identity_verified"],
            },
            "source": "ENSEMBLE_v1.0",
        }
        return contract
