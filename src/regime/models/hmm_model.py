"""
Gaussian Hidden Markov Model for Indian Equity Regime Detection.
Implements Expectation-Maximisation (Baum-Welch), Viterbi state decoding,
BIC model selection, and economic moment sorting to resolve label switching.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from .base import BaseRegimeModel, REGIME_NAMES

logger = logging.getLogger(__name__)


class GaussianRegimeHMM(BaseRegimeModel):
    r"""
    5-State Gaussian Emission Hidden Markov Model.

    Maths:
        Hidden Markov state $s_t \in \{0, \dots, K-1\}$, with transition matrix $A_{ij} = P(s_{t+1}=j \mid s_t=i)$.
        Emission distribution:
            $$y_t \mid s_t = k \sim \mathcal{N}(\mu_k, \Sigma_k)$$
        Marginal likelihood via Forward Algorithm:
            $$\alpha_t(j) = p(y_t \mid s_t=j) \sum_{i=0}^{K-1} \alpha_{t-1}(i) A_{ij}$$
        Bayesian Information Criterion (BIC):
            $$\text{BIC} = -2 \ln \mathcal{L} + p \ln(T)$$
        where $p = K(K-1) + (K-1) + 2Kd$ is the count of free parameters.

    Label Switching Resolution:
        States are sorted post-hoc by economic signature $(\mu_k, -\sigma_k)$:
        1. Risk-On: highest return, low/moderate volatility.
        2. Late-Cycle: positive return, low/moderate volatility, fatigue.
        3. Transitional: moderate return, intermediate volatility.
        4. Post-Shock: recovery return, elevated volatility.
        5. Risk-Off: negative return, highest volatility.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        n_iter: int = 60,
        tol: float = 1e-4,
        covar_floor: float = 2.5e-5,
        sticky_kappa: float = 25.0,
        dirichlet_prior: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, seed=seed)
        self.n_iter = n_iter
        self.tol = tol
        # Strict variance floor: daily vol >= 0.0050 (annualized >= 7.94%).
        # Never accept a collapsed state with near-zero variance or volatility 0.001.
        self.covar_floor = max(covar_floor, 2.5e-5)
        self.sticky_kappa = sticky_kappa
        self.dirichlet_prior = dirichlet_prior

        # Parameters
        self.pi: np.ndarray = np.ones(n_regimes) / n_regimes  # Initial distribution
        self.A: np.ndarray = np.eye(n_regimes) * 0.85 + (1 - np.eye(n_regimes)) * (0.15 / (n_regimes - 1))  # Transition matrix
        self.means: np.ndarray = np.zeros((n_regimes, 1))
        self.covars: np.ndarray = np.ones((n_regimes, 1, 1))

        # Mapping from fitted numeric state to canonical economic regime [0..4]
        self.state_to_regime: Dict[int, int] = {i: i for i in range(n_regimes)}
        self.log_likelihood_: float = -np.inf
        self.bic_: float = np.inf

    @staticmethod
    def _fast_logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
        """Fast vectorized logsumexp along specified axis."""
        a_max = np.max(a, axis=axis, keepdims=True)
        out = a_max + np.log(np.sum(np.exp(np.clip(a - a_max, -700, 0)), axis=axis, keepdims=True))
        return np.squeeze(out, axis=axis)

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[Any] = None) -> GaussianRegimeHMM:
        """
        Fit the Gaussian HMM using the Baum-Welch (EM) algorithm with sticky Dirichlet prior.

        Args:
            X: Returns or feature matrix of shape (T, d).
            y: Unused (unsupervised model).

        Returns:
            Fitted GaussianRegimeHMM instance.
        """
        arr = self._ensure_2d_array(X)
        valid_mask = ~np.isnan(arr).any(axis=1)
        arr = arr[valid_mask]
        T, d = arr.shape
        K = self.n_regimes

        rng = np.random.default_rng(self.seed)

        # Realistic economic initialization across return and volatility regimes
        self.means = np.zeros((K, d))
        self.covars = np.zeros((K, d, d))

        # Volatility proxy
        vol_1d = np.abs(arr[:, 0])
        q_bins = np.quantile(vol_1d, np.linspace(0.0, 1.0, K + 1))

        # Distinct initialisation: Risk-On (low vol, positive), to Risk-Off (high vol, negative)
        init_mus = [0.0012, 0.0006, 0.0001, -0.0005, -0.0025]
        init_vols = [0.006, 0.008, 0.010, 0.015, 0.022]

        for k in range(K):
            idx = (vol_1d >= q_bins[k]) & (vol_1d <= q_bins[k + 1])
            if np.sum(idx) > 20:
                self.means[k, 0] = np.mean(arr[idx, 0])
                var_est = np.var(arr[idx, 0])
            else:
                self.means[k, 0] = init_mus[k % len(init_mus)]
                var_est = (init_vols[k % len(init_vols)]) ** 2

            for dim in range(d):
                self.covars[k, dim, dim] = max(var_est, self.covar_floor)

        prev_ll = -np.inf

        for it in range(self.n_iter):
            # E-step: Forward-Backward pass
            log_B = self._compute_log_emission_probs(arr)
            log_alpha, ll = self._forward_pass(log_B)
            log_beta = self._backward_pass(log_B)

            # Posterior state probabilities gamma_t(i) = P(s_t = i | y)
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(np.clip(log_gamma, -700, 0))
            gamma_sum = np.sum(gamma, axis=1, keepdims=True)
            gamma = np.where(gamma_sum > 0, gamma / gamma_sum, 1.0 / K)

            # Vectorized posterior transition probabilities xi_t(i, j)
            log_A = np.log(np.clip(self.A, 1e-12, 1.0))
            log_xi = (
                log_alpha[:-1, :, None]
                + log_A[None, :, :]
                + log_B[1:, None, :]
                + log_beta[1:, None, :]
                - ll
            )
            xi = np.exp(np.clip(log_xi, -700, 0))
            xi_sum = np.sum(xi, axis=0)

            # M-step: Parameter updates with sticky Dirichlet prior
            self.pi = gamma[0] / max(np.sum(gamma[0]), 1e-12)
            prior_matrix = np.full((K, K), self.dirichlet_prior) + np.eye(K) * self.sticky_kappa
            self.A = (xi_sum + prior_matrix) / (
                gamma[:-1].sum(axis=0)[:, None] + np.sum(prior_matrix, axis=1, keepdims=True)
            )
            self.A = self.A / np.sum(self.A, axis=1, keepdims=True)

            # Update means and diagonal covariances
            for k in range(K):
                gamma_k = gamma[:, k]
                weight_k = np.sum(gamma_k) + 1e-12
                self.means[k] = np.sum(gamma_k[:, None] * arr, axis=0) / weight_k

                diff = arr - self.means[k]
                cov_k = np.zeros((d, d))
                for dim in range(d):
                    var_dim = np.sum(gamma_k * (diff[:, dim] ** 2)) / weight_k
                    cov_k[dim, dim] = max(var_dim, self.covar_floor)
                self.covars[k] = cov_k

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.log_likelihood_ = prev_ll
        n_params = K * (K - 1) + (K - 1) + 2 * K * d
        self.bic_ = -2.0 * prev_ll + n_params * np.log(T)

        # Resolve label switching
        self._resolve_label_switching()
        self.is_fitted = True
        logger.info("GaussianRegimeHMM fit complete: LL=%.2f, BIC=%.2f", self.log_likelihood_, self.bic_)
        return self

    def _resolve_label_switching(self) -> None:
        """
        Sort fitted states according to Sharpe/economic signature (mu / sigma).
        Maps highest risk-adjusted mean to Risk-On, worst return / severe vol to Risk-Off.
        """
        K = self.n_regimes
        summary = []
        for i in range(K):
            mu = float(self.means[i, 0])
            sig = float(np.sqrt(self.covars[i, 0, 0]))
            # Risk-adjusted signature: mu / sig
            sharpe_sig = (mu / max(sig, 1e-6))
            summary.append((i, mu, sig, sharpe_sig))

        # Sort descending by Sharpe signature: highest Sharpe (Risk-On) -> lowest Sharpe (Risk-Off)
        summary.sort(key=lambda x: x[3], reverse=True)

        reorder_idx = [summary[r][0] for r in range(K)]
        self.means = self.means[reorder_idx]
        self.covars = self.covars[reorder_idx]
        self.pi = self.pi[reorder_idx]
        self.A = self.A[np.ix_(reorder_idx, reorder_idx)]
        self.state_to_regime = {i: i for i in range(K)}

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Compute out-of-sample filtered regime probabilities P(s_t = k | y_{1:t}).
        """
        arr = self._ensure_2d_array(X)
        N, d = arr.shape
        K = self.n_regimes

        log_B = self._compute_log_emission_probs(arr)
        log_alpha, _ = self._forward_pass(log_B)

        log_probs = log_alpha - np.max(log_alpha, axis=1, keepdims=True)
        probs = np.exp(np.clip(log_probs, -700, 0))
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        nan_rows = np.isnan(arr).any(axis=1)
        if nan_rows.any():
            probs[nan_rows] = 1.0 / K

        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs

    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the KxK transition probability matrix labeled with canonical regimes."""
        return pd.DataFrame(
            self.A,
            index=self.regime_names,
            columns=self.regime_names,
        )

    def get_regime_durations(self) -> Dict[str, float]:
        """Calculate implied expected holding duration in days for each regime."""
        durations: Dict[str, float] = {}
        for k, name in enumerate(self.regime_names):
            p_stay = float(self.A[k, k])
            dur = 1.0 / max(1.0 - p_stay, 1e-4)
            durations[name] = float(np.round(dur, 2))
        return durations

    def get_stickiness_score(self) -> float:
        """Average diagonal self-transition probability."""
        return float(np.mean(np.diag(self.A)))

    def _compute_log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Compute log Gaussian emission densities for each state."""
        T, d = X.shape
        K = self.n_regimes
        log_B = np.zeros((T, K))

        for k in range(K):
            mu_k = self.means[k]
            var_k = np.diag(self.covars[k])
            diff = X - mu_k
            log_dens = -0.5 * np.sum(
                np.log(2.0 * np.pi * var_k) + (diff ** 2) / var_k,
                axis=1,
            )
            log_B[:, k] = np.nan_to_num(log_dens, nan=-100.0)

        return log_B

    def _forward_pass(self, log_B: np.ndarray) -> Tuple[np.ndarray, float]:
        """Log-space Forward algorithm using fast vector logsumexp."""
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))
        log_pi = np.log(np.clip(self.pi, 1e-12, 1.0))
        log_A = np.log(np.clip(self.A, 1e-12, 1.0))

        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = self._fast_logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0) + log_B[t]

        ll = float(self._fast_logsumexp(log_alpha[-1], axis=0))
        return log_alpha, ll

    def _backward_pass(self, log_B: np.ndarray) -> np.ndarray:
        """Log-space Backward algorithm using fast vector logsumexp."""
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(np.clip(self.A, 1e-12, 1.0))

        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = self._fast_logsumexp(log_A + log_B[t + 1] + log_beta[t + 1], axis=1)

        return log_beta
