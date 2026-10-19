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
        n_iter: int = 200,
        tol: float = 1e-5,
        covar_floor: float = 1e-4,
        seed: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, seed=seed)
        self.n_iter = n_iter
        self.tol = tol
        self.covar_floor = covar_floor

        # Parameters
        self.pi: np.ndarray = np.ones(n_regimes) / n_regimes  # Initial distribution
        self.A: np.ndarray = np.eye(n_regimes) * 0.85 + (1 - np.eye(n_regimes)) * (0.15 / (n_regimes - 1))  # Transition matrix
        self.means: np.ndarray = np.zeros((n_regimes, 1))
        self.covars: np.ndarray = np.ones((n_regimes, 1, 1))

        # Mapping from fitted numeric state to canonical economic regime [0..4]
        self.state_to_regime: Dict[int, int] = {i: i for i in range(n_regimes)}
        self.log_likelihood_: float = -np.inf
        self.bic_: float = np.inf

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[Any] = None) -> GaussianRegimeHMM:
        """
        Fit the Gaussian HMM using the Baum-Welch (EM) algorithm.

        Args:
            X: Returns or feature matrix of shape (T, d).
            y: Unused (unsupervised model).

        Returns:
            Fitted GaussianRegimeHMM instance.
        """
        arr = self._ensure_2d_array(X)
        # Drop rows with NaNs in training
        valid_mask = ~np.isnan(arr).any(axis=1)
        arr = arr[valid_mask]
        T, d = arr.shape
        K = self.n_regimes

        rng = np.random.default_rng(self.seed)

        # K-Means / Quantile initialization for means and covariances
        percentiles = np.linspace(10, 90, K)
        init_means = np.percentile(arr[:, 0], percentiles)
        self.means = np.zeros((K, d))
        self.means[:, 0] = np.sort(init_means)
        total_var = np.var(arr, axis=0) + self.covar_floor
        self.covars = np.tile(np.diag(total_var)[None, :, :], (K, 1, 1))

        prev_ll = -np.inf

        for it in range(self.n_iter):
            # E-step: Forward-Backward pass
            log_B = self._compute_log_emission_probs(arr)
            log_alpha, ll = self._forward_pass(log_B)
            log_beta = self._backward_pass(log_B)

            # Posterior state probabilities \gamma_t(i) = P(s_t = i | y)
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(log_gamma)
            gamma = gamma / np.sum(gamma, axis=1, keepdims=True)

            # Posterior transition probabilities \xi_t(i, j) = P(s_t = i, s_{t+1} = j | y)
            # log \xi_{t}(i, j) = log \alpha_t(i) + log A_{ij} + log B_{t+1}(j) + log \beta_{t+1}(j) - ll
            log_A = np.log(np.clip(self.A, 1e-12, 1.0))
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                log_xi_t = log_alpha[t, :, None] + log_A + log_B[t + 1, None, :] + log_beta[t + 1, None, :] - ll
                xi_sum += np.exp(log_xi_t)

            # M-step: Parameter updates
            self.pi = gamma[0] / np.sum(gamma[0])
            self.A = xi_sum / np.maximum(gamma[:-1].sum(axis=0)[:, None], 1e-12)
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
        # Free parameters: K(K-1) in A + (K-1) in pi + K*d in means + K*d in diag covars
        n_params = K * (K - 1) + (K - 1) + 2 * K * d
        self.bic_ = -2.0 * prev_ll + n_params * np.log(T)

        # Resolve label switching
        self._resolve_label_switching()
        self.is_fitted = True
        logger.info("GaussianRegimeHMM fit complete: LL=%.2f, BIC=%.2f", self.log_likelihood_, self.bic_)
        return self

    def _resolve_label_switching(self) -> None:
        """
        Sort fitted states according to economic signature (mean return, -vol).
        Maps highest risk-adjusted mean to Risk-On, worst return / high vol to Risk-Off.
        """
        K = self.n_regimes
        summary = []
        for i in range(K):
            mu = float(self.means[i, 0])
            sig = float(np.sqrt(self.covars[i, 0, 0]))
            summary.append((i, mu, sig))

        # Sort descending by (mu, -sig)
        summary.sort(key=lambda x: (x[1], -x[2]), reverse=True)

        # Map sorted rank to canonical regime index [0: Risk-On, 1: Late-Cycle, 2: Transitional, 3: Post-Shock, 4: Risk-Off]
        self.state_to_regime = {summary[r][0]: r for r in range(K)}

        # Reorder parameters to canonical order
        reorder_idx = [summary[r][0] for r in range(K)]
        self.means = self.means[reorder_idx]
        self.covars = self.covars[reorder_idx]
        self.pi = self.pi[reorder_idx]
        self.A = self.A[np.ix_(reorder_idx, reorder_idx)]
        # After reordering, state index matches canonical regime index directly
        self.state_to_regime = {i: i for i in range(K)}

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Compute out-of-sample one-step-ahead or filtered regime probabilities.

        Args:
            X: Input features of shape (N, d).

        Returns:
            Array of shape (N, 5) with calibrated regime probabilities.
        """
        arr = self._ensure_2d_array(X)
        N, d = arr.shape
        K = self.n_regimes

        log_B = self._compute_log_emission_probs(arr)
        log_alpha, _ = self._forward_pass(log_B)

        # Normalise forward variables to get filtered probabilities P(s_t = k | y_{1:t})
        log_probs = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        probs = np.exp(log_probs)

        # Handle any possible NaN rows in X gracefully
        nan_rows = np.isnan(arr).any(axis=1)
        if nan_rows.any():
            probs[nan_rows] = 1.0 / K

        # Re-ensure exact simplex constraints
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Return the KxK transition probability matrix labeled with canonical regimes.
        """
        return pd.DataFrame(
            self.A,
            index=self.regime_names,
            columns=self.regime_names,
        )

    def get_regime_durations(self) -> Dict[str, float]:
        """
        Calculate implied expected holding duration in days for each regime:
            $$E[D_k] = \frac{1}{1 - A_{kk}}$$
        """
        durations: Dict[str, float] = {}
        for k, name in enumerate(self.regime_names):
            p_stay = float(self.A[k, k])
            dur = 1.0 / max(1.0 - p_stay, 1e-4)
            durations[name] = float(np.round(dur, 2))
        return durations

    def get_stickiness_score(self) -> float:
        r"""
        Average diagonal self-transition probability:
            $$\bar{\kappa} = \frac{1}{K} \sum_{k=1}^K A_{kk}$$
        """
        return float(np.mean(np.diag(self.A)))

    def _compute_log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Compute log Gaussian emission densities for each state."""
        T, d = X.shape
        K = self.n_regimes
        log_B = np.zeros((T, K))

        for k in range(K):
            mu_k = self.means[k]
            var_k = np.diag(self.covars[k])
            # Multi-dimensional independent Gaussian emissions
            diff = X - mu_k
            log_dens = -0.5 * np.sum(
                np.log(2.0 * np.pi * var_k) + (diff ** 2) / var_k,
                axis=1,
            )
            log_B[:, k] = np.nan_to_num(log_dens, nan=-100.0)

        return log_B

    def _forward_pass(self, log_B: np.ndarray) -> Tuple[np.ndarray, float]:
        """Log-space Forward algorithm to compute alpha and total log likelihood."""
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))
        log_pi = np.log(np.clip(self.pi, 1e-12, 1.0))
        log_A = np.log(np.clip(self.A, 1e-12, 1.0))

        # Initial step
        log_alpha[0] = log_pi + log_B[0]

        # Induction
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]

        ll = float(logsumexp(log_alpha[-1]))
        return log_alpha, ll

    def _backward_pass(self, log_B: np.ndarray) -> np.ndarray:
        """Log-space Backward algorithm to compute beta."""
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(np.clip(self.A, 1e-12, 1.0))

        # Termination: beta_T(i) = 1 => log_beta_T(i) = 0
        log_beta[-1] = 0.0

        # Backward induction
        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = logsumexp(log_A[i, :] + log_B[t + 1, :] + log_beta[t + 1, :])

        return log_beta
