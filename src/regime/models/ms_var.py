"""
Markov-Switching Vector Autoregression (RS-VAR) for Multivariate Market Dynamics.
Models the joint regime-dependent dynamics of Nifty returns, realised volatility,
market breadth, institutional flows, currency, and bond yields.
Computes regime-conditional innovation covariances and impulse responses (PDF Section A8).
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


class MarkovSwitchingVAR(BaseRegimeModel):
    r"""
    Multivariate Regime-Switching Vector Autoregression (RS-VAR(1)).

    Maths:
        For multivariate feature vector $\mathbf{y}_t \in \mathbb{R}^d$:
        $$\mathbf{y}_t = \mathbf{c}_{s_t} + \mathbf{A}_{s_t} \mathbf{y}_{t-1} + \boldsymbol{\epsilon}_t$$
        $$\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{s_t})$$
        where latent state $s_t \in \{0, \dots, K-1\}$ follows a Markov chain with transition matrix $\mathbf{P}$.

    Cross-Asset Conditioning:
        In Risk-Off, cross-asset correlations spike, volatility innovations persist,
        and FII outflows induce amplified downward momentum compared to Risk-On.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        lags: int = 1,
        n_iter: int = 80,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, seed=seed)
        self.lags = lags
        self.n_iter = n_iter
        self.tol = tol

        # Model parameters
        self.P: np.ndarray = np.eye(n_regimes) * 0.88 + (1 - np.eye(n_regimes)) * (0.12 / (n_regimes - 1))
        self.pi: np.ndarray = np.ones(n_regimes) / n_regimes
        self.intercepts: Optional[np.ndarray] = None  # (K, d)
        self.AR: Optional[np.ndarray] = None  # (K, d, d)
        self.covars: Optional[np.ndarray] = None  # (K, d, d)

        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[Any] = None) -> MarkovSwitchingVAR:
        """
        Fit RS-VAR(1) using multivariate Expectation-Maximisation.

        Args:
            X: Feature matrix of shape (T, d).
            y: Unused.

        Returns:
            Fitted MarkovSwitchingVAR instance.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feat_{i}" for i in range(X.shape[1] if X.ndim > 1 else 1)]

        arr = self._ensure_2d_array(X)
        valid_mask = ~np.isnan(arr).any(axis=1)
        data = arr[valid_mask]
        T, d = data.shape
        K = self.n_regimes

        rng = np.random.default_rng(self.seed)

        # Pre-align lags: Y_t (T-1, d), Y_{t-1} (T-1, d)
        Y = data[1:]
        Y_lag = data[:-1]
        T_eff = len(Y)

        # Initialise regime clustering based on first component (returns)
        pct_quantiles = np.linspace(15, 85, K)
        thresholds = np.percentile(Y[:, 0], pct_quantiles)
        clusters = np.digitize(Y[:, 0], thresholds)

        self.intercepts = np.zeros((K, d))
        self.AR = np.zeros((K, d, d))
        self.covars = np.zeros((K, d, d))

        for k in range(K):
            sub_mask = (clusters == k)
            if np.sum(sub_mask) < d + 2:
                sub_mask = np.ones(T_eff, dtype=bool)

            Y_k = Y[sub_mask]
            Y_lag_k = Y_lag[sub_mask]

            # Ridge regression for initial VAR parameters
            reg_lambda = 0.1
            X_design = np.hstack([np.ones((len(Y_lag_k), 1)), Y_lag_k])
            XtX = X_design.T @ X_design + reg_lambda * np.eye(d + 1)
            XtY = X_design.T @ Y_k
            beta = np.linalg.solve(XtX, XtY)  # (d+1, d)

            self.intercepts[k] = beta[0]
            self.AR[k] = beta[1:].T  # (d, d)

            resids = Y_k - (X_design @ beta)
            cov_k = (resids.T @ resids) / max(len(Y_k), 1) + 1e-4 * np.eye(d)
            self.covars[k] = cov_k

        # EM iterations
        prev_ll = -np.inf
        for it in range(self.n_iter):
            # Compute log multivariate normal likelihood for each time step and regime
            log_B = np.zeros((T_eff, K))
            for k in range(K):
                mu_pred = self.intercepts[k] + Y_lag @ self.AR[k].T  # (T_eff, d)
                diff = Y - mu_pred

                # Precision and log determinant
                cov_reg = self.covars[k] + 1e-4 * np.eye(d)
                sign, logdet = np.linalg.slogdet(cov_reg)
                if sign <= 0:
                    cov_reg += 1e-3 * np.eye(d)
                    _, logdet = np.linalg.slogdet(cov_reg)

                inv_cov = np.linalg.pinv(cov_reg)
                quad = np.sum((diff @ inv_cov) * diff, axis=1)
                log_B[:, k] = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

            # Forward pass
            log_alpha = np.zeros((T_eff, K))
            log_pi = np.log(np.clip(self.pi, 1e-12, 1.0))
            log_P = np.log(np.clip(self.P, 1e-12, 1.0))

            log_alpha[0] = log_pi + log_B[0]
            for t in range(1, T_eff):
                log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_P, axis=0) + log_B[t]

            ll = float(logsumexp(log_alpha[-1]))

            # Backward pass
            log_beta = np.zeros((T_eff, K))
            log_beta[-1] = 0.0
            for t in range(T_eff - 2, -1, -1):
                log_beta[t] = logsumexp(log_P + log_B[t + 1, None, :] + log_beta[t + 1, None, :], axis=1)

            # State posteriors gamma_t(k)
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(np.clip(log_gamma, -700, 0))
            gamma = gamma / np.sum(gamma, axis=1, keepdims=True)

            # M-step: update transition matrix P
            xi_sum = np.zeros((K, K))
            for t in range(T_eff - 1):
                log_xi_t = log_alpha[t, :, None] + log_P + log_B[t + 1, None, :] + log_beta[t + 1, None, :] - ll
                xi_sum += np.exp(np.clip(log_xi_t, -700, 0))

            self.P = xi_sum / np.maximum(gamma[:-1].sum(axis=0)[:, None], 1e-12)
            self.P = self.P / np.sum(self.P, axis=1, keepdims=True)
            self.pi = gamma[0] / np.sum(gamma[0])

            # Update VAR parameters per regime with weighted least squares
            X_design = np.hstack([np.ones((T_eff, 1)), Y_lag])
            for k in range(K):
                w = gamma[:, k]
                w_sqrt = np.sqrt(np.maximum(w, 1e-8))[:, None]

                X_w = X_design * w_sqrt
                Y_w = Y * w_sqrt

                XtX = X_w.T @ X_w + 1e-2 * np.eye(d + 1)
                XtY = X_w.T @ Y_w
                beta_k = np.linalg.solve(XtX, XtY)

                self.intercepts[k] = beta_k[0]
                self.AR[k] = beta_k[1:].T

                resids_k = Y - (X_design @ beta_k)
                cov_k = (resids_k.T @ (resids_k * w[:, None])) / np.maximum(np.sum(w), 1e-4) + 1e-4 * np.eye(d)
                self.covars[k] = cov_k

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        # Sort regimes economically: (mu_return, -vol_return)
        summary = [(k, float(self.intercepts[k, 0]), float(np.sqrt(self.covars[k, 0, 0]))) for k in range(K)]
        summary.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        reorder_idx = [s[0] for s in summary]

        self.intercepts = self.intercepts[reorder_idx]
        self.AR = self.AR[reorder_idx]
        self.covars = self.covars[reorder_idx]
        self.P = self.P[np.ix_(reorder_idx, reorder_idx)]
        self.pi = self.pi[reorder_idx]

        self.is_fitted = True
        logger.info("RS-VAR(1) fitted: %d regimes across %d multivariate series.", K, d)
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Compute filtered multivariate regime probabilities P(s_t = k | y_{1:t}).

        Args:
            X: Input multivariate feature matrix of shape (N, d).

        Returns:
            Array of shape (N, 5) with calibrated regime probabilities.
        """
        arr = self._ensure_2d_array(X)
        N, d = arr.shape
        K = self.n_regimes

        if N < 2 or self.intercepts is None or self.covars is None:
            return np.full((N, K), 1.0 / K)

        Y = arr[1:]
        Y_lag = arr[:-1]
        T_eff = len(Y)

        log_B = np.zeros((T_eff, K))
        for k in range(K):
            mu_pred = self.intercepts[k] + Y_lag @ self.AR[k].T
            diff = Y - mu_pred
            cov_reg = self.covars[k] + 1e-4 * np.eye(d)
            inv_cov = np.linalg.pinv(cov_reg)
            _, logdet = np.linalg.slogdet(cov_reg)
            quad = np.sum((diff @ inv_cov) * diff, axis=1)
            log_B[:, k] = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

        log_alpha = np.zeros((T_eff, K))
        log_pi = np.log(np.clip(self.pi, 1e-12, 1.0))
        log_P = np.log(np.clip(self.P, 1e-12, 1.0))

        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T_eff):
            log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_P, axis=0) + log_B[t]

        log_probs = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        probs_eff = np.exp(np.clip(log_probs, -700, 0))
        probs_eff = probs_eff / np.sum(probs_eff, axis=1, keepdims=True)

        # Pad initial step with prior
        out_probs = np.vstack([self.pi[None, :], probs_eff])
        # Handle NaN rows
        nan_rows = np.isnan(arr).any(axis=1)
        if nan_rows.any():
            out_probs[nan_rows] = 1.0 / K
            out_probs = out_probs / np.sum(out_probs, axis=1, keepdims=True)

        return out_probs

    def get_regime_covariance(self, regime_idx: int) -> np.ndarray:
        """Return the (d, d) innovation covariance matrix for regime_idx."""
        if self.covars is None:
            raise ValueError("Model is not fitted.")
        return self.covars[regime_idx]

    def compute_impulse_response(
        self, shock_dim: int = 0, shock_magnitude: float = 1.0, horizon: int = 10
    ) -> Dict[str, np.ndarray]:
        r"""
        Compute regime-conditional impulse responses over a forward horizon:
            $$\mathbf{y}_{t+h} = \mathbf{A}_s^h \boldsymbol{\epsilon}_0$$
        Demonstrates how shocks (e.g. FII outflows or VIX spikes) propagate differently
        in Risk-On vs Risk-Off.
        """
        if self.AR is None or self.covars is None:
            raise ValueError("Model is not fitted.")

        d = self.AR.shape[1]
        irfs: Dict[str, np.ndarray] = {}

        for k, name in enumerate(self.regime_names):
            response = np.zeros((horizon, d))
            # Initial shock vector
            shock = np.zeros(d)
            shock[shock_dim] = shock_magnitude
            response[0] = shock

            A_k = self.AR[k]
            for h in range(1, horizon):
                response[h] = A_k @ response[h - 1]

            irfs[name] = response

        return irfs
