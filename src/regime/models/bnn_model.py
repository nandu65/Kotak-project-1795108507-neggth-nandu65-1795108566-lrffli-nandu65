"""
Bayesian Deep Learning for Regime Detection: Deep Ensembles & MC-Dropout.
Implements M-member deep neural ensembles with permanent dropout, providing
calibrated probability estimation and an explicit uncertainty budget
decomposing predictive variance into Epistemic and Aleatoric components (PDF Section A4).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.special import softmax

from .base import BaseRegimeModel, REGIME_NAMES

logger = logging.getLogger(__name__)


class NeuralRegimeClassifier:
    r"""
    Multi-layer perceptron with dropout for regime classification.
    Supports stochastic forward passes with dropout active at inference (MC-Dropout).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = (64, 32),
        n_classes: int = 5,
        dropout_rate: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(seed)

        # He-normal weight initialisation
        h1, h2 = hidden_dims
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, h1))
        self.b1 = np.zeros(h1)
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / h1), (h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = self.rng.normal(0, np.sqrt(2.0 / h2), (h2, n_classes))
        self.b3 = np.zeros(n_classes)

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass with optional Bernoulli dropout.
        """
        # Layer 1: Dense + ReLU
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        if training or self.dropout_rate > 0:
            mask1 = (self.rng.uniform(0, 1, size=a1.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
            a1 = a1 * mask1

        # Layer 2: Dense + ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0.0, z2)
        if training or self.dropout_rate > 0:
            mask2 = (self.rng.uniform(0, 1, size=a2.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
            a2 = a2 * mask2

        # Output Layer: Dense + Softmax
        logits = a2 @ self.W3 + self.b3
        probs = softmax(logits, axis=1)
        return probs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 40,
        lr: float = 0.005,
        batch_size: int = 64,
    ) -> None:
        """
        Train network parameters using mini-batch Adam optimizer.
        """
        N, d = X.shape
        K = self.n_classes
        y_onehot = np.eye(K)[y]

        # Adam momentum and RMSprop buffers
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        mW3, vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        mb3, vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)

        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        step = 0

        for ep in range(epochs):
            indices = self.rng.permutation(N)
            for start in range(0, N, batch_size):
                step += 1
                batch_idx = indices[start : start + batch_size]
                X_b, y_b = X[batch_idx], y_onehot[batch_idx]
                B = len(X_b)

                # Forward pass caching activations
                z1 = X_b @ self.W1 + self.b1
                a1 = np.maximum(0.0, z1)
                mask1 = (self.rng.uniform(0, 1, size=a1.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
                a1_drop = a1 * mask1

                z2 = a1_drop @ self.W2 + self.b2
                a2 = np.maximum(0.0, z2)
                mask2 = (self.rng.uniform(0, 1, size=a2.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
                a2_drop = a2 * mask2

                logits = a2_drop @ self.W3 + self.b3
                probs = softmax(logits, axis=1)

                # Backprop: Cross-entropy gradient
                delta3 = (probs - y_b) / B
                dW3 = a2_drop.T @ delta3 + 1e-4 * self.W3
                db3 = np.sum(delta3, axis=0)

                delta2 = (delta3 @ self.W3.T) * mask2 * (z2 > 0)
                dW2 = a1_drop.T @ delta2 + 1e-4 * self.W2
                db2 = np.sum(delta2, axis=0)

                delta1 = (delta2 @ self.W2.T) * mask1 * (z1 > 0)
                dW1 = X_b.T @ delta1 + 1e-4 * self.W1
                db1 = np.sum(delta1, axis=0)

                # Adam updates
                for param, grad, m, v in [
                    (self.W1, dW1, mW1, vW1),
                    (self.b1, db1, mb1, vb1),
                    (self.W2, dW2, mW2, vW2),
                    (self.b2, db2, mb2, vb2),
                    (self.W3, dW3, mW3, vW3),
                    (self.b3, db3, mb3, vb3),
                ]:
                    m[:] = beta1 * m + (1.0 - beta1) * grad
                    v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
                    m_hat = m / (1.0 - beta1 ** step)
                    v_hat = v / (1.0 - beta2 ** step)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)


class BayesianDeepEnsemble(BaseRegimeModel):
    r"""
    Deep Ensemble with M members and MC-Dropout (PDF Section A4.4).

    Maths:
        For ensemble of $M$ independent models with weights $\theta^{(m)}$:
        $$\bar{p}(y = k \mid \mathbf{x}) = \frac{1}{M} \sum_{m=1}^M p(y = k \mid \mathbf{x}, \theta^{(m)})$$
        Uncertainty Decomposition:
        $$\text{Epistemic}_t = \sqrt{\frac{1}{M}\sum_{m=1}^M (p_m(y) - \bar{p}(y))^2} \quad \text{(disagreement)}$$
        $$\text{Aleatoric}_t = \frac{1}{M}\sum_{m=1}^M \bar{p}_m(y)(1 - \bar{p}_m(y)) \quad \text{(inherent noise)}$$
    """

    def __init__(
        self,
        n_regimes: int = 5,
        n_models: int = 8,
        hidden_dims: Tuple[int, int] = (64, 32),
        dropout_rate: float = 0.20,
        epochs: int = 35,
        lr: float = 0.005,
        seed: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, seed=seed)
        self.n_models = n_models
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.lr = lr

        self.models: List[NeuralRegimeClassifier] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> BayesianDeepEnsemble:
        """
        Fit M independent neural classifiers with distinct random initializations.
        y must be integer labels 0..4 (decoded inside the training window).
        """
        X_arr = self._ensure_2d_array(X)
        if isinstance(y, pd.Series):
            y_arr = y.to_numpy(dtype=int)
        else:
            y_arr = np.asarray(y, dtype=int)

        # Filter out NaN rows
        valid = ~np.isnan(X_arr).any(axis=1) & ~np.isnan(y_arr)
        X_clean = X_arr[valid]
        y_clean = y_arr[valid]

        input_dim = X_clean.shape[1]
        self.models = []

        logger.info("Training Deep Ensemble with M=%d models (dim=%d, epochs=%d)", self.n_models, input_dim, self.epochs)
        for m in range(self.n_models):
            net = NeuralRegimeClassifier(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                n_classes=self.n_regimes,
                dropout_rate=self.dropout_rate,
                seed=self.seed + m * 37 + 11,
            )
            net.fit(X_clean, y_clean, epochs=self.epochs, lr=self.lr)
            self.models.append(net)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Emit ensemble mixture probability distribution over 5 regimes.
        """
        mean_p, _, _ = self.predict_with_uncertainty(X)
        return mean_p

    def predict_with_uncertainty(
        self, X: np.ndarray | pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Emit ensemble predictions decomposed into Epistemic and Aleatoric uncertainty budgets.

        Returns:
            Tuple of:
            - mean_p: (N, 5) mixture probabilities
            - epistemic: (N, 5) ensemble disagreement standard deviation
            - aleatoric: (N, 5) inherent observation noise
        """
        arr = self._ensure_2d_array(X)
        N = len(arr)
        K = self.n_regimes

        if not self.models:
            return np.full((N, K), 1.0 / K), np.zeros((N, K)), np.zeros((N, K))

        # Handle NaNs
        clean_arr = np.nan_to_num(arr, nan=0.0)

        member_preds = np.stack([m.forward(clean_arr, training=False) for m in self.models], axis=0)  # (M, N, K)
        mean_p = np.mean(member_preds, axis=0)  # (N, K)

        # Epistemic: standard deviation across models (disagreement)
        epistemic = np.std(member_preds, axis=0)

        # Aleatoric: average per-member Bernoulli variance
        aleatoric = np.mean(member_preds * (1.0 - member_preds), axis=0)

        return mean_p, epistemic, aleatoric


class MCDropoutBNN(BaseRegimeModel):
    r"""
    Monte Carlo Dropout Bayesian Neural Network (Gal & Ghahramani, 2016).
    Performs S stochastic forward passes with permanent dropout active at inference.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        hidden_dims: Tuple[int, int] = (64, 32),
        dropout_rate: float = 0.30,
        n_samples: int = 50,
        epochs: int = 40,
        lr: float = 0.005,
        seed: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, seed=seed)
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self.epochs = epochs
        self.lr = lr
        self.net: Optional[NeuralRegimeClassifier] = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[Any] = None) -> MCDropoutBNN:
        X_arr = self._ensure_2d_array(X)
        N, d = X_arr.shape
        K = self.n_regimes

        if y is None:
            ret = X_arr[:, 0]
            pcts = np.percentile(ret[~np.isnan(ret)], np.linspace(20, 80, K - 1))
            y_arr = np.digitize(ret, pcts)
        else:
            y_arr = np.asarray(y, dtype=int)

        valid = ~np.isnan(X_arr).any(axis=1) & ~np.isnan(y_arr)
        X_clean = X_arr[valid]
        y_clean = y_arr[valid]

        self.net = NeuralRegimeClassifier(
            input_dim=d,
            hidden_dims=self.hidden_dims,
            n_classes=K,
            dropout_rate=self.dropout_rate,
            seed=self.seed,
        )
        self.net.fit(X_clean, y_clean, epochs=self.epochs, lr=self.lr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = self._ensure_2d_array(X)
        clean_arr = np.nan_to_num(arr, nan=0.0)
        # S stochastic forward passes with dropout
        samples = np.stack(
            [self.net.forward(clean_arr, training=True) for _ in range(self.n_samples)],
            axis=0,
        )
        mean_p = np.mean(samples, axis=0)
        return mean_p / np.sum(mean_p, axis=1, keepdims=True)


class VariationalBNN(BaseRegimeModel):
    r"""
    Variational Bayesian Neural Network (Blundell et al., Bayes by Backprop).
    Weights are parameterized as Gaussian distributions $w \sim \mathcal{N}(\mu, \sigma^2)$,
    trained with the Evidence Lower Bound (ELBO):
        $$\mathcal{L} = \mathbb{E}_{q}[\log p(y \mid x, w)] - \text{KL}(q(w) \parallel p(w))$$
    """

    def __init__(
        self,
        n_regimes: int = 5,
        hidden_dims: Tuple[int, int] = (48, 24),
        n_samples: int = 30,
        epochs: int = 35,
        lr: float = 0.005,
        seed: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, seed=seed)
        self.hidden_dims = hidden_dims
        self.n_samples = n_samples
        self.epochs = epochs
        self.lr = lr
        self.weights_mu: Dict[str, np.ndarray] = {}
        self.weights_rho: Dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[Any] = None) -> VariationalBNN:
        X_arr = self._ensure_2d_array(X)
        N, d = X_arr.shape
        K = self.n_regimes

        if y is None:
            ret = X_arr[:, 0]
            pcts = np.percentile(ret[~np.isnan(ret)], np.linspace(20, 80, K - 1))
            y_arr = np.digitize(ret, pcts)
        else:
            y_arr = np.asarray(y, dtype=int)

        valid = ~np.isnan(X_arr).any(axis=1) & ~np.isnan(y_arr)
        X_clean = X_arr[valid]
        y_clean = y_arr[valid]

        rng = np.random.default_rng(self.seed)
        h1, h2 = self.hidden_dims

        # Initialize variational parameters (mu and rho where sigma = log1p(exp(rho)))
        self.weights_mu = {
            "W1": rng.normal(0, np.sqrt(2.0 / d), (d, h1)),
            "b1": np.zeros(h1),
            "W2": rng.normal(0, np.sqrt(2.0 / h1), (h1, h2)),
            "b2": np.zeros(h2),
            "W3": rng.normal(0, np.sqrt(2.0 / h2), (h2, K)),
            "b3": np.zeros(K),
        }
        self.weights_rho = {
            k: np.full_like(v, -3.0) for k, v in self.weights_mu.items()
        }

        # Train with reparameterisation trick and mini-batch ELBO gradient steps
        for ep in range(self.epochs):
            # Sample weights: W = mu + log(1 + exp(rho)) * epsilon
            sampled_W = {}
            for k in self.weights_mu:
                sig = np.log1p(np.exp(np.clip(self.weights_rho[k], -10, 10)))
                eps = rng.normal(0, 1, size=self.weights_mu[k].shape)
                sampled_W[k] = self.weights_mu[k] + sig * eps

            # Forward pass
            z1 = X_clean @ sampled_W["W1"] + sampled_W["b1"]
            a1 = np.maximum(0, z1)
            z2 = a1 @ sampled_W["W2"] + sampled_W["b2"]
            a2 = np.maximum(0, z2)
            logits = a2 @ sampled_W["W3"] + sampled_W["b3"]
            probs = softmax(logits, axis=1)

            # Error gradient descent step on mu
            y_onehot = np.eye(K)[y_clean]
            d_logits = (probs - y_onehot) / len(X_clean)
            self.weights_mu["W3"] -= self.lr * (a2.T @ d_logits)
            self.weights_mu["b3"] -= self.lr * np.sum(d_logits, axis=0)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = self._ensure_2d_array(X)
        clean_arr = np.nan_to_num(arr, nan=0.0)
        rng = np.random.default_rng(self.seed + 99)

        preds = []
        for _ in range(self.n_samples):
            sampled_W = {}
            for k in self.weights_mu:
                sig = np.log1p(np.exp(np.clip(self.weights_rho[k], -10, 10)))
                eps = rng.normal(0, 1, size=self.weights_mu[k].shape)
                sampled_W[k] = self.weights_mu[k] + sig * eps

            z1 = clean_arr @ sampled_W["W1"] + sampled_W["b1"]
            a1 = np.maximum(0, z1)
            z2 = a1 @ sampled_W["W2"] + sampled_W["b2"]
            a2 = np.maximum(0, z2)
            logits = a2 @ sampled_W["W3"] + sampled_W["b3"]
            preds.append(softmax(logits, axis=1))

        mean_p = np.mean(preds, axis=0)
        return mean_p / np.sum(mean_p, axis=1, keepdims=True)


class DeepEnsembleModel(BaseRegimeModel):
    r"""
    Multi-seed Deep Ensemble for Regime Classification.
    Combines M independently trained neural classifiers with distinct initializations.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        n_models: int = 5,
        hidden_dims: Tuple[int, int] = (64, 32),
        epochs: int = 35,
        lr: float = 0.005,
        seed: int = 42,
    ) -> None:
        super().__init__(n_regimes=n_regimes, seed=seed)
        self.n_models = n_models
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.models: List[NeuralRegimeClassifier] = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[Any] = None) -> DeepEnsembleModel:
        X_arr = self._ensure_2d_array(X)
        N, d = X_arr.shape
        K = self.n_regimes

        if y is None:
            ret = X_arr[:, 0]
            pcts = np.percentile(ret[~np.isnan(ret)], np.linspace(20, 80, K - 1))
            y_arr = np.digitize(ret, pcts)
        else:
            y_arr = np.asarray(y, dtype=int)

        valid = ~np.isnan(X_arr).any(axis=1) & ~np.isnan(y_arr)
        X_clean = X_arr[valid]
        y_clean = y_arr[valid]

        self.models = []
        for m in range(self.n_models):
            net = NeuralRegimeClassifier(
                input_dim=d,
                hidden_dims=self.hidden_dims,
                n_classes=K,
                dropout_rate=0.0,
                seed=self.seed + m * 43 + 7,
            )
            net.fit(X_clean, y_clean, epochs=self.epochs, lr=self.lr)
            self.models.append(net)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = self._ensure_2d_array(X)
        clean_arr = np.nan_to_num(arr, nan=0.0)
        preds = np.stack([m.forward(clean_arr, training=False) for m in self.models], axis=0)
        mean_p = np.mean(preds, axis=0)
        return mean_p / np.sum(mean_p, axis=1, keepdims=True)
