"""
Sequential Monte Carlo (Particle Filter) for Online Regime State Tracking.
Maintains an empirical particle approximation of the regime posterior in real-time
between nightly batch refits, with systematic resampling on ESS depletion (PDF Section A9.1).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class RegimeParticleFilter:
    r"""
    Bootstrap Particle Filter for K-Regime Latent State Filtering.

    Maths:
        State space $s_t \in \{0, \dots, K-1\}$ with transition matrix $\mathbf{P}$.
        Posterior represented by $N$ weighted particles $\{s_t^{(i)}, w_t^{(i)}\}_{i=1}^N$.
        1. Propagation: $s_t^{(i)} \sim \text{Categorical}(\mathbf{P}_{s_{t-1}^{(i)}})$
        2. Measurement update:
           $$\tilde{w}_t^{(i)} = w_{t-1}^{(i)} \cdot \mathcal{N}(y_t; \mu_{s_t^{(i)}}, \sigma_{s_t^{(i)}}^2)$$
           $$w_t^{(i)} = \frac{\tilde{w}_t^{(i)}}{\sum_{j=1}^N \tilde{w}_t^{(j)}}$$
        3. Systematic Resampling triggered when Effective Sample Size (ESS) drops below $N / 2$:
           $$\text{ESS} = \frac{1}{\sum_{i=1}^N (w_t^{(i)})^2}$$
    """

    def __init__(
        self,
        P: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        n_particles: int = 5000,
        seed: int = 42,
    ) -> None:
        self.P = np.asarray(P, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)

        self.K = self.P.shape[0]
        self.N = n_particles
        self.rng = np.random.default_rng(seed)

        # Uniform particle initialization
        self.particles = self.rng.integers(0, self.K, size=self.N)
        self.weights = np.full(self.N, 1.0 / self.N)

    def step(self, obs: float) -> Tuple[np.ndarray, float]:
        """
        Advance particle filter by one observation step.

        Args:
            obs: Realised scalar observation (e.g. daily log return).

        Returns:
            Tuple of:
            - post: (K,) array of posterior regime probabilities.
            - ess: Current Effective Sample Size.
        """
        if np.isnan(obs):
            # If observation is missing, propagate without measurement update
            return self.get_regime_posterior(), float(self.N)

        # 1. State propagation through Markov transition rows
        new_particles = np.empty(self.N, dtype=int)
        for i in range(self.N):
            curr_state = self.particles[i]
            # Transition probability vector for particle i
            p_trans = self.P[curr_state]
            new_particles[i] = self.rng.choice(self.K, p=p_trans)
        self.particles = new_particles

        # 2. Measurement re-weighting via Gaussian emission density
        diff = (obs - self.mu[self.particles]) / self.sigma[self.particles]
        like = np.exp(-0.5 * (diff ** 2)) / (self.sigma[self.particles] * np.sqrt(2.0 * np.pi))
        like = np.maximum(like, 1e-12)

        self.weights *= like
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights = np.full(self.N, 1.0 / self.N)

        # 3. Compute Effective Sample Size (ESS)
        ess = 1.0 / np.sum(self.weights ** 2)

        # 4. Systematic resampling if ESS falls below N / 2
        if ess < self.N / 2.0:
            self.particles = self._systematic_resample()
            self.weights = np.full(self.N, 1.0 / self.N)

        post = self.get_regime_posterior()
        return post, float(ess)

    def get_regime_posterior(self) -> np.ndarray:
        """Calculate weighted empirical distribution across K regimes."""
        counts = np.bincount(self.particles, weights=self.weights, minlength=self.K)
        s = np.sum(counts)
        return counts / s if s > 0 else np.full(self.K, 1.0 / self.K)

    def _systematic_resample(self) -> np.ndarray:
        """Low-variance systematic resampling."""
        cdf = np.cumsum(self.weights)
        u0 = self.rng.uniform(0.0, 1.0 / self.N)
        u = u0 + np.arange(self.N) * (1.0 / self.N)

        idx = np.searchsorted(cdf, u)
        return self.particles[np.clip(idx, 0, self.N - 1)]
