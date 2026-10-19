"""
Bayesian Online Changepoint Detection (BOCPD; Adams & MacKay, 2007).
Implements exact Student-t posterior predictive density with Normal-Gamma conjugacy
to detect acute structural regime shifts (PDF Section A9.2 & Prompt Hard Rules).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class StudentTBOCPD:
    r"""
    Bayesian Online Changepoint Detection with Student-t Predictive Density.

    Maths:
        Conjugate Normal-Gamma prior on segment parameters $(\mu, \tau)$:
            $$\mu \mid \tau \sim \mathcal{N}\left(\mu_0, \frac{1}{\kappa_0 \tau}\right), \quad \tau \sim \text{Gamma}(\alpha_0, \beta_0)$$
        Posterior predictive distribution for new observation $x$ given run-length $r$:
            $$x \mid r \sim t_{2\alpha_r}\left(\mu_r, \sigma_r = \sqrt{\frac{\beta_r (\kappa_r + 1)}{\alpha_r \kappa_r}}\right)$$
        Recursive run-length posterior updating:
            Growth: $R_{r+1, t+1} = R_{r, t} \cdot \pi_t(x_t \mid r) \cdot (1 - H)$
            Changepoint: $R_{0, t+1} = \sum_{r=0}^t R_{r, t} \cdot \pi_t(x_t \mid r) \cdot H$
            where hazard $H = \frac{1}{\lambda}$ is constant exit rate.
    """

    def __init__(
        self,
        hazard: float = 1.0 / 100.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        cp_threshold: float = 0.25,
    ) -> None:
        self.hazard = hazard
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.cp_threshold = cp_threshold

        # Online state arrays
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])

        # Current run-length posterior vector (length: current_t + 1)
        self.R = np.array([1.0])
        self.history_cp_probs: List[float] = [0.0]

    def update(self, x: float) -> Tuple[float, np.ndarray]:
        r"""
        Process single streaming observation $x_t$ and update run-length posterior.

        Args:
            x: Realised observation.

        Returns:
            Tuple of:
            - cp_prob: Probability that time t is a changepoint (run length = 0).
            - R: Current run-length probability vector.
        """
        if np.isnan(x):
            return 0.0, self.R

        # 1. Evaluate Student-t posterior predictive density across all active run lengths
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        scale = np.maximum(scale, 1e-8)

        # Predictive under existing run lengths
        pred_dens = stats.t.pdf(x, df=df, loc=self.mu, scale=scale)
        pred_dens = np.maximum(pred_dens, 1e-12)

        # Predictive under changepoint hypothesis (evaluated against prior parameters)
        scale0 = np.sqrt(self.beta0 * (self.kappa0 + 1.0) / (self.alpha0 * self.kappa0))
        pred_cp = stats.t.pdf(x, df=2.0 * self.alpha0, loc=self.mu0, scale=scale0)
        pred_cp = np.maximum(pred_cp, 1e-12)

        # 2. Recursive growth and changepoint probabilities
        R_prev = self.R
        growth_probs = R_prev * pred_dens * (1.0 - self.hazard)
        cp_prob = float(np.sum(R_prev * self.hazard) * pred_cp)

        # New run-length distribution: [changepoint, growth_0, growth_1, ...]
        R_new = np.concatenate([[cp_prob], growth_probs])
        evidence = np.sum(R_new)
        if evidence > 0:
            R_new = R_new / evidence
        else:
            R_new = np.zeros_like(R_new)
            R_new[0] = 1.0

        self.R = R_new

        # 3. Update sufficient statistics (Normal-Gamma conjugacy)
        mu_new = (self.kappa * self.mu + x) / (self.kappa + 1.0)
        kappa_new = self.kappa + 1.0
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0))

        # Append prior statistics at index 0 for next changepoint
        self.mu = np.concatenate([[self.mu0], mu_new])
        self.kappa = np.concatenate([[self.kappa0], kappa_new])
        self.alpha = np.concatenate([[self.alpha0], alpha_new])
        self.beta = np.concatenate([[self.beta0], beta_new])

        # Track history
        curr_cp_prob = float(R_new[0])
        self.history_cp_probs.append(curr_cp_prob)

        return curr_cp_prob, self.R

    def is_changepoint_active(self) -> bool:
        """Returns True if current changepoint probability exceeds threshold."""
        return bool(self.R[0] > self.cp_threshold)

    def run_batch(self, series: np.ndarray | pd.Series) -> pd.DataFrame:
        """
        Execute BOCPD across an entire historical series.

        Returns:
            DataFrame with columns ['changepoint_prob', 'expected_run_length', 'is_changepoint'].
        """
        arr = np.asarray(series, dtype=float)
        T = len(arr)

        cp_probs = np.zeros(T)
        expected_rl = np.zeros(T)

        # Reset state
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])
        self.R = np.array([1.0])
        self.history_cp_probs = []

        for t in range(T):
            cp_p, r_vec = self.update(arr[t])
            cp_probs[t] = cp_p
            # Expected run-length E[r_t] = \sum r * P(r_t = r)
            expected_rl[t] = float(np.sum(np.arange(len(r_vec)) * r_vec))

        df_out = pd.DataFrame(
            {
                "changepoint_prob": cp_probs,
                "expected_run_length": expected_rl,
                "is_changepoint": (cp_probs > self.cp_threshold).astype(int),
            }
        )
        if isinstance(series, pd.Series):
            df_out.index = series.index
        return df_out
