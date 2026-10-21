"""
Gaussian Hidden Markov Model for Indian Equity Regime Detection (Step 4a).
Provides Gaussian HMM with Baum-Welch EM, BIC model selection, multiple random restarts,
and economic moment sorting computed strictly inside the training window.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .base import BaseRegimeModel, REGIME_NAMES
from .hmm_model import GaussianRegimeHMM

logger = logging.getLogger(__name__)


def fit_best_hmm(
    X: np.ndarray | pd.DataFrame,
    n_regimes: int = 5,
    n_restarts: int = 5,
    n_iter: int = 100,
    seed: int = 42,
) -> GaussianRegimeHMM:
    """
    Fit Gaussian HMM with multiple random restarts and select the model with maximum log-likelihood.
    """
    best_model = None
    best_ll = -np.inf

    for restart in range(n_restarts):
        model = GaussianRegimeHMM(
            n_regimes=n_regimes,
            n_iter=n_iter,
            seed=seed + restart * 13 + 7,
        )
        model.fit(X)
        if model.log_likelihood_ > best_ll:
            best_ll = model.log_likelihood_
            best_model = model

    logger.info("Best HMM selected across %d restarts: Log-Likelihood = %.2f, BIC = %.2f", n_restarts, best_model.log_likelihood_, best_model.bic_)
    return best_model


def select_k_by_bic(
    X: np.ndarray | pd.DataFrame,
    k_candidates: List[int] = [3, 4, 5, 7],
    seed: int = 42,
) -> Tuple[int, Dict[int, float]]:
    """
    Evaluate candidate regime counts K using Bayesian Information Criterion (BIC).
    """
    bic_scores = {}
    best_k = k_candidates[0]
    min_bic = np.inf

    for k in k_candidates:
        model = GaussianRegimeHMM(n_regimes=k, n_iter=60, seed=seed)
        model.fit(X)
        bic_scores[k] = model.bic_
        if model.bic_ < min_bic:
            min_bic = model.bic_
            best_k = k

    logger.info("BIC Model Selection: candidate scores: %s. Selected K=%d.", bic_scores, best_k)
    return best_k, bic_scores


__all__ = ["GaussianRegimeHMM", "fit_best_hmm", "select_k_by_bic"]
