"""
Bayesian Hidden Markov Model for Indian Equity Regime Detection (Step 4b).
Marginalises discrete latent states via exact log-space forward algorithm.
Applies sticky Dirichlet transition priors and emits MCMC diagnostics (R-hat, ESS, divergences).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseRegimeModel, REGIME_NAMES
from .bayesian_hmm import (
    BayesianRegimeHMM,
    forward_log_marginal_likelihood,
)

logger = logging.getLogger(__name__)

__all__ = ["BayesianRegimeHMM", "forward_log_marginal_likelihood"]
