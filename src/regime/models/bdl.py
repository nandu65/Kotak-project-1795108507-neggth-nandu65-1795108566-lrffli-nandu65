"""
Bayesian Deep Learning for Regime Detection: Deep Ensembles & MC-Dropout (Step 4d).
Ensemble of 5 neural network classifiers trained on walk-forward regime labels
decoded strictly from historical data within each window.
Decomposes predictive uncertainty into Epistemic (mutual information) and Aleatoric (expected entropy).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseRegimeModel, REGIME_NAMES
from .bnn_model import (
    BayesianDeepEnsemble,
    NeuralRegimeClassifier,
    MCDropoutBNN,
    VariationalBNN,
    DeepEnsembleModel,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BayesianDeepEnsemble",
    "NeuralRegimeClassifier",
    "MCDropoutBNN",
    "VariationalBNN",
    "DeepEnsembleModel",
]
