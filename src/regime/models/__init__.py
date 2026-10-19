"""Regime detection models package."""
from .base import BaseRegimeModel, REGIME_NAMES
from .hmm_model import GaussianRegimeHMM
from .bayesian_hmm import BayesianRegimeHMM
from .ms_var import MarkovSwitchingVAR
from .bnn_model import BayesianDeepEnsemble
from .foundation_model import FoundationRegimeHead

__all__ = [
    "BaseRegimeModel",
    "REGIME_NAMES",
    "GaussianRegimeHMM",
    "BayesianRegimeHMM",
    "MarkovSwitchingVAR",
    "BayesianDeepEnsemble",
    "FoundationRegimeHead",
]
