"""
Abstract Base Class and Contract for Regime Detection Models.
Standardises the fit(X, y) / predict_proba(X) / predict(X) interface across all
frequentist, Bayesian, multivariate, and deep learning regime estimators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

REGIME_NAMES: List[str] = [
    "Risk-On",
    "Late-Cycle",
    "Transitional",
    "Post-Shock",
    "Risk-Off",
]


class BaseRegimeModel(ABC):
    """
    Abstract interface for Indian equity regime classifiers.
    All models must output a (N, 5) probability distribution over the 5 defined regimes.
    """

    def __init__(self, n_regimes: int = 5, seed: int = 42) -> None:
        self.n_regimes = n_regimes
        self.seed = seed
        self.is_fitted = False
        self.regime_names = REGIME_NAMES[:n_regimes]

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[np.ndarray | pd.Series] = None) -> BaseRegimeModel:
        """
        Fit model on training feature matrix.
        For unsupervised models (HMM/RS-VAR), y is ignored.
        For supervised models (BNN/Ensemble), y is the window-decoded regime target.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Emit calibrated regime probabilities of shape (N, 5).
        Probabilities must be non-negative and sum to 1.0 along axis 1.
        """
        pass

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Emit most likely regime index (0 to 4) via argmax.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_regime_labels(self, X: np.ndarray | pd.DataFrame) -> List[str]:
        """
        Emit string regime labels for each sample.
        """
        preds = self.predict(X)
        return [self.regime_names[p] for p in preds]

    def _ensure_2d_array(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Helper to ensure input is a 2D float64 numpy array."""
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
