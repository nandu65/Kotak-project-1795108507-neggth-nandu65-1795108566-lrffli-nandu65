"""
Post-Hoc Probability Calibration Layer for Bayesian Regime Detection Engine.

Maps raw state probabilities emitted by unsupervised or Bayesian regime estimators
(HMM, MS-VAR, BNN ensemble) to independent ground-truth rule-based regime targets
defined in Brief Section A1.4.

Fitted STRICTLY on the in-sample training window (2010–2018) with zero look-ahead bias,
then evaluated out-of-sample (2019–2024).
"""
from __future__ import annotations

import logging
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class PostHocProbabilityCalibrator:
    """
    Multiclass post-hoc probability calibration using regularized multinomial
    logistic regression fitted strictly on in-sample training window probabilities.
    """

    def __init__(
        self,
        n_classes: int = 5,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.n_classes = n_classes
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.calibrator: Optional[LogisticRegression] = None
        self.is_fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> PostHocProbabilityCalibrator:
        """
        Fit the calibrator on in-sample training window probabilities and ground-truth targets.

        Parameters
        ----------
        probs : np.ndarray of shape (N, n_classes)
            Raw predicted probabilities from the underlying regime model.
        labels : np.ndarray of shape (N,)
            Independent ground-truth regime class labels (0 to n_classes-1).
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)

        if probs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"probs length ({probs.shape[0]}) does not match labels length ({labels.shape[0]})"
            )

        # Clip probabilities to avoid numerical instability
        eps = 1e-12
        probs_clean = np.clip(probs, eps, 1.0 - eps)
        # Normalize rows to sum to 1.0
        probs_clean = probs_clean / np.sum(probs_clean, axis=1, keepdims=True)

        self.calibrator = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.calibrator.fit(probs_clean, labels)
        self.is_fitted = True
        logger.info(
            "PostHocProbabilityCalibrator successfully fitted on %d samples across %d classes.",
            len(labels),
            len(np.unique(labels)),
        )
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate out-of-sample probabilities.

        Parameters
        ----------
        probs : np.ndarray of shape (M, n_classes)
            Raw predicted probabilities.

        Returns
        -------
        calibrated_probs : np.ndarray of shape (M, n_classes)
            Calibrated probabilities summing to 1.0 across rows.
        """
        if not self.is_fitted or self.calibrator is None:
            raise RuntimeError("Calibrator must be fitted before predict_proba can be called.")

        probs = np.asarray(probs, dtype=np.float64)
        eps = 1e-12
        probs_clean = np.clip(probs, eps, 1.0 - eps)
        probs_clean = probs_clean / np.sum(probs_clean, axis=1, keepdims=True)

        cal_probs = self.calibrator.predict_proba(probs_clean)

        # Ensure output has all n_classes columns even if some were absent in training
        if cal_probs.shape[1] < self.n_classes:
            full_probs = np.zeros((cal_probs.shape[0], self.n_classes), dtype=np.float64)
            for idx, c in enumerate(self.calibrator.classes_):
                full_probs[:, c] = cal_probs[:, idx]
            cal_probs = full_probs

        # Final normalization
        row_sums = np.sum(cal_probs, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cal_probs / row_sums

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Emit discrete regime class via argmax on calibrated probabilities."""
        cal_probs = self.predict_proba(probs)
        return np.argmax(cal_probs, axis=1)
