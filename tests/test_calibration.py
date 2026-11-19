"""
Tests for Post-Hoc Probability Calibration Layer.
"""
import numpy as np
import pytest
from src.regime.models.calibration import PostHocProbabilityCalibrator


def test_calibrator_initialization_and_fit():
    np.random.seed(42)
    N_train = 500
    K = 5
    raw_probs = np.random.dirichlet(np.ones(K), size=N_train)
    labels = np.random.choice(K, size=N_train)

    calibrator = PostHocProbabilityCalibrator(n_classes=K, C=1.0)
    assert not calibrator.is_fitted

    calibrator.fit(raw_probs, labels)
    assert calibrator.is_fitted

    N_test = 100
    test_raw = np.random.dirichlet(np.ones(K), size=N_test)
    cal_probs = calibrator.predict_proba(test_raw)

    assert cal_probs.shape == (N_test, K)
    assert np.all(cal_probs >= 0.0)
    assert np.allclose(np.sum(cal_probs, axis=1), 1.0)

    preds = calibrator.predict(test_raw)
    assert len(preds) == N_test
    assert np.all((preds >= 0) & (preds < K))


def test_calibrator_requires_fit():
    calibrator = PostHocProbabilityCalibrator(n_classes=5)
    with pytest.raises(RuntimeError, match="must be fitted"):
        calibrator.predict_proba(np.zeros((10, 5)))
