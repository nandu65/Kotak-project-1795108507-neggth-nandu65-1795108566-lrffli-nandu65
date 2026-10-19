"""
Unit tests for Conformal Prediction and Calibration Layer:
- SplitConformalClassifier
- AdaptivePredictionSets (APS)
- AdaptiveConformalInference (ACI)
- ECE and Reliability Curve
"""
import numpy as np
import pytest
from src.regime.conformal.predictor import (
    AdaptiveConformalInference,
    AdaptivePredictionSets,
    SplitConformalClassifier,
    compute_expected_calibration_error,
    compute_reliability_curve,
)


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(42)
    n_cal = 200
    n_test = 100
    K = 5

    # True labels
    y_cal = rng.integers(0, K, size=n_cal)
    y_test = rng.integers(0, K, size=n_test)

    # Simulated moderately calibrated probabilities
    def make_probs(labels, n):
        P = np.zeros((n, K))
        for i in range(n):
            p = rng.dirichlet(np.ones(K) * 0.5)
            # Give higher mass to true class
            p[labels[i]] += 0.8
            P[i] = p / np.sum(p)
        return P

    cal_probs = make_probs(y_cal, n_cal)
    test_probs = make_probs(y_test, n_test)

    return cal_probs, y_cal, test_probs, y_test


def test_split_conformal_coverage(calibration_data):
    cal_probs, y_cal, test_probs, y_test = calibration_data
    nominal_alpha = 0.10  # 90% target coverage

    conformal = SplitConformalClassifier(alpha=nominal_alpha)
    q_hat = conformal.calibrate(cal_probs, y_cal)
    pred_sets = conformal.predict_set(test_probs)

    assert pred_sets.shape == test_probs.shape
    # Check empirical coverage on test set: fraction of test labels inside the set
    covered = np.array([pred_sets[i, y_test[i]] for i in range(len(y_test))])
    empirical_coverage = float(np.mean(covered))

    # Empirical coverage should be close to or above nominal (>= 85% with sample variance)
    assert empirical_coverage >= 0.85, f"Coverage was too low: {empirical_coverage}"


def test_adaptive_prediction_sets(calibration_data):
    cal_probs, y_cal, test_probs, y_test = calibration_data
    nominal_alpha = 0.10

    aps = AdaptivePredictionSets(alpha=nominal_alpha)
    aps.calibrate(cal_probs, y_cal)
    in_set_mask, set_labels = aps.predict_set(test_probs)

    assert in_set_mask.shape == test_probs.shape
    assert len(set_labels) == len(test_probs)
    # Check that every set has at least 1 regime and at most 5 regimes
    for s in set_labels:
        assert 1 <= len(s) <= 5

    covered = np.array([in_set_mask[i, y_test[i]] for i in range(len(y_test))])
    emp_cov = float(np.mean(covered))
    assert emp_cov >= 0.85


def test_adaptive_conformal_inference_online():
    # Simulate non-stationary stream with alternating coverage shocks
    aci = AdaptiveConformalInference(alpha_target=0.10, gamma=0.02)

    # 50 consecutive miscoverages (e.g. regime shock)
    for _ in range(50):
        aci.step(is_covered=False)

    # ACI must adapt by widening target (alpha_t decreases towards 0.001 to demand wider sets)
    assert aci.alpha_t < 0.05

    # 100 consecutive covered steps
    for _ in range(100):
        aci.step(is_covered=True)

    # After recovery, alpha_t rises back towards target
    assert aci.alpha_t > 0.05


def test_ece_and_reliability_curve():
    # Perfectly calibrated forecast: confidence equals empirical accuracy
    n = 200
    K = 5
    y_true = np.zeros(n, dtype=int)
    probs = np.full((n, K), 0.05)
    probs[:, 0] = 0.80  # 80% confidence in class 0
    # Set exactly 80% of labels to 0
    y_true[:160] = 0
    y_true[160:] = 1

    ece = compute_expected_calibration_error(probs, y_true, n_bins=5)
    # For 80% accuracy matching 80% confidence, ECE should be near 0
    assert ece < 0.05

    acc, conf, cnt = compute_reliability_curve(probs, y_true, n_bins=5)
    assert len(acc) == 5
    assert np.sum(cnt) == n
