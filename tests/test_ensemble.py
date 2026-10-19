"""
Unit tests for Ensemble Layer and Proper Scoring Rules.
"""
import numpy as np
import pytest
from src.regime.ensemble.stacker import (
    BayesianModelAverager,
    ConstrainedStacker,
    EnsembleRegimeEngine,
    compute_brier_decomposition,
    compute_brier_score,
    compute_log_loss,
    compute_ranked_probability_score,
    compute_skill_score,
)


def test_ranked_probability_score_ordering():
    # If true regime is 0 (Risk-On):
    y_true = np.array([0])

    # Prediction A predicts Late-Cycle (1) with 100% certainty (adjacent)
    probs_A = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    rps_A = compute_ranked_probability_score(probs_A, y_true, K=5)

    # Prediction B predicts Risk-Off (4) with 100% certainty (extreme distance)
    probs_B = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    rps_B = compute_ranked_probability_score(probs_B, y_true, K=5)

    # Log loss treats both wrong predictions equally if confident,
    # but RPS must strictly penalise Risk-Off much more heavily than Late-Cycle!
    assert rps_A < rps_B, f"RPS did not respect regime ordering: rps_A={rps_A}, rps_B={rps_B}"


def test_murphy_brier_decomposition():
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.1, 0.9, 100)
    y = (rng.uniform(0, 1, 100) < probs).astype(int)

    decomp = compute_brier_decomposition(probs, y, n_bins=5)
    # Check that Brier == Reliability - Resolution + Uncertainty (within tolerance)
    expected_brier = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
    assert np.isclose(decomp["brier"], expected_brier, atol=1e-5)


def test_bayesian_model_averager():
    bma = BayesianModelAverager()
    # Model 1 has much higher log likelihood than Model 2
    lls = np.array([-10.0, -50.0, -100.0])
    w = bma.fit(lls)

    assert len(w) == 3
    assert np.isclose(np.sum(w), 1.0)
    assert (w >= 0.0).all()
    # Model 0 should dominate
    assert w[0] > 0.99


def test_constrained_stacker():
    rng = np.random.default_rng(42)
    N, K, M = 100, 5, 3
    y_true = rng.integers(0, K, size=N)

    # Generate 3 models: Model 0 is smart, Model 1 & 2 are noisy
    base_probs = np.zeros((M, N, K))
    for i in range(N):
        # Model 0 assigns 70% to true class
        p0 = np.full(K, 0.075)
        p0[y_true[i]] = 0.70
        base_probs[0, i] = p0

        # Model 1 and 2 assign random probabilities
        p1 = rng.dirichlet(np.ones(K))
        p2 = rng.dirichlet(np.ones(K))
        base_probs[1, i] = p1
        base_probs[2, i] = p2

    stacker = ConstrainedStacker(seed=42)
    weights = stacker.fit(base_probs, y_true)

    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)
    assert (weights >= 0.0).all()
    # Model 0 should receive dominant weight
    assert weights[0] > 0.70

    combined_p = stacker.predict_proba(base_probs)
    assert combined_p.shape == (N, K)
    assert np.allclose(np.sum(combined_p, axis=1), 1.0)


def test_ensemble_regime_engine_contract():
    model_names = ["HMM", "BHMM", "RSVAR", "BNN", "CHRONOS"]
    engine = EnsembleRegimeEngine(model_names=model_names, seed=42)

    # Synthetic day predictions across 5 models: (5, 5)
    model_probs = np.array(
        [
            [0.60, 0.20, 0.10, 0.05, 0.05],  # HMM: Risk-On
            [0.55, 0.25, 0.10, 0.05, 0.05],  # BHMM: Risk-On
            [0.45, 0.35, 0.10, 0.05, 0.05],  # RSVAR: Risk-On
            [0.50, 0.30, 0.10, 0.05, 0.05],  # BNN: Risk-On
            [0.52, 0.28, 0.10, 0.05, 0.05],  # CHRONOS: Risk-On
        ]
    )

    contract = engine.predict_contract(model_probs, date_str="2026-10-19")

    assert contract["date"] == "2026-10-19"
    assert contract["dominant_regime"] == "Risk-On"
    assert "regime_probs" in contract
    assert len(contract["regime_probs"]) == 5
    assert contract["epistemic_uncertainty"] >= 0.0
    assert contract["aleatoric_uncertainty"] >= 0.0
    # Sum of probabilities across regimes must be 1.0
    assert np.isclose(sum(contract["regime_probs"].values()), 1.0, atol=1e-4)
