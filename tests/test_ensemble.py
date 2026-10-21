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

    # Uncertainty budget contract verification
    assert "uncertainty_budget" in contract
    ub = contract["uncertainty_budget"]
    assert "total_entropy" in ub and "aleatoric_entropy" in ub and "epistemic_mutual_info" in ub
    assert ub["identity_verified"] is True


def test_hand_computed_scoring_rules():
    """Verify analytical hand-computed test cases for scoring rules."""
    # Case 1: Perfect prediction
    p_perfect = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    y_zero = np.array([0])
    assert compute_ranked_probability_score(p_perfect, y_zero, K=5) == 0.0
    assert compute_brier_score(p_perfect, y_zero, K=5) == 0.0
    assert compute_log_loss(p_perfect, y_zero) < 1e-6

    # Case 2: Hand-computed Brier
    # p = [0.8, 0.1, 0.1, 0.0, 0.0], y = 0
    # Sum of squared errors = (0.8 - 1)^2 + 0.1^2 + 0.1^2 = 0.04 + 0.01 + 0.01 = 0.06
    p_hand = np.array([[0.8, 0.1, 0.1, 0.0, 0.0]])
    assert np.isclose(compute_brier_score(p_hand, y_zero, K=5), 0.06, atol=1e-6)

    # Case 3: Hand-computed RPS for 1-step error vs 4-step error
    # y = 0 => CDF_true = [1, 1, 1, 1, 1]
    # Predict state 1 (1-step error): CDF_pred = [0, 1, 1, 1, 1]
    # Diff = [1, 0, 0, 0, 0] => Sum diff^2 = 1. RPS = 1 / (K-1) = 1/4 = 0.25
    p_state1 = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    assert np.isclose(compute_ranked_probability_score(p_state1, y_zero, K=5), 0.25, atol=1e-6)

    # Predict state 4 (4-step error): CDF_pred = [0, 0, 0, 0, 1]
    # Diff = [1, 1, 1, 1, 0] => Sum diff^2 = 4. RPS = 4 / (K-1) = 4/4 = 1.0
    p_state4 = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    assert np.isclose(compute_ranked_probability_score(p_state4, y_zero, K=5), 1.0, atol=1e-6)


def test_uncertainty_decomposition_identity():
    """Verify information-theoretic identity: Total Entropy H = Expected Entropy E[H] + Mutual Information I."""
    from src.regime.ensemble.stacker import decompose_uncertainty_entropy

    rng = np.random.default_rng(42)
    # 5 models predicting over 5 regimes
    M, K = 5, 5
    model_probs = np.array([rng.dirichlet(np.ones(K) * 2.0) for _ in range(M)])
    weights = rng.dirichlet(np.ones(M) * 3.0)

    res = decompose_uncertainty_entropy(model_probs, weights=weights)
    H = res["total_entropy"]
    E_H = res["aleatoric_entropy"]
    I = res["epistemic_mutual_info"]

    assert res["identity_verified"] is True
    assert np.isclose(H, E_H + I, atol=1e-5), f"Decomposition identity failed: H={H}, E_H+I={E_H+I}"
    assert H >= E_H, "Total entropy must be >= expected entropy (Jensen's inequality)"
    assert I >= 0.0, "Mutual information must be non-negative"


def test_calibration_ece_and_benchmarks():
    """Verify ECE calculation, reliability curve generation, and baseline benchmarks."""
    from src.regime.ensemble.stacker import (
        ClimatologyBenchmark,
        PersistenceBenchmark,
        VolThresholdBenchmark,
        compute_ece,
        generate_reliability_curve,
    )

    # 1. Calibration ECE
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    probs = np.eye(5)[y_true]  # 100% accurate, 100% confidence
    ece_zero = compute_ece(probs, y_true, n_bins=5)
    assert np.isclose(ece_zero, 0.0, atol=1e-5)

    # Reliability curve file creation
    rel_res = generate_reliability_curve(probs, y_true, save_path="reports/figures/test_reliability.png")
    assert "ece" in rel_res
    assert rel_res["ece"] == 0.0

    # 2. Climatology Benchmark
    clim = ClimatologyBenchmark(K=5).fit(y_true)
    clim_p = clim.predict_proba(N=5)
    assert clim_p.shape == (5, 5)
    assert np.allclose(clim_p.sum(axis=1), 1.0)
    assert np.allclose(clim.probs_, 0.2)

    # 3. Persistence Benchmark
    pers = PersistenceBenchmark(K=5)
    pers_p = pers.predict_proba(np.array([0, 4]))
    assert pers_p.shape == (2, 5)
    assert np.argmax(pers_p[0]) == 0
    assert np.argmax(pers_p[1]) == 4

    # 4. Vol Threshold Benchmark
    vol_series = np.array([0.08, 0.12, 0.18, 0.25, 0.40])
    vol_bench = VolThresholdBenchmark(K=5).fit(vol_series)
    vol_p = vol_bench.predict_proba(vol_series)
    assert vol_p.shape == (5, 5)
    assert np.allclose(vol_p.sum(axis=1), 1.0)

