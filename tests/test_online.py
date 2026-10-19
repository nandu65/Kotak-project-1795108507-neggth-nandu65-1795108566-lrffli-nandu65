"""
Unit tests for Sequential Online Inference Layer:
- RegimeParticleFilter
- StudentTBOCPD
- OnlineBatchReconciler
"""
import numpy as np
import pytest
from src.regime.online.particle_filter import RegimeParticleFilter
from src.regime.online.bocpd import StudentTBOCPD
from src.regime.online.reconciliation import OnlineBatchReconciler, compute_total_variation_gap


def test_particle_filter():
    K = 5
    P = np.eye(K) * 0.90 + (1 - np.eye(K)) * (0.10 / (K - 1))
    mu = np.array([0.002, 0.0005, 0.0, -0.001, -0.003])
    sigma = np.array([0.008, 0.010, 0.012, 0.018, 0.025])

    pf = RegimeParticleFilter(P, mu, sigma, n_particles=1000, seed=42)

    # Step through 10 observations with positive returns (should tilt to Risk-On)
    for _ in range(10):
        post, ess = pf.step(0.002)

    assert post.shape == (5,)
    assert np.isclose(np.sum(post), 1.0)
    assert ess > 0
    # State 0 (Risk-On) should have highest probability
    assert post[0] > post[-1]


def test_bocpd_student_t_changepoint_detection():
    # 50 calm days followed by an abrupt crash shock of -5 standard deviations
    rng = np.random.default_rng(42)
    calm = rng.normal(0.0, 0.01, 50)
    crash = rng.normal(-0.06, 0.04, 10)  # COVID / IL&FS style breakdown
    series = np.concatenate([calm, crash])

    bocpd = StudentTBOCPD(hazard=1.0 / 80.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=0.0005)
    results = bocpd.run_batch(series)

    assert len(results) == 60
    assert "changepoint_prob" in results.columns
    assert "is_changepoint" in results.columns

    # During calm period, changepoint probability should be low
    calm_mean_cp = results["changepoint_prob"].iloc[:45].mean()
    assert calm_mean_cp < 0.20

    # At crash onset (index 50-52), changepoint probability must spike sharply
    crash_max_cp = results["changepoint_prob"].iloc[50:55].max()
    assert crash_max_cp > 0.30, f"BOCPD failed to flag crash shock: {crash_max_cp}"


def test_online_batch_reconciliation():
    reconciler = OnlineBatchReconciler(gap_threshold=0.15, cp_threshold=0.25)

    # 1. Healthy scenario
    p_batch = np.array([0.60, 0.20, 0.10, 0.05, 0.05])
    p_online = np.array([0.58, 0.22, 0.10, 0.05, 0.05])
    health = reconciler.evaluate_health(p_batch, p_online, changepoint_prob=0.05)
    assert health["is_healthy"] is True
    assert health["retrain_recommended"] is False

    # 2. Severe intraday divergence scenario
    p_divergent = np.array([0.10, 0.10, 0.20, 0.30, 0.30])  # Flipped to Risk-Off intraday
    health_div = reconciler.evaluate_health(p_batch, p_divergent, changepoint_prob=0.08)
    assert health_div["is_healthy"] is False
    assert "online_batch_divergence" in health_div["retraining_triggers"]

    # 3. Changepoint trigger scenario
    health_cp = reconciler.evaluate_health(p_batch, p_online, changepoint_prob=0.65)
    assert health_cp["is_healthy"] is False
    assert "regime_changepoint" in health_cp["retraining_triggers"]
