"""
Unit tests for Step 4 Regime Models:
- GaussianRegimeHMM
- BayesianRegimeHMM
- MarkovSwitchingVAR
- BayesianDeepEnsemble
- FoundationRegimeHead
"""
import numpy as np
import pytest
from src.regime.models.hmm_model import GaussianRegimeHMM
from src.regime.models.bayesian_hmm import BayesianRegimeHMM
from src.regime.models.ms_var import MarkovSwitchingVAR
from src.regime.models.bnn_model import BayesianDeepEnsemble
from src.regime.models.foundation_model import FoundationRegimeHead


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    T = 200
    # Simulate return series with 2 alternating volatility states
    rets = np.concatenate(
        [
            rng.normal(0.001, 0.008, 100),
            rng.normal(-0.002, 0.025, 100),
        ]
    )
    # Multivariate features (T, 3)
    vix = np.concatenate([rng.uniform(12, 16, 100), rng.uniform(22, 35, 100)])
    breadth = np.concatenate([rng.uniform(60, 80, 100), rng.uniform(20, 40, 100)])
    X_multi = np.column_stack([rets, vix, breadth])
    return rets.reshape(-1, 1), X_multi


def test_gaussian_hmm(synthetic_data):
    rets, _ = synthetic_data
    hmm = GaussianRegimeHMM(n_regimes=5, n_iter=30, seed=42)
    hmm.fit(rets)

    assert hmm.is_fitted
    probs = hmm.predict_proba(rets)
    assert probs.shape == (len(rets), 5)
    # Check simplex constraints
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0.0).all()

    # Label switching check: canonical states are ordered by economic return signature
    # State 0 (Risk-On) mean should be greater than State 4 (Risk-Off) mean
    assert hmm.means[0, 0] >= hmm.means[-1, 0]

    # Implied durations
    durations = hmm.get_regime_durations()
    assert len(durations) == 5
    assert all(d > 0 for d in durations.values())


def test_bayesian_hmm(synthetic_data):
    rets, _ = synthetic_data
    bhmm = BayesianRegimeHMM(n_regimes=5, n_draws=40, n_chains=2, seed=42)
    bhmm.fit(rets)

    assert bhmm.is_fitted
    probs = bhmm.predict_proba(rets)
    assert probs.shape == (len(rets), 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Diagnostics check
    diag = bhmm.diagnostics
    assert "r_hat" in diag
    assert "ess" in diag
    assert "divergences" in diag

    # Credible intervals check
    ci = bhmm.predict_proba_with_ci(rets, level=0.90)
    assert "mean" in ci
    assert "lower" in ci
    assert "upper" in ci
    assert ci["lower"].shape == (len(rets), 5)
    assert ci["upper"].shape == (len(rets), 5)


def test_markov_switching_var(synthetic_data):
    _, X_multi = synthetic_data
    rsvar = MarkovSwitchingVAR(n_regimes=5, lags=1, n_iter=20, seed=42)
    rsvar.fit(X_multi)

    assert rsvar.is_fitted
    probs = rsvar.predict_proba(X_multi)
    assert probs.shape == (len(X_multi), 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    # Cross-asset covariance recovery
    cov0 = rsvar.get_regime_covariance(0)
    assert cov0.shape == (3, 3)

    # Impulse responses
    irfs = rsvar.compute_impulse_response(shock_dim=0, horizon=5)
    assert len(irfs) == 5
    assert irfs["Risk-On"].shape == (5, 3)


def test_bayesian_deep_ensemble(synthetic_data):
    rets, X_multi = synthetic_data
    # Use decoded HMM states as targets
    hmm = GaussianRegimeHMM(n_regimes=5, n_iter=20, seed=42).fit(rets)
    y_labels = hmm.predict(rets)

    bnn = BayesianDeepEnsemble(n_regimes=5, n_models=4, epochs=15, seed=42)
    bnn.fit(X_multi, y_labels)

    assert bnn.is_fitted
    mean_p, epistemic, aleatoric = bnn.predict_with_uncertainty(X_multi)

    assert mean_p.shape == (len(X_multi), 5)
    assert epistemic.shape == (len(X_multi), 5)
    assert aleatoric.shape == (len(X_multi), 5)
    assert np.allclose(mean_p.sum(axis=1), 1.0, atol=1e-4)
    # Disagreement & noise are non-negative
    assert (epistemic >= 0.0).all()
    assert (aleatoric >= 0.0).all()


def test_foundation_regime_head(synthetic_data):
    rets, _ = synthetic_data
    hmm = GaussianRegimeHMM(n_regimes=5, n_iter=20, seed=42).fit(rets)
    y_labels = hmm.predict(rets)

    f_head = FoundationRegimeHead(n_regimes=5, window_size=32, d_model=16, seed=42)
    f_head.fit(rets, y_labels)

    assert f_head.is_fitted
    probs = f_head.predict_proba(rets)
    assert probs.shape == (len(rets), 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    # Probing diagnostic
    probe_diag = f_head.probe_diagnostics(rets, rets[:, 0])
    assert "probe_r2" in probe_diag
    assert probe_diag["probe_r2"] >= 0.0
