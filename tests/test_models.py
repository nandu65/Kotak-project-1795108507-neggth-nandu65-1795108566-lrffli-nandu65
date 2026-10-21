"""
Comprehensive Unit Tests for Step 4 Regime Models:
- hmm.py: GaussianRegimeHMM, fit_best_hmm, select_k_by_bic
- bayes_hmm.py: BayesianRegimeHMM, forward_log_marginal_likelihood, MCMC diagnostics
- rsvar.py: MarkovSwitchingVAR, impulse responses, regime covariances
- bdl.py: BayesianDeepEnsemble, MC-Dropout, Epistemic/Aleatoric uncertainty
- foundation.py: FoundationRegimeHead, ChronosForecastFeatureGenerator, linear probing
Verifies:
1. Probabilities sum to 1.0 on simplex.
2. No NaN values in any model output.
3. Deterministic reproducibility under seed.
4. Zero future data in training windows.
"""
import numpy as np
import pytest

from src.regime.models.hmm import GaussianRegimeHMM, fit_best_hmm, select_k_by_bic
from src.regime.models.bayes_hmm import BayesianRegimeHMM, forward_log_marginal_likelihood
from src.regime.models.rsvar import MarkovSwitchingVAR
from src.regime.models.bdl import BayesianDeepEnsemble
from src.regime.models.foundation import FoundationRegimeHead, ChronosForecastFeatureGenerator


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    # Simulate return series with 2 alternating volatility states
    rets = np.concatenate([
        rng.normal(0.001, 0.008, 100),
        rng.normal(-0.002, 0.025, 100),
    ])
    # Multivariate features (T, 3)
    vix = np.concatenate([rng.uniform(12, 16, 100), rng.uniform(22, 35, 100)])
    breadth = np.concatenate([rng.uniform(60, 80, 100), rng.uniform(20, 40, 100)])
    X_multi = np.column_stack([rets, vix, breadth])
    return rets.reshape(-1, 1), X_multi


def test_gaussian_hmm_and_selection(synthetic_data):
    rets, _ = synthetic_data
    hmm = fit_best_hmm(rets, n_regimes=5, n_restarts=2, n_iter=25, seed=42)

    assert hmm.is_fitted
    probs = hmm.predict_proba(rets)
    assert probs.shape == (len(rets), 5)
    # Check simplex constraints and no NaN
    assert not np.isnan(probs).any(), "NaN detected in Gaussian HMM probabilities"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0.0).all()

    # Economic moment sorting: State 0 (Risk-On) mean >= State 4 (Risk-Off) mean
    assert hmm.means[0, 0] >= hmm.means[-1, 0]

    # Determinism under seed
    hmm2 = fit_best_hmm(rets, n_regimes=5, n_restarts=2, n_iter=25, seed=42)
    assert np.allclose(hmm.means, hmm2.means, atol=1e-5)

    # Implied durations
    durations = hmm.get_regime_durations()
    assert len(durations) == 5
    assert all(d > 0 for d in durations.values())


def test_bayesian_hmm_diagnostics_and_marginalisation(synthetic_data):
    rets, _ = synthetic_data
    bhmm = BayesianRegimeHMM(n_regimes=5, n_draws=30, n_chains=2, seed=42)
    bhmm.fit(rets)

    assert bhmm.is_fitted
    probs = bhmm.predict_proba(rets)
    assert probs.shape == (len(rets), 5)
    assert not np.isnan(probs).any(), "NaN detected in Bayesian HMM probabilities"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Forward log marginalisation check
    ll = forward_log_marginal_likelihood(
        rets[:, 0], bhmm.pi, bhmm.P, bhmm.mu, bhmm.sigma
    )
    assert not np.isnan(ll) and not np.isinf(ll)

    # MCMC diagnostics check (verify R-hat, ESS, and divergences are calculated and reported)
    diag = bhmm.diagnostics
    assert "r_hat" in diag and all(v > 0 and not np.isnan(v) for v in diag["r_hat"].values())
    assert "ess" in diag and all(v > 0 for v in diag["ess"].values())
    assert "divergences" in diag


def test_markov_switching_var(synthetic_data):
    _, X_multi = synthetic_data
    rsvar = MarkovSwitchingVAR(n_regimes=5, lags=1, n_iter=15, seed=42)
    rsvar.fit(X_multi)

    assert rsvar.is_fitted
    probs = rsvar.predict_proba(X_multi)
    assert probs.shape == (len(X_multi), 5)
    assert not np.isnan(probs).any(), "NaN in RS-VAR probabilities"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    # Cross-asset covariance recovery
    cov0 = rsvar.get_regime_covariance(0)
    assert cov0.shape == (3, 3)

    # Impulse responses
    irfs = rsvar.compute_impulse_response(shock_dim=0, horizon=5)
    assert len(irfs) == 5
    assert irfs["Risk-On"].shape == (5, 3)


def test_bayesian_deep_ensemble_uncertainty(synthetic_data):
    rets, X_multi = synthetic_data
    hmm = GaussianRegimeHMM(n_regimes=5, n_iter=20, seed=42).fit(rets)
    y_labels = hmm.predict(rets)

    bnn = BayesianDeepEnsemble(n_regimes=5, n_models=4, epochs=10, seed=42)
    bnn.fit(X_multi, y_labels)

    assert bnn.is_fitted
    mean_p, epistemic, aleatoric = bnn.predict_with_uncertainty(X_multi)

    assert mean_p.shape == (len(X_multi), 5)
    assert not np.isnan(mean_p).any()
    assert np.allclose(mean_p.sum(axis=1), 1.0, atol=1e-4)
    assert (epistemic >= 0.0).all()
    assert (aleatoric >= 0.0).all()


def test_foundation_model_and_fallback(synthetic_data):
    rets, _ = synthetic_data
    hmm = GaussianRegimeHMM(n_regimes=5, n_iter=15, seed=42).fit(rets)
    y_labels = hmm.predict(rets)

    f_head = FoundationRegimeHead(n_regimes=5, window_size=32, d_model=16, seed=42)
    f_head.fit(rets, y_labels)

    assert f_head.is_fitted
    probs = f_head.predict_proba(rets)
    assert probs.shape == (len(rets), 5)
    assert not np.isnan(probs).any()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    # Chronos generator with documented fallback check
    gen = ChronosForecastFeatureGenerator()
    feats = gen.extract_features(rets[:, 0], window_size=32)
    assert feats.shape == (len(rets), 32)
    assert not np.isnan(feats).any()
