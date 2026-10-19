"""
Unit tests for Backtesting, Tactical Allocation Overlay, and Monte Carlo Engine:
- AllocationOverlay
- WalkForwardBacktester
- Deflated Sharpe Ratio
- RegimeConditionedMC
"""
import numpy as np
import pandas as pd
import pytest
from src.regime.backtest.engine import (
    AllocationOverlay,
    RegimeConditionedMC,
    WalkForwardBacktester,
    compute_deflated_sharpe_ratio,
    compute_probability_of_backtest_overfitting,
)


def test_allocation_overlay_bounds_and_no_trade_band():
    overlay = AllocationOverlay(min_equity=0.70, max_equity=1.00, no_trade_band=0.015, cost_bps=15.0)

    # 1. 100% Risk-On probability with high conviction (set size = 1)
    p_risk_on = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    w, turn, cost = overlay.compute_target_weight(p_risk_on, conformal_set_size=1, current_weight=0.85)

    assert w > 0.85
    assert w <= 1.00
    assert turn > 0
    assert cost > 0

    # 2. Rebalancing with tiny difference < 1.5% must trigger no-trade band
    # If target is 0.92 and current is 0.915 (diff = 0.005 < 0.015), weight must stay 0.915 with zero turnover
    w_band, turn_band, cost_band = overlay.compute_target_weight(
        p_risk_on, conformal_set_size=1, current_weight=w - 0.005
    )
    assert w_band == w - 0.005
    assert turn_band == 0.0
    assert cost_band == 0.0

    # 3. 100% Risk-Off probability
    p_risk_off = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    w_off, turn_off, cost_off = overlay.compute_target_weight(p_risk_off, conformal_set_size=1, current_weight=0.85)
    assert w_off >= 0.70
    assert w_off < 0.85


def test_deflated_sharpe_ratio_discounting():
    # As the number of trials increases, DSR must decrease (penalising multiple testing)
    dsr_1_trial = compute_deflated_sharpe_ratio(sharpe=0.10, n_obs=1000, n_trials=1)
    dsr_100_trials = compute_deflated_sharpe_ratio(sharpe=0.10, n_obs=1000, n_trials=100)

    assert dsr_1_trial > dsr_100_trials


def test_walk_forward_backtester():
    dates = pd.date_range("2022-01-01", periods=250, freq="B")
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.0005, 0.012, len(dates)), index=dates)

    # Synthetic alternating regime probabilities: (250, 5)
    probs = np.zeros((len(dates), 5))
    probs[:150, 0] = 0.80  # Risk-On first 150 days
    probs[:150, 1:] = 0.05
    probs[150:, 4] = 0.80  # Risk-Off last 100 days
    probs[150:, :4] = 0.05

    backtester = WalkForwardBacktester()
    res = backtester.run_backtest(rets, probs)

    assert "metrics" in res
    assert "paths" in res

    metrics = res["metrics"]
    assert "information_ratio" in metrics
    assert "strategy_sharpe" in metrics
    assert "strategy_max_drawdown" in metrics
    assert "deflated_sharpe_ratio" in metrics
    assert "annual_turnover" in metrics


def test_regime_conditioned_monte_carlo():
    K = 5
    P = np.eye(K) * 0.90 + (1.0 - np.eye(K)) * (0.10 / (K - 1))
    mu = np.array([0.0008, 0.0004, 0.0, 0.0005, -0.0012])
    sig = np.array([0.008, 0.010, 0.012, 0.018, 0.024])

    mc = RegimeConditionedMC(transition_matrix=P, regime_means=mu, regime_vols=sig, K=K)
    init_p = np.array([0.70, 0.15, 0.10, 0.03, 0.02])  # Risk-On initial state

    paths, states = mc.simulate(init_p, horizon=100, n_sims=500, seed=42)
    assert paths.shape == (500, 100)
    assert states.shape == (500, 100)

    var_stats = mc.compute_regime_var(paths)
    assert "var_95" in var_stats
    assert "cvar_95" in var_stats
    assert var_stats["cvar_95"] <= var_stats["var_95"]  # Expected shortfall is more severe than VaR

    narrative = mc.generate_ic_narrative(var_stats, current_dominant_regime="Risk-On")
    assert len(narrative) == 4
    assert any("Risk-On" in line for line in narrative)
