"""Backtesting, tactical overlay, and Monte Carlo simulation package."""
from .engine import (
    AllocationOverlay,
    RegimeConditionedMC,
    WalkForwardBacktester,
    compute_deflated_sharpe_ratio,
    compute_probability_of_backtest_overfitting,
    evaluate_stress_episodes,
    run_block_bootstrap_monte_carlo,
)

__all__ = [
    "AllocationOverlay",
    "WalkForwardBacktester",
    "RegimeConditionedMC",
    "compute_deflated_sharpe_ratio",
    "compute_probability_of_backtest_overfitting",
    "run_block_bootstrap_monte_carlo",
    "evaluate_stress_episodes",
]

