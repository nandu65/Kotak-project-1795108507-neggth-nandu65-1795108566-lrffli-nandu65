"""Backtesting, tactical overlay, and Monte Carlo simulation package."""
from .engine import (
    AllocationOverlay,
    RegimeConditionedMC,
    WalkForwardBacktester,
    compute_deflated_sharpe_ratio,
    compute_probability_of_backtest_overfitting,
)

__all__ = [
    "AllocationOverlay",
    "WalkForwardBacktester",
    "RegimeConditionedMC",
    "compute_deflated_sharpe_ratio",
    "compute_probability_of_backtest_overfitting",
]
