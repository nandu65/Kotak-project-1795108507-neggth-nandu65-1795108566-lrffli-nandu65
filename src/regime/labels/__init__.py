"""
Independent Rule-Based Ground Truth Regime Classification Module.
Strictly based on Project Brief Section A1.4 indicator table.
"""
from .rule_based import compute_rule_based_regimes, REGIME_NAMES

__all__ = ["compute_rule_based_regimes", "REGIME_NAMES"]
