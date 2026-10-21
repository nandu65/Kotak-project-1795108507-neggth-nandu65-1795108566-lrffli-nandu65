"""Ensemble and proper scoring rules module."""
from .stacker import (
    BayesianModelAverager,
    ClimatologyBenchmark,
    ConstrainedStacker,
    EnsembleRegimeEngine,
    PersistenceBenchmark,
    VolThresholdBenchmark,
    compute_brier_decomposition,
    compute_brier_score,
    compute_ece,
    compute_log_loss,
    compute_ranked_probability_score,
    compute_skill_score,
    decompose_uncertainty_entropy,
    generate_reliability_curve,
    log_ensemble_to_mlflow,
)

__all__ = [
    "BayesianModelAverager",
    "ConstrainedStacker",
    "EnsembleRegimeEngine",
    "ClimatologyBenchmark",
    "PersistenceBenchmark",
    "VolThresholdBenchmark",
    "compute_log_loss",
    "compute_brier_score",
    "compute_ranked_probability_score",
    "compute_brier_decomposition",
    "compute_skill_score",
    "compute_ece",
    "decompose_uncertainty_entropy",
    "generate_reliability_curve",
    "log_ensemble_to_mlflow",
]

