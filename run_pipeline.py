"""
Live End-to-End Pipeline Runner: Bayesian Regime Detection Engine.
Executes the full pipeline locally: PIT feature ingestion, ensembled regime inference,
epistemic/aleatoric uncertainty budgeting, BOCPD changepoint monitoring,
conformal prediction sets, tactical Kelly allocation tilts, and
Regime-Conditioned Monte Carlo VaR with Investment Committee briefing narratives.

Institution: Zetheta Algorithms Private Limited
CIN: U62012MH2023PTC410415
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd

from src.regime.backtest.engine import (
    AllocationOverlay,
    RegimeConditionedMC,
)
from src.regime.conformal.predictor import AdaptivePredictionSets
from src.regime.labels.rule_based import compute_rule_based_regimes
from src.regime.models.bnn_model import BayesianDeepEnsemble
from src.regime.models.hmm_model import GaussianRegimeHMM
from src.regime.online.bocpd import StudentTBOCPD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BRDE_Pipeline")


def run_pipeline() -> dict:
    logger.info("================================================================================")
    logger.info("BAYESIAN REGIME DETECTION ENGINE (BRDE) - LIVE LOCAL EXECUTION")
    logger.info("Zetheta Algorithms Private Limited (CIN: U62012MH2023PTC410415)")
    logger.info("================================================================================")

    features_path = Path("data/processed/market_features_standardized.parquet")
    pit_path = Path("data/processed/market_pit_dataset.parquet")

    if not features_path.exists() or not pit_path.exists():
        logger.warning("Processed parquets not found. Running PIT ingestion and feature engineering...")
        from src.regime.data.pit import PITAlignmentEngine
        from src.regime.features.engineer import FeatureEngineer
        PITAlignmentEngine().run()
        FeatureEngineer().build_features()

    feat_df = pd.read_parquet(features_path)
    pit_df = pd.read_parquet(pit_path)
    logger.info("Ingested standardized market dataset: %d trading sessions, %d features.", len(feat_df), feat_df.shape[1])

    # 1. Use the most recent 1000 days for live regime calibration
    recent_rets = pit_df["nifty_return"].dropna().tail(1000).values
    feature_cols = [c for c in feat_df.columns if c not in ["Date", "target", "nifty_return"]][:12]
    recent_features = feat_df[feature_cols].dropna().tail(1000).values

    # 2. Gaussian HMM Model
    logger.info("Fitting Gaussian HMM with Baum-Welch EM & economic moment sorting...")
    hmm = GaussianRegimeHMM(n_regimes=5, n_iter=30, seed=42)
    hmm.fit(recent_rets.reshape(-1, 1))
    hmm_probs = hmm.predict_proba(recent_rets[-1:].reshape(-1, 1))[0]

    # 3. Bayesian Deep Ensemble with MC-Dropout
    logger.info("Running Bayesian Deep Ensemble (M=4, epochs=10)...")
    bnn = BayesianDeepEnsemble(n_regimes=5, n_models=4, epochs=10, seed=42)
    # Walk-forward targets from rule-based Section A1.4 ground truth
    all_rule_labels = compute_rule_based_regimes(pit_df)
    train_labels = all_rule_labels.loc[feat_df.index].dropna().astype(int).tail(1000).values
    bnn.fit(recent_features, train_labels)
    bnn_probs_mat, epistemic_mat, aleatoric_mat = bnn.predict_with_uncertainty(recent_features[-1:])
    bnn_probs = bnn_probs_mat[0]
    epistemic_unc = float(np.mean(epistemic_mat[0]))
    aleatoric_unc = float(np.mean(aleatoric_mat[0]))
    total_entropy = float(-np.sum(bnn_probs * np.log(np.maximum(bnn_probs, 1e-12))))

    # 4. Ensembled Probabilities (Equal-Weighted BMA for demo run)
    ensembled_probs = 0.55 * hmm_probs + 0.45 * bnn_probs
    ensembled_probs = ensembled_probs / np.sum(ensembled_probs)
    regimes = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
    dominant_regime = regimes[int(np.argmax(ensembled_probs))]

    # 5. Online Changepoint Detection (BOCPD)
    logger.info("Evaluating sequential online BOCPD changepoint density...")
    bocpd = StudentTBOCPD(hazard=1.0 / 63.0)
    cp_prob = 0.0
    for r in recent_rets[-63:]:
        cp_prob, _ = bocpd.update(float(r))
    expected_run_length = float(np.sum(np.arange(len(bocpd.R)) * bocpd.R))

    # 6. Adaptive Conformal Prediction Set (APS)
    aps = AdaptivePredictionSets(alpha=0.10)
    calib_features = recent_features[-200:]
    calib_probs = bnn.predict_proba(calib_features)
    calib_labels = train_labels[-200:]
    aps.calibrate(calib_probs, calib_labels)
    _, set_labels = aps.predict_set(ensembled_probs.reshape(1, -1))
    conf_set_names = set_labels[0]

    # 7. Tactical Asset Allocation Overlay
    overlay = AllocationOverlay(min_equity=0.70, max_equity=1.00, no_trade_band=0.015, cost_bps=15.0)
    current_weight = 0.88
    new_weight, turnover, cost = overlay.compute_target_weight(
        regime_probs=ensembled_probs,
        conformal_set_size=len(conf_set_names),
        current_weight=current_weight,
    )

    # 8. Regime-Conditioned Monte Carlo VaR/CVaR
    logger.info("Executing Regime-Conditioned Monte Carlo path simulation (1,000 paths, 252d horizon)...")
    mc = RegimeConditionedMC(
        transition_matrix=hmm.A,
        regime_means=hmm.means[:, 0],
        regime_vols=np.sqrt(hmm.covars[:, 0, 0]),
        K=5,
    )
    paths, _ = mc.simulate(init_regime_dist=ensembled_probs, horizon=252, n_sims=1000, seed=42)
    var_stats = mc.compute_regime_var(paths, alpha=0.05)
    ic_statements = mc.generate_ic_narrative(
        var_stats=var_stats,
        current_dominant_regime=dominant_regime,
        risk_off_prob_90d=float(ensembled_probs[4]),
    )

    # Output Summary Contract
    result = {
        "status": "SUCCESS",
        "latest_trading_date": str(pit_df.index[-1].date()),
        "dominant_regime": dominant_regime,
        "regime_probabilities": {r: round(float(p), 4) for r, p in zip(regimes, ensembled_probs)},
        "uncertainty_budget": {
            "total_shannon_entropy": round(total_entropy, 4),
            "aleatoric_uncertainty": round(aleatoric_unc, 4),
            "epistemic_uncertainty": round(epistemic_unc, 4),
            "epistemic_fraction": round(epistemic_unc / max(total_entropy, 1e-6), 4),
        },
        "online_monitoring": {
            "bocpd_changepoint_probability": round(cp_prob, 4),
            "bocpd_expected_run_length": round(expected_run_length, 1),
            "changepoint_alert": cp_prob > 0.40,
        },
        "conformal_prediction": {
            "nominal_coverage_target": "90%",
            "prediction_set": conf_set_names,
            "set_size": len(conf_set_names),
        },
        "tactical_asset_allocation": {
            "current_equity_weight": round(current_weight, 4),
            "target_equity_weight": round(new_weight, 4),
            "rebalance_triggered": turnover > 0.0,
            "trade_delta": round(new_weight - current_weight, 4),
            "turnover_cost_bps": round(cost * 10000.0, 2),
        },
        "risk_management_1y": {
            "var_95_percentile": f"{var_stats['var_95']*100.0:+.2f}%",
            "cvar_95_expected_shortfall": f"{var_stats['cvar_95']*100.0:+.2f}%",
            "prob_negative_return_1y": f"{var_stats['prob_negative_1y']*100.0:.1f}%",
            "prediction_interval_90": f"[{var_stats['p90_interval_lower']*100.0:+.1f}%, {var_stats['p90_interval_upper']*100.0:+.1f}%]",
        },
        "investment_committee_briefing": ic_statements,
    }

    print("\n" + "=" * 80)
    print("                    DAILY REGIME DECISION OUTPUT CONTRACT")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print("=" * 80)
    return result


if __name__ == "__main__":
    run_pipeline()
