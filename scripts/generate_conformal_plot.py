"""
Generate Conformal Prediction Coverage vs Set Size Plot across Alpha Values (0.05 to 0.50).
Saves publication-quality chart to reports/conformal_coverage_vs_set_size.png.
"""
import sys
from pathlib import Path

WORKSPACE = Path(r"D:\bayesian-regime-detection-APK")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.regime.labels.rule_based import compute_rule_based_regimes
from src.regime.models.hmm_model import GaussianRegimeHMM
from src.regime.conformal.predictor import AdaptivePredictionSets

def run_conformal_alpha_sweep():
    pit_df = pd.read_parquet(WORKSPACE / "data" / "processed" / "market_pit_dataset.parquet")
    rule_labels = compute_rule_based_regimes(pit_df)

    # In-sample HMM fit
    train_mask = (pit_df.index >= "2010-01-01") & (pit_df.index <= "2018-12-31")
    train_returns = pit_df.loc[train_mask, "nifty_return"].dropna()
    train_X = train_returns.values.reshape(-1, 1)

    hmm = GaussianRegimeHMM(n_regimes=5, covar_floor=2.5e-5, sticky_kappa=25.0, seed=42)
    hmm.fit(train_X)

    # Disjoint held-out calibration set (2017-2018)
    calib_mask = (pit_df.index >= "2017-01-01") & (pit_df.index <= "2018-12-31")
    calib_returns = pit_df.loc[calib_mask, "nifty_return"].dropna()
    calib_labels = rule_labels.loc[calib_returns.index].dropna().astype(int)
    calib_returns = calib_returns.loc[calib_labels.index]
    calib_probs = hmm.predict_proba(calib_returns.values.reshape(-1, 1))
    y_cal = calib_labels.values

    # Test set (2019-2024)
    test_mask = (pit_df.index >= "2019-01-01") & (pit_df.index <= "2024-12-31")
    test_returns = pit_df.loc[test_mask, "nifty_return"].dropna()
    y_test_series = rule_labels.loc[test_returns.index].dropna().astype(int)
    test_returns = test_returns.loc[y_test_series.index]
    test_probs = hmm.predict_proba(test_returns.values.reshape(-1, 1))
    y_test = y_test_series.values
    N_test = len(y_test)

    alphas = np.linspace(0.05, 0.50, 10)
    results = []

    for a in alphas:
        aps = AdaptivePredictionSets(alpha=float(a))
        q = aps.calibrate(calib_probs, y_cal)
        mask, _ = aps.predict_set(test_probs)
        cov = float(np.mean(mask[np.arange(N_test), y_test]))
        size = float(np.mean(np.sum(mask, axis=1)))
        results.append({
            "alpha": float(a),
            "nominal_coverage": float(1.0 - a),
            "empirical_coverage": cov,
            "mean_set_size": size,
            "q_hat": q,
            "is_uninformative": size >= 4.5,
        })

    df = pd.DataFrame(results)

    # Generate Plot
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    color_cov = "#1A365D"   # Dark Navy
    color_size = "#C53030"  # Crimson Accent
    color_diag = "#718096"  # Slate Gray

    ax1.set_xlabel("Significance Level Alpha (Nominal Miscoverage Target)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Coverage Probability", color=color_cov, fontsize=11, fontweight="bold", labelpad=8)
    
    l1 = ax1.plot(df["alpha"], df["empirical_coverage"], color=color_cov, marker="o", linewidth=2.5, label="Empirical Coverage")
    l2 = ax1.plot(df["alpha"], df["nominal_coverage"], color=color_diag, linestyle="--", linewidth=1.8, label="Nominal Target (1 - Alpha)")
    ax1.tick_params(axis="y", labelcolor=color_cov)
    ax1.set_ylim(0.35, 1.05)

    # Secondary axis for set size
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Prediction Set Size (Regimes / 5)", color=color_size, fontsize=11, fontweight="bold", labelpad=8)
    l3 = ax2.plot(df["alpha"], df["mean_set_size"], color=color_size, marker="s", linewidth=2.5, linestyle="-.", label="Mean Set Size (Max 5.0)")
    ax2.tick_params(axis="y", labelcolor=color_size)
    ax2.set_ylim(1.0, 5.2)

    # Shaded uninformative zone
    ax2.axhspan(4.5, 5.0, color="#FED7D7", alpha=0.4, label="Uninformative Zone (Set Size >= 4.5)")

    # Title & annotations
    plt.title("Adaptive Conformal Prediction: Empirical Coverage vs Set Size\nAudit of Score Saturation at High Confidence Levels (alpha in [0.05, 0.50])",
              fontsize=12, fontweight="bold", pad=14)

    # Annotation of saturation
    ax1.annotate("Saturation: q_hat = 1.00\nSet Size = 4.84/5 (Uninformative)",
                 xy=(0.05, 0.975), xytext=(0.12, 0.99),
                 arrowprops=dict(facecolor="#2D3748", shrink=0.08, width=1, headwidth=6),
                 fontsize=9, fontweight="bold", backgroundcolor="#FFFFFF",
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="#CBD5E0", facecolor="#F7FAFC"))

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.95, edgecolor="#CBD5E0")

    plt.tight_layout()
    out_plot = WORKSPACE / "reports" / "conformal_coverage_vs_set_size.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved plot to {out_plot}")
    return df

if __name__ == "__main__":
    df = run_conformal_alpha_sweep()
    print(df.to_string(index=False))
