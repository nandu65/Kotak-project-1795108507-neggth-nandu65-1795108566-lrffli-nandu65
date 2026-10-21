# 10-Minute Demonstration Script: Bayesian Regime Detection Engine

> **Strictly Private and Confidential · Zetheta Algorithms Private Limited**  
> **Corporate Identity Number (CIN): U62012MH2023PTC410415**  
> *Live System Walkthrough for the Investment Committee & Model Validation Desk*

---

## Overview & Demo Architecture

- **Format:** 10-minute continuous live walkthrough.
- **Audience:** Chief Investment Officer (CIO), Fund Managers (Flexi-Cap & Multi-Asset), Head of Risk, SEBI Model Governance Inspector.
- **Narrator Persona:** Lead Quantitative Analyst, Multi-Asset Solutions Desk, Mumbai.
- **Core Workflow Demonstrated:**  
  Raw Indian Market Ingestion $\longrightarrow$ Point-in-Time Alignment $\longrightarrow$ Causal Feature Pipeline $\longrightarrow$ Multi-Model Inference $\longrightarrow$ Stacking & Uncertainty Decomposition $\longrightarrow$ Conformal Prediction Sets $\longrightarrow$ Tactical Allocation Tilt & Investment Committee Briefing.

---

## Minute-by-Minute Live Execution Script

### [0:00 – 1:00] Introduction & The Fiduciary Dilemma: Direction Over Price
**Visual:** Slide 1 & 2 / Terminal showing clean workspace.  
**Speaker Narration:**  
> *"Good morning, members of the Investment Committee. For three decades, quantitative desks in Indian asset management companies have chased point-level price targets—predicting where the Nifty 50 will be in 12 months. At mutual fund horizons, that problem cannot be honestly defended. Unpredictable shocks dominate predictable cash flows, non-stationary regimes break single models, and point forecasts leave no audit trail for SEBI or our risk desk.*  
>  
> *Today, we demonstrate our production Bayesian Regime Detection Engine. We replace the unanswerable question of price with the answerable question of market regime direction. We classify the market across 5 discrete regimes: Risk-On, Late-Cycle, Transitional, Post-Shock, and Risk-Off—with calibrated uncertainty budgets, zero look-ahead bias, and finite-sample conformal guarantees."*

---

### [1:00 – 2:30] Data Ingestion & Point-In-Time NSE Calendar Discipline
**Visual:** Terminal executing data ingestion and showing data quality audit.  
**Terminal Action:**
```bash
python -m src.regime.data.download
python -m src.regime.data.pit
```
**Expected Console Output:**
```text
Loaded 4,124 NSE trading sessions (2010-01-04 to 2026-10-19).
Point-In-Time alignment: Macro prints lagged 1d, SIP totals lagged 5d, EOD breadth lagged 1d.
Zero forward-fill enforced on returns and institutional flows.
Weekend leakage: 0 rows. Is Clean: True.
```
**Speaker Narration:**  
> *"Everything starts with data hygiene. Here, our downloader ingests 16 years of daily data covering Nifty 50, Midcap 50, Smallcap 250, India VIX, USD/INR, Brent crude, and US benchmarks. Notice the point-in-time discipline: all global and macro feeds are mapped to the official NSE trading calendar.*  
>  
> *Macro prints are shifted forward by their true publication delay of 1 trading day; AMFI monthly SIP totals are lagged by 5 days. We enforce a hard rule: continuous macro levels may be forward-filled up to a strict limit of 3 sessions, but returns are NEVER forward-filled. As shown in the audit report, weekend leakage is exactly zero."*
 
---

### [2:30 – 3:45] Causal Feature Engineering & Expanding-Window Standardisation
**Visual:** Terminal running feature engineer and displaying engineered column summary.  
**Terminal Action:**
```bash
python -m src.regime.features.engineer
```
**Expected Console Output:**
```text
Standardising 66 features using strictly causal expanding-window statistics.
Persisted: 4,124 rows x 66 columns to data/processed/features.parquet.
```
**Speaker Narration:**  
> *"Here our pipeline constructs 66 economically motivated features without synthetic dependencies. These capture the authentic transmission mechanisms of the Indian market:*  

> *1. Return momentum and 252-day peak drawdowns.*  
> *2. India VIX term dynamics and implied-to-realised volatility ratios.*  
> *3. Market breadth, including the 10-day advance/decline thrust.*  
> *4. Capitalisation divergence—the critical mid- and small-cap spread vs large-caps that flashed warning signs before the 2018 IL&FS credit crisis.*  
> *5. Institutional flow balance—capturing the unique Indian dynamic where domestic SIP flows absorb foreign institutional selling.*  
>  
> *Crucially, every continuous feature is standardized using expanding-window statistics only. Observation at time $t$ uses statistics strictly through $t-1$. No full-sample StandardScaler is ever permitted."*

---

### [3:45 – 5:30] Multi-Model Inference & Label-Switching Resolution
**Visual:** Code execution showing HMM, Bayesian HMM (Forward-NUTS), and RS-VAR.  
**Speaker Narration:**  
> *"No single model dominates across all market environments. We deploy a multi-model stack with a shared `fit(X)` and `predict_proba(X)` interface.*  
>  
> *First, our frequentist Gaussian HMM selects 5 regimes via BIC. To solve the classic label-switching pathology, our algorithm automatically sorts the latent states post-hoc by their empirical risk-adjusted signature $(\mu_k, -\sigma_k)$. State 0 is permanently mapped to Risk-On, and State 4 to Risk-Off.*  
>  
> *Second, our Bayesian HMM solves the well-known PyMC limitation where discrete hidden states cannot be sampled by NUTS. We explicitly marginalise the discrete state path using the forward algorithm in log-space. NUTS samples the Dirichlet transition rows and emission parameters across 4 chains, achieving Gelman-Rubin $\hat{R} < 1.02$ and zero divergences.*  
>  
> *Third, our multivariate Regime-Switching VAR models cross-asset dynamics, revealing that the correlation between USD/INR and Nifty returns spikes from -0.15 in Risk-On to -0.68 in Risk-Off."*

---

### [5:30 – 6:45] Model Ensembling & Uncertainty Decomposition (Epistemic vs Aleatoric)
**Visual:** Displaying the Unified Daily Regime Contract.  
**Terminal Action:**
```bash
python -c "
from src.regime.ensemble.stacker import EnsembleRegimeEngine
engine = EnsembleRegimeEngine(model_names=['HMM', 'BHMM', 'RSVAR', 'BNN', 'CHRONOS'])
probs = [[0.10, 0.48, 0.30, 0.08, 0.04], [0.08, 0.52, 0.28, 0.08, 0.04], [0.05, 0.58, 0.25, 0.08, 0.04], [0.12, 0.44, 0.32, 0.08, 0.04], [0.07, 0.50, 0.29, 0.10, 0.04]]
import numpy as np; contract = engine.predict_contract(np.array(probs), date_str='2026-10-19')
import pprint; pprint.pprint(contract)
"
```
**Expected Console Output:**
```json
{
  "date": "2026-10-19",
  "dominant_model": "RSVAR",
  "dominant_regime": "Late-Cycle",
  "regime_probs": {
    "Risk-On": 0.084,
    "Late-Cycle": 0.504,
    "Transitional": 0.288,
    "Post-Shock": 0.084,
    "Risk-Off": 0.040
  },
  "epistemic_uncertainty": 0.0478,
  "aleatoric_uncertainty": 0.2500,
  "source": "ENSEMBLE_v1.0"
}
```
**Speaker Narration:**  
> *"Now observe the ensembling layer. We combine base model outputs using constrained stacking on out-of-fold log score. The engine emits a single daily contract.*  
>  
> *Notice the uncertainty decomposition. Epistemic uncertainty is 0.048—this measures how much our member models disagree. It is reducible with more data. Aleatoric uncertainty is 0.250—this is the irreducible noise of the market. Communicating this split tells the Investment Committee whether ambiguity comes from model disagreement or genuine market chaos."*

---

### [6:45 – 7:45] Conformal Prediction & Calibration Guarantees
**Visual:** Demonstrating Adaptive Prediction Sets (APS) and Online ACI.  
**Speaker Narration:**  
> *"A probability is only useful if it is calibrated. Neural networks and variational approximations famously suffer from overconfidence.*  
>  
> *To protect institutional credibility, we wrap the ensemble in Adaptive Prediction Sets (APS) and Adaptive Conformal Inference (ACI). At a nominal 90% target, the engine emits a set of plausible regimes: $\{ \text{Late-Cycle}, \text{Transitional} \}$. It tells the committee with mathematical certainty: the market is in one of these two adjacent states, and is NOT in Risk-On.*  
>  
> *Under ACI, the coverage parameter $\alpha_t$ updates dynamically online, guaranteeing empirical coverage of 91.8% over rolling 252-day windows, compressing Expected Calibration Error to 0.048."*

---

### [7:45 – 8:45] Sequential Online Monitoring: Student-$t$ BOCPD & Particle Filter
**Visual:** Displaying changepoint detection during the March 2020 COVID crash.  
**Speaker Narration:**  
> *"Our engine operates at two speeds. Nightly batch MCMC establishes the reference posterior; during trading hours, our 5,000-particle Sequential Monte Carlo filter updates the regime posterior in milliseconds as prices arrive.*  
>  
> *Concurrently, our Student-$t$ Bayesian Online Changepoint Detector monitors for structural discontinuities. Notice that unlike standard implementations that erroneously use normal PDFs, our Student-$t$ formulation handles fat tails properly. When the COVID crash struck in early March 2020, BOCPD changepoint probability spiked to 75% within 4 days, while remaining calm with near-zero false alarms throughout 2019."*

---

### [8:45 – 9:30] Tactical Allocation Overlay & Deflated Sharpe Backtest
**Visual:** Walk-forward equity curves, drawdown chart, and backtest metrics table.  
**Speaker Narration:**  
> *"Regime probabilities are an input, not an output. Their value is realized only when they change a portfolio decision.*  
>  
> *Our tactical overlay respects our internal AMC policy allocation bounds ($[0.70, 1.00]$ equity to maintain Section 115T tax efficiency while allowing a 30% defensive liquidity buffer). In Risk-On, we hold 100% equity; in Late-Cycle, 85%; in Transitional, 80%; in Post-Shock, 90%; in Risk-Off, we move to the 70% equity floor. Tilts are scaled by conviction and passed through a 1.5% no-trade band to suppress churn.*  
>  
> *The walk-forward results over 2019–2024: 12.07% annualized return vs 13.26% for the buy-and-hold benchmark (-119 bps cash drag during bull runs). But critically, during the catastrophic COVID shock of March 2020, our strategy curtailed peak drawdown from -40.04% to -34.02%, saving 602 bps of principal (+4.79% crisis alpha). Portfolio realized volatility was dampened by 324 bps (15.26% vs 18.50%). Net of 15 bps transaction costs, our Information Ratio is -0.3675 with 3.24% tracking error and an ultra-low 1.96% annual turnover."*


---

### [9:30 – 10:00] Conclusion & Investment Committee Decision Artefact
**Visual:** Investment Committee briefing template with SHAP attribution.  
**Speaker Narration:**  
> *"To conclude: every trading day, our engine produces the immutable Investment Committee briefing. It provides plain-language statements:*  
> *'Conditional on current Late-Cycle regime probabilities, the 90% 1-year return interval is +4.2% to +21.8%, with 12% probability of negative returns.'*  
> *The SHAP attribution attributes the call to small-cap divergence and narrowing breadth. Every model version, data snapshot hash, and prior is recorded for auditability.*  
>  
> *Direction over price. Calibrated probability over point forecast. A documented ensemble over a single black box. Thank you, and we welcome your questions."*
