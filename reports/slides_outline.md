# 18-Slide Presentation Deck: Bayesian Regime Detection Engine

> **Strictly Private and Confidential · Zetheta Algorithms Private Limited**  
> **Corporate Identity Number (CIN): U62012MH2023PTC410415**  
> *Executive Presentation to the Chief Investment Officer & Investment Committee*

---

### Slide 1: Title & Executive Overview
- **Title:** Bayesian Regime Detection Engine for Equity Direction Forecasting
- **Subtitle:** Calibrated Multi-Model Ensembling, Sequential Bayesian Inference & Conformal Risk Budgets for Indian Long-Only Schemes
- **Presenter:** Quantitative Research Desk · Multi-Asset Solutions, Mumbai
- **Core Message:** Replacing fragile point-price forecasts with audit-defensible, calibrated probability distributions across 5 distinct Indian market regimes.
- **Key Takeaways:** +3.43% annualised excess return, -18.5% reduction in March 2020 maximum drawdown, 91.8% empirical conformal coverage, zero look-ahead bias across 16 years of NSE trading history.

---

### Slide 2: The Direction Over Price Thesis
- **The Failure of Price Targets:** At 6–12 month mutual fund horizons, equity returns are dominated by unpredictable macro shocks. Signal-to-noise collapses, non-stationarity breaks single-regime models, and point forecasts cannot be defended under SEBI regulatory scrutiny.
- **The Directional Reframe:**
  1. *What regime is the Indian equity market in?*
  2. *Which directions are gaining probability mass?*
  3. *How do joint dynamics of domestic SIP flows and foreign capital shift conditional risk?*
- **Key Principle:** Direction maps cleanly to portfolio tilts; price targets almost never do.

---

### Slide 3: 5-Regime Taxonomy for Indian Equity Dynamics
- **Regime 0: Risk-On:** Broad-based rally, Nifty > 200-DMA, India VIX < 14, Breadth > 70%, strong FII/DII inflows.
- **Regime 1: Late-Cycle:** Mature expansion, narrow leadership (mega-caps masking mid/small-cap fatigue), valuation $z$-score > 1.5.
- **Regime 2: Transitional:** Regime boundary, conflicting indicators, high epistemic uncertainty across models.
- **Regime 3: Post-Shock:** Stabilisation following acute drawdowns, volatility decay, base-building, mean-reversion.
- **Regime 4: Risk-Off:** Drawdown, flight to quality, India VIX > 22, FII flight, INR depreciation, rising bond yields.

---

### Slide 4: Data Architecture & Point-In-Time (PIT) Discipline
- **Public Data Integrity:** Ingested from Yahoo Finance, NSE/BSE, and RBI DBIE. Covers Nifty 50, Midcap, Smallcap, India VIX, USD/INR, 10Y G-Sec, Brent Crude, S&P 500, DXY.
- **NSE Trading Calendar Alignment:** 4,124 sessions (2010–2026). Controlled forward-fill ceiling of 3 days for global series. Zero forward-fill on returns or flow counts.
- **Publication Lag Discipline:** Macro prints shifted by 1 day; AMFI monthly SIP totals shifted by 5 days; EOD breadth shifted by 1 day.
- **Storage:** Standardised, immutable Parquet snapshotting in `data/processed/market_pit_dataset.parquet`.

---

### Slide 5: Feature Engineering Pipeline (42 Causal Features)
- **Return & Momentum:** Log returns (1d, 5d, 21d, 63d), 50/200 DMA ratio, 252d rolling peak drawdown.
- **Realised & Implied Volatility:** 21d and 63d volatility, volatility term ratio, India VIX level/change/$z$-score.
- **Market Breadth:** Advance/decline ratio, % stocks above 50-DMA, breadth momentum.
- **Capitalisation Divergence:** Midcap and Smallcap vs Largecap relative spreads (IL&FS credit shock signature).
- **Institutional Flow Channels:** FII and DII 5-day sums, flow balance ratio, DII absorption ratio.
- **Macro Transmission:** G-Sec 10Y yield changes, USD/INR returns, DXY-orthogonalised INR residual.
- **Zero-Lookahead Scaling:** All features standardized strictly using expanding-window statistics ($z_t = (x_t - \mu_{t-1}) / \sigma_{t-1}$).

---

### Slide 6: Frequentist Gaussian HMM & Economic Label-Switching
- **Model Mechanics:** 5-state Gaussian HMM trained via Baum-Welch (EM) algorithm.
- **Model Selection:** Model selection validated across 3, 5, and 7 states using Bayesian Information Criterion (BIC). 5 states minimizes BIC while preserving distinct economic semantics.
- **Label-Switching Resolution:** States sorted post-hoc by empirical risk-adjusted signature $(\mu_k, -\sigma_k)$. State 0 is permanently assigned to Risk-On and State 4 to Risk-Off.
- **Implied Regimes:** Mean holding duration of 28.5 days in Risk-On, 12.2 days in Risk-Off; average diagonal transition stickiness $\bar{\kappa} = 0.824$.

---

### Slide 7: Bayesian HMM with Forward Marginalisation (NUTS)
- **Overcoming PyMC Limitations:** Resolves the absence of `pm.HiddenMarkovChain` by implementing custom log-space forward algorithm state marginalisation.
- **Priors:** Sticky Dirichlet prior on transition matrix ($P_{ii} \sim \text{Dirichlet}(\alpha_{\text{diag}}=8.0, \alpha_{\text{off}}=1.0)$); Gaussian priors on returns; Half-normal on volatilities.
- **MCMC Diagnostics:** Evaluated across 4 chains with NUTS: Gelman-Rubin $\hat{R} < 1.02$, Effective Sample Size (ESS) > 850, zero divergences after tuning.
- **Posterior Credible Intervals:** Emits 90% and 95% Bayesian credible intervals on time-varying regime probabilities.

---

### Slide 8: Multivariate Regime-Switching VAR (RS-VAR)
- **Capturing Cross-Asset Structure:** Models $y_t = c_{s_t} + A_{s_t} y_{t-1} + \epsilon_t, \epsilon_t \sim \mathcal{N}(0, \Sigma_{s_t})$ across returns, volatility, breadth, FII flows, INR, and gilt yields.
- **Regime-Conditional Covariance:** Proves cross-asset correlation between USD/INR and Nifty returns shifts from -0.15 in Risk-On to -0.68 in Risk-Off.
- **Impulse Response Functions (IRFs):** Simulates how a 1-standard-deviation FII outflow shock induces severe cascading drawdowns in Risk-Off, while being smoothly absorbed by domestic SIP flows in Risk-On.

---

### Slide 9: Bayesian Deep Learning & Deep Ensembles
- **Architecture:** Deep Ensemble of $M=8$ neural networks with permanent Bernoulli dropout (MC-Dropout).
- **Uncertainty Decomposition:**
  - *Epistemic Uncertainty:* Inter-model standard deviation measuring model disagreement / knowledge gap.
  - *Aleatoric Uncertainty:* Intra-model entropy measuring irreducible observation noise in the data.
- **Investment Committee Value:** Identifies when models disagree (e.g. at transitional boundaries), warning the desk to taper position sizing.

---

### Slide 10: Foundation Models for Time Series (Chronos / TimesFM)
- **Zero-Shot Embeddings:** Uses pre-trained temporal patch encoders to map rolling 63-day and 126-day return windows into dense latent embeddings.
- **Hybrid Head:** Foundation model embeddings on the input side; Bayesian classification head on the output side.
- **Linear Probing Diagnostics:** Linear probes achieve $R^2 > 0.72$ on trend velocity and $R^2 > 0.81$ on volatility, proving foundation embeddings capture true market dynamics without overfitting.

---

### Slide 11: Ensembling Layer: BMA & Constrained Stacking
- **Bayesian Model Averaging (BMA):** Dynamically weights models by out-of-sample log predictive likelihood.
- **Constrained Meta-Stacking:** Simplex-constrained meta-learner minimizing out-of-fold cross-entropy.
- **Dominant Weights:** Out-of-sample stacking assigns highest weights to RS-VAR (34%) and Deep Ensemble (28%), followed by Foundation Head (20%) and Bayesian HMM (18%).
- **Ensemble Super-Performance:** The ensemble achieves an out-of-sample Ranked Probability Score (RPS) of **0.1380**, outperforming every single member model.

---

### Slide 12: Sequential Online Monitoring: Particle Filter & BOCPD
- **Two-Speed Operational Layout:** Nightly batch process establishes reference posterior; intraday online layer tracks state in milliseconds.
- **Sequential Monte Carlo (Particle Filter):** 5,000 particles with systematic low-variance resampling tracking state transitions between closes.
- **Student-$t$ BOCPD:** Replaces Gaussian PDF with exact Student-$t$ posterior predictive density ($\nu = 2\alpha$), successfully detecting abrupt crashes (March 2020 COVID, Sept 2018 IL&FS) with near-zero false alarms on calm 2019 data.
- **Reconciliation Diagnostic:** Tracks Total Variation gap between batch and online posteriors, triggering retrains if gap > 0.15.

---

### Slide 13: Conformal Prediction & Calibration Guarantees
- **The Miscalibration Hazard:** Raw neural networks and variational heads underestimate tail risk.
- **Adaptive Prediction Sets (APS):** Emits conformal regime prediction sets guaranteeing finite-sample coverage at nominal $1 - \alpha = 90\%$.
- **Adaptive Conformal Inference (ACI):** Dynamically adjusts online miscoverage target $\alpha_t$ to maintain 90% empirical coverage even during structural breaks.
- **Calibration Metrics:** Expected Calibration Error (ECE) compressed from 0.1820 (raw baseline) to **0.0480** (conformalized ensemble).

---

### Slide 14: Walk-Forward Backtest & Allocation Overlay Results
- **Walk-Forward Discipline:** Rolling 750-day training windows, quarterly 63-day test steps, strictly zero look-ahead bias.
- **Tactical Allocation Overlay:** Long-only bounds $[0.70, 1.00]$, conviction-scaled Kelly tilts, 1.5% no-trade turnover band, 15 bps Indian transaction costs.
- **Performance Highlights (2019–2024):**
  - Strategy Annualised Return: **+18.85%** vs Benchmark **+15.42%** (+3.43% alpha).
  - Annualised Volatility: **14.12%** vs Benchmark **18.25%** (significant volatility dampening).
  - Information Ratio: **+0.642**; Sharpe Ratio: **0.875** vs Benchmark 0.489.
  - Deflated Sharpe Ratio (DSR): **0.968** (passes Bailey-de Prado multiple testing threshold).

---

### Slide 15: Indian Market Case Studies & Replications
- **2013 Taper Tantrum:** Currency/rates transmission; detected EM stress via USD/INR and G-Sec yields while Nifty looked deceptively stable.
- **2017–2018 Mid-Cap Shock:** Small/Mid-cap divergence feature flagged Late-Cycle exhaustion 6 weeks before large-cap Nifty corrected.
- **2018 IL&FS Credit Freeze:** Non-bank credit spread widening and breadth collapse triggered Risk-Off regime shift in mid-caps.
- **2020 COVID-19 Pandemic:** BOCPD fired changepoint within 4 days of March onset; trimmed equity beta to 70%, shielding portfolio from a -37.6% drawdown (strategy drew down only -19.1%).
- **2024 Election Verdict Event:** Calendar-aware conviction halver widened conformal prediction sets pre-event, preventing panic-selling on June 4.

---

### Slide 16: Gamified Simulation Concept: Regime Lab (9 Levels)
- **Pedagogical Gap:** Buy-side desks lack simulation tools for teaching calibrated Bayesian regime inference.
- **9-Level Progression System:**
  - Levels 1–2: Indicator Reader & Macro Strategist (Breadth, Flows, FX).
  - Levels 3–4: HMM Builder & Bayesian Sceptic (Frequentist HMM & PyMC NUTS).
  - Levels 5–6: Multivariate Modeller & Deep Learner (RS-VAR & BNN).
  - Levels 7–8: Foundation Practitioner & Ensembler (Chronos, BMA, Stacking).
  - Level 9: Regulator's Choice (Simulated SEBI Inspection & Full Lineage Audit).
- **1000-Point Scoring Framework:** Accuracy (200), Calibration (200), Bayesian Rigour (200), Ensembling (150), Allocation (100), Defensibility (100), Speed (50).

---

### Slide 17: Model Risk Governance & SEBI Audit-Defensibility
- **Regulatory Alignment:** Built to satisfy SEBI Circulars on Algorithm Governance and AMFI Monthly Risk-O-Meter disclosures.
- **Daily Immutable Output Contract:** Every daily prediction records the model commit hash, training data snapshot hash, priors, calibration quantiles, and SHAP feature rationale.
- **Champion-Challenger Promotion:** Retraining triggers strictly gated by PSI drift (< 0.25), TV reconciliation gap (< 0.15), and out-of-sample RPS superiority.

---

### Slide 18: Summary, Production Serving & Roadmap
- **Production Two-Speed Serving Architecture:**
  - `POST /regime/score`: Returns calibrated regime probabilities, conformal set, uncertainty budget, and health flags.
  - `GET /regime/health`: Real-time health monitoring (BOCPD status, TV gap, 30d coverage).
- **Core Conclusion:** Direction over price. Calibrated probability over point forecast. A documented ensemble over a single black box.
- **Next Steps:** Integration into live AMC order management systems, daily automated dispatch of Investment Committee briefing artefacts.
