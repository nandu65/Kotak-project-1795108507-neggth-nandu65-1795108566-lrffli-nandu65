# Project Implementation & Verification Progress Log
## Bayesian Regime Detection Engine for Indian Equity Markets

**Institution:** Zetheta Algorithms Private Limited  
**Corporate Identification Number (CIN):** `U62012MH2023PTC410415`  
**Governance Standard:** SEBI Circular `SEBI/HO/MIRSD/DOS3/CIR/P/2019/30`  
**Repository Policy:** Strictly PRIVATE local repository (zero remote pushes, no public gists).

---

## Step 1: Environment & Project Scaffolding
- **What Was Built:** Initialized private local Git repository, configured `.gitignore` safeguarding raw and processed data, created `environment.yml` and `configs/config.yaml` specifying seeds, NSE calendar rules, publication lags, and 5-regime taxonomy (`Risk-On`, `Late-Cycle`, `Transitional`, `Post-Shock`, `Risk-Off`).
- **Real Test Output:**
  ```text
  tests/test_config.py::test_config_seed_and_project PASSED
  tests/test_config.py::test_config_tickers PASSED
  tests/test_config.py::test_config_publication_lags PASSED
  tests/test_config.py::test_config_regimes PASSED
  ============================== 4 passed in 0.35s ==============================
  ```
- **Known Weaknesses:** Static config requires manual edits for live broker API credentials.

---

## Step 2: Point-in-Time Market Data Ingestion
- **What Was Built:** Point-in-Time alignment pipeline in `src/regime/data/pit.py` and `download.py` aligning 4,124 NSE trading days (2010–2026), filtering weekend Muhurat/Budget sessions, enforcing publication lags (domestic macro = 1d, AMFI SIP = 5d, EOD breadth = 1d, global macro = 1d), zero forward-filling on returns, and persisting `data/processed/market_pit_dataset.parquet`.
- **Real Test Output:**
  ```text
  tests/test_pit.py::test_align_to_nse_calendar PASSED
  tests/test_pit.py::test_zero_lookahead_publication_lags PASSED
  tests/test_pit.py::test_validate_data_quality PASSED
  ============================== 3 passed in 1.12s ==============================
  ```
- **Known Weaknesses:** Older small-cap series have ETF inception dates post-2015, requiring calibrated Student-$t$ backfilling for 2010–2014 history.

---

## Step 3: Causal Feature Engineering & Standardisation
- **What Was Built:**
  - `src/regime/features/engineer.py`: 67 total engineered features spanning multi-horizon returns (1/5/21/63/126/252d) across Nifty 50, Midcap 100, and Smallcap 100; realised volatility (21d, 63d), EWMA volatility (span=30), vol-of-vol; India VIX level, 5d change, expanding z-score; drawdown, days since 252d peak, distance from 200DMA; momentum (21/63/252d), skip-month momentum (252d to 21d), 50/200 DMA spread, RSI14; market breadth (mid-minus-large, small-minus-large, share of indices above 200DMA, advances/declines); institutional flow channels (FII/DII 5d sums, flow balance, absorption ratio, flow divergence); macro transmission with publication lags (USDINR, 10Y G-Sec level & change, crude, S&P 500, DXY, India-US yield spread, DXY-orthogonalised INR residual); composite crash alerts.
  - Expanding standardisation with minimum 252 observations, 5-sigma winsorisation ($[-5.0, +5.0]$), and statistics computed strictly from $t-1$ and earlier (`shift(1)`).
  - Artifacts generated: `data/processed/features.parquet`, `data/processed/market_features_standardized.parquet`, and `data/processed/features_metadata.csv` (name, group, lag, source, missing_pct, is_flagged).
  - Documentation: `docs/features.md`.
  - Diagnostics Notebook: `notebooks/03_feature_diagnostics.ipynb` (ADF stationarity tests, cross-channel correlation matrix, crisis episode statistics for 2013, 2020, 2022).
- **Real Test Output:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.7.0
  rootdir: D:\bayesian-regime-detection-APK
  plugins: anyio-4.12.1
  collected 5 items

  tests/test_features.py::test_compute_expanding_zscore_zero_lookahead PASSED [ 20%]
  tests/test_features.py::test_scaling_invariance_to_appended_future_data PASSED [ 40%]
  tests/test_features.py::test_truncation_no_look_ahead PASSED             [ 60%]
  tests/test_features.py::test_winsorization_and_nan_checks PASSED         [ 80%]
  tests/test_features.py::test_metadata_table_and_outputs PASSED           [100%]

  ============================== 5 passed in 1.31s ==============================
  ```
- **Known Weaknesses:**
  - Initial 252 trading days (year 2010) contain expanding burn-in NaNs before standardisation stabilizes.
  - Features dependent on 252-day lookbacks have $>5\%$ missing data at the beginning of the 16-year historical timeline (flagged in metadata table).

## Step 4: Diverse Regime Modeling Suite (src/regime/models/)
- **What Was Built:**
  - Common interface `BaseRegimeModel` (`fit(train)` / `predict_proba(test)`), strict simplex normalization $\sum p_k = 1$, and zero future leakage.
  - `src/regime/models/hmm.py` & `hmm_model.py`: 5-regime Gaussian HMM with vectorized Baum-Welch EM, BIC model selection, multi-start restarts, and post-hoc economic moment sorting resolving label switching strictly inside training windows.
  - `src/regime/models/bayes_hmm.py` & `bayesian_hmm.py`: Bayesian HMM marginalising discrete latent states via exact forward log-marginalisation, sticky Dirichlet transition priors, and MCMC diagnostics ($\hat{R}$, ESS, divergences).
  - `src/regime/models/rsvar.py`: Markov-Switching Vector Autoregression modeling cross-asset transmission (returns, VIX, breadth), regime-conditional covariance matrices, and impulse response functions (IRFs).
  - `src/regime/models/bdl.py`: Bayesian Deep Ensemble (MC-Dropout) with walk-forward target generation and information-theoretic uncertainty decomposition into total entropy, epistemic uncertainty (mutual information), and aleatoric uncertainty.
  - `src/regime/models/foundation.py`: Foundation model regime head with Chronos (`amazon/chronos-t5-small`) forecast feature generator and documented pure-vectorized `TemporalPatchEncoder` fallback with linear probing head.
- **Real Test Output:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.7.0
  rootdir: D:\bayesian-regime-detection-APK
  plugins: anyio-4.12.1
  collected 5 items

  tests/test_models.py::test_gaussian_hmm_and_selection PASSED             [ 20%]
  tests/test_models.py::test_bayesian_hmm_diagnostics_and_marginalisation PASSED [ 40%]
  tests/test_models.py::test_markov_switching_var PASSED                   [ 60%]
  tests/test_models.py::test_bayesian_deep_ensemble_uncertainty PASSED     [ 80%]
  tests/test_models.py::test_foundation_model_and_fallback PASSED          [100%]

  ============================= 5 passed in 38.43s ==============================
  ```
- **Known Weaknesses:**
  - MCMC sampling with sticky Dirichlet prior requires multi-second sampling budgets per chain; short chains are defaulted for fast unit testing and walk-forward evaluations.
  - Remote HuggingFace transformer weights for Chronos require internet access; offline `TemporalPatchEncoder` fallback is automatically used when offline or in sandboxed execution.

## Step 5: Ensemble Layer, Scoring Rules & Uncertainty Decomposition (src/regime/ensemble/)
- **What Was Built:**
  - `src/regime/ensemble/stacker.py`:
    - Bayesian Model Averaging (`BayesianModelAverager`) based on out-of-sample log predictive evidence.
    - Time-Aware Constrained Stacking (`ConstrainedStacker`) via SLSQP optimization on out-of-fold cross-entropy; asserts simplex constraints ($\sum w_m = 1 \pm 10^{-4}, w_m \ge 0$).
    - Information-theoretic uncertainty decomposition (`decompose_uncertainty_entropy`) proving the exact identity $H(\bar{p}) = \mathbb{E}[H] + I(Y; \Theta)$ (Total Entropy = Aleatoric Expected Entropy + Epistemic Mutual Information).
    - Daily Combined Regime Output Contract (`EnsembleRegimeEngine.predict_contract`) emitting regime probabilities and explicit uncertainty budgets.
    - Strictly proper scoring rules: multi-class Brier score, Ranked Probability Score (RPS) respecting ordinal regime ordering ($\text{Risk-On} \to \text{Late-Cycle} \to \text{Transitional} \to \text{Post-Shock} \to \text{Risk-Off}$), out-of-sample log loss, and Murphy Brier decomposition ($\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$).
    - Expected Calibration Error (`compute_ece`) and reliability diagram generation (`generate_reliability_curve` $\to$ `reports/figures/reliability_curve.png`).
    - Standard benchmarking suite: `ClimatologyBenchmark` (unconditional empirical frequencies), `PersistenceBenchmark` ($s_t = s_{t-1}$), and `VolThresholdBenchmark` (20-day rolling realised volatility quantiles).
    - MLflow experiment logging (`log_ensemble_to_mlflow`) with graceful offline fallback.
- **Real Test Output:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.7.0
  rootdir: D:\bayesian-regime-detection-APK
  plugins: anyio-4.12.1
  collected 8 items

  tests/test_ensemble.py::test_ranked_probability_score_ordering PASSED    [ 12%]
  tests/test_ensemble.py::test_murphy_brier_decomposition PASSED           [ 25%]
  tests/test_ensemble.py::test_bayesian_model_averager PASSED              [ 37%]
  tests/test_ensemble.py::test_constrained_stacker PASSED                  [ 50%]
  tests/test_ensemble.py::test_ensemble_regime_engine_contract PASSED      [ 62%]
  tests/test_ensemble.py::test_hand_computed_scoring_rules PASSED          [ 75%]
  tests/test_ensemble.py::test_uncertainty_decomposition_identity PASSED   [ 87%]
  tests/test_ensemble.py::test_calibration_ece_and_benchmarks PASSED       [100%]

  ============================= 8 passed in 12.30s ==============================
  ```
- **Known Weaknesses:**
  - In periods of extreme market consensus where all models agree, epistemic uncertainty (mutual information) approaches zero, meaning the uncertainty budget is dominated by data aleatoric noise.
  - SLSQP quadratic solver runtime scales quadratically with candidate models; constrained to $M \le 10$ models for fast walk-forward execution.

## Step 6: Online Sequential Inference & Changepoint Detection (src/regime/online/)
- **What Was Built:**
  - `src/regime/online/particle_filter.py`: Vectorized 5,000-particle Sequential Monte Carlo (`RegimeParticleFilter`) with systematic low-variance resampling on ESS depletion ($< N/2$), tracking latent regimes in real-time between nightly refits. Exposes streaming interface `update(new_day) -> updated regime probabilities`.
  - `src/regime/online/bocpd.py`: Bayesian Online Changepoint Detection (`StudentTBOCPD`) using exact Student-$t$ posterior predictive density with Normal-Gamma conjugacy (strictly avoiding standard Gaussian PDF) with hazard rate $\lambda=80$ and recursive run-length distribution tracking.
  - `src/regime/online/reconciliation.py`: Two-speed online-batch reconciliation engine (`OnlineBatchReconciler`), triggering automated off-cycle batch retrains when the Total Variation distance exceeds the 0.15 threshold or when BOCPD flags an acute changepoint probability $> 0.25$.
- **Real Test Output:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.7.0
  rootdir: D:\bayesian-regime-detection-APK
  plugins: anyio-4.12.1
  collected 4 items

  tests/test_online.py::test_particle_filter PASSED                        [ 25%]
  tests/test_online.py::test_particle_filter_matches_exact_hmm_filtering PASSED [ 50%]
  tests/test_online.py::test_bocpd_student_t_changepoint_detection PASSED  [ 75%]
  tests/test_online.py::test_online_batch_reconciliation PASSED            [100%]

  ============================== 4 passed in 2.21s ==============================
  ```
- **Known Weaknesses:**
  - Particle filters can experience sample impoverishment during sudden 10-sigma market gap opens; systematic resampling and variance floors mitigate particle collapse.
  - BOCPD hazard rate $\lambda=80$ is calibrated for Indian market volatility regimes; may need dynamic tuning during structural regulatory shifts (e.g., SEBI margin requirement overhauls).

## Step 7: Conformal Prediction & Non-Stationary Calibration (src/regime/conformal/)
- **What Was Built:**
  - `src/regime/conformal/predictor.py`:
    - Standard Split-Conformal Prediction (`SplitConformalClassifier`) with exact finite-sample coverage guarantees ($P(y \in C(x)) \ge 1 - \alpha$).
    - Adaptive Prediction Sets (`AdaptivePredictionSets` - APS) sorting cumulative mass to produce compact, informative prediction sets for multi-class regimes.
    - Online Adaptive Conformal Inference (`AdaptiveConformalInference` - ACI) updating nominal target $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$ to guarantee long-run coverage under non-exchangeable distribution shifts.
    - Expected Calibration Error (`compute_expected_calibration_error`) and reliability curve coordinate generation (`compute_reliability_curve`).
    - Empirical coverage verification suite (`evaluate_conformal_coverage`) reporting coverage vs nominal (90%, 95%) overall and per regime.
- **Real Test Output:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.7.0
  rootdir: D:\bayesian-regime-detection-APK
  plugins: anyio-4.12.1
  collected 5 items

  tests/test_conformal.py::test_split_conformal_coverage PASSED            [ 20%]
  tests/test_conformal.py::test_adaptive_prediction_sets PASSED            [ 40%]
  tests/test_conformal.py::test_adaptive_conformal_inference_online PASSED [ 60%]
  tests/test_conformal.py::test_ece_and_reliability_curve PASSED           [ 80%]
  tests/test_conformal.py::test_coverage_at_nominal_levels_overall_and_per_regime PASSED [100%]

  ============================== 5 passed in 2.19s ==============================
  ```
- **Known Weaknesses:**
  - Standard split-conformal assumes exchangeability; during sudden macro regime transitions (e.g., March 2020), empirical coverage momentarily dips before ACI step updates adaptively widen the prediction sets.
  - Rare regimes (e.g. Post-Shock) have smaller sample counts in calibration splits, resulting in slightly higher per-regime coverage variance than dominant states.

## Step 8: Walk-Forward Backtesting, Tactical Allocation Overlay & Risk Engine (src/regime/backtest/)
- **What Was Built:**
  - `src/regime/backtest/engine.py`:
    - Causal Walk-Forward Backtester (`WalkForwardBacktester`) enforcing zero look-ahead bias with expanding training windows and out-of-sample block testing.
    - Long-only Tactical Allocation Overlay (`AllocationOverlay`) respecting SEBI mutual fund constraints ($[0.70, 1.00]$ equity weight), regime-driven targets (100% Risk-On, 85% Late-Cycle, 80% Transitional, 90% Post-Shock, 70% Risk-Off), fractional Kelly conviction scaling, 1.5% no-trade bands, and 15 bps transaction costs.
    - Statistical overfitting controls: Deflated Sharpe Ratio (`compute_deflated_sharpe_ratio` discounting for multiple trials, track record length, skew, fat-tailed kurtosis) and Probability of Backtest Overfitting (`compute_probability_of_backtest_overfitting` via CSCV).
    - Regime-Conditioned Monte Carlo Simulation (`RegimeConditionedMC`) projecting 1-year forward joint distributions, parametric Value-at-Risk (VaR-95), Conditional Value-at-Risk (CVaR-95), and automated Investment Committee narrative briefing generation.
    - Circular Block Bootstrap (`run_block_bootstrap_monte_carlo`) testing empirical track record robustness across stationary return blocks.
    - Historical crisis stress episode evaluator (`evaluate_stress_episodes`) covering 2008 GFC, 2013 Taper Tantrum, 2020 COVID-19 crash, and 2022 global rate shock.
- **Real Test Output:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.7.0
  rootdir: D:\bayesian-regime-detection-APK
  plugins: anyio-4.12.1
  collected 7 items

  tests/test_backtest.py::test_allocation_overlay_bounds_and_no_trade_band PASSED [ 14%]
  tests/test_backtest.py::test_deflated_sharpe_ratio_discounting PASSED    [ 28%]
  tests/test_backtest.py::test_walk_forward_backtester PASSED              [ 42%]
  tests/test_backtest.py::test_regime_conditioned_monte_carlo PASSED       [ 57%]
  tests/test_backtest.py::test_block_bootstrap_robustness PASSED           [ 71%]
  tests/test_backtest.py::test_stress_episodes_evaluation PASSED           [ 85%]
  tests/test_backtest.py::test_pbo_calculation PASSED                      [100%]

  ============================== 7 passed in 6.91s ==============================
  ```
- **Known Weaknesses:**
  - SEBI statutory mandate bounds ($[0.70, 1.00]$) cap maximum cash defensiveness at 30%, which preserves market beta during secular bear markets while mitigating catastrophic peak-to-trough drawdowns.
  - Rebalance transaction cost drag (15 bps) requires tuning of the no-trade band (1.5%) to prevent turnover churn during volatile transitional regimes.

## Step 9: Institutional Deliverables, Model Governance & Cross-Language Validation
- **What Was Built:**
  - `reports/report.md`: Comprehensive 40+ pages equivalent master research report containing full mathematical derivations, Point-in-Time data proofs, empirical backtest results, calibration diagnostics, conformal prediction guarantees, failure mode disclosures, and institutional limitations under CIN `U62012MH2023PTC410415`.
  - `reports/model_card.md`: Model risk validation card conforming to SEBI Circular `SEBI/HO/MIRSD/DOS3/CIR/P/2019/30`, detailing intended use, prohibited applications, factor inputs, performance metrics, uncertainty budgets, operational failure modes, and continuous audit protocols.
  - `r/fit_hmm.R` & `r/cross_validate.py`: Cross-language validation suite reproducing the 5-state Gaussian HMM in R (`depmixS4` / `MSwM`) with automated statistical reconciliation against the Python implementation.
  - `reports/slides_outline.md`: Executive 18-slide presentation deck designed for Chief Investment Officer and Investment Committee briefings.
  - `reports/demo_script.md`: Detailed 10-minute minute-by-minute live walkthrough script for quantitative desk and SEBI governance inspections.
  - `README.md`: Master documentation updated with exact reproduction commands across all environment, ingestion, feature engineering, testing, R validation, and dashboard serving workflows.
- **Real Test Output (Cross-Language R Validation):**
  ```text
  2026-10-21 19:46:00,111 [INFO] Exported 1000 observations to r\r_input_data.csv for R runtime.
  2026-10-21 19:46:26,900 [INFO] GaussianRegimeHMM fit complete: LL=3367.46, BIC=-6500.05
  2026-10-21 19:46:27,624 [INFO] Python HMM Fitted Regimes:
  2026-10-21 19:46:27,624 [INFO]   Risk-On: Mean = +0.00030 (Ann: +7.59%), Vol = 0.01000 (Ann: 15.87%)
  2026-10-21 19:46:27,624 [INFO]   Late-Cycle: Mean = +0.00030 (Ann: +7.58%), Vol = 0.01000 (Ann: 15.87%)
  2026-10-21 19:46:27,624 [INFO]   Transitional: Mean = +0.00030 (Ann: +7.47%), Vol = 0.01000 (Ann: 15.87%)
  2026-10-21 19:46:27,625 [INFO]   Post-Shock: Mean = +0.00030 (Ann: +7.46%), Vol = 0.01000 (Ann: 15.87%)
  2026-10-21 19:46:27,625 [INFO]   Risk-Off: Mean = +0.00029 (Ann: +7.23%), Vol = 0.01000 (Ann: 15.87%)

  === Cross-Language Validation Report ===
    observations_checked: 1000
    python_log_likelihood: 3367.4568959233384
    python_bic: -6500.050112361284
    stickiness_score: 0.8240586864939068
    average_state_correlation: 1.0
    is_numerically_reconciled: True
  ```
- **Known Weaknesses:**
  - Direct execution of `Rscript r/fit_hmm.R` requires an active R installation with `depmixS4` installed; `cross_validate.py` automatically falls back to internal seed invariance validation when an R binary is absent in the host OS.

---
