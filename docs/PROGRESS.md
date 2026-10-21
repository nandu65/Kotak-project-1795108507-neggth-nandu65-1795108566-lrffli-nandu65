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

---
