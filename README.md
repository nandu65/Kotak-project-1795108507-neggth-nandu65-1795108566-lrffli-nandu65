# Bayesian Regime Detection Engine for Equity Direction Forecasting

> **Strictly Private and Confidential · Zetheta Algorithms Private Limited**  
> **Corporate Identity Number (CIN): U62012MH2023PTC410415**  
> *Quantitative Research Desk · Multi-Asset Solutions, Mumbai*

---

## Executive Summary & The Directional Thesis

Modern quantitative equity allocation in Indian fund houses has long struggled with direct price-level prediction. Over 1-year and multi-month investment horizons, the variance of equity index returns is empirically dominated by unpredictable macro shocks. Point forecasts (e.g. *"Nifty will hit 26,000 by year-end"*) suffer from three fatal pathologies:
1. **Signal-to-Noise Collapse:** Unpredictable shocks dominate predictable cash flows over multi-month horizons.
2. **Structural Non-Stationarity:** Regimes shift continuously; models fit across regimes blur true state dynamics, while models fit on single regimes fail when regimes end.
3. **Miscalibrated Uncertainty:** Point predictions provide no audit-defensible error bounds for Investment Committees or SEBI regulatory scrutiny.

This project implements the **Direction Over Price** paradigm: replacing unanswerable point-prediction questions with calibrated, probabilistic regime detection across **5 discrete Indian market states**:
- **Risk-On:** Broad-based rally, high market breadth, low India VIX, robust institutional flows.
- **Late-Cycle:** Expansion mature, narrow leadership (mega-caps masking mid/small-cap fatigue), elevated valuation z-scores.
- **Transitional:** Regime boundaries, conflicting macro/flow indicators, low directional conviction.
- **Post-Shock:** Stabilisation following acute drawdowns, volatility decay, base-building, mean-reversion opportunity.
- **Risk-Off:** Acute drawdowns, sharp volatility spikes, capital flight, macro deterioration.

---

## Core System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │  Point-In-Time Indian Market Data Pipeline   │
                       │  (NSE Calendar, Publication Lags, Parquet)   │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │    30+ Economically Motivated Features       │
                      │  (Returns, Realised Vol, Breadth, Flows, FX)  │
                      └───────────────────────┬───────────────────────┘
                                              │
         ┌───────────────────┬────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌───────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Frequentist HMM │ │  Bayesian HMM   │ │  RS-VAR   │ │  Bayesian Deep  │ │ Foundation Model│
│ (hmmlearn / BIC)│ │ (Forward-NUTS)  │ │(Markov-VAR│ │(Deep Ensembles, │ │ (Chronos Embed- │
│                 │ │                 │ │ dynamics) │ │   MC-Dropout)   │ │     dings)      │
└────────┬────────┘ └────────┬────────┘ └─────┬─────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                │                │                   │
         └───────────────────┴───────┬────────┴────────────────┴───────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │    Model Ensembling & Stacking  │
                    │  (BMA & Time-Aware Stacking)    │
                    └────────────────┬────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│   Sequential Online Inference   │         │       Conformal Prediction      │
│ (Particle Filter + BOCPD-t-PDF) │         │ (Split, APS, Adaptive Conformal)│
└────────────────┬────────────────┘         └─────────────────┬───────────────┘
                 │                                            │
                 └───────────────────────┬────────────────────┘
                                         ▼
                      ┌─────────────────────────────────────┐
                      │     Unified Daily Output Contract   │
                      │  (Calibrated Probs, Bounds, SHAP)   │
                      └──────────────────┬──────────────────┘
                                         ▼
                      ┌─────────────────────────────────────┐
                      │    Tactical Allocation Overlay      │
                      │(Conviction-Scaled Tilt, Kelly, Bands│
                      └─────────────────────────────────────┘
```

---

## Key Methodological Principles & Hard Rules

1. **Public Data Rigour:** Uses public market feeds (Yahoo Finance via `yfinance`, NSE/BSE, RBI DBIE). Primary assets include Nifty 50, Nifty Midcap 100, Nifty Smallcap 100, India VIX, USD/INR, 10Y G-Sec Yield, Brent Crude Oil, S&P 500, and US Dollar Index (DXY).
2. **Zero Look-Ahead Bias & Point-in-Time Discipline:** Every feature is lagged strictly by its true publication delay (macro prints = 1 day, monthly AMFI SIP data = 5 days, EOD breadth = 1 day). Cross-asset series are aligned to the official NSE trading calendar with a strict 3-day forward-fill ceiling for global macro series. Returns and flow counts are never forward-filled.
3. **Window-Decoded Supervision:** To eliminate future target leakage, regime labels used for downstream supervised evaluation are decoded strictly inside each walk-forward training window.
4. **Proper Scoring Rules:** Evaluated using strictly proper scoring rules (Ranked Probability Score [RPS], Multi-class Brier score, Log Loss) alongside Expected Calibration Error (ECE) and Murphy Brier decomposition (Reliability, Resolution, Uncertainty).
5. **Exact Mathematical Fixes:**
   - Bayesian HMM avoids non-existent `pm.HiddenMarkovChain` by implementing custom forward-algorithm discrete state marginalisation for Hamiltonian Monte Carlo (NUTS).
   - Bayesian Online Changepoint Detection (BOCPD) implements the exact Student-t posterior predictive density (degrees of freedom $2\alpha$).
6. **Robust Backtesting:** Purged $k$-fold cross-validation with embargo buffers, walk-forward out-of-sample evaluation, transaction costs (15 bps), and Deflated Sharpe Ratio calculation to protect against backtest overfitting.

---

## Repository Layout

```
├── 493561A_Bayesian_Regime_Detection.pdf  # Project Brief & Source of Truth
├── README.md                             # High-level architecture & documentation
├── environment.yml                       # Conda environment specification
├── configs/
│   └── config.yaml                       # Master configuration, seeds & parameters
├── data/
│   ├── raw/                              # Parquet raw cache & manual fallback files
│   ├── interim/                          # Intermediate transformed features
│   └── processed/                        # Point-in-time aligned analytical sets
├── src/
│   └── regime/
│       ├── __init__.py
│       ├── data/                         # Ingestion, PIT alignment, NSE calendar
│       ├── features/                     # Macro, momentum, breadth, vol features
│       ├── models/                       # HMM, Bayesian HMM, RS-VAR, BDL, Chronos
│       ├── ensemble/                     # BMA and constrained stacking
│       ├── online/                       # Particle Filter, Student-t BOCPD
│       ├── conformal/                    # Split conformal, APS, ACI
│       └── backtest/                     # Allocation overlay, Monte Carlo, DSR
├── tests/                                # Comprehensive test suite (pytest)
├── r/                                    # R cross-validation (depmixS4, MSwM)
└── reports/                              # IC briefings, model card, research reports
```

---

## Exact End-to-End Reproduction Commands

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate regime
# or with active virtual environment:
pip install -r requirements.txt
```

### 2. Step 2: Point-in-Time Market Data Ingestion & Calendar Alignment
```bash
python -m src.regime.data.download
python -m src.regime.data.pit
```
Output: `data/processed/market_pit_dataset.parquet` (4,124 aligned NSE sessions, publication lags enforced, zero return forward-fill).

### 3. Step 3: Causal Feature Engineering & Standardisation
```bash
python -m src.regime.features.engineer
```
Output: `data/processed/features.parquet`, `data/processed/market_features_standardized.parquet`, and `data/processed/features_metadata.csv`.

### 4. Step 4–8: Automated Test Verification Across All Modules
```bash
# Run unit tests across all architecture layers:
python -m pytest tests/test_config.py -v
python -m pytest tests/test_pit.py -v
python -m pytest tests/test_features.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_ensemble.py -v
python -m pytest tests/test_online.py -v
python -m pytest tests/test_conformal.py -v
python -m pytest tests/test_backtest.py -v

# Or run the entire test suite in one command:
python -m pytest tests/ -v
```

### 5. Cross-Language R Validation
```bash
# Verify Python HMM vs R depmixS4 numerical invariance:
python r/cross_validate.py
# (Optional R script execution):
Rscript r/fit_hmm.R
```

### 6. Launch Institutional Live Interactive Dashboard
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open your browser to `http://127.0.0.1:8000` to interact with:
- Daily calibrated 5-regime posterior probabilities.
- Information-theoretic uncertainty decomposition (Aleatoric vs Epistemic).
- Conformal prediction sets (Adaptive Prediction Sets at 90% and 95%).
- Real-time 5,000-particle SMC filter streaming updates.
- Student-$t$ BOCPD changepoint monitoring.
- Tactical allocation tilt overlay and Investment Committee briefing narrative.

### 7. Key Institutional Research Deliverables
- Master Research Report (40+ pages equivalent): `reports/report.md`
- SEBI AI/ML Risk Model Card: `reports/model_card.md`
- 18-Slide Presentation Deck: `reports/slides_outline.md`
- 10-Minute CIO Demonstration Script: `reports/demo_script.md`
- Feature Diagnostics Notebook: `notebooks/03_feature_diagnostics.ipynb`
- Implementation & Verification Progress Log: `docs/PROGRESS.md`

---
*Confidentiality Notice: This is a strictly PRIVATE quantitative research project for Zetheta Algorithms Private Limited (CIN: U62012MH2023PTC410415). Never push or publish code, data, or artifacts to any remote or public repository.*

