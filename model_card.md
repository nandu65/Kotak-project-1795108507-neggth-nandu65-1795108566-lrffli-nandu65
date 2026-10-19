# Model Card: Bayesian Regime Detection Engine (v1.0.0)

> **Strictly Private and Confidential · Zetheta Algorithms Private Limited**  
> **Corporate Identity Number (CIN): U62012MH2023PTC410415**  
> *Quantitative Research Desk · Multi-Asset Solutions, Mumbai*

---

## 1. Model Details

- **Model Name:** Bayesian Regime Detection Engine for Indian Equity Direction Forecasting (`bayesian-regime-detection-APK`)
- **Version:** `1.0.0-PROD` (Release Date: October 2026)
- **Model Type:** Multi-Model Bayesian Ensemble combining Hidden Markov Models, Bayesian HMMs with Forward Marginalisation, Regime-Switching VAR(1), Deep Neural Ensembles with MC-Dropout, and Foundation Temporal Patch Heads, wrapped in Adaptive Conformal Prediction (APS / ACI).
- **Target Asset Universe:** Indian Equity Markets (Nifty 50, Nifty Midcap 100, Nifty Smallcap 100, India VIX, USD/INR, 10Y G-Sec Yield, Brent Crude Oil, S&P 500, US Dollar Index).
- **Primary Fiduciary Objective:** Direction over price. Emitting time-varying, calibrated probability distributions across 5 discrete Indian market regimes with explicit aleatoric and epistemic uncertainty budgets for long-only mutual fund tactical asset allocation.
- **Auditing & Governance:** Designed in alignment with SEBI's AI/ML Governance Directives, AMFI Risk-O-Meter guidelines, and Model Risk Management Tier 1 compliance.

---

## 2. Intended Use & Fiduciary Scope

### Primary Intended Use
- **Tactical Asset Allocation (TAA):** Informing equity beta tilts ($w_{\text{equity}} \in [0.70, 1.00]$) and defensive cash buffer sizing ($w_{\text{cash}} \in [0.00, 0.30]$) across Indian large-cap, flexi-cap, and multi-asset mutual fund schemes.
- **Sleeve Weighting & Cap-Rotation:** Guiding allocation between large-cap, mid-cap, and small-cap sleeves based on capitalisation divergence signals and credit-spread stresses.
- **Risk-O-Meter Input Generation:** Conditioning monthly portfolio stress testing and liquidity-adjusted Value-at-Risk (VaR) projections on regime-conditioned Monte Carlo simulations.
- **Real-Time Intraday Monitoring:** Tracking regime state between nightly batch refits via an intraday Sequential Monte Carlo (Particle Filter) and Student-$t$ Bayesian Online Changepoint Detection (BOCPD).

### Out-of-Scope & Misuse Caveats
- **High-Frequency Intraday Execution:** The engine is calibrated to daily closing bars aligned to the NSE trading calendar; it must not be used for tick-level or latency-sensitive execution.
- **Direct Point Forecasting:** The engine strictly refuses point-level price targets ($P_t = \text{target}$); all outputs are probability distributions over market direction.
- **Unconstrained Leveraged Long/Short Strategies:** The allocation overlay enforces long-only constraints consistent with SEBI scheme categorisation rules.

---

## 3. Regime Taxonomy & Economic Grounding

The engine partitions Indian equity dynamics into 5 distinct, mutually exclusive regimes:

| Regime ID | Canonical Name | Economic State | Core Feature Signatures | Tactical Action |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Risk-On** | Broad-based structural rally, positive earnings revisions, domestic & foreign liquidity expansion | Nifty > 200-DMA, India VIX < 14, FII inflows positive, Breadth > 70% above 50-DMA | 100% Equity beta, extend momentum |
| **1** | **Late-Cycle** | Mature expansion, narrow leadership (mega-caps masking mid/small fatigue), valuation exhaustion | Nifty > 200-DMA, Breadth narrowing (< 45%), Valuation $z$-score > 1.5, Smallcap divergence | Trim beta to 85%, rotate to quality |
| **2** | **Transitional** | Regime boundary, conflicting macro/flow indicators, low directional conviction | India VIX 14–22, mixed breadth, elevated epistemic uncertainty across ensemble | 80% Equity beta, neutral stance |
| **3** | **Post-Shock** | Stabilisation following acute drawdown, volatility decay, institutional base-building | Volatility decaying from peaks, oversold breadth, DII flow absorption active | 90% Equity, accumulate on dips |
| **4** | **Risk-Off** | Acute drawdown, flight to quality, systemic deleveraging, macro/currency deterioration | Nifty < 200-DMA, India VIX > 22, FII flight, INR depreciation, gilt yields rising | 70% Equity (30% Cash buffer floor) |

---

## 4. Point-In-Time (PIT) Data Lineage

- **Master Trading Calendar:** Strictly anchored to official NSE trading sessions (January 2010 to October 2026, totaling 4,124 trading sessions).
- **Zero Look-Ahead Publication Lags:**
  - Macro releases (GDP, IIP, CPI): Lagged by 1 trading day ($t-1$).
  - AMFI Monthly SIP totals: Lagged by 5 trading days ($t-5$).
  - Market Breadth (NSE advances/declines): Lagged by 1 trading day ($t-1$).
  - Global Overseas Series (S&P 500, DXY, Brent Crude): Lagged by 1 trading day ($t-1$).
- **Missing Value & Forward-Fill Discipline:**
  - Continuous global macro series forward-filled up to a strict ceiling of 3 sessions ($L \le 3$).
  - Daily log returns, FII/DII flow counts, and market breadth are **never forward-filled**.
  - All feature standardisation is executed using **strictly causal expanding-window statistics** ($z_t = (x_t - \mu_{t-1}) / \sigma_{t-1}$), eliminating full-sample scaling leakage.

---

## 5. Model Architecture & Ensembling Design

```
                         Input: 42 PIT Aligned Causal Features
                                          │
       ┌──────────────────┬───────────────┼───────────────┬──────────────────┐
       ▼                  ▼               ▼               ▼                  ▼
┌──────────────┐   ┌──────────────┐ ┌───────────┐ ┌──────────────┐   ┌──────────────┐
│ Gaussian HMM │   │ Bayesian HMM │ │  RS-VAR   │ │ Bayesian DL  │   │  Foundation  │
│ (hmmlearn /  │   │ (Forward-NUTS│ │ (Markov-  │ │(Deep Ensemble│   │ Temporal Head│
│ Baum-Welch)  │   │ marginalise) │ │   VAR)    │ │+ MC-Dropout) │   │ (Chronos/TF) │
└──────┬───────┘   └──────┬───────┘ └─────┬─────┘ └──────┬───────┘   └──────┬───────┘
       │                  │               │              │                  │
       └──────────────────┴───────┬───────┴──────────────┴──────────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │  Constrained Stacking &   │
                    │ Bayesian Model Averaging  │
                    └─────────────┬─────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│   Sequential Online Monitoring  │       │       Conformal Prediction      │
│ (Particle Filter + BOCPD-t-PDF) │       │   (Split, APS, Adaptive ACI)    │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                   Unified Daily Regime Output Contract
                   (Calibrated Probs, Bounds, SHAP, Health)
```

1. **Gaussian HMM (`GaussianRegimeHMM`):** Baseline frequentist Baum-Welch EM with economic moment sorting.
2. **Bayesian HMM (`BayesianRegimeHMM`):** Solves `pm.HiddenMarkovChain` limitation via exact log-space forward algorithm marginalisation, sampled with NUTS across 4 chains.
3. **Markov-Switching VAR (`MarkovSwitchingVAR`):** Models cross-asset dynamics ($y_t = c_{s_t} + A_{s_t} y_{t-1} + \epsilon_t$), recovering regime-conditional covariance and impulse response functions.
4. **Bayesian Deep Ensemble (`BayesianDeepEnsemble`):** $M=8$ neural networks with permanent Bernoulli dropout, providing clean separation between Epistemic and Aleatoric uncertainty.
5. **Foundation Model Head (`FoundationRegimeHead`):** Multi-scale temporal patch tokenisation with linear probing diagnostics.
6. **Ensemble Stacking (`ConstrainedStacker`):** Simplex-constrained meta-learner optimizing out-of-fold multi-class cross-entropy.
7. **Online Monitoring:** Particle Filter (5,000 particles with systematic resampling) + Student-$t$ BOCPD with hazard $H = 1/100$.
8. **Conformal Wrapper (`AdaptivePredictionSets` & `AdaptiveConformalInference`):** Guarantees finite-sample coverage at nominal $1 - \alpha = 90\%$.

---

## 6. Quantitative Scoring & Calibration Performance

### Out-of-Sample Proper Scoring Rules (Walk-Forward 2019–2024)

| Model Architecture | Ranked Probability Score (RPS) ↓ | Multi-Class Brier Score ↓ | Log Loss ↓ | Expected Calibration Error (ECE) ↓ | APS Coverage (Nominal 90%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Climatology Baseline** | 0.2840 | 0.6850 | 1.4820 | 0.2450 | 72.4% |
| **Persistence Baseline** | 0.1980 | 0.4420 | 1.1250 | 0.1820 | 81.0% |
| **Volatility-Threshold Rule** | 0.2150 | 0.4950 | 1.2100 | 0.2100 | 78.5% |
| **Frequentist Gaussian HMM** | 0.1620 | 0.3810 | 0.9420 | 0.0890 | 88.2% |
| **Bayesian HMM (Forward-NUTS)**| 0.1540 | 0.3650 | 0.8950 | 0.0710 | 89.6% |
| **Multivariate RS-VAR(1)** | 0.1490 | 0.3520 | 0.8650 | 0.0680 | 90.4% |
| **Bayesian Deep Ensemble** | 0.1460 | 0.3440 | 0.8410 | 0.0620 | 91.1% |
| **Chronos Foundation Head** | 0.1480 | 0.3490 | 0.8520 | 0.0650 | 90.8% |
| **Ensemble (Stacking + BMA)** | **0.1380** | **0.3280** | **0.8050** | **0.0480** | **91.8%** |

*RPS Skill Score of Ensemble over Persistence Baseline:* **+30.3%**.  
*RPS Skill Score of Ensemble over Climatology Baseline:* **+51.4%**.

---

## 7. Tactical Asset Allocation Backtest Performance

Evaluated over rolling out-of-sample walk-forward windows (2019–2024, 1,486 trading days) comparing the **Regime-Driven Tactical Allocation Overlay** against a **100% Equity (Nifty 50) Buy & Hold Benchmark**:

| Metric | Buy & Hold Benchmark (Nifty 50) | Regime Tactical Overlay (Calibrated) | Tactical Delta / Edge |
| :--- | :---: | :---: | :---: |
| **Annualised Return** | +15.42% | **+18.85%** | **+3.43% p.a.** |
| **Annualised Volatility** | 18.25% | **14.12%** | **-4.13% (de-risked)** |
| **Sharpe Ratio ($r_f = 6.5\%$)** | 0.489 | **0.875** | **+0.386** |
| **Information Ratio (IR)** | — | **+0.642** | High institutional skill |
| **Tracking Error** | — | 5.34% | Controlled active risk |
| **Maximum Drawdown (COVID March 2020)** | **-37.6%** | **-19.1%** | **+18.5% capital defense** |
| **Annualised Portfolio Turnover** | 0.0% | 42.5% | Well within turnover budget |
| **Transaction Costs Deducted** | 0.0 bps | 15.0 bps / turnover | Realistic Indian net return |
| **Deflated Sharpe Ratio (DSR)** | 0.420 | **0.968** | Robust to multiple testing ($p < 0.05$) |
| **Probability of Backtest Overfitting (PBO)**| — | **0.125** | Low risk of spurious fit |

---

## 8. Model Risk Governance & Limitations

### Known Weaknesses & Edge Cases
1. **Transitional State Ambiguity:** The boundary between Late-Cycle and Transitional regimes exhibits the highest epistemic uncertainty ($\sigma_{\text{epistemic}} > 0.08$). During macro transition phases, prediction sets widen from singleton $\{s_t\}$ to pairs $\{ \text{Late-Cycle}, \text{Transitional} \}$.
2. **Sudden Event Discontinuities (Scheduled Election / Budget Verdicts):** Smooth regime models cannot anticipate discrete counting day jumps (e.g. 4 June 2024 Election verdict). The calendar-aware conviction halver (`event_adjusted_conviction`) and Student-$t$ BOCPD must be active to widen intervals pre-event.
3. **Turnover Bleed in Choppy Markets:** During sideways markets with oscillating indicators, the no-trade band ($1.5\%$) and hysteresis bands are critical to prevent whipsaw turnover costs.

### Production Serving Contract (`POST /regime/score`)
Every day at 18:30 IST, the engine emits the immutable JSON contract:
- `DateKey`: integer date key (`YYYYMMDD`)
- `RegimeProbs`: vector of 5 calibrated probabilities summing to 1.0
- `ConformalSet`: list of regime names guaranteeing 90% finite-sample coverage
- `EpistemicUncertainty`: ensemble disagreement standard deviation
- `AleatoricUncertainty`: per-member entropy
- `DominantModel`: highest-weight model generating the call
- `HealthFlags`: BOCPD changepoint status, online/batch TV gap, maximum feature PSI
- `SHAPAttribution`: top positive and negative contributing indicators.
