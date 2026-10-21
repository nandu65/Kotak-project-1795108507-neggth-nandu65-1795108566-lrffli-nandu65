# Feature Engineering Dictionary & Methodology
## Bayesian Regime Detection Engine for Indian Equity Markets

**Institution:** Zetheta Algorithms Private Limited  
**Corporate Identification Number (CIN):** `U62012MH2023PTC410415`  
**Data Lineage Artifact:** `data/processed/features.parquet`  
**Metadata Table:** `data/processed/features_metadata.csv`  

---

## 1. Overview & Architectural Philosophy

The Feature Engineering Pipeline transforms raw Point-in-Time (PIT) market feeds into 35+ economically defensible, stationary, and causal input signals. Financial time-series regime modeling requires strict discipline against three insidious pitfalls:
1. **Look-Ahead Bias:** Future statistical moments contaminating historical z-scores.
2. **Publication Lag Ignorance:** Assuming macroeconomic indicators (inflation, SIP flows, EOD market breadth) are accessible instantaneously at market open.
3. **Non-Stationarity & Distribution Shift:** Raw price and yield levels exhibiting unit roots, breaking probabilistic density estimation.

All features are aligned strictly to the National Stock Exchange (NSE) trading calendar (4,124 sessions from 2010 to 2026) with explicit publication lags and causal expanding-window standardisation.

---

## 2. Expanding-Window Standardisation Methodology

Traditional quantitative pipelines apply full-sample standardisation:
$$z_t = \frac{x_t - \mu_{0:T}}{\sigma_{0:T}}$$
This leaks future market crashes and volatility spikes into historical data, artificially flattering backtest accuracy.

### Causal Expanding-Window Formulation
BRDE implements strictly causal expanding-window standardisation where statistics at time $t$ are computed using data strictly up to $t-1$:
$$\hat{\mu}_{t-1} = \frac{1}{t-1}\sum_{i=1}^{t-1} x_i$$
$$\hat{\sigma}_{t-1} = \sqrt{\frac{1}{t-2}\sum_{i=1}^{t-1} (x_i - \hat{\mu}_{t-1})^2}$$
$$z_t = \text{clip}\left(\frac{x_t - \hat{\mu}_{t-1}}{\hat{\sigma}_{t-1} + \epsilon}, -5.0, +5.0\right)$$

- **Burn-In Requirement:** Minimum 252 trading sessions ($\approx 1$ calendar year) required before emitting continuous standardized values.
- **5-Sigma Winsorisation:** Extreme market outliers beyond 5 standard deviations are winsorised to $[-5.0, +5.0]$ to prevent numerical degeneracy in downstream Gaussian Mixture HMM and Deep Ensemble likelihoods.
- **Invariance Guarantee:** Appending future observations $t+1, \dots, t+k$ leaves historical z-scores $z_{1:t}$ bit-for-bit unchanged.

---

## 3. Feature Taxonomy & Economic Rationale

The engine computes 67 total feature columns structured across 8 orthogonal risk channels:

### Channel 1: Multi-Horizon Returns (18 Features)
Evaluates price velocity and return momentum across the market capitalization hierarchy:
- **Nifty 50 Large-Cap:** `ret_1d`, `ret_5d`, `ret_21d`, `ret_63d`, `ret_126d`, `ret_252d`
- **Nifty Midcap 100:** `midcap_ret_1d`, `midcap_ret_5d`, `midcap_ret_21d`, `midcap_ret_63d`, `midcap_ret_126d`, `midcap_ret_252d`
- **Nifty Smallcap 100:** `smallcap_ret_1d`, `smallcap_ret_5d`, `smallcap_ret_21d`, `smallcap_ret_63d`, `smallcap_ret_126d`, `smallcap_ret_252d`

*Economic Rationale:* Multi-horizon cumulative log returns capture early momentum acceleration during bull markets and decoupling between large and smaller cap tiers.

### Channel 2: Realised Volatility & Term Dynamics (5 Features)
- `vol_21d`: 21-day annualized Parkinson/Close-to-Close realized volatility:
  $$\sigma_{21d} = \sqrt{252 \cdot \frac{1}{21}\sum_{i=1}^{21} r_i^2}$$
- `vol_63d`: 63-day quarterly realized volatility baseline.
- `vol_ewma`: Exponentially Weighted Moving Average (EWMA) volatility with $\text{span} = 30$ ($\lambda \approx 0.94$).
- `vol_of_vol`: 21-day rolling standard deviation of 21-day realized volatility, measuring uncertainty of the volatility regime itself.
- `vol_ratio`: Short-term to medium-term volatility term structure ratio (`vol_21d / vol_63d`). A ratio $> 1.4$ signals volatility inversion typical of acute crashes.

### Channel 3: India VIX Implied Moments (5 Features)
- `vix_level`: Raw closing level of NSE India VIX.
- `vix_change_5d`: 5-day absolute difference in India VIX.
- `vix_change_21d`: 21-day momentum of implied volatility.
- `vix_ratio_vol`: Implied-to-realized volatility premium ($\text{VIX} / \sigma_{21d}$). High premia indicate excessive fear pricing.
- `vix_z`: Causal expanding z-score of India VIX ($t-1$ statistics).
- `vol_regime_code`: Discrete scenario classification (0: Low $<13$, 1: Normal $13–18$, 2: Elevated $18–24$, 3: Crisis $\ge 24$).

### Channel 4: Drawdown & Trend Position (6 Features)
- `drawdown`: Peak-to-trough percentage drawdown from rolling 252-day high:
  $$\text{DD}_t = \frac{P_t - \max_{i \in [t-252, t]} P_i}{\max_{i \in [t-252, t]} P_i}$$
- `days_since_peak`: Number of trading days elapsed since the rolling 252-day peak. Protracted durations ($> 120$ days) without new highs indicate structural regime exhaustion.
- `above_200dma`: Binary regime trend filter ($\mathbb{I}(P_t > \text{MA}_{200})$).
- `dist_200dma`: Percentage distance from the 200-day moving average.
- `ma_50_200`: 50-day / 200-day moving average spread (Golden Cross / Death Cross metric).

### Channel 5: Momentum, Skip-Month & Technical Features (5 Features)
- `mom_21d`, `mom_63d`, `mom_252d`: Multi-period momentum returns.
- `mom_skip_month`: 12-month momentum excluding the most recent month ($P_{t-21} / P_{t-252} - 1$), removing short-term 1-month reversal noise (Jegadeesh & Titman, 1993).
- `rsi_14`: 14-day Wilder's Relative Strength Index bounded in $[0, 100]$.

### Channel 6: Market Breadth & Capitalization Divergence (7 Features)
- `adv_dec_ratio`: Daily advances divided by declines ($\text{Advances} / \text{Declines}$).
- `pct_above_50dma`: Percentage of index constituents trading above their 50-day moving average.
- `breadth_mid_minus_large`: 21-day return spread between Midcap 100 and Nifty 50.
- `breadth_small_minus_large`: 21-day return spread between Smallcap 100 and Nifty 50.
- `small_large_spread_63d`: Quarterly small-cap divergence.
- `share_above_200dma`: Fraction of market indices (Large, Mid, Small) concurrently above their 200DMAs ($0.0, 0.33, 0.67, 1.0$).
- `cap_divergence_warning`: Binary warning triggered when small-cap 21d relative underperformance exceeds $-5\%$.

### Channel 7: Institutional Flow Channels (6 Features)
- `fii_flow_5d`: Rolling 5-day sum of Foreign Portfolio Investor net equity purchases (₹ Crores).
- `dii_flow_5d`: Rolling 5-day sum of Domestic Institutional Investor net equity purchases (₹ Crores).
- `flow_balance_ratio`: Ratio of DII buying to absolute FII selling ($\text{DII} / |\text{FII}|$).
- `flow_divergence`: Characteristic Indian regime pattern where domestic mutual funds absorb foreign outflows ($\mathbb{I}(\text{DII} > 0 \land \text{FII} < 0)$).
- `absorption_ratio`: Rate at which domestic SIP inflows cushion institutional liquidation.

### Channel 8: Macro Transmission & Global Linkages (7 Features)
- `gilt_level` & `gilt_change_21d`: India 10-Year Government Securities yield and monthly change.
- `usdinr_level` & `usdinr_change_21d`: USD/INR exchange rate and 21-day depreciation rate.
- `crude_change_21d`: Brent crude oil 21-day return (hydrocarbon import bill pressure).
- `sp500_change_21d`: S&P 500 21-day return (global equity risk appetite).
- `dxy_change_21d`: US Dollar Index change.
- `yield_spread_india_us`: Spread between Indian Gilt and US Treasury reference rates.
- `inr_dxy_residual_21d`: USD/INR return orthogonalized against DXY changes via expanding-window beta.

### Channel 9: Composite Crisis Signals (2 Features)
- `crash_alert_count`: Integer count ($0$ to $4$) of independent stress channels concurrently firing (VIX surge, breadth collapse, FII flight, rupee stress).
- `acute_risk_off_flag`: Binary indicator firing when $\ge 3$ stress channels trigger simultaneously.

---

## 4. Publication Lag & Quality Gate Summary

All features have been vetted against Point-in-Time availability:
- **Zero Lag (EOD Market Close):** Nifty returns, realized volatility, India VIX.
- **1-Day Publication Lag (T+1 Market Open):** Market advances/declines, FII/DII flow totals, global closes (S&P 500, DXY, Brent crude), RBI macro releases.
- **5-Day Publication Lag:** AMFI mutual fund monthly SIP inflow data.

Features with $> 5\%$ missing historical values (primarily 252-day burn-in lookbacks at the series inception in 2010) are explicitly documented and flagged in `data/processed/features_metadata.csv`.
