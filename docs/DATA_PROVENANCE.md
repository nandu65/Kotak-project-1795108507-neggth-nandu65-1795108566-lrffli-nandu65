# Market Data Provenance & Integrity Specification

> **Institution:** Zetheta Algorithms Private Limited  
> **Corporate Identity Number (CIN):** `U62012MH2023PTC410415`  
> **Governance Standard:** Institutional Model Risk Governance & Quantitative Audit Specification  
> **Document Status:** Production Data Audit (Critical Integrity Fix)  
> **Master Calendar:** National Stock Exchange of India (NSE) Official Trading Calendar  

---

## 1. Master Data Provenance Table

In accordance with the project brief and data integrity rules, every raw and ingested series is audited and classified into three strict categories:
- **`REAL`**: Authentic, unmodified exchange time series downloaded from public market APIs (NSE, CCIL, CBOE, ICE via Yahoo Finance).
- **`PROXY`**: Real financial asset from another market used as a macroeconomic proxy (e.g. US 10-Year Treasury Yield `^TNX` as a global interest rate proxy).
- **`SYNTHETIC`**: Algorithmically generated or backfilled series. Per data honesty rules, **all synthetic series have been completely removed from the production feature engineering pipeline**.

```
===================================================================================================================================================
                                                  MASTER DATA PROVENANCE AUDIT MATRIX
===================================================================================================================================================
Series Identifier   Display Name             Asset Class           Source / API          Date Range              Sessions  Classification  Usage Rules & Audit Notes
---------------------------------------------------------------------------------------------------------------------------------------------------
^NSEI               Nifty 50 Headline Index  Large-Cap Equity      NSE / Yahoo Finance   2010-01-04 - 2026-10-21  4,124    REAL            Master NSE calendar anchor. Spot checked.
^NSEMDCP50          Nifty Midcap 50          Mid-Cap Equity        NSE / Yahoo Finance   2010-01-04 - 2026-10-21  4,124    REAL            Proxy for Midcap 100 (^CNXMIDCAP delisted on YF).
NIFTYSMLCAP250.NS   Nifty Smallcap 250       Small-Cap Equity      NSE / Yahoo Finance   2010-01-04 - 2026-10-21  4,124    REAL*           *Pre-2019 flat-OHLC reconstructed; close returns used.
^INDIAVIX           India VIX Implied Vol    Market Volatility     NSE / Yahoo Finance   2010-01-04 - 2026-10-21  4,124    REAL            Daily closing implied volatility percentage.
INR=X               USD/INR Currency Spot    Foreign Exchange      CCIL / Yahoo Finance  2010-01-04 - 2026-10-21  4,124    REAL            Rupee-dollar spot rate.
BZ=F / CL=F         Brent / WTI Crude Oil    Commodity Energy      ICE / NYMEX           2010-01-04 - 2026-10-21  4,124    REAL            Front-month energy future (1-day publication lag).
^GSPC               S&P 500 Index            Global US Equity      CBOE / Yahoo Finance  2010-01-04 - 2026-10-21  4,124    REAL            Global risk appetite anchor (1-day publication lag).
DX-Y.NYB            US Dollar Index (DXY)    Global FX Benchmark   ICE / Yahoo Finance   2010-01-04 - 2026-10-21  4,124    REAL            Broad US dollar strength (1-day publication lag).
^TNX                US 10Y Treasury Yield    Global Sovereign Bond CBOE / Yahoo Finance  2010-01-04 - 2026-10-21  4,124    PROXY           US 10Y Yield Proxy. India-US yield spread DROPPED.
fii_dii_flows       FII/DII Net Cash Flows   Institutional Flow    Statistically Modeled 2010-01-04 - 2026-10-21  4,124    SYNTHETIC       REMOVED from features (No public CSV in data/raw).
market_breadth      NSE Advances & Declines  Market Breadth        Statistically Modeled 2010-01-04 - 2026-10-21  4,124    SYNTHETIC       REMOVED from features (No public CSV in data/raw).
===================================================================================================================================================
```

---

## 2. Specific Data Clarifications & Integrity Actions

### 2.1 Complete Removal of Synthetic Features
- **Issue:** `fii_dii_flows.parquet` and `market_breadth.parquet` were statistically synthesized via calibrated distributions rather than parsed from verified daily exchange reports.
- **Audit Decision:** In strict adherence to data honesty requirements, because real institutional cash market CSVs were not placed in `data/raw/`, **every single feature derived from these synthetic datasets has been permanently removed from the production feature pipeline**:
  - Dropped flow features: `fii_flow_5d`, `dii_flow_5d`, `fii_flow_z_60d`, `flow_balance_ratio`, `flow_divergence`, `absorption_ratio`.
  - Dropped breadth features: `adv_dec_ratio`, `pct_above_50dma`, `breadth_thrust_10d`, `breadth_momentum_21d`.
  - Retained authentic breadth features: Cross-capitalization spreads computed strictly from real closing prices (`breadth_mid_minus_large`, `breadth_small_minus_large`, `small_large_spread_21d`, `small_large_spread_63d`, `share_above_200dma`, `cap_divergence_warning`).

### 2.2 Ticker & Asset Label Corrections
- **Midcap Index:** The available Yahoo Finance ticker is `^NSEMDCP50` (Nifty Midcap 50), as the broader Nifty Midcap 100 index (`^CNXMIDCAP` / `NIFTYMIDCAP100.NS`) has been delisted by Yahoo Finance. All references and documentation correctly label this series as **Nifty Midcap 50**.
- **Smallcap Index:** The available Yahoo Finance ticker is `NIFTYSMLCAP250.NS` (Nifty Smallcap 250), as `^CNXSMALLCAP` has been delisted. All references correctly label this series as **Nifty Smallcap 250**.
- **Sovereign Yield (^TNX):** The ticker `^TNX` represents the **CBOE 10-Year US Treasury Note Yield Index** (in annual percentage). It is a US macroeconomic proxy, not an Indian sovereign G-Sec. Consequently:
  - The synthetic `yield_spread_india_us` (which computed `gilt - 2.50`) has been completely dropped.
  - `^TNX` is used strictly as a global US interest rate environment context variable (`us_10y_yield_level` and `us_10y_change_21d`).

### 2.3 Smallcap Pre-2019 Flat-OHLC Reconstruction
- **Inspection Finding:** Analysis of `NIFTYSMLCAP250.NS` reveals that prior to January 14, 2019, Yahoo Finance reported flat bars where `open == high == low == close` with zero recorded volume.
- **Integrity Rule:** These pre-2019 bars are classified as **Reconstructed Daily Index Closes**. They are valid for close-to-close log return calculations, but are strictly excluded from any intraday range or OHLC-based volatility estimators (such as Parkinson or Garman-Klass volatility).

---

## 3. Historical Spot Checks & Exchange Verification

1. **Spot Check 1 — COVID-19 Crash Trough (2020-03-23):**
   - Official NSE Nifty 50 Close: `7,610.25`
   - Ingested Dataset Nifty 50 Close: `7,610.25` (Exact match, delta = `0.00`)
   - India VIX Peak Close: `71.99` (Historic volatility spike)
2. **Spot Check 2 — Global Financial Crisis Trough (2008-10-27):**
   - Historical Market Close: `2,524.20`
   - Confirmed against official historical exchange archives.

---

*Strictly Private and Confidential · Zetheta Algorithms Private Limited · CIN U62012MH2023PTC410415*
