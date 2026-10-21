# R Verification Codebase & Run Instructions
## Bayesian Regime Detection Engine

**Corporate Identification Number (CIN):** `U62012MH2023PTC410415`  
**Host Runtime Status:** **NOT RUN** (R / Rscript is not installed on the Windows evaluation host).

---

### Overview of R Scripts

This directory contains R implementations fulfilling Deliverable 3 of Section D3:
1. `fit_hmm.R`: 5-state Gaussian Hidden Markov Model and Markov-switching regression via `depmixS4` and `MSwM`.
2. `stan_regime.stan` & `fit_stan.R`: Fully Bayesian Gaussian HMM with sticky Dirichlet priors, NUTS MCMC sampling, and R-hat/ESS diagnostics.
3. `bcp_changepoint.R`: Bayesian Online Changepoint Detection via `bcp` (Barry & Hartigan) and PELT variance shift detection via `changepoint`.
4. `conformal_prediction.R`: Split-conformal prediction sets and empirical coverage auditing at 90% and 95% nominal levels.
5. `cross_validate.py`: Cross-language reconciliation driver comparing R regime probabilities against the Python engine.

---

### Instructions for Running R Scripts

To execute these scripts in an environment with R installed:

```bash
# 1. Install required R packages
Rscript -e 'install.packages(c("depmixS4", "MSwM", "rstan", "bcp", "changepoint"), repos="https://cloud.r-project.org")'

# 2. Export clean data from Python
python r/cross_validate.py

# 3. Run HMM estimation in R
Rscript r/fit_hmm.R

# 4. Run Bayesian Stan MCMC
Rscript r/fit_stan.R

# 5. Run Changepoint Analysis
Rscript r/bcp_changepoint.R

# 6. Run Conformal Prediction
Rscript r/conformal_prediction.R

# 7. Run cross-validation reconciliation
python r/cross_validate.py
```

### Audit Disclosure
Because the host environment does not have R or `Rscript` installed in `PATH`, the numerical cross-validation check is formally reported as **NOT RUN** across `docs/COMPLIANCE.md`, `reports/report.md`, and `reports/model_card.md`. No synthetic "seed invariance" fallback is claimed as reconciliation.
