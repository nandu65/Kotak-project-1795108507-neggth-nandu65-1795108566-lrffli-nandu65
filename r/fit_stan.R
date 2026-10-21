# ==============================================================================
# Bayesian Regime Detection Engine - R Stan / rstanarm Verification Script
# Strictly Private and Confidential · Zetheta Algorithms Private Limited · CIN U62012MH2023PTC410415
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("rstan")) {
    install.packages("rstan", repos = "https://cloud.r-project.org")
    library(rstan)
  }
})

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

cat("=== Fitting Bayesian HMM via Stan ===\n")
csv_path <- "r_input_data.csv"
if (!file.exists(csv_path)) {
  stop("Input data file r_input_data.csv not found. Run python r/cross_validate.py first.")
}

df <- read.csv(csv_path)
y <- na.omit(df$nifty_return)
T_len <- length(y)
K <- 5

stan_data <- list(
  T = T_len,
  K = K,
  y = as.vector(y),
  alpha_prior = rep(1.0, K),
  kappa = 8.0
)

cat(sprintf("Running No-U-Turn Sampler (NUTS) across 4 chains for %d observations...\n", T_len))
fit <- stan(
  file = "stan_regime.stan",
  data = stan_data,
  iter = 1000,
  warmup = 500,
  chains = 4,
  seed = 42
)

print(fit, pars = c("mu", "sigma", "A"))

# Extract posterior draws and compute MCMC diagnostics
summary_fit <- summary(fit)$summary
rhats <- summary_fit[, "Rhat"]
neffs <- summary_fit[, "n_eff"]

cat(sprintf("Max R-hat: %.4f (Target < 1.05)\n", max(rhats, na.rm = TRUE)))
cat(sprintf("Min ESS: %.1f (Target > 400)\n", min(neffs, na.rm = TRUE)))

write.csv(as.data.frame(summary_fit), "stan_posterior_summary.csv")
cat("Saved stan_posterior_summary.csv successfully.\n")
