# ==============================================================================
# Bayesian Regime Detection Engine - R Bayesian Changepoint Detection Script
# Strictly Private and Confidential · Zetheta Algorithms Private Limited · CIN U62012MH2023PTC410415
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("bcp")) {
    install.packages("bcp", repos = "https://cloud.r-project.org")
    library(bcp)
  }
  if (!require("changepoint")) {
    install.packages("changepoint", repos = "https://cloud.r-project.org")
    library(changepoint)
  }
})

cat("=== Bayesian Changepoint Analysis (bcp / changepoint) ===\n")
csv_path <- "r_input_data.csv"
if (!file.exists(csv_path)) {
  stop("Input data file r_input_data.csv not found.")
}

df <- read.csv(csv_path)
y <- na.omit(df$nifty_return)

# 1. Barry and Hartigan Bayesian Changepoint model
cat("Fitting Barry & Hartigan Bayesian Changepoint model...\n")
bcp_fit <- bcp(y, p0 = 0.02, burnin = 500, mcmc = 2000)

posterior_prob <- bcp_fit$posterior.prob
high_prob_cps <- which(posterior_prob > 0.30)
cat(sprintf("Detected %d candidate changepoints with posterior probability > 0.30.\n", length(high_prob_cps)))

# 2. PELT Changepoint in Variance (PELT algorithm)
cat("Fitting PELT variance changepoint model...\n")
cpt_var <- cpt.var(y, method = "PELT", penalty = "SIC")
cpt_locs <- cpts(cpt_var)
cat(sprintf("PELT identified %d structural variance shifts.\n", length(cpt_locs)))

output_cp <- data.frame(
  step = 1:length(y),
  bcp_prob = posterior_prob
)
write.csv(output_cp, "r_changepoint_output.csv", row.names = FALSE)
cat("Exported changepoint probabilities to r_changepoint_output.csv.\n")
