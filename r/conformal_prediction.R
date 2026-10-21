# ==============================================================================
# Bayesian Regime Detection Engine - R Conformal Prediction Script
# Strictly Private and Confidential · Zetheta Algorithms Private Limited · CIN U62012MH2023PTC410415
# ==============================================================================

cat("=== Conformal Prediction in R (Finite-Sample Valid Sets) ===\n")

csv_path <- "r_input_data.csv"
if (!file.exists(csv_path)) {
  stop("Input data file r_input_data.csv not found.")
}

df <- read.csv(csv_path)
y <- na.omit(df$nifty_return)
n <- length(y)

# Calibration split (50/50 split)
n_cal <- floor(n / 2)
cal_y <- y[1:n_cal]
test_y <- y[(n_cal + 1):n]

# Simple conformal prediction intervals for forward returns
# Non-conformity score: absolute residual from rolling median
med_val <- median(cal_y)
scores <- sort(abs(cal_y - med_val))

# Nominal coverage 90% (alpha = 0.10) and 95% (alpha = 0.05)
for (nominal in c(0.90, 0.95)) {
  alpha <- 1 - nominal
  q_idx <- ceiling((n_cal + 1) * (1 - alpha))
  q_idx <- min(max(q_idx, 1), n_cal)
  cutoff <- scores[q_idx]
  
  # Evaluate coverage on test set
  test_covered <- abs(test_y - med_val) <= cutoff
  emp_cov <- mean(test_covered)
  
  cat(sprintf("Nominal: %2.0f%% | Conformal Cutoff: %.5f | Test Empirical Coverage: %.2f%%\n",
              nominal * 100, cutoff, emp_cov * 100))
}

cat("Conformal evaluation complete.\n")
