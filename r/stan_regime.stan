// Stan Model for Bayesian Gaussian Hidden Markov Model with Sticky Dirichlet Prior
// Strictly Private and Confidential · Zetheta Algorithms Private Limited · CIN U62012MH2023PTC410415

data {
  int<lower=1> T;                // Number of time steps
  int<lower=1> K;                // Number of regimes (K=5)
  vector[T] y;                   // Return series
  vector[K] alpha_prior;         // Dirichlet base hyperparameter
  real<lower=0> kappa;           // Sticky diagonal bonus
}

parameters {
  simplex[K] pi_init;            // Initial state distribution
  simplex[K] A[K];               // Transition probability matrix
  ordered[K] mu;                 // Ordered regime means to prevent label switching
  vector<lower=1e-4>[K] sigma;   // Regime emission volatilities
}

model {
  // Priors
  pi_init ~ dirichlet(rep_vector(1.0, K));
  for (k in 1:K) {
    vector[K] dir_prior = alpha_prior;
    dir_prior[k] += kappa;       // Sticky diagonal self-transition reinforcement
    A[k] ~ dirichlet(dir_prior);
    sigma[k] ~ cauchy(0, 0.02);
  }
  mu ~ normal(0, 0.01);

  // Exact forward log-marginalisation (forward algorithm)
  {
    vector[K] log_alpha;
    vector[K] log_alpha_next;
    
    // t = 1
    for (k in 1:K) {
      log_alpha[k] = log(pi_init[k]) + normal_lpdf(y[1] | mu[k], sigma[k]);
    }
    
    // t = 2 .. T
    for (t in 2:T) {
      for (k in 1:K) {
        vector[K] acc;
        for (j in 1:K) {
          acc[j] = log_alpha[j] + log(A[j][k]);
        }
        log_alpha_next[k] = log_sum_exp(acc) + normal_lpdf(y[t] | mu[k], sigma[k]);
      }
      log_alpha = log_alpha_next;
    }
    
    target += log_sum_exp(log_alpha);
  }
}
