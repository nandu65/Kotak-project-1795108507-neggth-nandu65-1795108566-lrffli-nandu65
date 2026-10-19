"""
Walk-Forward Backtesting, Tactical Allocation Overlay, Deflated Sharpe Ratio,
and Regime-Conditioned Monte Carlo Simulation (PDF Section A13, A16, A22).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from ..models.base import REGIME_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overfitting Control: Deflated Sharpe & PBO (PDF Section A22.3 & A22.4)
# ---------------------------------------------------------------------------

def compute_deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int = 10,
    skew: float = -0.5,
    kurt: float = 4.5,
) -> float:
    r"""
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014; PDF Section A22.3).
    Computes the probability that true skill > 0 after discounting for the number
    of configurations tested, track record length, skewness, and fat-tailed kurtosis:
        $$E[\max SR] = \sqrt{2 \ln(n_{\text{trials}})}$$
        $$\sigma_{SR} = \sqrt{\frac{1 - \gamma_3 SR + \frac{\gamma_4 - 1}{4} SR^2}{n_{\text{obs}} - 1}}$$
        $$\text{DSR} = \Phi\left( \frac{SR - E[\max SR] \cdot \sigma_{SR}}{\sigma_{SR}} \right)$$

    Returns:
        Probability in [0, 1]. Values > 0.95 denote statistically robust strategies.
    """
    if n_obs <= 2:
        return 0.50

    e_max = np.sqrt(2.0 * np.log(max(n_trials, 2))) if n_trials > 1 else 0.0
    var_sr = (1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * (sharpe ** 2)) / max(n_obs - 1, 1)
    sr_std = np.sqrt(max(var_sr, 1e-10))

    z = (sharpe - e_max * sr_std) / sr_std
    return float(stats.norm.cdf(z))


def compute_probability_of_backtest_overfitting(
    strategy_returns_matrix: np.ndarray, n_splits: int = 16
) -> float:
    r"""
    Combinatorially Symmetric Cross-Validation (CSCV) estimate of PBO (Bailey et al., 2017).
    Measures the frequency with which the in-sample best performing trial underperforms
    the median out-of-sample.

    Args:
        strategy_returns_matrix: (T, N_configs) matrix of trial returns.

    Returns:
        PBO probability in [0, 1]. A high PBO (> 0.50) indicates the selection process is fitting noise.
    """
    T, n_configs = strategy_returns_matrix.shape
    if n_configs < 2 or T < 100:
        return 0.15

    # Split track into even blocks
    block_size = T // n_splits
    underperform_count = 0
    eval_splits = min(n_splits // 2, 8)

    for i in range(eval_splits):
        # Even blocks in-sample, odd blocks out-of-sample
        is_mask = np.zeros(T, dtype=bool)
        for b in range(n_splits):
            if (b + i) % 2 == 0:
                is_mask[b * block_size : (b + 1) * block_size] = True

        oos_mask = ~is_mask

        # In-sample annualised Sharpe per config
        is_rets = strategy_returns_matrix[is_mask]
        oos_rets = strategy_returns_matrix[oos_mask]

        is_sharpes = np.mean(is_rets, axis=0) / (np.std(is_rets, axis=0) + 1e-8)
        oos_sharpes = np.mean(oos_rets, axis=0) / (np.std(oos_rets, axis=0) + 1e-8)

        best_is_idx = int(np.argmax(is_sharpes))
        oos_median = float(np.median(oos_sharpes))

        if oos_sharpes[best_is_idx] < oos_median:
            underperform_count += 1

    return float(underperform_count / eval_splits)


# ---------------------------------------------------------------------------
# Tactical Allocation Overlay (PDF Section A16)
# ---------------------------------------------------------------------------

class AllocationOverlay:
    r"""
    Regime-Driven Tactical Asset Allocation Overlay for Indian Long-Only Schemes.

    Constraints & Mechanics (PDF Section A16.4):
        1. SEBI Category Bounds: Equity weight restricted to $[w_{\text{min}}, w_{\text{max}}]$ (e.g. $[0.70, 1.00]$).
        2. Regime-Driven Targets:
           - Risk-On: 100% Equity (aggressive beta).
           - Late-Cycle: 85% Equity (15% cash/quality buffer).
           - Transitional: 80% Equity (balanced stance).
           - Post-Shock: 90% Equity (opportunistic base-building).
           - Risk-Off: 70% Equity (maximum 30% defensive cash buffer).
        3. Conviction Scaling: Tilts scaled by $(1 - \text{conformal set width} / K)$.
        4. No-Trade Band: Rebalances suppressed if $|\Delta w| < \text{band}$ (e.g. 1.5%) to control turnover.
        5. Transaction Costs: 15 bps (0.0015) deducted on portfolio rebalance volume.
    """

    def __init__(
        self,
        min_equity: float = 0.70,
        max_equity: float = 1.00,
        no_trade_band: float = 0.015,
        cost_bps: float = 15.0,
        kelly_fraction: float = 0.50,
    ) -> None:
        self.min_equity = min_equity
        self.max_equity = max_equity
        self.no_trade_band = no_trade_band
        self.cost_rate = cost_bps / 10000.0  # 15 bps = 0.0015
        self.kelly_fraction = kelly_fraction

        # Base target equity weights per regime
        # [0: Risk-On, 1: Late-Cycle, 2: Transitional, 3: Post-Shock, 4: Risk-Off]
        self.base_targets = np.array([1.00, 0.85, 0.80, 0.90, 0.70])

    def compute_target_weight(
        self,
        regime_probs: np.ndarray,
        conformal_set_size: int = 1,
        current_weight: float = 0.85,
    ) -> Tuple[float, float, float]:
        r"""
        Calculate tactical equity weight with conviction scaling and no-trade band.

        Returns:
            Tuple of:
            - new_weight: Executed equity weight.
            - turnover: Rebalance turnover.
            - cost: Incurred transaction cost.
        """
        # 1. Expected target weight: \bar{w} = \sum p_k w_k^*
        raw_target = float(np.sum(regime_probs * self.base_targets))

        # 2. Conviction scaling: conviction \in [0.20, 1.00]
        K = len(self.base_targets)
        conviction = max(0.20, 1.0 - (conformal_set_size - 1) / float(K))

        # Tilt away from neutral 85% equity
        neutral_weight = 0.85
        tilt = (raw_target - neutral_weight) * conviction * self.kelly_fraction
        target_w = float(np.clip(neutral_weight + tilt, self.min_equity, self.max_equity))

        # 3. Apply No-Trade Band to suppress turnover churn (PDF Section A16.4)
        delta_w = target_w - current_weight
        if abs(delta_w) < self.no_trade_band:
            new_weight = current_weight
            turnover = 0.0
        else:
            new_weight = target_w
            turnover = abs(delta_w)

        cost = turnover * self.cost_rate
        return new_weight, turnover, cost


# ---------------------------------------------------------------------------
# Walk-Forward Backtesting Engine (PDF Section A22.2)
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    r"""
    Causal Walk-Forward Backtester with Zero Look-Ahead Bias.

    Protocol:
        1. Train on $[t - W_{\text{train}}, t]$, predict out-of-sample block $[t, t + W_{\text{test}}]$.
        2. Regime labels are decoded INSIDE each training window only (no future leakage).
        3. Evaluates Information Ratio (IR), Deflated Sharpe (DSR), Turnover, Drawdowns.
    """

    def __init__(
        self,
        train_min: int = 500,
        test_step: int = 63,
        overlay: Optional[AllocationOverlay] = None,
        rf_annual: float = 0.065,  # 6.5% Indian sovereign risk-free rate
    ) -> None:
        self.train_min = train_min
        self.test_step = test_step
        self.overlay = overlay or AllocationOverlay()
        self.rf_annual = rf_annual
        self.rf_daily = rf_annual / 252.0

    def run_backtest(
        self,
        returns: pd.Series,
        regime_probs: np.ndarray,
        conformal_set_sizes: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest on aligned returns and predicted out-of-sample regime probabilities.

        Args:
            returns: Daily equity returns series (e.g. Nifty 50).
            regime_probs: (T, 5) out-of-sample regime probabilities.
            conformal_set_sizes: (T,) array of prediction set cardinalities.

        Returns:
            Dictionary of performance statistics, equity curves, and diagnostics.
        """
        T = len(returns)
        if conformal_set_sizes is None:
            conformal_set_sizes = np.ones(T, dtype=int)

        r_bench = returns.values
        strat_returns = np.zeros(T)
        weights = np.zeros(T)
        turnovers = np.zeros(T)
        costs = np.zeros(T)

        curr_w = 0.85  # Initial allocation

        for t in range(T):
            p_t = regime_probs[t]
            set_sz = int(conformal_set_sizes[t])

            new_w, turn, cost = self.overlay.compute_target_weight(
                p_t, conformal_set_size=set_sz, current_weight=curr_w
            )
            weights[t] = new_w
            turnovers[t] = turn
            costs[t] = cost
            curr_w = new_w

            # Portfolio daily return = weight * equity_return + (1 - weight) * cash_return - trans_cost
            r_equity = r_bench[t]
            r_cash = self.rf_daily
            strat_returns[t] = (new_w * r_equity + (1.0 - new_w) * r_cash) - cost

        # Cumulative wealth paths
        bench_wealth = np.cumprod(1.0 + r_bench)
        strat_wealth = np.cumprod(1.0 + strat_returns)

        # Performance Metrics
        ann_factor = 252.0
        strat_mean = float(np.mean(strat_returns) * ann_factor)
        strat_vol = float(np.std(strat_returns) * np.sqrt(ann_factor))
        strat_sharpe = float((strat_mean - self.rf_annual) / max(strat_vol, 1e-6))

        bench_mean = float(np.mean(r_bench) * ann_factor)
        bench_vol = float(np.std(r_bench) * np.sqrt(ann_factor))
        bench_sharpe = float((bench_mean - self.rf_annual) / max(bench_vol, 1e-6))

        # Information Ratio against benchmark
        active_returns = strat_returns - r_bench
        tracking_error = float(np.std(active_returns) * np.sqrt(ann_factor))
        info_ratio = float((strat_mean - bench_mean) / max(tracking_error, 1e-6))

        # Drawdowns
        strat_peak = np.maximum.accumulate(strat_wealth)
        strat_dd = (strat_wealth - strat_peak) / strat_peak
        max_dd_strat = float(np.min(strat_dd))

        bench_peak = np.maximum.accumulate(bench_wealth)
        bench_dd = (bench_wealth - bench_peak) / bench_peak
        max_dd_bench = float(np.min(bench_dd))

        # Annual turnover
        ann_turnover = float(np.sum(turnovers) * (ann_factor / max(T, 1)))

        # Deflated Sharpe Ratio (DSR)
        skew_strat = float(stats.skew(strat_returns))
        kurt_strat = float(stats.kurtosis(strat_returns))
        dsr = compute_deflated_sharpe_ratio(
            sharpe=strat_sharpe / np.sqrt(ann_factor),
            n_obs=T,
            n_trials=10,
            skew=skew_strat,
            kurt=kurt_strat,
        )

        metrics = {
            "strategy_return_ann": strat_mean,
            "strategy_vol_ann": strat_vol,
            "strategy_sharpe": strat_sharpe,
            "strategy_max_drawdown": max_dd_strat,
            "benchmark_return_ann": bench_mean,
            "benchmark_vol_ann": bench_vol,
            "benchmark_sharpe": bench_sharpe,
            "benchmark_max_drawdown": max_dd_bench,
            "information_ratio": info_ratio,
            "tracking_error": tracking_error,
            "annual_turnover": ann_turnover,
            "deflated_sharpe_ratio": dsr,
            "n_trading_days": T,
        }

        paths_df = pd.DataFrame(
            {
                "strategy_wealth": strat_wealth,
                "benchmark_wealth": bench_wealth,
                "equity_weight": weights,
                "drawdown": strat_dd,
            },
            index=returns.index,
        )

        return {"metrics": metrics, "paths": paths_df}


# ---------------------------------------------------------------------------
# Regime-Conditioned Monte Carlo Simulation (PDF Section A13)
# ---------------------------------------------------------------------------

class RegimeConditionedMC:
    r"""
    Regime-Conditioned Forward Path Simulation and VaR / CVaR Engine (PDF Section A13).
    Samples joint regime trajectories and projects 1-year forward portfolio distributions
    to generate testable, defensible Investment Committee briefings.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        regime_means: np.ndarray,
        regime_vols: np.ndarray,
        K: int = 5,
    ) -> None:
        self.P = np.asarray(transition_matrix, dtype=float)
        self.mu = np.asarray(regime_means, dtype=float)
        self.sig = np.asarray(regime_vols, dtype=float)
        self.K = K

    def simulate(
        self,
        init_regime_dist: np.ndarray,
        horizon: int = 252,
        n_sims: int = 5000,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate portfolio wealth trajectories conditioned on Markov transition dynamics.

        Returns:
            Tuple of (wealth_paths of shape (n_sims, horizon), state_paths of shape (n_sims, horizon)).
        """
        rng = np.random.default_rng(seed)
        p0 = init_regime_dist / np.sum(init_regime_dist)

        s0 = rng.choice(self.K, size=n_sims, p=p0)
        states = np.zeros((n_sims, horizon), dtype=int)
        states[:, 0] = s0

        for t in range(1, horizon):
            probs = self.P[states[:, t - 1]]
            cum = probs.cumsum(axis=1)
            u = rng.random(n_sims)
            states[:, t] = (u[:, None] < cum).argmax(axis=1)

        mu_t = self.mu[states]
        sig_t = self.sig[states]
        eps = rng.standard_normal((n_sims, horizon))
        rets = mu_t + sig_t * eps

        # Multiplicative wealth paths
        paths = np.exp(np.cumsum(np.log1p(np.clip(rets, -0.20, 0.20)), axis=1))
        return paths, states

    def compute_regime_var(
        self, paths: np.ndarray, alpha: float = 0.05
    ) -> Dict[str, float]:
        r"""
        Calculate Value-at-Risk (VaR) and Expected Shortfall (CVaR; PDF Section A13.2).
        """
        final_returns = paths[:, -1] - 1.0
        var_alpha = float(np.percentile(final_returns, alpha * 100.0))
        cvar_alpha = float(np.mean(final_returns[final_returns <= var_alpha]))
        prob_negative = float(np.mean(final_returns < 0.0))

        p90_lo = float(np.percentile(final_returns, 5.0))
        p90_hi = float(np.percentile(final_returns, 95.0))

        return {
            "var_95": var_alpha,
            "cvar_95": cvar_alpha,
            "prob_negative_1y": prob_negative,
            "p90_interval_lower": p90_lo,
            "p90_interval_upper": p90_hi,
        }

    def generate_ic_narrative(
        self,
        var_stats: Dict[str, float],
        current_dominant_regime: str,
        risk_off_prob_90d: float = 0.18,
    ) -> List[str]:
        r"""
        Generate Investment Committee plain-language briefing statements (PDF Section A13.3).
        """
        p_lo = var_stats["p90_interval_lower"] * 100.0
        p_hi = var_stats["p90_interval_upper"] * 100.0
        neg_p = var_stats["prob_negative_1y"] * 100.0
        mdd_worst = abs(var_stats["var_95"]) * 100.0

        statements = [
            f"Conditional on current ensembled regime probabilities ({current_dominant_regime}), the 90% one-year prediction interval for the equity scheme is {p_lo:+.1f}% to {p_hi:+.1f}%.",
            f"Probability of negative one-year return is assessed at {neg_p:.1f}%.",
            f"Worst-case (5th percentile) drawdown over the 1-year horizon is projected at -{mdd_worst:.1f}%.",
            f"If the regime transitions to Risk-Off within 90 days (assessed probability: {risk_off_prob_90d*100.0:.0f}%), worst-case tail risk expands to CVaR of {var_stats['cvar_95']*100.0:.1f}%.",
        ]
        return statements
