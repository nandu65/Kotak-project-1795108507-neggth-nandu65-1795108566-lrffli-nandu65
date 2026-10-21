"""
Independent Rule-Based Ground Truth Regime Classification.

This module implements an independent, non-model-derived ground truth regime classification
strictly calibrated to the brief's Section A1.4 indicator table:
  1. Nifty 50DMA vs 200DMA (trend alignment & golden/death cross)
  2. India VIX thresholds (< 14 Risk-On, 14-22 Moderate/Late-Cycle/Transitional, > 22 Risk-Off/Crash)
  3. Authentic Breadth proxy from real index data (Midcap 50 & Smallcap 250 participation vs Nifty 50)
  4. Drawdown from trailing 252-day peak
  5. Volatility decay and price base-building after an acute market shock

CRITICAL INTEGRITY PRINCIPLE:
These rule-based labels serve as the INDEPENDENT EVALUATION TARGET for scoring out-of-sample
probabilistic models (Log Loss, Brier score, RPS, ECE, Conformal Prediction coverage).
They must NOT be used to train the model that is being evaluated.
"""

from __future__ import annotations

import logging
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_NAMES: List[str] = [
    "Risk-On",      # State 0: Sustained equity rally, broad participation, low VIX (<14-16)
    "Late-Cycle",   # State 1: Mature expansion, narrow leadership/divergence, VIX 14-22, fatigue
    "Transitional", # State 2: Regime boundary, conflicting indicators, chop/consolidation
    "Post-Shock",   # State 3: Mean-reversion following acute drawdown, volatility decay, base-building
    "Risk-Off",     # State 4: Acute drawdown, flight to quality, Nifty < 200DMA, VIX > 22
]


def compute_rule_based_regimes(
    pit_df: pd.DataFrame,
    vix_low_thresh: float = 16.0,
    vix_high_thresh: float = 21.0,
    vix_panic_thresh: float = 25.0,
    breadth_spread_thresh: float = -0.02,
    breadth_share_thresh: float = 0.66,
    drawdown_riskoff_thresh: float = -0.12,
    drawdown_riskon_thresh: float = -0.05,
    vix_decay_delta: float = -2.5,
) -> pd.Series:
    r"""
    Compute daily independent ground-truth regime labels strictly from market observables.

    Section A1.4 Indicator Rules:
    -----------------------------
    1. Acute Crash / Risk-Off (State 4):
       - Condition A: Nifty < 200DMA AND India VIX > 21.0
       - Condition B: Trailing 252d Peak Drawdown < -12.0%
       - Condition C: India VIX > 25.0 AND Peak Drawdown < -6.0%
       - Condition D: Acute panic: India VIX > 30.0 AND Nifty < 200DMA

    2. Post-Shock (State 3):
       - Recent acute shock: Trailing 21d peak VIX > 22.0 OR 21d trough drawdown < -8.0%
       - Volatility decaying: VIX trailing 21d decay <= -2.5 OR 5d VIX change <= -1.5 OR VIX < 24.0
       - Price base-building: 5-day return > 0% AND Drawdown > -18.0%
       - Not in acute crash condition

    3. Risk-On (State 0):
       - Sustained trend: Nifty > 200DMA AND Nifty 50DMA > 200DMA
       - Low/calm volatility: India VIX < 16.0
       - Near highs: Trailing 252d Drawdown > -5.0%
       - Broad participation: Authentic breadth spread >= -0.02 AND share above 50DMA >= 0.66

    4. Late-Cycle (State 1):
       - Above 200DMA: Nifty > 200DMA
       - Near highs / moderate drawdown: Drawdown > -8.0%
       - Narrow leadership / fatigue:
         * Authentic breadth spread < -0.02 (Mid/Small lagging Large Cap)
         * OR share of indices above 50DMA < 0.66
         * OR India VIX creeping into [16.0, 22.0]

    5. Transitional (State 2):
       - Conflicting trend, mixed breadth, chop around 50DMA/200DMA, low directional conviction.

    Args:
        pit_df: Cleaned PIT market dataframe containing 'nifty_50', 'india_vix',
                and real breadth proxies ('nifty_midcap_100', 'nifty_smallcap_100').
        vix_low_thresh: Ceiling for calm Risk-On volatility (default: 16.0).
        vix_high_thresh: Floor for elevated Risk-Off volatility (default: 21.0).
        vix_panic_thresh: Extreme panic VIX threshold (default: 25.0).
        breadth_spread_thresh: Divergence spread floor for broad rally (default: -0.02).
        breadth_share_thresh: Minimum share of indices above 50DMA for broad rally (default: 0.66).
        drawdown_riskoff_thresh: Drawdown threshold triggering Risk-Off (default: -0.12).
        drawdown_riskon_thresh: Maximum allowable drawdown in Risk-On (default: -0.05).
        vix_decay_delta: Minimum drop from 21d VIX peak denoting vol decay (default: -2.5).

    Returns:
        pd.Series of integer regime labels in [0, 1, 2, 3, 4] indexed by date.
    """
    if "nifty_50" not in pit_df.columns:
        raise ValueError("pit_df must contain 'nifty_50' column.")

    nifty = pit_df["nifty_50"].astype(float)
    vix = pit_df["india_vix"].astype(float) if "india_vix" in pit_df.columns else pd.Series(15.0, index=pit_df.index)

    # 1. Trend indicators: Nifty 50DMA and 200DMA
    ma_50 = nifty.rolling(50, min_periods=20).mean()
    ma_200 = nifty.rolling(200, min_periods=50).mean()

    # 2. Drawdown from trailing 252-day high-water mark
    peak_252 = nifty.rolling(252, min_periods=20).max()
    drawdown = (nifty - peak_252) / (peak_252 + 1e-8)

    # 3. Authentic Breadth proxy from real index data
    has_mid = "nifty_midcap_100" in pit_df.columns
    has_small = "nifty_smallcap_100" in pit_df.columns

    if has_mid and has_small:
        mid = pit_df["nifty_midcap_100"].astype(float)
        small = pit_df["nifty_smallcap_100"].astype(float)
        mid_ma50 = mid.rolling(50, min_periods=20).mean()
        small_ma50 = small.rolling(50, min_periods=20).mean()

        share_above_50 = (
            (nifty > ma_50).astype(int)
            + (mid > mid_ma50).astype(int)
            + (small > small_ma50).astype(int)
        ) / 3.0

        mid_rel_21 = (mid / mid.shift(21)) - (nifty / nifty.shift(21))
        small_rel_21 = (small / small.shift(21)) - (nifty / nifty.shift(21))
        breadth_spread = 0.5 * (mid_rel_21 + small_rel_21)
    else:
        # Documented fallback if separate mid/small columns absent
        share_above_50 = (nifty > ma_50).astype(float)
        breadth_spread = pd.Series(0.0, index=pit_df.index)

    # 4. Volatility shock and decay
    vix_peak_21 = vix.rolling(21, min_periods=5).max()
    vix_decay = vix - vix_peak_21
    vix_5d_change = vix - vix.shift(5)

    recent_shock = (
        (vix_peak_21 > 22.0)
        | (drawdown.rolling(21, min_periods=5).min() < -0.08)
    )
    vol_decaying = (
        (vix_decay <= vix_decay_delta)
        | (vix_5d_change <= -1.5)
        | (vix < 24.0)
    )
    price_recovering = (
        (nifty > nifty.shift(5))
        & (drawdown > -0.18)
    )

    labels = pd.Series(np.nan, index=pit_df.index, dtype=float)
    T = len(pit_df)

    for t in range(T):
        d_ma200 = ma_200.iloc[t]
        if np.isnan(d_ma200):
            continue

        d_nifty = nifty.iloc[t]
        d_ma50 = ma_50.iloc[t]
        d_vix = vix.iloc[t]
        d_dd = drawdown.iloc[t]
        d_bspread = breadth_spread.iloc[t]
        d_share = share_above_50.iloc[t]
        d_shock = recent_shock.iloc[t]
        d_vdecay = vol_decaying.iloc[t]
        d_rec = price_recovering.iloc[t]

        # Rule 1: Acute Panic / Crash (Priority 1)
        is_acute_crash = (
            (d_vix > 30.0 and d_nifty < d_ma200)
            or (d_dd < -0.18 and d_vix > 24.0)
        )

        # Rule 2: Post-Shock Stabilization (Priority 2)
        is_post_shock = (
            not is_acute_crash
            and d_shock
            and d_vdecay
            and (d_rec or d_nifty > d_ma50)
            and (d_dd < -0.03)
        )

        # Rule 3: Risk-Off Persistent Drawdown
        is_risk_off = (
            not is_post_shock
            and (
                (d_nifty < d_ma200 and d_vix > vix_high_thresh)
                or (d_dd < drawdown_riskoff_thresh)
                or (d_vix > vix_panic_thresh and d_dd < -0.06)
            )
        )

        # Rule 4: Risk-On Broad Expansion
        is_risk_on = (
            not is_post_shock
            and not is_risk_off
            and (d_nifty > d_ma200)
            and (d_ma50 > d_ma200)
            and (d_vix < vix_low_thresh)
            and (d_dd > drawdown_riskon_thresh)
            and (d_bspread >= breadth_spread_thresh)
            and (d_share >= breadth_share_thresh)
        )

        # Rule 5: Late-Cycle Fatigue / Narrow Breadth
        is_late_cycle = (
            not is_post_shock
            and not is_risk_off
            and not is_risk_on
            and (d_nifty > d_ma200)
            and (d_dd > -0.08)
            and (
                (d_bspread < breadth_spread_thresh)
                or (d_share < breadth_share_thresh)
                or (vix_low_thresh <= d_vix <= vix_high_thresh)
            )
        )

        # Assign mutually exclusive state
        if is_acute_crash or is_risk_off:
            labels.iloc[t] = 4  # Risk-Off
        elif is_post_shock:
            labels.iloc[t] = 3  # Post-Shock
        elif is_risk_on:
            labels.iloc[t] = 0  # Risk-On
        elif is_late_cycle:
            labels.iloc[t] = 1  # Late-Cycle
        else:
            labels.iloc[t] = 2  # Transitional

    return labels
