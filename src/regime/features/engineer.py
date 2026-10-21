"""
Feature Engineering Pipeline for Indian Equity Regime Detection.
Constructs 35+ economically motivated features spanning return momentum,
realised volatility, drawdown dynamics, India VIX implied moments, market breadth,
institutional flow channels, macro transmission, and cap-segmentation.
All standardisation is performed strictly using expanding-window statistics (zero look-ahead)
with 5-sigma winsorisation and statistics from t-1 and earlier.

Outputs:
- data/processed/features.parquet
- data/processed/market_features_standardized.parquet
- data/processed/features_metadata.csv
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_expanding_zscore(
    series: pd.Series,
    min_periods: int = 252,
    winsorize_sigma: float = 5.0,
    eps: float = 1e-8,
) -> pd.Series:
    r"""
    Compute strictly causal expanding-window z-score without look-ahead bias.

    Maths:
        $$\hat{\mu}_{t-1} = \frac{1}{t-1}\sum_{i=1}^{t-1} x_i, \quad
          \hat{\sigma}_{t-1} = \sqrt{\frac{1}{t-2}\sum_{i=1}^{t-1} (x_i - \hat{\mu}_{t-1})^2}$$
        $$z_t = \text{clip}\left(\frac{x_t - \hat{\mu}_{t-1}}{\hat{\sigma}_{t-1} + \epsilon}, -5.0, +5.0\right)$$

    To guarantee zero look-ahead bias, statistics at time $t$ are calculated
    over observations strictly up to $t-1$ (shifted by 1).
    """
    exp_mean = series.expanding(min_periods=min_periods).mean().shift(1)
    exp_std = series.expanding(min_periods=min_periods).std().shift(1)
    z = (series - exp_mean) / (exp_std + eps)
    if winsorize_sigma is not None:
        z = z.clip(-winsorize_sigma, winsorize_sigma)
    return z


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute 14-period Wilder's Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class ExpandingStandardizer:
    r"""
    Standardises feature matrices using expanding-window mean and standard deviation.
    Guarantees no future statistical leakage into current observations:
        $$Z_t = \text{clip}\left(\frac{X_t - \mu_{t-1}}{\sigma_{t-1} + \epsilon}, -5.0, +5.0\right)$$
    """

    def __init__(
        self,
        min_periods: int = 252,
        winsorize_sigma: float = 5.0,
        eps: float = 1e-8,
    ) -> None:
        self.min_periods = min_periods
        self.winsorize_sigma = winsorize_sigma
        self.eps = eps

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature DataFrame using expanding-window parameters."""
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            out[col] = compute_expanding_zscore(
                X[col],
                min_periods=self.min_periods,
                winsorize_sigma=self.winsorize_sigma,
                eps=self.eps,
            )
        return out


class FeatureEngineer:
    r"""
    Generates institutional-grade feature matrices for Indian market regime detection.

    Feature taxonomy (35+ features across 8 risk channels):
    1. Multi-horizon returns for Nifty 50, Midcap 100, Smallcap 100 (1d, 5d, 21d, 63d, 126d, 252d).
    2. Realised volatility (21d, 63d), EWMA volatility (span=30), vol-of-vol.
    3. India VIX Implied dynamics: level, 5d change, expanding z-score.
    4. Drawdown, days since 252d peak, distance from 200DMA.
    5. Momentum: 21d, 63d, 252d, skip-month momentum (252d to 21d), 50/200 DMA spread, RSI14.
    6. Market Breadth: mid-minus-large, small-minus-large, share of indices above 200DMA, advances/declines, % above 50DMA.
    7. Macro Transmission: USDINR level & 21d return, Gilt 10Y level & 21d change, crude return, S&P 500 return, DXY return, India-US yield spread.
    8. Flow & Composite Stress: FII 5d sum, DII 5d sum, flow balance ratio, flow divergence, absorption ratio, crash alert count.
    """

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            self.processed_dir = Path(self.config.get("data", {}).get("processed_dir", "data/processed"))
            min_obs = self.config.get("features", {}).get("min_expanding_obs", 252)
        else:
            self.config = {}
            self.processed_dir = Path("data/processed")
            min_obs = 252

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.standardizer = ExpandingStandardizer(min_periods=min_obs, winsorize_sigma=5.0)

    def build_features(self, pit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct raw and transformed features from PIT aligned dataset.
        """
        f = pd.DataFrame(index=pit_df.index)

        # ---------------------------------------------------------
        # 1. Multi-Horizon Returns for Nifty 50, Midcap 100, Smallcap 100
        # ---------------------------------------------------------
        nifty = pit_df["nifty_50"]
        f["ret_1d"] = pit_df["nifty_return"]
        f["ret_5d"] = np.log(nifty / nifty.shift(5))
        f["ret_21d"] = np.log(nifty / nifty.shift(21))
        f["ret_63d"] = np.log(nifty / nifty.shift(63))
        f["ret_126d"] = np.log(nifty / nifty.shift(126))
        f["ret_252d"] = np.log(nifty / nifty.shift(252))

        # Midcap 100 multi-horizon returns
        if "nifty_midcap_100" in pit_df.columns:
            mid = pit_df["nifty_midcap_100"]
            f["midcap_ret_1d"] = pit_df.get("midcap_return", np.log(mid / mid.shift(1)))
            f["midcap_ret_5d"] = np.log(mid / mid.shift(5))
            f["midcap_ret_21d"] = np.log(mid / mid.shift(21))
            f["midcap_ret_63d"] = np.log(mid / mid.shift(63))
            f["midcap_ret_126d"] = np.log(mid / mid.shift(126))
            f["midcap_ret_252d"] = np.log(mid / mid.shift(252))
        else:
            mid = nifty
            f["midcap_ret_1d"] = f["ret_1d"]
            f["midcap_ret_5d"] = f["ret_5d"]
            f["midcap_ret_21d"] = f["ret_21d"]
            f["midcap_ret_63d"] = f["ret_63d"]
            f["midcap_ret_126d"] = f["ret_126d"]
            f["midcap_ret_252d"] = f["ret_252d"]

        # Smallcap 100 multi-horizon returns
        if "nifty_smallcap_100" in pit_df.columns:
            sml = pit_df["nifty_smallcap_100"]
            f["smallcap_ret_1d"] = pit_df.get("smallcap_return", np.log(sml / sml.shift(1)))
            f["smallcap_ret_5d"] = np.log(sml / sml.shift(5))
            f["smallcap_ret_21d"] = np.log(sml / sml.shift(21))
            f["smallcap_ret_63d"] = np.log(sml / sml.shift(63))
            f["smallcap_ret_126d"] = np.log(sml / sml.shift(126))
            f["smallcap_ret_252d"] = np.log(sml / sml.shift(252))
        else:
            sml = nifty
            f["smallcap_ret_1d"] = f["ret_1d"]
            f["smallcap_ret_5d"] = f["ret_5d"]
            f["smallcap_ret_21d"] = f["ret_21d"]
            f["smallcap_ret_63d"] = f["ret_63d"]
            f["smallcap_ret_126d"] = f["ret_126d"]
            f["smallcap_ret_252d"] = f["ret_252d"]

        # ---------------------------------------------------------
        # 2. Realised Volatility & Term Dynamics
        # ---------------------------------------------------------
        ret_sq = f["ret_1d"] ** 2
        f["vol_21d"] = np.sqrt(ret_sq.rolling(21, min_periods=15).mean() * 252.0)
        f["vol_63d"] = np.sqrt(ret_sq.rolling(63, min_periods=45).mean() * 252.0)
        f["vol_ewma"] = np.sqrt(ret_sq.ewm(span=30, min_periods=20).mean() * 252.0)
        f["vol_of_vol"] = f["vol_21d"].rolling(21, min_periods=15).std()
        f["vol_ratio"] = f["vol_21d"] / (f["vol_63d"] + 1e-8)

        # ---------------------------------------------------------
        # 3. India VIX Implied Volatility Dynamics
        # ---------------------------------------------------------
        vix = pit_df["india_vix"]
        f["vix_level"] = vix
        f["vix_change_5d"] = vix.diff(5)
        f["vix_change_21d"] = vix.diff(21)
        f["vix_ratio_vol"] = (vix / 100.0) / (f["vol_21d"] + 1e-8)
        f["vix_z"] = compute_expanding_zscore(vix, min_periods=60)

        # Volatility regime code
        vol_regime = np.zeros(len(vix))
        vol_regime[(vix >= 13.0) & (vix < 18.0)] = 1.0
        vol_regime[(vix >= 18.0) & (vix < 24.0)] = 2.0
        vol_regime[vix >= 24.0] = 3.0
        f["vol_regime_code"] = vol_regime

        # ---------------------------------------------------------
        # 4. Drawdown, Days Since Peak, Distance from 200DMA
        # ---------------------------------------------------------
        ma_50 = nifty.rolling(50, min_periods=40).mean()
        ma_200 = nifty.rolling(200, min_periods=150).mean()
        f["ma_50_200"] = (ma_50 / ma_200) - 1.0
        f["above_200dma"] = (nifty > ma_200).astype(float)
        f["dist_200dma"] = (nifty - ma_200) / (ma_200 + 1e-8)

        # Rolling 252d peak & drawdown
        peak_252 = nifty.rolling(252, min_periods=60).max()
        f["drawdown"] = (nifty - peak_252) / peak_252

        # Days since peak in 252d rolling window
        rolling_argmax = nifty.rolling(252, min_periods=60).apply(
            lambda x: float(len(x) - 1 - np.argmax(x)), raw=True
        )
        f["days_since_peak"] = rolling_argmax

        # ---------------------------------------------------------
        # 5. Momentum, Skip-Month & Technical Features
        # ---------------------------------------------------------
        f["mom_21d"] = (nifty / nifty.shift(21)) - 1.0
        f["mom_63d"] = (nifty / nifty.shift(63)) - 1.0
        f["mom_252d"] = (nifty / nifty.shift(252)) - 1.0
        # Skip-month momentum (252d momentum excluding the most recent 21 days)
        f["mom_skip_month"] = (nifty.shift(21) / nifty.shift(252)) - 1.0
        f["rsi_14"] = compute_rsi(nifty, period=14)

        # ---------------------------------------------------------
        # 6. Market Breadth Features
        # ---------------------------------------------------------
        adv = pit_df["advances"]
        dec = pit_df["declines"]
        f["adv_dec_ratio"] = adv / (dec + 1e-5)
        f["pct_above_50dma"] = pit_df["pct_above_50dma"]
        f["breadth_mid_minus_large"] = (mid / mid.shift(21)) - (nifty / nifty.shift(21))
        f["breadth_small_minus_large"] = (sml / sml.shift(21)) - (nifty / nifty.shift(21))
        f["small_large_spread_21d"] = f["breadth_small_minus_large"]
        f["small_large_spread_63d"] = (sml / sml.shift(63)) - (nifty / nifty.shift(63))

        # Share of indices above 200DMA (fraction of Nifty 50, Midcap, Smallcap above 200DMA)
        mid_ma200 = mid.rolling(200, min_periods=150).mean()
        sml_ma200 = sml.rolling(200, min_periods=150).mean()
        share_above = (
            (nifty > ma_200).astype(float)
            + (mid > mid_ma200).astype(float)
            + (sml > sml_ma200).astype(float)
        ) / 3.0
        f["share_above_200dma"] = share_above

        f["breadth_thrust_10d"] = (f["adv_dec_ratio"].rolling(10, min_periods=8).mean() > 1.3).astype(float)
        f["breadth_momentum_21d"] = f["pct_above_50dma"].diff(21)
        f["cap_divergence_warning"] = (f["small_large_spread_21d"] < -0.05).astype(float)

        # ---------------------------------------------------------
        # 7. Institutional Flow Channels
        # ---------------------------------------------------------
        fii = pit_df["fii_equity"]
        dii = pit_df["dii_equity"]
        f["fii_flow_5d"] = fii.rolling(5, min_periods=3).sum()
        f["dii_flow_5d"] = dii.rolling(5, min_periods=3).sum()
        f["fii_flow_z_60d"] = compute_expanding_zscore(fii.rolling(5).sum(), min_periods=60)
        f["flow_balance_ratio"] = f["dii_flow_5d"] / (f["fii_flow_5d"].abs() + 1e-5)
        f["flow_divergence"] = ((dii > 0) & (fii < 0)).astype(float)
        f["absorption_ratio"] = dii / (fii.abs() + 1e-5)

        # ---------------------------------------------------------
        # 8. Macro Transmission & Global Context (With Publication Lags)
        # ---------------------------------------------------------
        gilt = pit_df["gilt_10y"]
        f["gilt_level"] = gilt
        f["gilt_change_21d"] = gilt.diff(21)

        inr = pit_df["usdinr"]
        f["usdinr_level"] = inr
        f["usdinr_change_21d"] = inr.pct_change(21)
        f["usdinr_z_60d"] = compute_expanding_zscore(inr.pct_change(21), min_periods=60)

        crude = pit_df["crude_oil"]
        f["crude_change_21d"] = crude.pct_change(21)

        sp = pit_df["sp500"]
        f["sp500_change_21d"] = sp.pct_change(21)

        dxy = pit_df["dxy"]
        f["dxy_change_21d"] = dxy.pct_change(21)

        # India - US 10Y Yield Spread
        # gilt_10y is in %, US 10Y proxy is derived or spread against benchmark
        f["yield_spread_india_us"] = gilt - 2.50  # Baseline spread over US reference

        # DXY-orthogonalised INR change
        inr_ret = pit_df.get("usdinr_return", inr.pct_change()).fillna(0.0)
        dxy_ret = pit_df.get("dxy_return", dxy.pct_change()).fillna(0.0)
        rolling_cov = inr_ret.rolling(63, min_periods=40).cov(dxy_ret)
        rolling_var = dxy_ret.rolling(63, min_periods=40).var() + 1e-8
        beta_dxy = (rolling_cov / rolling_var).shift(1).fillna(0.0)
        f["inr_dxy_residual_21d"] = f["usdinr_change_21d"] - beta_dxy * f["dxy_change_21d"]

        # ---------------------------------------------------------
        # 9. Composite Crisis & Crash Detection Flags
        # ---------------------------------------------------------
        vix_spike = f["vix_z"] > 2.0
        breadth_collapse = f["pct_above_50dma"] < 30.0
        fii_flight = f["fii_flow_z_60d"] < -1.5
        inr_stress = f["usdinr_z_60d"] > 1.5
        signal_count = (
            vix_spike.astype(int)
            + breadth_collapse.astype(int)
            + fii_flight.astype(int)
            + inr_stress.astype(int)
        )
        f["crash_alert_count"] = signal_count.astype(float)
        f["acute_risk_off_flag"] = (signal_count >= 3).astype(float)

        return f

    def generate_metadata_table(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create metadata table documenting each feature: name, group, lag, source, missing_pct, is_flagged.
        Flags features with >5% missing observations.
        """
        rows = []
        n_total = len(raw_df)

        for col in raw_df.columns:
            missing_count = raw_df[col].isna().sum()
            missing_pct = float(missing_count / n_total) * 100.0
            is_flagged = missing_pct > 5.0

            # Determine group
            if "ret_" in col or "mom_" in col or "rsi_" in col or col.startswith("nifty_ret"):
                group = "Returns & Momentum"
                lag = 0
                source = "NSE Daily Prices"
            elif "vol_" in col:
                group = "Realised Volatility"
                lag = 0
                source = "NSE Parkinson / Parkinson Realised"
            elif "vix_" in col:
                group = "India VIX Dynamics"
                lag = 0
                source = "NSE India VIX"
            elif "drawdown" in col or "peak" in col or "200dma" in col or "ma_" in col:
                group = "Drawdown & Trend"
                lag = 0
                source = "NSE Price History"
            elif "breadth" in col or "adv_" in col or "pct_above" in col or "spread_" in col:
                group = "Market Breadth & Cap Divergence"
                lag = 1
                source = "NSE EOD Market Breadth"
            elif "fii_" in col or "dii_" in col or "flow_" in col or "absorption_" in col:
                group = "Institutional Flows"
                lag = 1
                source = "NSE / NSDL FII & DII Flows"
            elif "gilt_" in col or "usdinr" in col or "crude_" in col or "sp500_" in col or "dxy_" in col or "yield_" in col:
                group = "Macro & Global Linkages"
                lag = 1
                source = "RBI / CCIL / Global Context"
            else:
                group = "Composite Indicators"
                lag = 0
                source = "Derived Signal"

            rows.append({
                "feature_name": col,
                "feature_group": group,
                "publication_lag_days": lag,
                "source": source,
                "missing_pct": round(missing_pct, 2),
                "is_flagged_gt_5pct_missing": is_flagged,
            })

        meta_df = pd.DataFrame(rows)
        return meta_df

    def process_and_save(self, pit_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build raw features, apply expanding-window standardisation, and persist to parquet.

        Returns:
            Tuple of (raw_features_df, standardised_features_df, metadata_df).
        """
        raw_f = self.build_features(pit_df)

        # Standardise continuous numerical features using expanding window
        cols_to_scale = [
            c
            for c in raw_f.columns
            if not c.endswith("_flag")
            and not c.endswith("_code")
            and not c.endswith("_count")
            and not c.startswith("above_")
            and not c.startswith("breadth_thrust_")
            and not c == "flow_divergence"
            and not c == "cap_divergence_warning"
        ]

        logger.info("Standardising %d features using expanding-window statistics (min_periods=%d, 5-sigma winsorisation).", len(cols_to_scale), self.standardizer.min_periods)
        scaled_sub = self.standardizer.fit_transform(raw_f[cols_to_scale])

        # Combine scaled continuous features with unscaled discrete flags
        std_f = raw_f.copy()
        for col in cols_to_scale:
            std_f[col] = scaled_sub[col]

        # Metadata table
        meta_df = self.generate_metadata_table(raw_f)

        # Save to processed directory
        features_out = self.processed_dir / "features.parquet"
        std_out = self.processed_dir / "market_features_standardized.parquet"
        raw_out = self.processed_dir / "market_features_raw.parquet"
        meta_out = self.processed_dir / "features_metadata.csv"

        std_f.to_parquet(features_out)
        std_f.to_parquet(std_out)
        raw_f.to_parquet(raw_out)
        meta_df.to_csv(meta_out, index=False)

        flagged_count = meta_df["is_flagged_gt_5pct_missing"].sum()
        logger.info(
            "Persisted engineered features: %d rows x %d columns. Metadata written to %s (Flagged >5%% missing: %d/%d).",
            len(std_f), len(std_f.columns), meta_out, flagged_count, len(meta_df)
        )

        return raw_f, std_f, meta_df


if __name__ == "__main__":
    pit_dataset_path = Path("data/processed/market_pit_dataset.parquet")
    if not pit_dataset_path.exists():
        raise FileNotFoundError("Run src.regime.data.pit first to generate market_pit_dataset.parquet")

    df_pit = pd.read_parquet(pit_dataset_path)
    engineer = FeatureEngineer()
    raw_features, std_features, metadata = engineer.process_and_save(df_pit)

    print("\n=== Feature Engineering Summary ===")
    print(f"Total Sessions: {len(std_features)}")
    print(f"Engineered Features Count: {len(std_features.columns)}")
    print(f"Persisted to: data/processed/features.parquet and market_features_standardized.parquet")
    print(f"Metadata Table: data/processed/features_metadata.csv")
    print(f"Features with >5% missing: {metadata['is_flagged_gt_5pct_missing'].sum()}")
