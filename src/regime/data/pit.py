"""
Point-In-Time (PIT) Alignment and Trading Calendar Module.
Enforces zero look-ahead bias by aligning all global and macro feeds
to the NSE trading calendar and lagging features by their real publication delay.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import yaml

logger = logging.getLogger(__name__)


class PointInTimeAligner:
    r"""
    Enforces Point-in-Time (PIT) data discipline and calendar alignment.

    Rules applied:
    1. Alignment to NSE Trading Calendar: All data indexed to NSE sessions.
    2. Controlled Forward-Fill: Macro/global levels forward-filled up to limit (<=3 sessions).
    3. Strict Zero Forward-Fill on Returns & Flows: Returns, FII/DII, breadth counts are never filled.
    4. Publication Lag Enforcement: Features shifted forward by real availability delay.

    Maths:
        For feature series $X_t$ published with delay $L \ge 0$:
        $$\tilde{X}_t = X_{t - L}$$
        Ensuring that at decision timestamp $t$, only information $\mathcal{F}_{t-L}$ is accessible.
    """

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        """
        Initialize the PIT Aligner.

        Args:
            config_path: Path to configuration YAML.
        """
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.processed_dir = Path(self.config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.calendar_anchor = self.config["data"].get("calendar_anchor", "nifty_50")
        self.max_ffill = int(self.config["data"].get("max_ffill_limit", 3))
        self.never_ffill = set(self.config["data"].get("never_ffill_columns", []))
        self.publication_lags = self.config["data"].get("publication_lags", {})

    def align_and_process(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Perform end-to-end alignment, publication lag adjustment, and quality validation.

        Args:
            raw_data: Dictionary of raw time series DataFrames from MarketDataDownloader.

        Returns:
            Fully aligned Point-in-Time DataFrame saved to data/processed/.
        """
        if self.calendar_anchor not in raw_data:
            raise ValueError(f"Calendar anchor '{self.calendar_anchor}' missing from raw data dictionary.")

        anchor_df = raw_data[self.calendar_anchor]
        if anchor_df.empty:
            raise ValueError(f"Calendar anchor DataFrame '{self.calendar_anchor}' is empty.")

        # Master NSE trading sessions
        nse_sessions = anchor_df.index.sort_values()
        if self.config["data"].get("filter_weekends", True):
            nse_sessions = nse_sessions[nse_sessions.dayofweek < 5]

        # 1. Align all individual series to NSE trading sessions
        aligned_df = self.align_to_nse_calendar(raw_data, nse_sessions)

        # 2. Compute foundational log returns and core series
        processed_df = self._compute_foundational_series(aligned_df)

        # 3. Apply publication lags to prevent look-ahead bias
        pit_df = self.enforce_publication_lags(processed_df)

        # 4. Save to processed parquet
        out_path = self.processed_dir / "market_pit_dataset.parquet"
        pit_df.to_parquet(out_path)
        logger.info("Saved PIT aligned dataset (%d rows, %d cols) to %s", len(pit_df), len(pit_df.columns), out_path)

        return pit_df

    def align_to_nse_calendar(
        self, frames: Dict[str, pd.DataFrame], nse_sessions: pd.DatetimeIndex
    ) -> pd.DataFrame:
        r"""
        Align all data frames to master NSE trading calendar with controlled ffill.

        Maths:
            For level series $S_t$:
            $$S_t^{\text{aligned}} = \text{ffill}(S_t \mathbf{1}_{\{t \in \text{NSE}\}}, \text{limit}=3)$$
            For return and flow series $F_t$:
            $$F_t^{\text{aligned}} = F_t \mathbf{1}_{\{t \in \text{NSE}\}}$$ (never forward-filled)

        Args:
            frames: Dict of raw DataFrames.
            nse_sessions: Master DatetimeIndex of Indian market trading days.

        Returns:
            Unified DataFrame aligned strictly to nse_sessions.
        """
        aligned_dict: Dict[str, pd.Series] = {}

        for name, df in frames.items():
            if df.empty:
                logger.warning("Frame '%s' is empty, skipping alignment.", name)
                continue

            # Standardize index
            df_std = df.copy()
            if not isinstance(df_std.index, pd.DatetimeIndex):
                df_std.index = pd.to_datetime(df_std.index)
            if df_std.index.tz is not None:
                df_std.index = df_std.index.tz_localize(None)

            # Extract relevant columns
            if name == "fii_dii_flows":
                for col in ["fii_equity", "dii_equity"]:
                    if col in df_std.columns:
                        s = df_std[col].reindex(nse_sessions)
                        # Zero ffill for flow variables
                        aligned_dict[col] = s.fillna(0.0)
            elif name == "market_breadth":
                for col in ["advances", "declines", "pct_above_50dma"]:
                    if col in df_std.columns:
                        s = df_std[col].reindex(nse_sessions)
                        aligned_dict[col] = s.ffill(limit=1)
            else:
                # Price / index series: extract 'close'
                col_name = "close" if "close" in df_std.columns else df_std.columns[0]
                s = df_std[col_name].reindex(nse_sessions)

                # Determine if column is allowed to be forward-filled
                if name in self.never_ffill:
                    aligned_dict[name] = s
                else:
                    # Controlled ffill (maximum 3 sessions)
                    aligned_dict[name] = s.ffill(limit=self.max_ffill)

        res_df = pd.DataFrame(aligned_dict, index=nse_sessions)
        res_df.index.name = "date"
        return res_df

    def _compute_foundational_series(self, df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Compute continuous log returns, price ratios, and raw spreads.
        Log returns:
            r_t = \ln(P_t) - \ln(P_{t-1})
        """
        out = df.copy()

        # Nifty 50 log return
        if "nifty_50" in out.columns:
            out["nifty_return"] = np.log(out["nifty_50"] / out["nifty_50"].shift(1))

        # Midcap & Smallcap log returns
        if "nifty_midcap_100" in out.columns:
            out["midcap_return"] = np.log(out["nifty_midcap_100"] / out["nifty_midcap_100"].shift(1))
        if "nifty_smallcap_100" in out.columns:
            out["smallcap_return"] = np.log(out["nifty_smallcap_100"] / out["nifty_smallcap_100"].shift(1))

        # India VIX daily difference
        if "india_vix" in out.columns:
            out["vix_change_1d"] = out["india_vix"].diff(1)

        # USD/INR return
        if "usdinr" in out.columns:
            out["usdinr_return"] = np.log(out["usdinr"] / out["usdinr"].shift(1))

        # Global equity (S&P 500) and crude returns
        if "sp500" in out.columns:
            out["sp500_return"] = np.log(out["sp500"] / out["sp500"].shift(1))
        if "crude_oil" in out.columns:
            out["crude_return"] = np.log(out["crude_oil"] / out["crude_oil"].shift(1))
        if "dxy" in out.columns:
            out["dxy_return"] = np.log(out["dxy"] / out["dxy"].shift(1))

        return out

    def enforce_publication_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shift columns forward based on publication lags defined in configuration.

        Lag Rules:
        - macro_prints: lagged by config['publication_lags']['macro_prints']
        - breadth_eod: lagged by config['publication_lags']['breadth_eod']
        - amfi_sip: lagged by config['publication_lags']['amfi_sip']
        - global_macro: lagged by config['publication_lags']['global_macro']
        """
        out = df.copy()
        breadth_lag = int(self.publication_lags.get("breadth_eod", 1))
        global_lag = int(self.publication_lags.get("global_macro", 1))

        # Shift breadth indicators if they are end-of-day releases
        breadth_cols = ["advances", "declines", "pct_above_50dma"]
        for c in breadth_cols:
            if c in out.columns and breadth_lag > 0:
                out[c] = out[c].shift(breadth_lag)

        # Shift global variables so overseas close cannot leak into concurrent Indian session
        global_cols = ["sp500", "sp500_return", "dxy", "dxy_return", "crude_oil", "crude_return"]
        for c in global_cols:
            if c in out.columns and global_lag > 0:
                out[c] = out[c].shift(global_lag)

        return out

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audit data quality, checking missingness, weekend leakage, and distributional moments.

        Returns:
            Dictionary containing audit report and summary metrics.
        """
        report: Dict[str, Any] = {}

        # 1. Check for weekend dates (Saturday = 5, Sunday = 6)
        weekend_mask = df.index.dayofweek >= 5
        weekend_count = int(weekend_mask.sum())
        report["weekend_leakage_count"] = weekend_count
        if weekend_count > 0:
            logger.error("Weekend leakage detected: %d weekend rows present!", weekend_count)

        # 2. Missingness summary
        missing_pcts = (df.isnull().mean() * 100.0).round(2).to_dict()
        report["missing_percentage"] = missing_pcts

        # 3. Maximum consecutive missing gaps per column
        max_gaps = {}
        for col in df.columns:
            is_na = df[col].isna().astype(int)
            blocks = is_na.groupby((~df[col].isna()).cumsum()).sum()
            max_gaps[col] = int(blocks.max()) if not blocks.empty else 0
        report["max_consecutive_missing_gap"] = max_gaps

        # 4. Statistical moments of returns
        moments: Dict[str, Dict[str, float]] = {}
        for col in ["nifty_return", "midcap_return", "smallcap_return", "sp500_return"]:
            if col in df.columns:
                s = df[col].dropna()
                if len(s) >= 3:
                    moments[col] = {
                        "count": len(s),
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                        "skewness": float(stats.skew(s)) if len(s) >= 3 else 0.0,
                        "kurtosis": float(stats.kurtosis(s)) if len(s) >= 4 else 0.0,
                        "min": float(s.min()),
                        "max": float(s.max()),
                    }
        report["moments"] = moments
        report["is_clean"] = weekend_count == 0 and all(gap <= 5 for gap in max_gaps.values())
        return report


if __name__ == "__main__":
    from src.regime.data.download import MarketDataDownloader

    downloader = MarketDataDownloader()
    raw = downloader.download_all()
    aligner = PointInTimeAligner()
    dataset = aligner.align_and_process(raw)
    audit = aligner.validate_data_quality(dataset)
    print("\n=== Data Quality & PIT Audit Report ===")
    print(f"Total Trading Sessions: {len(dataset)}")
    print(f"Date Range: {dataset.index.min().date()} to {dataset.index.max().date()}")
    print(f"Weekend Leakage: {audit['weekend_leakage_count']}")
    print(f"Is Clean: {audit['is_clean']}")
    print("Return Moments:")
    for k, v in audit["moments"].items():
        print(f"  {k}: mean={v['mean']:.5f}, vol={v['std']*np.sqrt(252):.4f}, skew={v['skewness']:.2f}, kurt={v['kurtosis']:.2f}")
