"""
Unit tests for FeatureEngineer and ExpandingStandardizer.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.regime.features.engineer import ExpandingStandardizer, FeatureEngineer, compute_expanding_zscore


def test_compute_expanding_zscore_zero_lookahead():
    # If x has an extreme jump at t=10, the z-score at t=9 must NOT reflect the jump at t=10
    n = 20
    series = pd.Series([10.0] * 10 + [1000.0] * 10)
    z = compute_expanding_zscore(series, min_periods=5)

    # Values before min_periods should be NaN
    assert np.isnan(z.iloc[0])
    assert np.isnan(z.iloc[4])

    # At t=9 (before the spike at t=10), z-score is calculated on the steady series
    # At t=10, x_10 is 1000.0, but mu_9 and sigma_9 are from [10, ..., 10]
    assert np.isclose(z.iloc[9], 0.0, atol=1e-4)
    # At t=10, the spike is detected
    assert z.iloc[10] > 10.0


def test_expanding_standardizer(tmp_path):
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "feat_a": np.random.normal(5.0, 2.0, 100),
            "feat_b": np.random.exponential(1.0, 100),
        },
        index=dates,
    )
    scaler = ExpandingStandardizer(min_periods=20)
    res = scaler.fit_transform(df)

    assert res.shape == df.shape
    # First 20 rows are NaN because of min_periods and shift(1)
    assert res.iloc[:20].isna().all().all()
    # Row 21 onwards is non-NaN
    assert res.iloc[21:].notna().all().all()


def test_feature_engineer_build_features(tmp_path):
    # Create synthetic pit dataset for testing
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    nifty = 18000.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, len(dates))))

    pit_df = pd.DataFrame(
        {
            "nifty_50": nifty,
            "nifty_return": np.concatenate([[np.nan], np.diff(np.log(nifty))]),
            "nifty_midcap_100": nifty * 0.8,
            "midcap_return": np.concatenate([[np.nan], np.diff(np.log(nifty * 0.8))]),
            "nifty_smallcap_100": nifty * 0.6,
            "smallcap_return": np.concatenate([[np.nan], np.diff(np.log(nifty * 0.6))]),
            "india_vix": np.random.uniform(12.0, 25.0, len(dates)),
            "advances": np.random.randint(500, 1500, len(dates)),
            "declines": np.random.randint(500, 1500, len(dates)),
            "pct_above_50dma": np.random.uniform(20.0, 80.0, len(dates)),
            "fii_equity": np.random.normal(-100.0, 1000.0, len(dates)),
            "dii_equity": np.random.normal(800.0, 600.0, len(dates)),
            "gilt_10y": np.random.uniform(6.8, 7.4, len(dates)),
            "usdinr": np.random.uniform(80.0, 84.0, len(dates)),
            "usdinr_return": np.random.normal(0.0, 0.003, len(dates)),
            "crude_oil": np.random.uniform(70.0, 95.0, len(dates)),
            "sp500": np.random.uniform(4000.0, 5000.0, len(dates)),
            "dxy": np.random.uniform(100.0, 106.0, len(dates)),
            "dxy_return": np.random.normal(0.0, 0.004, len(dates)),
        },
        index=dates,
    )

    engineer = FeatureEngineer(config_path="configs/config.yaml")
    features = engineer.build_features(pit_df)

    assert len(features) == len(pit_df)
    # Check that we built 30+ features
    assert len(features.columns) >= 30

    # Check key feature columns exist
    expected_cols = [
        "ret_1d",
        "ret_5d",
        "ret_21d",
        "vol_21d",
        "vol_63d",
        "vix_level",
        "vix_z",
        "vol_regime_code",
        "adv_dec_ratio",
        "pct_above_50dma",
        "fii_flow_5d",
        "dii_flow_5d",
        "flow_balance_ratio",
        "gilt_level",
        "small_large_spread_21d",
        "inr_dxy_residual_21d",
        "crash_alert_count",
    ]
    for col in expected_cols:
        assert col in features.columns, f"Expected feature {col} missing from features"
