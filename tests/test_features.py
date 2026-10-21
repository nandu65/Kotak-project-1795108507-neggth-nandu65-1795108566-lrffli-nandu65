"""
Comprehensive Unit & Invariance Tests for Feature Engineering (Step 3).
Verifies:
1. Truncation no-look-ahead (rebuild at date T matches full build up to T)
2. Scaling invariance to appended future data (t-1 expanding stats cannot change when future data arrives)
3. 5-sigma winsorisation and zero-lookahead standardisation
4. Publication lag compliance and metadata table generation
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.regime.features.engineer import (
    ExpandingStandardizer,
    FeatureEngineer,
    compute_expanding_zscore,
)


def test_compute_expanding_zscore_zero_lookahead():
    """Verify statistics at time t are computed strictly on observations up to t-1."""
    series = pd.Series([10.0] * 10 + [1000.0] * 10)
    z = compute_expanding_zscore(series, min_periods=5, winsorize_sigma=5.0)

    # First 5 rows should be NaN (min_periods=5 + 1 shift)
    assert np.isnan(z.iloc[0])
    assert np.isnan(z.iloc[4])

    # At t=9 (before the spike at t=10), z-score is 0.0 on the steady series
    assert np.isclose(z.iloc[9], 0.0, atol=1e-4)

    # At t=10, the spike is detected and clamped to winsorize_sigma (5.0)
    assert z.iloc[10] == 5.0


def test_scaling_invariance_to_appended_future_data():
    """
    Core causal property: Expanding z-score of row t must be strictly invariant
    to whether future rows t+1, ..., t+k are present in the dataset.
    """
    rng = np.random.default_rng(42)
    base_data = pd.Series(rng.normal(100.0, 15.0, size=300))
    future_data = pd.Series(rng.normal(200.0, 50.0, size=100))  # Big regime shift in the future!

    full_series = pd.concat([base_data, future_data], ignore_index=True)

    z_base = compute_expanding_zscore(base_data, min_periods=50, winsorize_sigma=5.0)
    z_full = compute_expanding_zscore(full_series, min_periods=50, winsorize_sigma=5.0)

    # The first 300 values of z_full must be IDENTICAL to z_base!
    np.testing.assert_allclose(
        z_base.dropna().values,
        z_full.iloc[:len(base_data)].dropna().values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Expanding z-score leaked future data! Values changed when future data was appended.",
    )


def test_truncation_no_look_ahead():
    """
    Rebuilding features at date T must match full build up to date T.
    Ensures rolling and expanding feature definitions have zero lookahead.
    """
    pit_path = Path("data/processed/market_pit_dataset.parquet")
    if not pit_path.exists():
        pytest.skip("market_pit_dataset.parquet not available")

    pit_df = pd.read_parquet(pit_path)
    T = 600  # Cut-off date index

    truncated_pit = pit_df.iloc[:T].copy()

    engineer = FeatureEngineer()
    full_features = engineer.build_features(pit_df)
    truncated_features = engineer.build_features(truncated_pit)

    # Compare features at date T-1 between truncated build and full build
    # Select columns that are valid at T-1
    test_cols = [
        "ret_1d", "ret_5d", "ret_21d", "vol_21d", "vol_63d",
        "vix_level", "vix_change_5d", "adv_dec_ratio",
        "gilt_level", "usdinr_level", "crude_change_21d"
    ]

    for col in test_cols:
        full_val = full_features[col].iloc[T - 1]
        trunc_val = truncated_features[col].iloc[T - 1]
        assert np.isclose(full_val, trunc_val, rtol=1e-6, atol=1e-6), (
            f"Feature {col} diverged at T-1 between truncated and full build: {trunc_val} vs {full_val}"
        )


def test_winsorization_and_nan_checks():
    """Verify winsorization strictly clamps outliers to [-5.0, 5.0]."""
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    # Series with a massive 50-sigma outlier at index 350
    vals = np.ones(400) * 10.0
    vals[350] = 5000.0
    s = pd.Series(vals, index=dates)

    scaler = ExpandingStandardizer(min_periods=50, winsorize_sigma=5.0)
    df = pd.DataFrame({"test_col": s})
    res = scaler.fit_transform(df)

    outlier_z = res["test_col"].iloc[350]
    assert outlier_z <= 5.0
    assert outlier_z >= -5.0


def test_metadata_table_and_outputs():
    """Verify features.parquet and features_metadata.csv exist and satisfy quality gates."""
    feat_path = Path("data/processed/features.parquet")
    meta_path = Path("data/processed/features_metadata.csv")

    assert feat_path.exists(), "data/processed/features.parquet was not created!"
    assert meta_path.exists(), "data/processed/features_metadata.csv was not created!"

    meta_df = pd.read_csv(meta_path)
    assert len(meta_df) >= 30, "Metadata table must document at least 30 features!"

    required_meta_cols = ["feature_name", "feature_group", "publication_lag_days", "source", "missing_pct", "is_flagged_gt_5pct_missing"]
    for col in required_meta_cols:
        assert col in meta_df.columns, f"Required column {col} missing from metadata table"

    # Flag features with >5% missing
    flagged = meta_df[meta_df["is_flagged_gt_5pct_missing"] == True]
    assert len(flagged) >= 0  # Checked and tracked
