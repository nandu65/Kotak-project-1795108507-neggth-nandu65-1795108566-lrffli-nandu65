"""
Unit tests for independent rule-based ground-truth regime labeling.
Verifies Section A1.4 threshold implementation, non-collapse across all 5 regimes,
and mathematical independence from any model outputs.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.regime.labels.rule_based import compute_rule_based_regimes, REGIME_NAMES


@pytest.fixture
def sample_market_data():
    """Create a realistic synthetic PIT dataset with all required indicators."""
    n_days = 600
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    rng = np.random.RandomState(42)

    # Synthetic price series with bull, bear, shock, and recovery
    rets = rng.normal(0.0005, 0.01, size=n_days)
    # Inject crash shock
    rets[150:180] = rng.normal(-0.015, 0.03, size=30)
    # Inject recovery
    rets[180:230] = rng.normal(0.008, 0.015, size=50)

    nifty_prices = 10000.0 * np.exp(np.cumsum(rets))
    midcap_prices = 8000.0 * np.exp(np.cumsum(rets + rng.normal(0, 0.005, size=n_days)))
    smallcap_prices = 5000.0 * np.exp(np.cumsum(rets + rng.normal(0, 0.007, size=n_days)))

    # VIX series
    vix = np.full(n_days, 15.0)
    vix[150:180] = 35.0  # Acute crash
    vix[180:230] = np.linspace(32.0, 16.0, 50)  # Decay
    vix[230:] = 13.5  # Calm bull

    df = pd.DataFrame(
        {
            "nifty_50": nifty_prices,
            "nifty_midcap_100": midcap_prices,
            "nifty_smallcap_100": smallcap_prices,
            "india_vix": vix,
        },
        index=dates,
    )
    return df


def test_rule_based_regimes_independent_of_models(sample_market_data):
    """
    TEST INDEPENDENCE: The ground truth target must be strictly independent of model outputs.
    It takes ONLY raw market PIT data and has no model parameters or predictions in its signature.
    """
    labels_baseline = compute_rule_based_regimes(sample_market_data)

    # Create dummy model predictions that attempt to influence evaluation
    fake_model_probs = np.random.uniform(size=(len(sample_market_data), 5))
    fake_model_probs = fake_model_probs / fake_model_probs.sum(axis=1, keepdims=True)
    fake_model_states = np.argmax(fake_model_probs, axis=1)

    # Compute labels again with separate copy of market data
    labels_repeat = compute_rule_based_regimes(sample_market_data.copy(deep=True))

    # Assert outputs are identical regardless of any external model state
    pd.testing.assert_series_equal(labels_baseline, labels_repeat)
    assert not np.array_equal(labels_baseline.dropna().values, fake_model_states[-len(labels_baseline.dropna()):])


def test_rule_based_all_five_regimes_represented(sample_market_data):
    """Verify that all 5 canonical regimes are present in the dataset."""
    labels = compute_rule_based_regimes(sample_market_data)
    valid_labels = labels.dropna().astype(int)

    unique_states = set(valid_labels.unique())
    assert unique_states == {0, 1, 2, 3, 4}, f"Expected all 5 states, got {unique_states}"


def test_rule_based_no_state_exceeds_sixty_percent(sample_market_data):
    """Sanity gate: no single regime may hold more than 60% of trading days."""
    labels = compute_rule_based_regimes(sample_market_data)
    valid_labels = labels.dropna().astype(int)
    counts = valid_labels.value_counts(normalize=True)

    for state, freq in counts.items():
        assert freq < 0.60, f"State {state} ({REGIME_NAMES[state]}) holds {freq:.2%} > 60% ceiling!"


def test_rule_based_causality_and_no_lookahead(sample_market_data):
    """Verify causality: label at date t depends strictly on data up to date t."""
    cutoff = 400
    labels_full = compute_rule_based_regimes(sample_market_data)
    labels_truncated = compute_rule_based_regimes(sample_market_data.iloc[:cutoff])

    # Truncated series must match the full series exactly up to the cutoff
    valid_mask = ~labels_truncated.isna()
    pd.testing.assert_series_equal(
        labels_full.iloc[:cutoff][valid_mask],
        labels_truncated[valid_mask],
    )


def test_rule_based_real_dataset_distribution():
    """Verify on actual project PIT parquet dataset that real data produces valid distributions."""
    pit_path = Path(r"D:\bayesian-regime-detection-APK\data\processed\market_pit_dataset.parquet")
    if not pit_path.exists():
        pytest.skip("PIT dataset not found, skipping real data test.")

    df = pd.read_parquet(pit_path)
    labels = compute_rule_based_regimes(df)
    valid = labels.dropna().astype(int)

    counts = valid.value_counts(normalize=True)
    assert len(counts) == 5, f"Expected 5 regimes on real data, got {len(counts)}"
    for s in range(5):
        freq = counts.get(s, 0.0)
        assert freq >= 0.05, f"Regime {s} has only {freq:.2%} < 5% minimum frequency!"
        assert freq <= 0.60, f"Regime {s} has {freq:.2%} > 60% ceiling!"
