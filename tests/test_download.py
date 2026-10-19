"""
Unit tests for MarketDataDownloader.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.regime.data.download import MarketDataDownloader


@pytest.fixture
def downloader(tmp_path):
    # Use existing config but create a downloader instance
    d = MarketDataDownloader(config_path="configs/config.yaml")
    # Redirect raw_dir to temporary path for isolated testing
    d.raw_dir = tmp_path / "raw"
    d.raw_dir.mkdir(parents=True, exist_ok=True)
    return d


def test_standardize_df(downloader):
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    raw_df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [95, 96, 97, 98, 99],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000, 1200, 1100, 1300, 1400],
        },
        index=dates,
    )
    std = downloader._standardize_df(raw_df)
    assert list(std.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(std.index, pd.DatetimeIndex)
    assert std.index.tz is None


def test_calibrated_fallback_generation(downloader):
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = downloader._generate_calibrated_fallback("test_ticker", dates)
    assert len(df) == 10
    assert "close" in df.columns
    assert "open" in df.columns
    assert "volume" in df.columns
    assert (df["high"] >= df["low"]).all()


def test_manual_fallback_loading(downloader):
    # Place a dummy CSV in the temp raw directory
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    manual_df = pd.DataFrame({"close": [10, 20, 30]}, index=dates)
    csv_path = downloader.raw_dir / "custom_asset.csv"
    manual_df.to_csv(csv_path)

    loaded = downloader._load_manual_fallback("custom_asset")
    assert loaded is not None
    assert len(loaded) == 3
    assert "close" in loaded.columns


def test_fetch_fii_dii_flows(downloader):
    dates = pd.date_range("2024-01-01", periods=15, freq="B")
    flows = downloader.fetch_fii_dii_flows(dates, force_refresh=True)
    assert len(flows) == 15
    assert "fii_equity" in flows.columns
    assert "dii_equity" in flows.columns
    # Check that parquet cache was written
    cache_file = downloader.raw_dir / "fii_dii_flows.parquet"
    assert cache_file.exists()


def test_fetch_market_breadth(downloader):
    dates = pd.date_range("2024-01-01", periods=15, freq="B")
    breadth = downloader.fetch_market_breadth(dates, force_refresh=True)
    assert len(breadth) == 15
    assert "advances" in breadth.columns
    assert "declines" in breadth.columns
    assert "pct_above_50dma" in breadth.columns
    assert (breadth["advances"] + breadth["declines"] == 2000).all()
