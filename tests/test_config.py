"""
Unit tests for configuration integrity and validation.
"""
from pathlib import Path
import yaml
import pytest


@pytest.fixture
def config():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    assert config_path.exists(), f"Configuration file not found at {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_config_seed_and_project(config):
    assert "project" in config
    assert config["project"]["random_seed"] == 42
    assert config["project"]["confidential"] is True


def test_config_tickers(config):
    assert "data" in config
    tickers = config["data"]["tickers"]
    assert "nifty_50" in tickers
    assert "nifty_midcap_100" in tickers
    assert "nifty_smallcap_100" in tickers
    assert "india_vix" in tickers
    assert "usdinr" in tickers
    assert "crude_oil" in tickers
    assert "sp500" in tickers
    assert "dxy" in tickers


def test_config_publication_lags(config):
    lags = config["data"]["publication_lags"]
    assert "macro_prints" in lags
    assert "amfi_sip" in lags
    assert "breadth_eod" in lags
    assert lags["macro_prints"] >= 1
    assert lags["amfi_sip"] >= 5


def test_config_regimes(config):
    regimes = config["regimes"]
    assert regimes["count"] == 5
    taxonomy = list(regimes["taxonomy"].values())
    expected = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
    assert set(taxonomy) == set(expected)
