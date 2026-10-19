"""
Unit tests for PointInTimeAligner.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.regime.data.pit import PointInTimeAligner


@pytest.fixture
def aligner(tmp_path):
    a = PointInTimeAligner(config_path="configs/config.yaml")
    a.processed_dir = tmp_path / "processed"
    a.processed_dir.mkdir(parents=True, exist_ok=True)
    return a


@pytest.fixture
def sample_raw_frames():
    # 5 NSE trading sessions (Mon-Fri)
    nse_dates = pd.date_range("2024-01-08", periods=5, freq="B")  # Mon to Fri

    nifty_df = pd.DataFrame(
        {"close": [21500.0, 21550.0, 21600.0, 21520.0, 21700.0]},
        index=nse_dates,
    )
    vix_df = pd.DataFrame(
        {"close": [14.0, 13.8, 14.2, 15.1, 14.5]},
        index=nse_dates,
    )
    # Global series missing Wednesday (holiday overseas)
    global_dates = nse_dates.drop(nse_dates[2])
    sp500_df = pd.DataFrame(
        {"close": [4700.0, 4710.0, 4725.0, 4740.0]},
        index=global_dates,
    )

    flows_df = pd.DataFrame(
        {"fii_equity": [-500.0, 200.0, -100.0, -800.0, 400.0], "dii_equity": [800.0, 600.0, 950.0, 1100.0, 500.0]},
        index=nse_dates,
    )

    breadth_df = pd.DataFrame(
        {"advances": [1200, 1100, 1300, 900, 1400], "declines": [800, 900, 700, 1100, 600], "pct_above_50dma": [60.0, 58.0, 62.0, 54.0, 65.0]},
        index=nse_dates,
    )

    return {
        "nifty_50": nifty_df,
        "india_vix": vix_df,
        "sp500": sp500_df,
        "fii_dii_flows": flows_df,
        "market_breadth": breadth_df,
    }


def test_align_to_nse_calendar(aligner, sample_raw_frames):
    nse_sessions = sample_raw_frames["nifty_50"].index
    aligned = aligner.align_to_nse_calendar(sample_raw_frames, nse_sessions)

    assert len(aligned) == 5
    assert (aligned.index == nse_sessions).all()
    # Wednesday (idx 2) was missing in sp500: check that controlled ffill filled it
    assert not np.isnan(aligned.loc[nse_sessions[2], "sp500"])
    assert aligned.loc[nse_sessions[2], "sp500"] == 4710.0


def test_zero_lookahead_publication_lags(aligner, sample_raw_frames):
    processed = aligner.align_and_process(sample_raw_frames)

    # Breadth lag is 1 session: index 0 breadth should be NaN (shifted)
    assert np.isnan(processed["pct_above_50dma"].iloc[0])
    # Global macro lag is 1 session: index 0 sp500 should be NaN (shifted)
    assert np.isnan(processed["sp500"].iloc[0])
    # Next day (index 1) gets the day 0 breadth value
    assert processed["pct_above_50dma"].iloc[1] == 60.0


def test_validate_data_quality(aligner, sample_raw_frames):
    processed = aligner.align_and_process(sample_raw_frames)
    audit = aligner.validate_data_quality(processed)

    assert audit["weekend_leakage_count"] == 0
    assert "moments" in audit
    assert "nifty_return" in audit["moments"]
    assert audit["moments"]["nifty_return"]["count"] > 0
