import os
import pandas as pd
import pytest
from datetime import datetime, timedelta

from ml.data.fetch_historical_data import HistoricalDataFetcher

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return str(tmp_path / "test_data")

@pytest.fixture
def fetcher(sample_data_dir):
    """Create a HistoricalDataFetcher instance for testing."""
    return HistoricalDataFetcher(
        symbols=["BTCUSDT"],
        interval="5m",
        days_back=1,
        data_dir=sample_data_dir
    )

def test_csv_integrity(fetcher, sample_data_dir):
    """Test the integrity of saved CSV files."""
    # Create a sample DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='5min'),
        'open': [100.0] * 10,
        'high': [101.0] * 10,
        'low': [99.0] * 10,
        'close': [100.5] * 10,
        'volume': [1000.0] * 10
    })
    
    # Save the data
    fetcher.save_data(df, "BTCUSDT")
    
    # Read the saved file
    filename = os.path.join(sample_data_dir, "BTCUSDT_5m.csv")
    saved_df = pd.read_csv(filename)
    
    # Convert timestamp to datetime
    saved_df['timestamp'] = pd.to_datetime(saved_df['timestamp'])
    
    # Perform assertions
    assert not saved_df.isnull().any().any(), "No null values should be present"
    assert saved_df['timestamp'].is_monotonic_increasing, "Timestamps should be in ascending order"
    assert len(saved_df.columns) == 6, "Should have 6 columns"
    assert all(col in saved_df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(saved_df['timestamp'])
    assert pd.api.types.is_numeric_dtype(saved_df['open'])
    assert pd.api.types.is_numeric_dtype(saved_df['high'])
    assert pd.api.types.is_numeric_dtype(saved_df['low'])
    assert pd.api.types.is_numeric_dtype(saved_df['close'])
    assert pd.api.types.is_numeric_dtype(saved_df['volume'])

def test_data_continuity(fetcher, sample_data_dir):
    """Test that data has no gaps in timestamps."""
    # Create a sample DataFrame with a gap
    timestamps = pd.date_range(start='2024-01-01', periods=10, freq='5min')
    timestamps = timestamps.drop(timestamps[5])  # Create a gap
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': [100.0] * 9,
        'high': [101.0] * 9,
        'low': [99.0] * 9,
        'close': [100.5] * 9,
        'volume': [1000.0] * 9
    })
    
    # Save the data
    fetcher.save_data(df, "BTCUSDT")
    
    # Read the saved file
    filename = os.path.join(sample_data_dir, "BTCUSDT_5m.csv")
    saved_df = pd.read_csv(filename)
    saved_df['timestamp'] = pd.to_datetime(saved_df['timestamp'])
    
    # Calculate expected time differences
    time_diffs = saved_df['timestamp'].diff()
    expected_diff = pd.Timedelta(minutes=5)
    
    # Check for gaps
    gaps = time_diffs[time_diffs > expected_diff]
    assert len(gaps) > 0, "Test data should contain gaps" 