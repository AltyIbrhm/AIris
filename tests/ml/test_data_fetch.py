import os
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ml.data.fetch_historical_data import HistoricalDataFetcher

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return str(tmp_path / "test_data")

@pytest.fixture
def mock_klines_data():
    """Create mock klines data."""
    return [
        [1625097600000, "35000", "35100", "34900", "35050", "100", 1625097899999, "3500000", 100, "50", "1750000", "0"],
        [1625097900000, "35050", "35200", "35000", "35150", "150", 1625098199999, "5250000", 150, "75", "2625000", "0"]
    ]

@pytest.fixture
def fetcher(sample_data_dir, mock_klines_data):
    """Create a HistoricalDataFetcher instance for testing."""
    with patch.dict(os.environ, {
        'BINANCE_API_KEY': 'test_key',
        'BINANCE_API_SECRET': 'test_secret'
    }):
        with patch('binance.client.Client') as mock_client:
            # Configure the mock client
            mock_instance = mock_client.return_value
            mock_instance.get_historical_klines.return_value = mock_klines_data
            
            fetcher = HistoricalDataFetcher(
                symbols=["BTCUSDT"],
                interval="5m",
                days_back=1,
                data_dir=sample_data_dir
            )
            fetcher.client = mock_instance  # Replace the client with our mock
            return fetcher

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

def test_fetch_klines(fetcher, mock_klines_data):
    """Test the fetch_klines method."""
    df = fetcher.fetch_klines("BTCUSDT")
    
    # Verify the DataFrame structure
    assert len(df) == 2  # We mocked 2 klines
    assert all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
    assert all(pd.api.types.is_numeric_dtype(df[col]) for col in ['open', 'high', 'low', 'close', 'volume'])

def test_fetch_all(fetcher, mock_klines_data, sample_data_dir):
    """Test the fetch_all method."""
    fetcher.fetch_all()
    
    # Verify that files were created
    filename = os.path.join(sample_data_dir, "BTCUSDT_5m.csv")
    assert os.path.exists(filename)
    
    # Verify the content
    df = pd.read_csv(filename)
    assert len(df) == 2  # We mocked 2 klines
    assert all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def test_missing_api_credentials():
    """Test that appropriate error is raised when API credentials are missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="BINANCE_API_KEY and BINANCE_API_SECRET must be set"):
            HistoricalDataFetcher(symbols=["BTCUSDT"])

def test_invalid_symbol(fetcher):
    """Test handling of invalid symbol."""
    # Configure mock to raise an exception for invalid symbol
    fetcher.client.get_historical_klines.side_effect = Exception("Invalid symbol")
    
    with pytest.raises(Exception, match="Invalid symbol"):
        fetcher.fetch_klines("INVALID") 