import pandas as pd
import numpy as np
import pytest
from ml.labels.label_generator import generate_labels, calculate_future_returns

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    data = {
        'timestamp': dates,
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100)
    }
    df = pd.DataFrame(data)
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df

def test_future_returns_calculation(sample_ohlcv_data):
    """Test future returns calculation."""
    future_bars = 5
    future_returns = calculate_future_returns(sample_ohlcv_data, future_bars)
    
    # Check that future returns are properly calculated
    for i in range(len(sample_ohlcv_data) - future_bars):
        expected_return = (sample_ohlcv_data['close'].iloc[i + future_bars] - 
                         sample_ohlcv_data['close'].iloc[i]) / sample_ohlcv_data['close'].iloc[i]
        assert abs(future_returns.iloc[i] - expected_return) < 1e-10, "Future returns calculation incorrect"
    
    # Check that last future_bars rows are NaN
    assert future_returns.iloc[-future_bars:].isna().all(), "Last rows should be NaN"

def test_label_generation(sample_ohlcv_data):
    """Test label generation with different parameters."""
    # Test with default parameters
    df_labeled, stats = generate_labels(sample_ohlcv_data)
    
    # Check that labels are present
    assert 'label' in df_labeled.columns, "Label column missing"
    assert 'label_text' in df_labeled.columns, "Label text column missing"
    
    # Check label values
    assert df_labeled['label'].isin([-1, 0, 1]).all(), "Invalid label values"
    assert df_labeled['label_text'].isin(['BUY', 'SELL', 'HOLD']).all(), "Invalid label text"
    
    # Check label statistics
    assert stats['total_samples'] > 0, "No samples in statistics"
    assert 0 <= stats['buy_ratio'] <= 1, "Invalid buy ratio"
    assert 0 <= stats['sell_ratio'] <= 1, "Invalid sell ratio"
    assert 0 <= stats['hold_ratio'] <= 1, "Invalid hold ratio"
    assert abs(stats['buy_ratio'] + stats['sell_ratio'] + stats['hold_ratio'] - 1) < 1e-10, "Ratios don't sum to 1"

def test_label_thresholds(sample_ohlcv_data):
    """Test that labels are generated based on ATR thresholds."""
    # Create a controlled dataset
    df = sample_ohlcv_data.copy()
    df['close'] = 100  # Set constant price
    df['high'] = 101
    df['low'] = 99
    
    # Calculate ATR
    atr = 1.0  # With constant high/low, ATR will be 1
    atr_pct = atr / 100  # 1%
    
    # Set future prices to create specific returns
    future_bars = 5
    df.loc[0, 'close'] = 100  # Current price
    df.loc[future_bars, 'close'] = 102  # +2% return (should be BUY)
    df.loc[future_bars + 1, 'close'] = 98  # -2% return (should be SELL)
    df.loc[future_bars + 2, 'close'] = 100.5  # +0.5% return (should be HOLD)
    
    # Generate labels
    df_labeled, _ = generate_labels(df, future_bars=future_bars, atr_multiplier=0.75)
    
    # Check labels
    assert df_labeled['label'].iloc[0] == 1, "Strong positive return should be BUY"
    assert df_labeled['label'].iloc[1] == -1, "Strong negative return should be SELL"
    assert df_labeled['label'].iloc[2] == 0, "Weak return should be HOLD"

def test_label_distribution(sample_ohlcv_data):
    """Test that label distribution is reasonable."""
    df_labeled, stats = generate_labels(sample_ohlcv_data)
    
    # Check that we don't have extreme class imbalance
    assert 0.1 <= stats['buy_ratio'] <= 0.4, "Buy ratio outside reasonable range"
    assert 0.1 <= stats['sell_ratio'] <= 0.4, "Sell ratio outside reasonable range"
    assert 0.3 <= stats['hold_ratio'] <= 0.8, "Hold ratio outside reasonable range"

def test_data_continuity(sample_ohlcv_data):
    """Test that data continuity is maintained."""
    df_labeled, _ = generate_labels(sample_ohlcv_data)
    
    # Check timestamp continuity
    time_diffs = df_labeled['timestamp'].diff()
    expected_diff = pd.Timedelta(minutes=5)
    assert all(diff == expected_diff for diff in time_diffs[1:]), "Timestamps not continuous"
    
    # Check that no data points were lost except for the last future_bars
    assert len(df_labeled) == len(sample_ohlcv_data) - 10, "Incorrect number of rows after label generation"

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test with empty DataFrame
    with pytest.raises(Exception):
        generate_labels(pd.DataFrame())
    
    # Test with missing required columns
    with pytest.raises(Exception):
        generate_labels(pd.DataFrame({'timestamp': pd.date_range(start='2024-01-01', periods=10)}))
    
    # Test with invalid future_bars
    with pytest.raises(Exception):
        generate_labels(sample_ohlcv_data, future_bars=-1)
    
    # Test with invalid atr_window
    with pytest.raises(Exception):
        generate_labels(sample_ohlcv_data, atr_window=0)
    
    # Test with invalid atr_multiplier
    with pytest.raises(Exception):
        generate_labels(sample_ohlcv_data, atr_multiplier=-1) 