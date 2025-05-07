import pandas as pd
import numpy as np
import pytest
from ml.labels.label_generator import generate_labels, calculate_future_returns

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    # Create more realistic price movements
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, 0.015, 100)  # 1.5% daily volatility
    price = 100 * np.exp(np.cumsum(returns))
    
    data = {
        'timestamp': dates,
        'close': price,
        'volume': np.random.normal(1000, 100, 100)
    }
    
    # Add realistic high-low ranges based on volatility
    daily_vol = np.std(returns)
    data['high'] = data['close'] * (1 + daily_vol)
    data['low'] = data['close'] * (1 - daily_vol)
    data['open'] = data['close'] * (1 + np.random.normal(0, daily_vol/2, 100))
    
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

def test_label_thresholds():
    """Test that labels are generated based on ATR thresholds."""
    # Create a controlled dataset with stable ATR
    periods = 30  # More periods for ATR stability
    
    # Create initial price series with some volatility
    np.random.seed(42)  # For reproducibility
    base_price = 100.0
    daily_volatility = 0.01  # 1% daily volatility (reduced from 2%)
    
    # Generate prices with some volatility for ATR stability
    prices = []
    current_price = base_price
    for _ in range(periods):
        current_price *= (1 + np.random.normal(0, daily_volatility))
        prices.append(current_price)
    
    # Create DataFrame with OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': 1000.0
    })
    
    # Add high-low ranges based on volatility
    df['high'] = df['close'] * (1 + daily_volatility)
    df['low'] = df['close'] * (1 - daily_volatility)
    
    future_bars = 5
    atr_window = 14  # Standard ATR window
    atr_multiplier = 0.75
    
    # Let ATR stabilize for first 14 bars, then test scenarios
    test_start = 15
    
    # Set up test cases after ATR stabilization:
    # Test case 1: Strong up move (+2%)
    current_idx = test_start
    future_idx = current_idx + future_bars
    current_price = 100.0
    df.loc[current_idx, 'close'] = current_price
    df.loc[current_idx, 'high'] = current_price * (1 + daily_volatility)
    df.loc[current_idx, 'low'] = current_price * (1 - daily_volatility)
    df.loc[future_idx, 'close'] = current_price * 1.02  # +2% move
    
    # Test case 2: Strong down move (-2%)
    current_idx = test_start + 1
    future_idx = current_idx + future_bars
    df.loc[current_idx, 'close'] = current_price
    df.loc[current_idx, 'high'] = current_price * (1 + daily_volatility)
    df.loc[current_idx, 'low'] = current_price * (1 - daily_volatility)
    df.loc[future_idx, 'close'] = current_price * 0.98  # -2% move
    
    # Test case 3: Weak move (+0.5%)
    current_idx = test_start + 2
    future_idx = current_idx + future_bars
    df.loc[current_idx, 'close'] = current_price
    df.loc[current_idx, 'high'] = current_price * (1 + daily_volatility)
    df.loc[current_idx, 'low'] = current_price * (1 - daily_volatility)
    df.loc[future_idx, 'close'] = current_price * 1.005  # +0.5% move
    
    # Generate labels
    df_labeled, stats = generate_labels(
        df,
        future_bars=future_bars,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier
    )
    
    # Calculate ATR for debugging
    from ta.volatility import AverageTrueRange
    atr = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=atr_window,
        fillna=True
    ).average_true_range()
    
    # Print debug information
    print("\nDebug Information:")
    print(f"ATR window: {atr_window}")
    print(f"ATR multiplier: {atr_multiplier}")
    print(f"Future bars: {future_bars}")
    print(f"\nTest point {test_start}:")
    print(f"Current price: {df.loc[test_start, 'close']:.2f}")
    print(f"Future price: {df.loc[test_start + future_bars, 'close']:.2f}")
    print(f"Return: {((df.loc[test_start + future_bars, 'close'] / df.loc[test_start, 'close']) - 1) * 100:.2f}%")
    print(f"ATR: {atr.iloc[test_start]:.4f}")
    print(f"ATR%: {(atr.iloc[test_start] / df.loc[test_start, 'close']) * 100:.2f}%")
    print(f"Threshold: {(atr.iloc[test_start] / df.loc[test_start, 'close']) * atr_multiplier * 100:.2f}%")
    
    # Check labels at our test points
    assert df_labeled['label'].iloc[test_start] == 1, "Strong positive return (2%) should be BUY"
    assert df_labeled['label'].iloc[test_start + 1] == -1, "Strong negative return (-2%) should be SELL"
    assert df_labeled['label'].iloc[test_start + 2] == 0, "Weak return (0.5%) should be HOLD"
    
    # Verify label text matches
    assert df_labeled['label_text'].iloc[test_start] == 'BUY', "Label text should be BUY"
    assert df_labeled['label_text'].iloc[test_start + 1] == 'SELL', "Label text should be SELL"
    assert df_labeled['label_text'].iloc[test_start + 2] == 'HOLD', "Label text should be HOLD"

def test_label_distribution(sample_ohlcv_data):
    """Test that label distribution is reasonable."""
    df_labeled, stats = generate_labels(sample_ohlcv_data)
    
    # Check that we don't have extreme class imbalance
    # In realistic market conditions, most periods should be HOLD
    assert 0.05 <= stats['buy_ratio'] <= 0.4, "Buy ratio outside reasonable range"
    assert 0.05 <= stats['sell_ratio'] <= 0.4, "Sell ratio outside reasonable range"
    assert 0.3 <= stats['hold_ratio'] <= 0.9, "Hold ratio outside reasonable range"

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