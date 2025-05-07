import os
import pandas as pd
import numpy as np
import pytest
from ml.features.feature_engineering import generate_features

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

def test_feature_generation(sample_ohlcv_data):
    """Test that features are generated correctly."""
    df = generate_features(sample_ohlcv_data)
    
    # Check that all required features are present
    required_features = [
        'ema_8', 'ema_21', 'ema_ratio',
        'rsi_14',
        'macd_line', 'macd_signal', 'macd_hist',
        'bb_b', 'bb_width', 'bb_zscore',
        'atr_14', 'atr_pct', 'high_low_pct',
        'momentum_10', 'rate_of_change_5',
        'log_return_1', 'log_return_5',
        'volume_zscore_10', 'obv'
    ]
    
    for feature in required_features:
        assert feature in df.columns, f"Missing feature: {feature}"
    
    # Check for no NaN values
    assert not df.isnull().values.any(), "DataFrame contains NaN values"
    
    # Check for no infinite values
    assert not np.isinf(df.select_dtypes(include=np.number)).values.any(), "DataFrame contains infinite values"
    
    # Check that all features are numeric
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    assert all(pd.api.types.is_numeric_dtype(df[col]) for col in feature_columns), "Non-numeric features found"
    
    # Check z-score normalized features
    zscore_features = [
        'ema_8', 'ema_21', 'ema_ratio',
        'macd_line', 'macd_signal', 'macd_hist',
        'bb_width', 'bb_zscore',
        'momentum_10', 'rate_of_change_5',
        'log_return_1', 'log_return_5',
        'volume_zscore_10', 'obv'
    ]
    for col in zscore_features:
        assert abs(df[col].mean()) < 0.1, f"Feature {col} mean is too far from 0"
        assert abs(df[col].std() - 1) < 0.1, f"Feature {col} std is too far from 1"
    
    # Check min-max scaled features
    minmax_features = ['bb_b', 'high_low_pct', 'atr_14', 'atr_pct']
    for col in minmax_features:
        assert df[col].min() >= -0.1 and df[col].max() <= 1.1, f"Feature {col} is not properly min-max scaled"

def test_feature_ranges(sample_ohlcv_data):
    """Test that features are within expected ranges."""
    df = generate_features(sample_ohlcv_data)
    
    # RSI should be between 0 and 100
    assert df['rsi_14'].min() >= 0 and df['rsi_14'].max() <= 100, "RSI out of range"
    
    # Bollinger %B should be between 0 and 1
    assert df['bb_b'].min() >= -0.1 and df['bb_b'].max() <= 1.1, "Bollinger %B out of range"
    
    # ATR percentage should be normalized between 0 and 1
    assert df['atr_pct'].min() >= -0.1 and df['atr_pct'].max() <= 1.1, "ATR percentage out of range"
    
    # High-low percentage should be normalized between 0 and 1
    assert df['high_low_pct'].min() >= -0.1 and df['high_low_pct'].max() <= 1.1, "High-low percentage out of range"

def test_data_continuity(sample_ohlcv_data):
    """Test that data continuity is maintained."""
    df = generate_features(sample_ohlcv_data)
    
    # Check timestamp continuity
    time_diffs = df['timestamp'].diff()
    expected_diff = pd.Timedelta(minutes=5)
    assert all(diff == expected_diff for diff in time_diffs[1:]), "Timestamps not continuous"
    
    # Check that no data points were lost
    assert len(df) > 0, "No data points in processed DataFrame"
    assert len(df) <= len(sample_ohlcv_data), "Processed DataFrame has more rows than input" 