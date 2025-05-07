import os
import pandas as pd
import numpy as np
import pytest
from ml.features.feature_engineering import generate_features
from ta.trend import EMAIndicator

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
        'volume_zscore_10', 'obv',
        # Add lagged features to required features
        'rsi_14_lag1', 'macd_hist_lag1', 'log_return_1_lag1', 'obv_lag1',
        # Regime flags
        'is_trending', 'high_volatility', 'low_volatility',
        'is_ranging', 'price_above_bb_mid', 'rsi_regime'
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
        'macd_line', 'macd_signal',
        'bb_width', 'bb_zscore',
        'momentum_10', 'rate_of_change_5',
        'volume_zscore_10'
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

def test_lagged_features(sample_ohlcv_data):
    """Test that lagged features are generated correctly."""
    df = generate_features(sample_ohlcv_data)
    
    # Check presence of lagged features
    lagged_features = ["rsi_14_lag1", "macd_hist_lag1", "log_return_1_lag1", "obv_lag1"]
    for feature in lagged_features:
        assert feature in df.columns, f"Missing lagged feature: {feature}"
        assert df[feature].isnull().sum() == 0, f"{feature} contains NaNs"
    
    # Check that lagged features are properly shifted
    for i in range(1, len(df)):
        # For non-normalized features, check exact matches
        assert abs(df['rsi_14_lag1'].iloc[i] - df['rsi_14'].iloc[i-1]) < 1e-10, "RSI lag not properly shifted"
        assert abs(df['obv_lag1'].iloc[i] - df['obv'].iloc[i-1]) < 1e-10, "OBV lag not properly shifted"
        assert abs(df['macd_hist_lag1'].iloc[i] - df['macd_hist'].iloc[i-1]) < 1e-10, "MACD hist lag not properly shifted"
        assert abs(df['log_return_1_lag1'].iloc[i] - df['log_return_1'].iloc[i-1]) < 1e-10, "Log return lag not properly shifted"

def test_regime_flags():
    """Test regime flag generation"""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Generate features
    df_features = generate_features(df)
    
    # Test binary flags
    binary_flags = ['is_trending', 'high_volatility', 'low_volatility', 'is_ranging', 'price_above_bb_mid']
    for flag in binary_flags:
        assert df_features[flag].isin([0, 1]).all(), f"Flag {flag} contains values other than 0 or 1"
    
    # Test RSI regime
    assert df_features['rsi_regime'].isin([0, 1, 2]).all(), "RSI regime contains values other than 0, 1, or 2"
    
    # Test trend flag logic
    # Calculate EMAs and handle NaN values
    ema_8 = EMAIndicator(df["close"], window=8).ema_indicator()
    ema_21 = EMAIndicator(df["close"], window=21).ema_indicator()
    ema_8 = ema_8.fillna(method='ffill').fillna(method='bfill')
    ema_21 = ema_21.fillna(method='ffill').fillna(method='bfill')
    
    for i in range(len(df_features)):
        if df_features['is_trending'].iloc[i] == 1:
            assert ema_8.iloc[i] > ema_21.iloc[i], "Trend flag logic incorrect"
        else:
            assert ema_8.iloc[i] <= ema_21.iloc[i], "Trend flag logic incorrect"
    
    # Test RSI regime boundaries
    for i in range(len(df_features)):
        rsi = df_features['rsi_14'].iloc[i]
        regime = df_features['rsi_regime'].iloc[i]
        if rsi <= 30:
            assert regime == 0, "RSI regime 0 boundary incorrect"
        elif rsi >= 70:
            assert regime == 2, "RSI regime 2 boundary incorrect"
        else:
            assert regime == 1, "RSI regime 1 boundary incorrect" 