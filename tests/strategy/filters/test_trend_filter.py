import pytest
import numpy as np
from strategy.filters.trend_filter import TrendFilter

def test_trend_filter_initialization():
    """Test trend filter initialization with default and custom parameters"""
    # Test default parameters
    tf = TrendFilter()
    assert tf.fast_period == 8
    assert tf.slow_period == 21
    
    # Test custom parameters
    tf = TrendFilter(fast_period=5, slow_period=15)
    assert tf.fast_period == 5
    assert tf.slow_period == 15

def test_ema_calculation():
    """Test EMA calculation with known price series"""
    tf = TrendFilter()
    
    # Test insufficient data
    prices = [100.0]
    assert tf.calculate_ema(prices, period=8) == 100.0
    
    # Test flat prices (EMA should equal price)
    prices = [100.0] * 10
    ema = tf.calculate_ema(prices, period=8)
    assert abs(ema - 100.0) < 0.0001
    
    # Test rising prices (EMA should lag behind)
    prices = [100.0 + i for i in range(10)]
    ema = tf.calculate_ema(prices, period=5)
    assert ema < prices[-1]
    
    # Test falling prices (EMA should lag behind)
    prices = [100.0 - i for i in range(10)]
    ema = tf.calculate_ema(prices, period=5)
    assert ema > prices[-1]

def test_trend_detection():
    """Test trend detection with various price patterns"""
    tf = TrendFilter(fast_period=3, slow_period=5)
    
    # Test uptrend
    prices = [100.0 + i for i in range(10)]
    fast_ema, slow_ema, trend = tf.check_trend(prices)
    assert trend == "uptrend"
    assert fast_ema > slow_ema
    
    # Test downtrend
    prices = [100.0 - i for i in range(10)]
    fast_ema, slow_ema, trend = tf.check_trend(prices)
    assert trend == "downtrend"
    assert fast_ema < slow_ema
    
    # Test sideways (should depend on last movement)
    prices = [100.0] * 10
    fast_ema, slow_ema, trend = tf.check_trend(prices)
    assert trend in ["uptrend", "downtrend"]
    assert abs(fast_ema - slow_ema) < 0.0001

def test_entry_signals():
    """Test entry signal generation based on trend alignment"""
    tf = TrendFilter()
    
    # Test BUY signals
    assert tf.check_entry(110.0, 100.0, "BUY") == True    # Clear uptrend
    assert tf.check_entry(90.0, 100.0, "BUY") == False    # Clear downtrend
    assert tf.check_entry(100.0, 100.0, "BUY") == False   # No trend
    
    # Test SELL signals
    assert tf.check_entry(90.0, 100.0, "SELL") == True    # Clear downtrend
    assert tf.check_entry(110.0, 100.0, "SELL") == False  # Clear uptrend
    assert tf.check_entry(100.0, 100.0, "SELL") == False  # No trend

def test_realistic_price_series():
    """Test trend detection with realistic price movements"""
    tf = TrendFilter(fast_period=8, slow_period=21)
    
    # Simulate a trending market with noise
    base_price = 100.0
    trend_coef = 0.2  # Strong trend coefficient
    volatility = 0.1
    np.random.seed(42)  # For reproducibility
    
    # Generate uptrend
    prices = []
    for i in range(30):
        noise = np.random.normal(0, volatility)
        price = base_price + trend_coef * i + noise
        prices.append(price)
    
    fast_ema, slow_ema, trend = tf.check_trend(prices)
    assert trend == "uptrend"  # Should detect uptrend despite noise
    
    # Generate downtrend
    prices = []
    for i in range(30):
        noise = np.random.normal(0, volatility)
        price = base_price - trend_coef * i + noise
        prices.append(price)
    
    fast_ema, slow_ema, trend = tf.check_trend(prices)
    assert trend == "downtrend"  # Should detect downtrend despite noise
    
    # Test string representation
    assert str(tf) == "TrendFilter(fast_period=8, slow_period=21)" 