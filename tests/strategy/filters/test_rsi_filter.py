import pytest
import numpy as np
from strategy.filters.rsi_filter import RSIFilter

def test_rsi_initialization():
    """Test RSI filter initialization with default and custom parameters"""
    # Test default parameters
    rsi = RSIFilter()
    assert rsi.period == 14
    assert rsi.oversold == 30
    assert rsi.overbought == 70
    
    # Test custom parameters
    rsi = RSIFilter(period=10, oversold=25, overbought=75)
    assert rsi.period == 10
    assert rsi.oversold == 25
    assert rsi.overbought == 75

def test_rsi_calculation():
    """Test RSI calculation with known price series"""
    rsi = RSIFilter(period=14)
    
    # Test insufficient data
    prices = [100.0, 101.0]
    assert rsi.calculate_rsi(prices) == 50.0
    
    # Test all rising prices (RSI should be high)
    prices = [100.0 + i for i in range(20)]
    assert rsi.calculate_rsi(prices) == 100.0
    
    # Test all falling prices (RSI should be low)
    prices = [100.0 - i for i in range(20)]
    rsi_value = rsi.calculate_rsi(prices)
    assert rsi_value < 10.0
    
    # Test alternating prices
    prices = [100.0 + (i % 2) for i in range(20)]
    rsi_value = rsi.calculate_rsi(prices)
    assert 45.0 <= rsi_value <= 55.0

def test_entry_signals():
    """Test entry signal generation based on RSI values"""
    rsi = RSIFilter(oversold=30, overbought=70)
    
    # Test BUY signals
    assert rsi.check_entry(25.0, "BUY") == True    # Oversold
    assert rsi.check_entry(35.0, "BUY") == False   # Not oversold
    assert rsi.check_entry(50.0, "BUY") == False   # Neutral
    
    # Test SELL signals
    assert rsi.check_entry(75.0, "SELL") == True   # Overbought
    assert rsi.check_entry(65.0, "SELL") == False  # Not overbought
    assert rsi.check_entry(50.0, "SELL") == False  # Neutral

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    rsi = RSIFilter()
    
    # Test exactly at thresholds
    assert rsi.check_entry(30.0, "BUY") == False   # At oversold threshold
    assert rsi.check_entry(70.0, "SELL") == False  # At overbought threshold
    
    # Test extreme RSI values
    assert rsi.check_entry(0.0, "BUY") == True     # Extremely oversold
    assert rsi.check_entry(100.0, "SELL") == True  # Extremely overbought
    
    # Test invalid signals
    assert rsi.check_entry(50.0, "HOLD") == False  # Invalid signal type
    assert rsi.check_entry(50.0, "") == False      # Empty signal

def test_realistic_price_series():
    """Test RSI calculation with realistic price movements"""
    rsi = RSIFilter(period=14)
    
    # Simulate a typical price series with trend and volatility
    base_price = 100.0
    trend = 0.1
    volatility = 0.5
    np.random.seed(42)  # For reproducibility
    
    prices = []
    for i in range(30):
        noise = np.random.normal(0, volatility)
        price = base_price + trend * i + noise
        prices.append(price)
    
    rsi_value = rsi.calculate_rsi(prices)
    assert 0.0 <= rsi_value <= 100.0  # RSI should be within valid range
    
    # Test string representation
    assert str(rsi) == "RSIFilter(period=14, oversold=30, overbought=70)" 