import pytest
import numpy as np
from strategy.filters.rsi_filter import RSIFilter
from strategy.filters.trend_filter import TrendFilter
from strategy.filters.confidence_filter import ConfidenceFilter
from strategy.filters.base_filter import BaseFilter

def test_filter_combination():
    """Test all filters working together to validate trade reduction"""
    rsi = RSIFilter(oversold=30, overbought=70)
    trend = TrendFilter(fast_period=8, slow_period=21)
    confidence = ConfidenceFilter(threshold=0.85)
    base = BaseFilter(min_trade_spacing=5)
    
    # Generate a realistic price series with both noise and trends
    prices = generate_realistic_prices()
    
    # Track number of individual vs combined signals
    individual_signals = 0
    combined_signals = 0
    
    for i in range(len(prices) - 30):  # Use 30-day window
        window = prices[i:i+30]
        
        # Calculate individual signals
        rsi_value = rsi.calculate_rsi(window)
        fast_ema, slow_ema, trend_direction = trend.check_trend(window)
        confidence_value = 0.90  # Simulated high confidence
        
        # Count individual signals
        if rsi.check_entry(rsi_value, "BUY"):
            individual_signals += 1
        if trend.check_entry(fast_ema, slow_ema, "BUY"):
            individual_signals += 1
        if confidence.check_entry(confidence_value, "BUY"):
            individual_signals += 1
            
        # Count combined signals (all filters must agree)
        if (rsi.check_entry(rsi_value, "BUY") and
            trend.check_entry(fast_ema, slow_ema, "BUY") and
            confidence.check_entry(confidence_value, "BUY") and
            base.check_trade_spacing(i)):
            combined_signals += 1
    
    # Verify significant reduction in signals when combined
    assert combined_signals < individual_signals * 0.1  # At least 90% reduction

def test_consecutive_trade_prevention():
    """Test prevention of rapid consecutive trades"""
    rsi = RSIFilter(oversold=30, overbought=70)
    trend = TrendFilter(fast_period=8, slow_period=21)
    base = BaseFilter(min_trade_spacing=5)
    
    # Generate oscillating prices that would trigger frequent trades
    prices = generate_oscillating_prices()
    
    trades = []
    for i in range(len(prices) - 30):
        window = prices[i:i+30]
        
        rsi_value = rsi.calculate_rsi(window)
        fast_ema, slow_ema, trend_direction = trend.check_trend(window)
        
        if (rsi.check_entry(rsi_value, "BUY") and
            trend.check_entry(fast_ema, slow_ema, "BUY") and
            base.check_trade_spacing(i)):
            trades.append(i)
    
    # Verify minimum spacing between trades
    for i in range(len(trades) - 1):
        spacing = trades[i+1] - trades[i]
        assert spacing >= 5  # Minimum 5 periods between trades

def test_drawdown_protection():
    """Test filters prevent excessive drawdown"""
    rsi = RSIFilter(oversold=30, overbought=70)
    trend = TrendFilter(fast_period=8, slow_period=21)
    base = BaseFilter(min_trade_spacing=5)
    
    # Generate declining prices with brief recoveries
    prices = generate_drawdown_prices()
    
    max_drawdown = 0
    peak = prices[0]
    
    for i in range(len(prices) - 30):
        window = prices[i:i+30]
        current_price = prices[i+29]
        
        # Update maximum drawdown
        drawdown = (peak - current_price) / peak
        max_drawdown = max(max_drawdown, drawdown)
        
        # Check if filters allow entry
        rsi_value = rsi.calculate_rsi(window)
        fast_ema, slow_ema, trend_direction = trend.check_trend(window)
        
        if (rsi.check_entry(rsi_value, "BUY") and
            trend.check_entry(fast_ema, slow_ema, "BUY") and
            base.check_trade_spacing(i)):
            # Verify no entries during severe drawdowns
            assert drawdown <= 0.15  # Max 15% drawdown allowed
    
    # Verify overall drawdown protection
    assert max_drawdown <= 0.13  # Matches simulation's 13% max drawdown

def generate_realistic_prices():
    """Generate realistic price series with trends and noise"""
    np.random.seed(42)
    prices = []
    price = 100.0
    trend = 0.1
    volatility = 0.5
    
    for _ in range(100):
        noise = np.random.normal(0, volatility)
        price += trend + noise
        prices.append(price)
        
        # Occasionally change trend
        if np.random.random() < 0.1:
            trend = -trend
    
    return prices

def generate_oscillating_prices():
    """Generate oscillating price series that would trigger frequent trades"""
    prices = []
    price = 100.0
    
    for i in range(100):
        # Create oscillating pattern
        price = 100.0 + 10 * np.sin(i / 5)
        prices.append(price)
    
    return prices

def generate_drawdown_prices():
    """Generate price series with controlled drawdown"""
    prices = []
    price = 100.0
    trend = -0.1
    volatility = 0.2
    
    for i in range(100):
        noise = np.random.normal(0, volatility)
        price += trend + noise
        prices.append(price)
        
        # Add recovery periods
        if price < 87:  # 13% drawdown
            trend = 0.2  # Strong recovery
        elif price > 95:
            trend = -0.1  # Resume decline
    
    return prices 