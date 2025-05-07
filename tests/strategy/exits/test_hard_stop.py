import pytest
import numpy as np
from strategy.exits.hard_stop import HardStop
from datetime import datetime, timedelta

def test_hard_stop_initialization():
    """Test hard stop initialization with default and custom parameters"""
    # Test default parameters
    hs = HardStop()
    assert hs.stop_loss_atr_mult == 2.0
    assert hs.atr_period == 14
    assert hs.max_loss_pct == 0.01
    assert hs.max_holding_time == 72
    assert hs.balance_override is None
    
    # Test custom parameters
    hs = HardStop(
        stop_loss_atr_mult=3.0,
        atr_period=10,
        max_loss_pct=0.02,
        max_holding_time=48,
        balance_override=10000.0
    )
    assert hs.stop_loss_atr_mult == 3.0
    assert hs.atr_period == 10
    assert hs.max_loss_pct == 0.02
    assert hs.max_holding_time == 48
    assert hs.balance_override == 10000.0

def test_atr_calculation():
    """Test ATR calculation with known price series"""
    hs = HardStop(atr_period=3)
    
    # Test insufficient data
    assert hs.calculate_atr([100], [99], [99.5]) == 0.0
    
    # Test known price series
    highs = [100, 101, 102, 101]
    lows = [98, 99, 100, 99]
    closes = [99, 100, 101, 100]
    
    atr = hs.calculate_atr(highs, lows, closes)
    assert 1.9 <= atr <= 2.1  # Should be around 2.0
    
    # Test flat prices (ATR should be near zero)
    highs = [100, 100, 100, 100]
    lows = [100, 100, 100, 100]
    closes = [100, 100, 100, 100]
    
    atr = hs.calculate_atr(highs, lows, closes)
    assert atr == 0.0

def test_stop_loss_initialization():
    """Test stop loss price calculation at trade initialization"""
    hs = HardStop(stop_loss_atr_mult=2.0)
    
    # Initialize long position
    highs = [100, 101, 102, 101]
    lows = [98, 99, 100, 99]
    closes = [99, 100, 101, 100]
    
    hs.initialize_trade(
        entry_price=100.0,
        side="BUY",
        high_prices=highs,
        low_prices=lows,
        close_prices=closes
    )
    
    atr = hs.calculate_atr(highs, lows, closes)
    expected_stop = 100.0 - (2.0 * atr)
    assert abs(hs.stop_loss_price - expected_stop) < 0.001
    
    # Initialize short position
    hs.initialize_trade(
        entry_price=100.0,
        side="SELL",
        high_prices=highs,
        low_prices=lows,
        close_prices=closes
    )
    
    expected_stop = 100.0 + (2.0 * atr)
    assert abs(hs.stop_loss_price - expected_stop) < 0.001

def test_balance_based_stop_loss():
    """Test stop loss adjustment based on balance percentage"""
    hs = HardStop(
        stop_loss_atr_mult=2.0,
        max_loss_pct=0.01,  # 1% max loss
        balance_override=10000.0
    )
    
    # Initialize long position
    highs = [100, 101, 102, 101]
    lows = [98, 99, 100, 99]
    closes = [99, 100, 101, 100]
    
    entry_price = 100.0
    hs.initialize_trade(
        entry_price=entry_price,
        side="BUY",
        high_prices=highs,
        low_prices=lows,
        close_prices=closes
    )
    
    # Stop loss should be at max 1% loss from entry
    expected_stop = entry_price * 0.99  # 1% below entry
    assert hs.stop_loss_price >= expected_stop  # ATR-based stop might be higher
    
    # Initialize short position
    hs.initialize_trade(
        entry_price=entry_price,
        side="SELL",
        high_prices=highs,
        low_prices=lows,
        close_prices=closes
    )
    
    # Stop loss should be at max 1% loss from entry
    expected_stop = entry_price * 1.01  # 1% above entry
    assert hs.stop_loss_price <= expected_stop  # ATR-based stop might be lower

def test_stop_loss_hit():
    """Test stop loss exit signals"""
    hs = HardStop(stop_loss_atr_mult=2.0)
    
    # Initialize long position
    highs = [100, 101, 102, 101]
    lows = [98, 99, 100, 99]
    closes = [99, 100, 101, 100]
    
    hs.initialize_trade(
        entry_price=100.0,
        side="BUY",
        high_prices=highs,
        low_prices=lows,
        close_prices=closes
    )
    stop_price = hs.stop_loss_price
    
    # Price above stop - no exit
    current_price = stop_price + 0.1
    exit_signal, exit_price, exit_reason = hs.check_exit(current_price)
    assert not exit_signal
    
    # Price hits stop - exit
    current_price = stop_price - 0.1
    exit_signal, exit_price, exit_reason = hs.check_exit(current_price)
    assert exit_signal
    assert exit_reason == "stop_loss"
    assert abs(exit_price - stop_price) < 0.001
    
    # Test short position
    hs.initialize_trade(
        entry_price=100.0,
        side="SELL",
        high_prices=highs,
        low_prices=lows,
        close_prices=closes
    )
    stop_price = hs.stop_loss_price
    
    # Price below stop - no exit
    current_price = stop_price - 0.1
    exit_signal, exit_price, exit_reason = hs.check_exit(current_price)
    assert not exit_signal
    
    # Price hits stop - exit
    current_price = stop_price + 0.1
    exit_signal, exit_price, exit_reason = hs.check_exit(current_price)
    assert exit_signal
    assert exit_reason == "stop_loss"
    assert abs(exit_price - stop_price) < 0.001

def test_time_exit():
    """Test time-based exit functionality"""
    hs = HardStop(max_holding_time=1)  # 1 hour max holding time
    
    # Initialize position
    highs = [100, 101, 102, 101]
    lows = [98, 99, 100, 99]
    closes = [99, 100, 101, 100]
    
    hs.initialize_trade(
        entry_price=100.0,
        side="BUY",
        high_prices=highs,
        low_prices=lows,
        close_prices=closes
    )
    
    # Simulate time passing
    hs.entry_time = datetime.now() - timedelta(hours=2)
    
    current_price = 101.0
    exit_signal, exit_price, exit_reason = hs.check_exit(current_price)
    assert exit_signal
    assert exit_reason == "time_exit"
    assert exit_price == current_price

def test_string_representation():
    """Test string representation of hard stop"""
    hs = HardStop(
        stop_loss_atr_mult=2.0,
        atr_period=14,
        max_loss_pct=0.01,
        max_holding_time=72
    )
    expected = "HardStop(stop_mult=2.0, atr_period=14, max_loss=0.01, max_time=72h)"
    assert str(hs) == expected 