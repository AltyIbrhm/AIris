import pytest
import numpy as np
from strategy.exits.trailing_stop import TrailingStop
from datetime import datetime, timedelta

def test_trailing_stop_initialization():
    """Test trailing stop initialization with default and custom parameters"""
    # Test default parameters
    ts = TrailingStop()
    assert ts.activation_threshold == 0.5
    assert ts.trail_distance_atr_mult == 0.8
    assert ts.atr_period == 14
    assert ts.min_profit_threshold == 0.003
    
    # Test custom parameters
    ts = TrailingStop(
        activation_threshold=0.6,
        trail_distance_atr_mult=1.0,
        atr_period=10,
        min_profit_threshold=0.005
    )
    assert ts.activation_threshold == 0.6
    assert ts.trail_distance_atr_mult == 1.0
    assert ts.atr_period == 10
    assert ts.min_profit_threshold == 0.005

def test_atr_calculation():
    """Test ATR calculation with known price series"""
    ts = TrailingStop(atr_period=3)
    
    # Test insufficient data
    assert ts.calculate_atr([100], [99], [99.5]) == 0.0
    
    # Test known price series
    highs = [100, 101, 102, 101]
    lows = [98, 99, 100, 99]
    closes = [99, 100, 101, 100]
    
    atr = ts.calculate_atr(highs, lows, closes)
    assert 1.9 <= atr <= 2.1  # Should be around 2.0
    
    # Test flat prices (ATR should be near zero)
    highs = [100, 100, 100, 100]
    lows = [100, 100, 100, 100]
    closes = [100, 100, 100, 100]
    
    atr = ts.calculate_atr(highs, lows, closes)
    assert atr == 0.0

def test_trailing_activation():
    """Test trailing stop activation based on profit threshold"""
    ts = TrailingStop(activation_threshold=0.5, min_profit_threshold=0.005)  # 0.5% min profit
    
    # Initialize long position
    ts.initialize_trade(entry_price=100.0, side="BUY")
    
    # Price moves up but not enough to activate trailing
    current_price = 100.3  # 0.3% profit
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 100.3],
        low_prices=[99, 100],
        close_prices=[99.5, 100.3]
    )
    assert not exit_signal
    assert trail_price == 100.0  # Trail should stay at entry price
    
    # Price moves up enough to activate trailing
    current_price = 100.6  # 0.6% profit
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 100.3, 100.6],
        low_prices=[99, 100, 100.3],
        close_prices=[99.5, 100.3, 100.6]
    )
    assert not exit_signal
    assert trail_price < current_price  # Trail should be below current price

def test_trailing_updates():
    """Test trailing stop updates as price moves"""
    ts = TrailingStop(activation_threshold=0.5, min_profit_threshold=0.005)
    
    # Initialize long position
    ts.initialize_trade(entry_price=100.0, side="BUY")
    
    # Move price up to activate trailing
    current_price = 101.0  # 1% profit
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101],
        low_prices=[99, 100],
        close_prices=[99.5, 101]
    )
    initial_trail = trail_price
    assert not exit_signal
    
    # Price moves up further
    current_price = 102.0
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101, 102],
        low_prices=[99, 100, 101],
        close_prices=[99.5, 101, 102]
    )
    assert not exit_signal
    assert trail_price > initial_trail  # Trail should move up
    
    # Price moves down but not enough to hit trail
    current_price = 101.5
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101, 102, 101.5],
        low_prices=[99, 100, 101, 101],
        close_prices=[99.5, 101, 102, 101.5]
    )
    assert not exit_signal
    last_trail = trail_price
    
    # Price moves down to hit trail
    current_price = last_trail - 0.1
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101, 102, 101.5, current_price],
        low_prices=[99, 100, 101, 101, current_price-1],
        close_prices=[99.5, 101, 102, 101.5, current_price]
    )
    assert exit_signal
    assert exit_reason == "trail_hit"
    assert abs(exit_price - last_trail) < 0.001  # Exit at trail price

def test_take_profit_activation():
    """Test trailing stop activation based on profit threshold"""
    ts = TrailingStop(activation_threshold=0.5, min_profit_threshold=0.01)  # Activate at 1% profit
    
    # Initialize long position
    ts.initialize_trade(entry_price=100.0, side="BUY")
    
    # Price moves up to activation level
    current_price = 101.0  # 1% profit
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101],
        low_prices=[99, 100],
        close_prices=[99.5, 101]
    )
    assert not exit_signal
    assert trail_price < current_price  # Trail should be below current price
    last_trail = trail_price
    
    # Price moves down to hit trail
    current_price = last_trail - 0.1
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101, current_price],
        low_prices=[99, 100, current_price-1],
        close_prices=[99.5, 101, current_price]
    )
    assert exit_signal
    assert exit_reason == "trail_hit"
    assert abs(exit_price - last_trail) < 0.001  # Exit at trail price

def test_short_position():
    """Test trailing stop functionality with short positions"""
    ts = TrailingStop(activation_threshold=0.5, min_profit_threshold=0.005)
    
    # Initialize short position
    ts.initialize_trade(entry_price=100.0, side="SELL")
    
    # Price moves down but not enough to activate trailing
    current_price = 99.7  # 0.3% profit
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 99.7],
        low_prices=[99, 99.5],
        close_prices=[99.5, 99.7]
    )
    assert not exit_signal
    assert trail_price == 100.0  # Trail should stay at entry price
    
    # Price moves down enough to activate trailing
    current_price = 99.4  # 0.6% profit
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 99.7, 99.4],
        low_prices=[99, 99.5, 99.3],
        close_prices=[99.5, 99.7, 99.4]
    )
    assert not exit_signal
    assert trail_price > current_price  # Trail should be above current price
    last_trail = trail_price
    
    # Price moves up to hit trail
    current_price = last_trail + 0.1
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 99.7, 99.4, current_price],
        low_prices=[99, 99.5, 99.3, current_price-1],
        close_prices=[99.5, 99.7, 99.4, current_price]
    )
    assert exit_signal
    assert exit_reason == "trail_hit"
    assert abs(exit_price - last_trail) < 0.001  # Exit at trail price

def test_max_loss_exit():
    """Test maximum loss exit functionality"""
    ts = TrailingStop(max_loss_pct=0.02)  # 2% max loss
    
    # Initialize long position
    ts.initialize_trade(entry_price=100.0, side="BUY")
    
    # Price drops to max loss
    current_price = 97.5  # 2.5% loss
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 99, 98, 97.5],
        low_prices=[99, 98, 97, 97],
        close_prices=[99.5, 98.5, 97.5, 97.5]
    )
    assert exit_signal
    assert exit_reason == "max_loss"
    assert exit_price == current_price
    
    # Test short position
    ts.initialize_trade(entry_price=100.0, side="SELL")
    
    # Price rises to max loss
    current_price = 102.5  # 2.5% loss
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101, 102, 102.5],
        low_prices=[99, 100, 101, 102],
        close_prices=[99.5, 100.5, 101.5, 102.5]
    )
    assert exit_signal
    assert exit_reason == "max_loss"
    assert exit_price == current_price

def test_time_exit():
    """Test time-based exit functionality"""
    ts = TrailingStop(max_holding_time=1)  # 1 hour max holding time
    
    # Initialize position
    ts.initialize_trade(entry_price=100.0, side="BUY")
    
    # Simulate time passing
    ts.entry_time = datetime.now() - timedelta(hours=2)
    
    current_price = 101.0
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101],
        low_prices=[99, 100],
        close_prices=[99.5, 101]
    )
    assert exit_signal
    assert exit_reason == "time_exit"
    assert exit_price == current_price

def test_position_size_adjustment():
    """Test trail distance adjustment based on position size"""
    ts = TrailingStop(
        trail_distance_atr_mult=1.0,
        position_size_mult=2.0  # Double trail distance for larger positions
    )
    
    # Initialize position with large size
    ts.initialize_trade(entry_price=100.0, side="BUY", position_size=2.0)
    
    # Move price up to activate trailing
    current_price = 105.0  # 5% profit, enough to ensure trail is above entry
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 102, 104, 105],
        low_prices=[99, 101, 103, 104],
        close_prices=[99.5, 101.5, 103.5, 105]
    )
    
    # Calculate expected trail distance
    atr = ts.calculate_atr([100, 102, 104, 105], [99, 101, 103, 104], [99.5, 101.5, 103.5, 105])
    trail_distance = atr * ts.trail_distance_atr_mult * ts.position_size_mult
    expected_trail = current_price - trail_distance
    
    # Trail price should be at expected trail since it's well above entry price
    assert abs(trail_price - expected_trail) < 0.001
    
    # Verify trail is above entry price
    assert trail_price > ts.entry_price
    
    # Verify trail distance is doubled due to position size multiplier
    standard_trail_distance = atr * ts.trail_distance_atr_mult
    actual_trail_distance = current_price - trail_price
    assert abs(actual_trail_distance - (2.0 * standard_trail_distance)) < 0.001

def test_exit_reasons():
    """Test different exit reasons are properly reported"""
    ts = TrailingStop(
        max_loss_pct=0.02,
        max_holding_time=1
    )
    
    # Test trail hit
    ts.initialize_trade(entry_price=100.0, side="BUY")
    current_price = 101.0
    exit_signal, trail_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101],
        low_prices=[99, 100],
        close_prices=[99.5, 101]
    )
    assert not exit_signal  # Should not exit yet
    
    # Price drops to hit trail
    current_price = trail_price - 0.1
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101, current_price],
        low_prices=[99, 100, current_price-1],
        close_prices=[99.5, 101, current_price]
    )
    assert exit_signal
    assert exit_reason == "trail_hit"
    
    # Test max loss
    ts.initialize_trade(entry_price=100.0, side="BUY")
    current_price = 97.5
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 99, 98, 97.5],
        low_prices=[99, 98, 97, 97],
        close_prices=[99.5, 98.5, 97.5, 97.5]
    )
    assert exit_signal
    assert exit_reason == "max_loss"
    
    # Test time exit
    ts.initialize_trade(entry_price=100.0, side="BUY")
    ts.entry_time = datetime.now() - timedelta(hours=2)
    current_price = 101.0
    exit_signal, exit_price, exit_reason = ts.update_trail(
        current_price=current_price,
        high_prices=[100, 101],
        low_prices=[99, 100],
        close_prices=[99.5, 101]
    )
    assert exit_signal
    assert exit_reason == "time_exit"

def test_string_representation():
    """Test string representation of trailing stop"""
    ts = TrailingStop(
        activation_threshold=0.5,
        trail_distance_atr_mult=0.8,
        atr_period=14,
        max_loss_pct=0.02,
        max_holding_time=48
    )
    expected = "TrailingStop(activation=0.5, trail_mult=0.8, atr_period=14, max_loss=0.02, max_time=48h)"
    assert str(ts) == expected 