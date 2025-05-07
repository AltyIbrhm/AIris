import pytest
import numpy as np
from datetime import datetime, timedelta
from strategy.exits.exit_manager import ExitManager

def test_exit_manager_initialization():
    """Test exit manager initialization with default and custom configurations"""
    # Test default initialization
    em = ExitManager()
    assert em.trailing is not None
    assert em.hard_stop is not None
    
    # Test custom configuration
    trailing_config = {
        "activation_threshold": 0.6,
        "trail_distance_atr_mult": 1.0,
        "atr_period": 10,
        "min_profit_threshold": 0.005
    }
    
    hard_stop_config = {
        "stop_loss_atr_mult": 3.0,
        "atr_period": 10,
        "max_loss_pct": 0.02,
        "max_holding_time": 48
    }
    
    em = ExitManager(trailing_config, hard_stop_config)
    assert em.trailing.activation_threshold == 0.6
    assert em.trailing.trail_distance_atr_mult == 1.0
    assert em.trailing.atr_period == 10
    assert em.trailing.min_profit_threshold == 0.005
    
    assert em.hard_stop.stop_loss_atr_mult == 3.0
    assert em.hard_stop.atr_period == 10
    assert em.hard_stop.max_loss_pct == 0.02
    assert em.hard_stop.max_holding_time == 48

def test_trade_initialization():
    """Test trade initialization with both exit strategies"""
    em = ExitManager()
    
    # Initialize trade
    entry_price = 100.0
    side = "BUY"
    high_prices = [100, 101, 102, 101]
    low_prices = [98, 99, 100, 99]
    close_prices = [99, 100, 101, 100]
    position_size = 2.0
    
    em.initialize_trade(
        entry_price=entry_price,
        side=side,
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices,
        position_size=position_size
    )
    
    # Verify state variables
    assert em.entry_price == entry_price
    assert em.position_side == side
    assert em.position_size == position_size
    assert em.entry_time is not None
    
    # Verify both stops are initialized
    assert em.trailing.entry_price == entry_price
    assert em.trailing.position_side == side
    assert em.trailing.position_size == position_size
    
    assert em.hard_stop.entry_price == entry_price
    assert em.hard_stop.position_side == side
    assert em.hard_stop.stop_loss_price > 0

def test_trailing_stop_exit():
    """Test trailing stop exit priority"""
    em = ExitManager()
    
    # Initialize trade
    entry_price = 100.0
    high_prices = [100, 101, 102, 101]
    low_prices = [98, 99, 100, 99]
    close_prices = [99, 100, 101, 100]
    
    em.initialize_trade(
        entry_price=entry_price,
        side="BUY",
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices
    )
    
    # Move price up to activate trailing
    current_price = 101.0  # 1% profit
    exit_signal, exit_price, exit_reason = em.check_exit(
        current_price=current_price,
        high_prices=high_prices + [current_price],
        low_prices=low_prices + [current_price-1],
        close_prices=close_prices + [current_price]
    )
    assert not exit_signal
    trail_price = exit_price
    
    # Price drops to hit trailing stop
    current_price = trail_price - 0.1
    exit_signal, exit_price, exit_reason = em.check_exit(
        current_price=current_price,
        high_prices=high_prices + [101.0, current_price],
        low_prices=low_prices + [100.0, current_price-1],
        close_prices=close_prices + [101.0, current_price]
    )
    assert exit_signal
    assert exit_reason == "trail_hit"
    assert abs(exit_price - trail_price) < 0.001

def test_hard_stop_exit():
    """Test hard stop exit when trailing not activated"""
    em = ExitManager()
    
    # Initialize trade
    entry_price = 100.0
    high_prices = [100, 101, 102, 101]
    lows = [98, 99, 100, 99]
    closes = [99, 100, 101, 100]
    
    em.initialize_trade(
        entry_price=entry_price,
        side="BUY",
        high_prices=high_prices,
        low_prices=lows,
        close_prices=closes
    )
    
    # Get initial stop levels
    stops = em.get_current_stops()
    hard_stop = stops["hard_stop"]
    
    # Price drops to hit hard stop
    current_price = hard_stop  # Use exact hard stop price
    exit_signal, exit_price, exit_reason = em.check_exit(
        current_price=current_price,
        high_prices=high_prices + [current_price],
        low_prices=lows + [current_price-1],
        close_prices=closes + [current_price]
    )
    assert exit_signal
    assert exit_reason == "max_loss"  # Hard stop uses max_loss as reason
    assert abs(exit_price - hard_stop) < 0.001

def test_time_exit():
    """Test time-based exit from either stop"""
    em = ExitManager(
        trailing_config={"max_holding_time": 1},  # 1 hour
        hard_stop_config={"max_holding_time": 2}  # 2 hours
    )
    
    # Initialize trade
    entry_price = 100.0
    high_prices = [100, 101, 102, 101]
    low_prices = [98, 99, 100, 99]
    close_prices = [99, 100, 101, 100]
    
    em.initialize_trade(
        entry_price=entry_price,
        side="BUY",
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices
    )
    
    # Set entry time to 1.5 hours ago (between trailing and hard stop times)
    em.entry_time = datetime.now() - timedelta(hours=1.5)
    em.trailing.entry_time = em.entry_time
    em.hard_stop.entry_time = em.entry_time
    
    # Check exit - should trigger trailing stop's time exit first
    current_price = 101.0
    exit_signal, exit_price, exit_reason = em.check_exit(
        current_price=current_price,
        high_prices=high_prices + [current_price],
        low_prices=low_prices + [current_price-1],
        close_prices=close_prices + [current_price]
    )
    assert exit_signal
    assert exit_reason == "time_exit"
    assert exit_price == current_price

def test_get_current_stops():
    """Test getting current stop levels"""
    em = ExitManager(
        trailing_config={
            "activation_threshold": 0.5,
            "min_profit_threshold": 0.005  # Activate at 0.5% profit
        }
    )
    
    # Initialize trade
    entry_price = 100.0
    high_prices = [100, 101, 102, 101]
    lows = [98, 99, 100, 99]
    closes = [99, 100, 101, 100]
    
    em.initialize_trade(
        entry_price=entry_price,
        side="BUY",
        high_prices=high_prices,
        low_prices=lows,
        close_prices=closes
    )
    
    # Get initial stops
    stops = em.get_current_stops()
    assert "trailing_stop" in stops
    assert "hard_stop" in stops
    assert abs(stops["trailing_stop"] - entry_price * (1 - 0.005)) < 0.001  # Initial trailing stop below entry
    assert stops["hard_stop"] < entry_price  # Hard stop below entry for long
    
    # Move price up to activate trailing
    current_price = 101.0  # 1% profit
    em.check_exit(
        current_price=current_price,
        high_prices=high_prices + [current_price],
        low_prices=lows + [current_price-1],
        close_prices=closes + [current_price]
    )
    
    # Get updated stops
    stops = em.get_current_stops()
    assert stops["trailing_stop"] < current_price  # Trailing stop should be below current price
    assert stops["trailing_stop"] > entry_price  # But above entry price
    assert stops["hard_stop"] < entry_price  # Hard stop stays fixed

def test_string_representation():
    """Test string representation of exit manager"""
    em = ExitManager()
    str_rep = str(em)
    assert "ExitManager" in str_rep
    assert "TrailingStop" in str_rep
    assert "HardStop" in str_rep 