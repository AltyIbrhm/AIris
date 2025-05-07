"""
Tests for the risk management module.
"""
import pytest
from datetime import datetime, timedelta
import json
import os
from risk.checker import RiskManager
from utils.portfolio import PortfolioTracker
from typing import Dict, Any

@pytest.fixture
def risk_config(tmp_path):
    """Create temporary risk config file."""
    config = {
        "default": {
            "min_confidence": 0.3,
            "max_open_positions": 1,
            "max_drawdown_percent": 10,
            "max_daily_loss": 300,
            "default_sl_percent": 2.0,
            "default_tp_percent": 4.0,
            "duplicate_signal_block_minutes": 5,
            "max_position_size_percent": 10
        },
        "trend_following": {
            "min_confidence": 0.4,
            "max_open_positions": 2,
            "default_sl_percent": 3.0,
            "default_tp_percent": 6.0,
            "max_position_size_percent": 15
        }
    }
    config_path = tmp_path / "risk_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return str(config_path)

@pytest.fixture
def portfolio():
    """Create a portfolio tracker instance."""
    return PortfolioTracker(initial_capital=10000.0)

@pytest.fixture
def risk_manager(portfolio, risk_config):
    """Create a risk manager instance."""
    return RiskManager(risk_config, portfolio)

@pytest.fixture
def valid_signal():
    """Create a valid trading signal."""
    return {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'direction': 'long',
        'confidence': 0.8,
        'price': 100.0,
        'stop_loss': 95.0,
        'take_profit': 110.0,
        'strategy': 'default',
        'timestamp': datetime.now()
    }

def test_valid_signal_default_strategy(risk_manager, valid_signal):
    """Test validation of signal with default strategy."""
    assert risk_manager.check(valid_signal) is True

def test_valid_signal_custom_strategy(risk_manager, valid_signal):
    """Test validation of signal with custom strategy."""
    valid_signal['strategy'] = 'trend_following'
    valid_signal['confidence'] = 0.5
    assert risk_manager.check(valid_signal) is True

def test_max_positions_default_strategy(portfolio, risk_manager, valid_signal):
    """Test maximum positions limit with default strategy."""
    # Open maximum allowed positions
    for i in range(3):
        portfolio.open_position(
            symbol=valid_signal['symbol'],
            entry_price=valid_signal['price'],
            size=0.1,
            direction=valid_signal['direction'],
            stop_loss=valid_signal['stop_loss'],
            take_profit=valid_signal['take_profit']
        )

    # Next signal should fail
    assert risk_manager.check(valid_signal) is False

def test_max_positions_custom_strategy(portfolio, risk_manager, valid_signal):
    """Test maximum positions limit with custom strategy."""
    valid_signal['strategy'] = 'trend_following'
    
    # Open maximum allowed positions
    for i in range(3):
        portfolio.open_position(
            symbol=valid_signal['symbol'],
            entry_price=valid_signal['price'],
            size=0.1,
            direction=valid_signal['direction'],
            stop_loss=valid_signal['stop_loss'],
            take_profit=valid_signal['take_profit']
        )

    # Next signal should fail
    assert risk_manager.check(valid_signal) is False

def test_drawdown_limit(portfolio, risk_manager, valid_signal):
    """Test drawdown limit check."""
    # Set initial capital and peak capital
    portfolio.current_capital = 10000.0
    portfolio.peak_capital[valid_signal['symbol']] = 10000.0
    
    # Open a position with a large size to ensure significant drawdown
    position = portfolio.open_position(
        symbol=valid_signal['symbol'],
        entry_price=100.0,
        size=100.0,  # Much larger size to trigger significant drawdown
        direction='long',
        stop_loss=95.0,
        take_profit=110.0
    )

    # Close with significant loss to trigger drawdown
    portfolio.close_position(valid_signal['symbol'], position, 90.0)  # 10% loss

    # Next signal should fail due to drawdown
    assert risk_manager.check(valid_signal) is False

def test_daily_loss_limit(portfolio, risk_manager, valid_signal):
    """Test daily loss limit check."""
    # Set initial capital
    portfolio.current_capital = 10000.0
    portfolio.peak_capital[valid_signal['symbol']] = 10000.0
    
    # Open and close positions with losses
    for i in range(3):
        position = portfolio.open_position(
            symbol=valid_signal['symbol'],
            entry_price=100.0,
            size=10.0,  # Large size to trigger significant loss
            direction='long',
            stop_loss=95.0,
            take_profit=110.0
        )
        portfolio.close_position(valid_signal['symbol'], position, 90.0)  # 10% loss each time

    # Next signal should fail due to daily loss
    assert risk_manager.check(valid_signal) is False

def test_duplicate_signal(risk_manager, valid_signal):
    """Test duplicate signal check."""
    # First signal should pass
    assert risk_manager.check(valid_signal) is True

    # Immediate duplicate should fail
    assert risk_manager.check(valid_signal) is False

def test_sl_tp_validation_default(risk_manager, valid_signal):
    """Test stop loss and take profit validation with default strategy."""
    assert risk_manager.check(valid_signal) is True

def test_sl_tp_validation_custom(risk_manager, valid_signal):
    """Test stop loss and take profit validation with custom strategy."""
    valid_signal['strategy'] = 'trend_following'
    assert risk_manager.check(valid_signal) is True

def test_confidence_threshold(risk_manager, valid_signal):
    """Test confidence threshold check."""
    valid_signal['confidence'] = 0.2  # Below minimum
    assert risk_manager.check(valid_signal) is False

def test_unknown_strategy(risk_manager, valid_signal):
    """Test handling of unknown strategy."""
    valid_signal['strategy'] = 'unknown_strategy'
    assert risk_manager.check(valid_signal) is True  # Should use default config

def test_missing_fields(risk_manager):
    """Test handling of missing required fields."""
    signal = {
        'symbol': 'BTC/USDT',
        'action': 'buy'  # Missing other required fields
    }
    assert risk_manager.check(signal) is False

def test_config_loading_error(portfolio, tmp_path):
    """Test error handling for invalid config file."""
    invalid_config_path = tmp_path / "invalid_config.json"
    risk_manager = RiskManager(str(invalid_config_path), portfolio)
    
    signal = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'direction': 'long',
        'confidence': 0.8,
        'price': 100.0,
        'stop_loss': 95.0,
        'take_profit': 110.0,
        'strategy': 'default',
        'timestamp': datetime.now()
    }
    
    assert risk_manager.check(signal) is False 