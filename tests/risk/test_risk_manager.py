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
    """Create portfolio tracker instance."""
    return PortfolioTracker(initial_capital=10000.0)

@pytest.fixture
def risk_manager(risk_config, portfolio):
    """Create risk manager instance."""
    return RiskManager(risk_config, portfolio)

def create_signal(strategy: str = "default", confidence: float = 0.8, **kwargs) -> Dict[str, Any]:
    """Helper function to create test signals."""
    signal = {
        'action': 'buy',
        'price': 100.0,
        'timestamp': datetime.now(),
        'confidence': confidence,
        'symbol': 'BTC/USD',
        'strategy': strategy
    }
    signal.update(kwargs)  # Add any additional parameters
    return signal

def test_valid_signal_default_strategy(risk_manager):
    """Test valid signal with default strategy passes all checks."""
    signal = create_signal(strategy="default")
    assert risk_manager.check(signal) is True

def test_valid_signal_custom_strategy(risk_manager):
    """Test valid signal with custom strategy passes all checks."""
    signal = create_signal(strategy="trend_following", confidence=0.5)
    assert risk_manager.check(signal) is True

def test_low_confidence_default_strategy(risk_manager):
    """Test signal with low confidence is rejected using default strategy."""
    signal = create_signal(strategy="default", confidence=0.2)
    assert risk_manager.check(signal) is False

def test_low_confidence_custom_strategy(risk_manager):
    """Test signal with low confidence is rejected using custom strategy threshold."""
    signal = create_signal(strategy="trend_following", confidence=0.35)
    assert risk_manager.check(signal) is False

def test_max_positions_default_strategy(risk_manager, portfolio):
    """Test position limit enforcement for default strategy."""
    # Open a position
    portfolio.open_position(
        symbol='BTC/USD',
        entry_price=100.0,
        size=1.0,
        direction='long',
        timestamp=datetime.now()
    )
    
    # Try to open another position
    signal = create_signal(strategy="default")
    assert risk_manager.check(signal) is False

def test_max_positions_custom_strategy(risk_manager, portfolio):
    """Test position limit enforcement for custom strategy."""
    # Open first position
    portfolio.open_position(
        symbol='BTC/USD',
        entry_price=100.0,
        size=1.0,
        direction='long',
        timestamp=datetime.now()
    )
    
    # Try to open second position (should be allowed for trend_following)
    signal = create_signal(strategy="trend_following")
    assert risk_manager.check(signal) is True

def test_drawdown_limit(risk_manager, portfolio):
    """Test drawdown limit enforcement."""
    # Simulate drawdown
    portfolio.current_capital = 8000.0  # 20% drawdown
    portfolio.peak_capital = 10000.0
    
    signal = create_signal()
    assert risk_manager.check(signal) is False

def test_daily_loss_limit(risk_manager, portfolio):
    """Test daily loss limit enforcement."""
    # Simulate daily loss
    portfolio.daily_pnl[datetime.now().strftime('%Y-%m-%d')] = -400.0
    
    signal = create_signal()
    assert risk_manager.check(signal) is False

def test_duplicate_signal(risk_manager):
    """Test duplicate signal blocking."""
    signal = create_signal()
    
    # First signal should pass
    assert risk_manager.check(signal) is True
    
    # Second signal within block window should fail
    assert risk_manager.check(signal) is False

def test_sl_tp_validation_default(risk_manager):
    """Test stop loss and take profit validation for default strategy."""
    signal = create_signal()
    assert risk_manager.check(signal) is True

def test_sl_tp_validation_custom(risk_manager):
    """Test stop loss and take profit validation for custom strategy."""
    signal = create_signal(strategy="trend_following")
    assert risk_manager.check(signal) is True

def test_position_sizing_default(risk_manager):
    """Test position size calculation for default strategy."""
    signal = create_signal(volatility=0.02)
    size = risk_manager.get_position_size(signal)
    assert size > 0
    assert size <= risk_manager.portfolio.current_capital * 0.1

def test_position_sizing_custom(risk_manager):
    """Test position size calculation for custom strategy."""
    signal = create_signal(strategy="trend_following", volatility=0.02)
    size = risk_manager.get_position_size(signal)
    assert size > 0
    assert size <= risk_manager.portfolio.current_capital * 0.15

def test_missing_fields(risk_manager):
    """Test signal validation with missing fields."""
    signal = {
        'action': 'buy',
        'price': 100.0
    }
    assert risk_manager.check(signal) is False

def test_missing_strategy(risk_manager):
    """Test signal validation with missing strategy."""
    signal = {
        'action': 'buy',
        'price': 100.0,
        'timestamp': datetime.now(),
        'confidence': 0.8,
        'symbol': 'BTC/USD'
    }
    assert risk_manager.check(signal) is False

def test_unknown_strategy(risk_manager):
    """Test handling of unknown strategy."""
    signal = create_signal(strategy="unknown_strategy")
    # Should fall back to default configuration
    assert risk_manager.check(signal) is True

def test_config_loading_error(portfolio, tmp_path):
    """Test error handling for invalid config file."""
    invalid_config = str(tmp_path / "invalid_config.json")
    risk_manager = RiskManager(invalid_config, portfolio)
    
    signal = create_signal()
    assert risk_manager.check(signal) is False 