"""
Tests for the exposure manager.
"""
import pytest
import time
from typing import Dict, Any
from risk.exposure import ExposureManager

@pytest.fixture
def position():
    """Sample position data for testing."""
    return {
        'symbol': 'BTCUSDT',
        'size': 0.01,  # 0.01 BTC at 50,000 USDT = 500 USDT
        'entry_price': 50000,
        'side': 'long'
    }

@pytest.fixture
def exposure_manager():
    """Create an exposure manager instance."""
    return ExposureManager({
        'max_position_size': 1000,  # 1,000 USDT
        'max_daily_trades': 5,
        'max_drawdown': 0.2
    })

def test_valid_position(exposure_manager, position):
    """Test evaluation of a valid position."""
    result = exposure_manager.evaluate_risk(position)
    assert result['allowed'] is True
    assert result['position_size'] == position['size'] * position['entry_price']
    assert result['daily_trades'] == 0

def test_exceed_position_size(exposure_manager, position):
    """Test position size limit."""
    position['size'] = 0.03  # 0.03 BTC at 50,000 USDT = 1,500 USDT
    result = exposure_manager.evaluate_risk(position)
    assert result['allowed'] is False
    assert 'Position size exceeds maximum' in result['reason']

def test_daily_trade_limit(exposure_manager, position):
    """Test daily trade limit."""
    # Simulate max daily trades
    for _ in range(5):
        exposure_manager.update_pnl(100)
    
    result = exposure_manager.evaluate_risk(position)
    assert result['allowed'] is False
    assert 'Daily trade limit reached' in result['reason']

def test_max_drawdown(exposure_manager, position):
    """Test maximum drawdown limit."""
    # Simulate large loss
    exposure_manager.update_pnl(-1000)
    
    result = exposure_manager.evaluate_risk(position)
    assert result['allowed'] is False
    assert 'Maximum drawdown reached' in result['reason']

def test_daily_reset(exposure_manager, position):
    """Test daily counter reset."""
    # Add some trades
    exposure_manager.update_pnl(100)
    exposure_manager.update_pnl(200)
    
    # Force reset by setting last_reset to 25 hours ago
    exposure_manager.last_reset = time.time() - 86400 - 3600
    
    result = exposure_manager.evaluate_risk(position)
    assert result['allowed'] is True
    assert result['daily_trades'] == 0  # Should be reset

def test_invalid_position(exposure_manager):
    """Test handling of invalid position data."""
    invalid_position = {'symbol': 'BTCUSDT'}  # Missing required fields
    result = exposure_manager.evaluate_risk(invalid_position)
    assert 'error' in result
 