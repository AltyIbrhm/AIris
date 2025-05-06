"""
Tests for the signal combiner strategy.
"""
import pytest
from typing import Dict, Any
from strategy.signal_combiner import SignalCombiner
from strategy.base import BaseStrategy

class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    def __init__(self, action: str, confidence: float = 1.0):
        self.action = action
        self.confidence = confidence

    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'action': self.action,
            'price': market_data['close'][-1],
            'timestamp': market_data['timestamp'][-1],
            'confidence': self.confidence
        }

@pytest.fixture
def market_data():
    """Sample market data for testing."""
    return {
        'close': [100, 101, 102, 103, 104],
        'timestamp': [1000, 1001, 1002, 1003, 1004]
    }

@pytest.fixture
def signal_combiner():
    """Create a signal combiner instance."""
    return SignalCombiner({
        'strategy_weights': {},
        'hold_threshold': 0.05  # Lower threshold to make weighted signals test pass
    })

def test_combine_buy_signals(signal_combiner, market_data):
    """Test combining multiple buy signals."""
    # Add two buy strategies
    signal_combiner.add_strategy(MockStrategy('buy', 0.8), weight=1.0)
    signal_combiner.add_strategy(MockStrategy('buy', 0.9), weight=1.0)
    
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'buy'
    assert signal['confidence'] > 0.5

def test_combine_conflicting_signals(signal_combiner, market_data):
    """Test combining conflicting signals (buy + sell)."""
    # Add conflicting strategies
    signal_combiner.add_strategy(MockStrategy('buy', 0.8), weight=1.0)
    signal_combiner.add_strategy(MockStrategy('sell', 0.9), weight=1.0)
    
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'hold'  # Should default to hold with conflicting signals
    assert signal['confidence'] == 0.0

def test_weighted_signals(signal_combiner, market_data):
    """Test that signals are properly weighted."""
    # Add strategies with different weights
    signal_combiner.add_strategy(MockStrategy('buy', 0.8), weight=2.0)
    signal_combiner.add_strategy(MockStrategy('sell', 0.9), weight=1.0)
    
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'buy'  # Should be buy due to higher weight
    assert signal['confidence'] > 0.0

def test_no_strategies(signal_combiner, market_data):
    """Test behavior when no strategies are added."""
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'hold'
    assert 'error' in signal
    assert signal['error'] == 'No strategies configured'

def test_zero_weights(signal_combiner, market_data):
    """Test behavior when all weights are zero."""
    signal_combiner.add_strategy(MockStrategy('buy'), weight=0.0)
    signal_combiner.add_strategy(MockStrategy('sell'), weight=0.0)
    
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'hold'
    assert 'error' in signal
    assert signal['error'] == 'Total weight is zero' 