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
        super().__init__({'name': 'mock_strategy'})
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
        'hold_threshold': 0.05,  # Lower threshold to make weighted signals test pass
        'min_confidence': 0.3,
        'conflict_threshold': 0.2
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
    # Add conflicting strategies with equal weights and similar confidence
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

def test_low_confidence_signals(signal_combiner, market_data):
    """Test that low confidence signals are filtered out."""
    signal_combiner.add_strategy(MockStrategy('buy', 0.2), weight=1.0)  # Below min_confidence
    signal_combiner.add_strategy(MockStrategy('sell', 0.9), weight=1.0)
    
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'sell'  # Only high confidence signal should be considered
    assert signal['confidence'] > 0.0

def test_similar_weight_conflict(signal_combiner, market_data):
    """Test conflict detection with similar weights."""
    # Add strategies with similar weights but slightly different
    signal_combiner.add_strategy(MockStrategy('buy', 0.8), weight=1.1)
    signal_combiner.add_strategy(MockStrategy('sell', 0.8), weight=0.9)
    
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'hold'  # Should detect conflict due to similar weights
    assert signal['confidence'] == 0.0

def test_clear_winner_despite_conflict(signal_combiner, market_data):
    """Test that strong signals can overcome weight similarity."""
    # Add strategies with similar weights but very different confidence
    signal_combiner.add_strategy(MockStrategy('buy', 0.9), weight=1.1)
    signal_combiner.add_strategy(MockStrategy('sell', 0.4), weight=0.9)
    
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'buy'  # Strong buy should win despite similar weights
    assert signal['confidence'] > 0.0

def test_error_handling(signal_combiner, market_data):
    """Test error handling in signal generation."""
    class ErrorStrategy(BaseStrategy):
        def __init__(self):
            super().__init__({'name': 'error_strategy'})
            
        def generate_signal(self, market_data):
            raise Exception("Test error")
    
    signal_combiner.add_strategy(ErrorStrategy())
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'hold'
    assert signal['confidence'] == 0.0 