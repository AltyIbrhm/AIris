"""
Tests for the SignalRouter implementation.
"""
import pytest
from typing import Dict, Any
from core.signal_router import DefaultSignalRouter
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
        'symbol': 'BTC/USDT',
        'close': [50000.0, 50100.0, 50200.0],
        'volume': [100.0, 110.0, 120.0],
        'timestamp': [1000, 2000, 3000]
    }

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'strategy_weights': {},
        'hold_threshold': 0.1
    }

@pytest.fixture
def signal_router(config):
    """Create a signal router instance."""
    return DefaultSignalRouter(config)

@pytest.mark.asyncio
async def test_empty_router(signal_router, market_data):
    """Test router behavior with no strategies."""
    signals = await signal_router.get_signals(market_data)
    assert len(signals) == 0

@pytest.mark.asyncio
async def test_single_strategy(signal_router, market_data):
    """Test router with a single strategy."""
    strategy = MockStrategy('buy', 0.8)
    signal_router.add_strategy(strategy)
    
    signals = await signal_router.get_signals(market_data)
    assert len(signals) == 1
    assert signals[0]['action'] == 'buy'
    assert signals[0]['confidence'] == 0.8

@pytest.mark.asyncio
async def test_multiple_strategies(signal_router, market_data):
    """Test router with multiple strategies."""
    # Add strategies with different signals
    signal_router.add_strategy(MockStrategy('buy', 0.8))
    signal_router.add_strategy(MockStrategy('sell', 0.6))
    
    signals = await signal_router.get_signals(market_data)
    assert len(signals) == 1  # Should combine into single signal
    assert signals[0]['action'] in ['buy', 'sell', 'hold']  # Actual action depends on combiner logic

@pytest.mark.asyncio
async def test_strategy_removal(signal_router, market_data):
    """Test removing strategies."""
    strategy = MockStrategy('buy', 0.8)
    signal_router.add_strategy(strategy)
    
    # Verify strategy is added
    signals = await signal_router.get_signals(market_data)
    assert len(signals) == 1
    
    # Remove strategy
    signal_router.remove_strategy(strategy.__class__.__name__)
    
    # Verify no signals after removal
    signals = await signal_router.get_signals(market_data)
    assert len(signals) == 0

@pytest.mark.asyncio
async def test_duplicate_strategy(signal_router, market_data):
    """Test adding duplicate strategy."""
    strategy1 = MockStrategy('buy', 0.8)
    strategy2 = MockStrategy('sell', 0.9)
    
    signal_router.add_strategy(strategy1)
    signal_router.add_strategy(strategy2)
    
    # Add duplicate of strategy1
    new_strategy1 = MockStrategy('buy', 0.7)
    signal_router.add_strategy(new_strategy1)
    
    signals = await signal_router.get_signals(market_data)
    assert len(signals) == 1  # Should still combine into single signal

@pytest.mark.asyncio
async def test_error_handling(signal_router, market_data):
    """Test error handling in signal generation."""
    class ErrorStrategy(BaseStrategy):
        def __init__(self):
            super().__init__({'name': 'error_strategy'})
            
        def generate_signal(self, market_data):
            raise Exception("Test error")
    
    signal_router.add_strategy(ErrorStrategy())
    signals = await signal_router.get_signals(market_data)
    assert len(signals) == 0  # Should handle error gracefully 