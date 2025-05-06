"""
Integration tests for the complete trading flow.
"""
import pytest
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

from strategy.signal_combiner import SignalCombiner
from strategy.ai_strategy import AIStrategy
from risk.exposure import ExposureManager
from core.interfaces import Exchange
from utils.logger import setup_logger

class MockExchange(Exchange):
    """Mock exchange for testing the complete flow."""
    def __init__(self):
        self.name = "mock_exchange"
        self.orders = []
        self.positions = {}
        self.balance = {'USDT': 10000.0}
    
    def get_market_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        return {
            'symbol': symbol,
            'close': [50000.0, 50100.0, 50200.0],
            'volume': [100.0, 110.0, 120.0],
            'timestamp': [
                int((datetime.now() - timedelta(minutes=i)).timestamp() * 1000)
                for i in range(3)
            ]
        }
    
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        self.orders.append(order)
        order_id = f"order_{len(self.orders)}"
        
        # Simulate order execution
        if order['side'] == 'buy':
            self.positions[order['symbol']] = {
                'size': order['quantity'],
                'entry_price': order['price'],
                'side': 'long'
            }
            self.balance['USDT'] -= order['quantity'] * order['price']
        else:
            if order['symbol'] in self.positions:
                del self.positions[order['symbol']]
                self.balance['USDT'] += order['quantity'] * order['price']
        
        return {
            'order_id': order_id,
            'status': 'filled',
            'filled_price': order['price'],
            'filled_quantity': order['quantity']
        }
    
    def get_balance(self) -> Dict[str, float]:
        return self.balance

class MockModel:
    """Mock ML model for testing."""
    def __init__(self, prediction: float):
        self.prediction = prediction
    
    def predict(self, data: Dict[str, Any]) -> float:
        return self.prediction

@pytest.fixture
def market_data():
    """Sample market data for testing."""
    return {
        'symbol': 'BTCUSDT',
        'close': [50000.0, 50100.0, 50200.0],
        'volume': [100.0, 110.0, 120.0],
        'timestamp': [
            int((datetime.now() - timedelta(minutes=i)).timestamp() * 1000)
            for i in range(3)
        ]
    }

@pytest.fixture
def exchange():
    """Create a mock exchange instance."""
    return MockExchange()

@pytest.fixture
def exposure_manager():
    """Create an exposure manager instance."""
    return ExposureManager({
        'max_position_size': 100000,  # 100,000 USDT
        'max_daily_trades': 5,
        'max_drawdown': 0.2
    })

@pytest.fixture
def signal_combiner():
    """Create a signal combiner instance."""
    return SignalCombiner({'strategy_weights': {}})

@pytest.fixture
def ai_strategy():
    """Create an AI strategy instance."""
    return AIStrategy({
        'model_path': 'models/trained/mock_model.h5',
        'prediction_threshold': 0.6
    })

@pytest.fixture
def logger(tmp_path):
    """Create a logger instance."""
    return setup_logger('test_trade_flow', str(tmp_path))

def test_complete_trade_flow(
    exchange,
    exposure_manager,
    signal_combiner,
    ai_strategy,
    market_data,
    logger
):
    """Test the complete trading flow from signal to execution."""
    # Setup AI strategy with mock model
    ai_strategy.model = MockModel(0.8)  # Strong buy signal
    signal_combiner.add_strategy(ai_strategy, weight=1.0)
    
    # Generate trading signal
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'buy'
    assert signal['confidence'] > 0.6
    
    # Check risk exposure
    position = {
        'symbol': market_data['symbol'],
        'size': 0.1,  # 0.1 BTC
        'entry_price': market_data['close'][-1],
        'side': 'long'
    }
    risk_check = exposure_manager.evaluate_risk(position)
    assert risk_check['allowed'] is True
    
    # Place order if risk check passes
    if risk_check['allowed']:
        order = {
            'symbol': market_data['symbol'],
            'side': signal['action'],
            'quantity': position['size'],
            'price': market_data['close'][-1]
        }
        order_result = exchange.place_order(order)
        
        # Verify order execution
        assert order_result['status'] == 'filled'
        assert order_result['filled_price'] == market_data['close'][-1]
        assert order_result['filled_quantity'] == position['size']
        
        # Verify position and balance updates
        assert market_data['symbol'] in exchange.positions
        assert exchange.positions[market_data['symbol']]['size'] == position['size']
        assert exchange.balance['USDT'] < 10000.0  # Balance should be reduced

def test_risk_rejection_flow(
    exchange,
    exposure_manager,
    signal_combiner,
    ai_strategy,
    market_data,
    logger
):
    """Test the flow when risk check rejects the trade."""
    # Setup AI strategy with mock model
    ai_strategy.model = MockModel(0.8)  # Strong buy signal
    signal_combiner.add_strategy(ai_strategy, weight=1.0)
    
    # Generate trading signal
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'buy'
    
    # Simulate exceeding position size limit
    position = {
        'symbol': market_data['symbol'],
        'size': 3.0,  # 3 BTC at ~50,000 USDT = 150,000 USDT (exceeds max_position_size)
        'entry_price': market_data['close'][-1],
        'side': 'long'
    }
    risk_check = exposure_manager.evaluate_risk(position)
    assert risk_check['allowed'] is False
    
    # Verify no order is placed
    initial_balance = exchange.balance['USDT']
    assert market_data['symbol'] not in exchange.positions
    assert len(exchange.orders) == 0
    assert exchange.balance['USDT'] == initial_balance

def test_conflicting_signals_flow(
    exchange,
    exposure_manager,
    signal_combiner,
    ai_strategy,
    market_data,
    logger
):
    """Test the flow with conflicting signals from different strategies."""
    # Setup AI strategy with mock model
    ai_strategy.model = MockModel(0.8)  # Strong buy signal
    signal_combiner.add_strategy(ai_strategy, weight=1.0)
    
    # Add another strategy with sell signal
    class MockSellStrategy:
        def generate_signal(self, data):
            return {'action': 'sell', 'confidence': 0.9}
    
    signal_combiner.add_strategy(MockSellStrategy(), weight=1.0)
    
    # Generate trading signal
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'hold'  # Should default to hold with conflicting signals
    
    # Verify no order is placed
    initial_balance = exchange.balance['USDT']
    assert market_data['symbol'] not in exchange.positions
    assert len(exchange.orders) == 0
    assert exchange.balance['USDT'] == initial_balance

def test_error_handling_flow(
    exchange,
    exposure_manager,
    signal_combiner,
    ai_strategy,
    market_data,
    logger
):
    """Test error handling in the trading flow."""
    # Setup AI strategy without model
    ai_strategy.model = None
    signal_combiner.add_strategy(ai_strategy, weight=1.0)
    
    # Generate trading signal
    signal = signal_combiner.generate_signal(market_data)
    assert signal['action'] == 'hold'
    assert 'error' in signal
    
    # Verify no order is placed
    initial_balance = exchange.balance['USDT']
    assert market_data['symbol'] not in exchange.positions
    assert len(exchange.orders) == 0
    assert exchange.balance['USDT'] == initial_balance 