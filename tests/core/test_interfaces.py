"""
Tests for the core interfaces.
"""
import pytest
from abc import ABC, abstractmethod
from core.interfaces import Strategy, Model, Exchange

class MockStrategy(Strategy):
    """Mock strategy implementation for testing."""
    def __init__(self):
        self.name = "mock_strategy"
    
    def generate_signal(self, data):
        return {'action': 'buy', 'confidence': 0.8}
    
    def update(self, data):
        pass

class MockModel(Model):
    """Mock model implementation for testing."""
    def __init__(self):
        self.name = "mock_model"
    
    def predict(self, data):
        return 0.5
    
    def train(self, data):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass

class MockExchange(Exchange):
    """Mock exchange implementation for testing."""
    def __init__(self):
        self.name = "mock_exchange"
    
    def get_market_data(self, symbol, timeframe):
        return {'price': 50000, 'volume': 1000}
    
    def place_order(self, order):
        return {'order_id': '123', 'status': 'filled'}
    
    def get_balance(self):
        return {'BTC': 1.0, 'USDT': 50000}

@pytest.fixture
def mock_strategy():
    """Create a mock strategy instance."""
    return MockStrategy()

@pytest.fixture
def mock_model():
    """Create a mock model instance."""
    return MockModel()

@pytest.fixture
def mock_exchange():
    """Create a mock exchange instance."""
    return MockExchange()

def test_strategy_interface(mock_strategy):
    """Test strategy interface implementation."""
    assert isinstance(mock_strategy, Strategy)
    assert mock_strategy.name == "mock_strategy"
    
    # Test signal generation
    signal = mock_strategy.generate_signal({'price': 50000})
    assert isinstance(signal, dict)
    assert 'action' in signal
    assert 'confidence' in signal

def test_model_interface(mock_model):
    """Test model interface implementation."""
    assert isinstance(mock_model, Model)
    assert mock_model.name == "mock_model"
    
    # Test prediction
    prediction = mock_model.predict([1, 2, 3])
    assert isinstance(prediction, (int, float))
    
    # Test save/load
    mock_model.save("test_model.pkl")
    mock_model.load("test_model.pkl")

def test_exchange_interface(mock_exchange):
    """Test exchange interface implementation."""
    assert isinstance(mock_exchange, Exchange)
    assert mock_exchange.name == "mock_exchange"
    
    # Test market data
    market_data = mock_exchange.get_market_data("BTCUSDT", "1h")
    assert isinstance(market_data, dict)
    assert 'price' in market_data
    assert 'volume' in market_data
    
    # Test order placement
    order = {'symbol': 'BTCUSDT', 'side': 'buy', 'quantity': 0.1}
    order_result = mock_exchange.place_order(order)
    assert isinstance(order_result, dict)
    assert 'order_id' in order_result
    assert 'status' in order_result
    
    # Test balance
    balance = mock_exchange.get_balance()
    assert isinstance(balance, dict)
    assert 'BTC' in balance
    assert 'USDT' in balance

def test_invalid_strategy_implementation():
    """Test invalid strategy implementation."""
    class InvalidStrategy(Strategy):
        pass
    
    with pytest.raises(TypeError):
        InvalidStrategy()

def test_invalid_model_implementation():
    """Test invalid model implementation."""
    class InvalidModel(Model):
        pass
    
    with pytest.raises(TypeError):
        InvalidModel()

def test_invalid_exchange_implementation():
    """Test invalid exchange implementation."""
    class InvalidExchange(Exchange):
        pass
    
    with pytest.raises(TypeError):
        InvalidExchange() 