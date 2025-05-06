"""
Tests for the AI-based trading strategy.
"""
import pytest
import numpy as np
from typing import Dict, Any
from strategy.ai_strategy import AIStrategy

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
        'close': [100, 101, 102, 103, 104],
        'timestamp': [1000, 1001, 1002, 1003, 1004]
    }

@pytest.fixture
def ai_strategy():
    """Create an AI strategy instance."""
    return AIStrategy({
        'model_path': 'models/trained/mock_model.h5',
        'prediction_threshold': 0.6
    })

def test_buy_signal(ai_strategy, market_data):
    """Test generating a buy signal."""
    ai_strategy.model = MockModel(0.8)  # Strong buy signal
    signal = ai_strategy.generate_signal(market_data)
    assert signal['action'] == 'buy'
    assert signal['confidence'] == 0.8
    assert 'prediction' in signal

def test_sell_signal(ai_strategy, market_data):
    """Test generating a sell signal."""
    ai_strategy.model = MockModel(-0.8)  # Strong sell signal
    signal = ai_strategy.generate_signal(market_data)
    assert signal['action'] == 'sell'
    assert signal['confidence'] == 0.8
    assert 'prediction' in signal

def test_hold_signal(ai_strategy, market_data):
    """Test generating a hold signal."""
    ai_strategy.model = MockModel(0.3)  # Weak signal
    signal = ai_strategy.generate_signal(market_data)
    assert signal['action'] == 'hold'
    assert signal['confidence'] == 0.3
    assert 'prediction' in signal

def test_no_model(ai_strategy, market_data):
    """Test behavior when model is not loaded."""
    ai_strategy.model = None
    signal = ai_strategy.generate_signal(market_data)
    assert signal['action'] == 'hold'
    assert 'error' in signal

def test_model_error(ai_strategy, market_data):
    """Test handling of model prediction errors."""
    class ErrorModel:
        def predict(self, data):
            raise Exception("Model error")

    ai_strategy.model = ErrorModel()
    signal = ai_strategy.generate_signal(market_data)
    assert signal['action'] == 'hold'
    assert 'error' in signal
    assert 'Model error' in signal['error'] 