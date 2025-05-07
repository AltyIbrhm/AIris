import pytest
import torch
import numpy as np
from ml.inference.live_inference import LiveInference
from utils.enums import SignalType

class MockMarketDataHandler:
    def __init__(self):
        self.data = {
            "open": np.array([100.0]),
            "high": np.array([105.0]),
            "low": np.array([95.0]),
            "close": np.array([102.0]),
            "volume": np.array([1000.0])
        }
    
    def get_latest_data(self):
        return self.data

class MockPortfolio:
    def __init__(self):
        self.account_value = 10000.0
        self.current_position = None
    
    def get_account_value(self):
        return self.account_value
    
    def open_position(self, side, size, entry_price, stop_loss, take_profit):
        self.current_position = {
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        return self.current_position
    
    def close_position(self, current_price):
        pnl = 0.0  # Mock PnL calculation
        self.current_position = None
        return pnl

class MockPerformanceTracker:
    def __init__(self, config):
        self.metrics = {}
    
    def update_metrics(self, pnl):
        pass
    
    def log_final_metrics(self):
        pass

class MockLSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 3)
        
    def forward(self, x):
        return self.linear(x.squeeze(1))

@pytest.fixture
def mock_config():
    return {
        "model": {
            "input_size": 8,
            "hidden_size": 64,
            "num_layers": 2
        },
        "data": {
            "lookback_window": 60,
            "feature_mean": [100.0, 10.0, 100.0, 0.0, 1000.0, 100000.0, 2.0, 50.0],
            "feature_std": [10.0, 2.0, 10.0, 0.01, 200.0, 20000.0, 0.5, 10.0]
        },
        "inference": {
            "confidence_threshold": 0.7
        }
    }

@pytest.fixture
def mock_risk_config():
    return {
        "trailing_stop": {
            "activation_threshold": 0.02,
            "trailing_distance": 0.01
        },
        "hard_stop": {
            "max_loss": 0.05
        },
        "position_sizing": {
            "risk_per_trade": 0.02,
            "atr_multiplier": 2.0,
            "risk_reward_ratio": 2.0,
            "min_confidence": 0.7,
            "max_confidence": 0.95,
            "min_risk_per_trade": 0.01,
            "max_risk_per_trade": 0.05,
            "max_holding_periods": 20
        }
    }

@pytest.fixture
def live_inference(mock_config, mock_risk_config, monkeypatch):
    # Mock the model loading and other dependencies
    def mock_load_model(*args, **kwargs):
        return MockLSTMClassifier(
            input_dim=mock_config["model"]["input_size"],
            hidden_dim=mock_config["model"]["hidden_size"],
            num_layers=mock_config["model"]["num_layers"]
        )
    
    def mock_load_config(*args, **kwargs):
        return mock_config
    
    def mock_load_risk_config(*args, **kwargs):
        return mock_risk_config
    
    monkeypatch.setattr(LiveInference, "_load_model", mock_load_model)
    monkeypatch.setattr(LiveInference, "_load_config", mock_load_config)
    monkeypatch.setattr(LiveInference, "_load_risk_config", mock_load_risk_config)
    
    instance = LiveInference("dummy_model_path")
    instance.market_data = MockMarketDataHandler()
    instance.portfolio = MockPortfolio()
    instance.performance_tracker = MockPerformanceTracker(mock_config)
    
    return instance

class TestPrepareFeatures:
    def test_feature_preparation_shape(self, live_inference):
        """Test that feature preparation returns correct tensor shape."""
        market_data = {
            "open": np.array([100.0]),
            "high": np.array([105.0]),
            "low": np.array([95.0]),
            "close": np.array([102.0]),
            "volume": np.array([1000.0])
        }
        
        live_inference.atr = 2.0  # Set ATR for testing
        features = live_inference._prepare_features(market_data)
        
        # Check tensor shape: (batch_size=1, seq_len=1, num_features=5)
        assert features.shape == (1, 1, 5)
        assert isinstance(features, torch.Tensor)
        
    def test_feature_normalization(self, live_inference):
        """Test that features are properly normalized."""
        market_data = {
            "open": np.array([100.0]),
            "high": np.array([105.0]),
            "low": np.array([95.0]),
            "close": np.array([102.0]),
            "volume": np.array([1000.0])
        }
        
        live_inference.atr = 2.0
        features = live_inference._prepare_features(market_data)
        
        # Check that features are normalized (mean close to 0, std close to 1)
        features_np = features.cpu().numpy()
        assert np.allclose(features_np.mean(), 0.0, atol=1e-6)
        assert np.allclose(features_np.std(), 1.0, atol=1e-6)
        
    def test_missing_atr(self, live_inference):
        """Test feature preparation when ATR is not available."""
        market_data = {
            "open": np.array([100.0]),
            "high": np.array([105.0]),
            "low": np.array([95.0]),
            "close": np.array([102.0]),
            "volume": np.array([1000.0])
        }
        
        live_inference.atr = None
        features = live_inference._prepare_features(market_data)
        
        # Should still work with zero padding for ATR features
        assert features.shape == (1, 1, 5)  # 5 features with zero padding for ATR
        
    def test_invalid_market_data(self, live_inference):
        """Test handling of invalid market data."""
        with pytest.raises(Exception):
            live_inference._prepare_features({})  # Empty market data

class TestInterpretPrediction:
    def test_high_confidence_long(self, live_inference):
        """Test interpretation of high confidence long prediction."""
        # Create prediction tensor with high confidence for long
        prediction = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32)
        live_inference.last_prediction = prediction
        signal = live_inference._interpret_prediction(prediction)
        
        assert signal == SignalType.LONG
        
    def test_high_confidence_short(self, live_inference):
        """Test interpretation of high confidence short prediction."""
        prediction = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32)
        live_inference.last_prediction = prediction
        signal = live_inference._interpret_prediction(prediction)
        
        assert signal == SignalType.SHORT
        
    def test_low_confidence(self, live_inference):
        """Test that low confidence predictions return neutral."""
        prediction = torch.tensor([[0.4, 0.3, 0.3]], dtype=torch.float32)
        live_inference.last_prediction = prediction
        signal = live_inference._interpret_prediction(prediction)
        
        assert signal == SignalType.NEUTRAL
        
    def test_invalid_prediction_shape(self, live_inference):
        """Test handling of invalid prediction shape."""
        prediction = torch.tensor([0.1, 0.1, 0.8], dtype=torch.float32)
        live_inference.last_prediction = prediction
        signal = live_inference._interpret_prediction(prediction)
        
        # Should handle error gracefully and return neutral
        assert signal == SignalType.NEUTRAL

class TestConfidenceBasedPositionSizing:
    def test_position_size_scaling(self, live_inference):
        """Test that position size scales with confidence."""
        # Test different confidence levels
        confidences = [0.7, 0.8, 0.9, 0.95]
        base_size = 0.02  # 2% base risk
        
        for conf in confidences:
            # Create prediction with specific confidence
            prediction = torch.tensor([[conf, 0.1, 0.1]], dtype=torch.float32)
            live_inference.last_prediction = prediction
            
            # Calculate position size
            position_size = live_inference.calculate_position_size(SignalType.LONG)
            
            # Position size should scale with confidence
            expected_size = base_size * (1.0 + (conf - 0.7) / (0.95 - 0.7))
            assert abs(position_size - expected_size) < 1e-6
            
    def test_min_max_position_size(self, live_inference):
        """Test that position size stays within configured limits."""
        # Test minimum confidence
        prediction = torch.tensor([[0.7, 0.1, 0.1]], dtype=torch.float32)
        live_inference.last_prediction = prediction
        min_size = live_inference.calculate_position_size(SignalType.LONG)
        assert min_size >= live_inference.risk_config["position_sizing"]["min_risk_per_trade"]
        
        # Test maximum confidence
        prediction = torch.tensor([[0.95, 0.1, 0.1]], dtype=torch.float32)
        live_inference.last_prediction = prediction
        max_size = live_inference.calculate_position_size(SignalType.LONG)
        assert max_size <= live_inference.risk_config["position_sizing"]["max_risk_per_trade"] 