import pytest
import torch
import numpy as np
import pandas as pd
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
            "input_size": 13,
            "hidden_size": 64,
            "num_layers": 2
        },
        "data": {
            "lookback_window": 60,
            "feature_mean": [
                102.0,  # close
                0.0,    # returns
                102000.0,  # volume_price
                1.0,    # ema_ratio
                102.0,  # ema_fast
                102.0,  # ema_slow
                50.0,   # rsi
                0.0,    # macd_line
                0.0,    # signal_line
                0.0,    # macd_hist
                0.02,   # bb_width
                0.5,    # bb_position
                0.0     # bb_pct
            ],
            "feature_std": [
                10.0,   # close
                0.01,   # returns
                10000.0,  # volume_price
                0.1,    # ema_ratio
                10.0,   # ema_fast
                10.0,   # ema_slow
                10.0,   # rsi
                0.1,    # macd_line
                0.1,    # signal_line
                0.1,    # macd_hist
                0.01,   # bb_width
                0.2,    # bb_position
                0.2     # bb_pct
            ]
        },
        "inference": {
            "confidence_threshold": 0.85
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
        # Create sample market data with enough history for indicators
        market_data = {
            "open": np.array([100.0] * 50),
            "high": np.array([105.0] * 50),
            "low": np.array([95.0] * 50),
            "close": np.array([102.0] * 50),
            "volume": np.array([1000.0] * 50)
        }
        
        features = live_inference._prepare_features(market_data)
        
        # Check tensor shape: (batch_size=1, seq_len=1, num_features=13)
        assert features.shape == (1, 1, 13)
        assert isinstance(features, torch.Tensor)
        
    def test_feature_normalization(self, live_inference):
        """Test that features are properly normalized."""
        market_data = {
            "open": np.array([100.0] * 50),
            "high": np.array([105.0] * 50),
            "low": np.array([95.0] * 50),
            "close": np.array([102.0] * 50),
            "volume": np.array([1000.0] * 50)
        }
        
        features = live_inference._prepare_features(market_data)
        features_np = features.cpu().numpy().squeeze()
        
        # Check that each feature is normalized
        for i, (mean, std) in enumerate(zip(
            live_inference.config["data"]["feature_mean"],
            live_inference.config["data"]["feature_std"]
        )):
            # Get the raw feature value
            raw_value = live_inference.last_raw_features[list(live_inference.last_raw_features.keys())[i]]
            
            # Calculate expected normalized value
            expected_normalized = (raw_value - mean) / std
            
            # Check if normalization was applied correctly
            assert np.allclose(features_np[i], expected_normalized, atol=1e-6), \
                f"Feature {i} not normalized correctly. Expected {expected_normalized}, got {features_np[i]}"
        
    def test_technical_indicators(self, live_inference):
        """Test that all technical indicators are calculated correctly."""
        # Create sample market data with enough history for indicators
        market_data = {
            "open": np.array([100.0] * 50),
            "high": np.array([105.0] * 50),
            "low": np.array([95.0] * 50),
            "close": np.array([102.0] * 50),
            "volume": np.array([1000.0] * 50)
        }
        
        features = live_inference._prepare_features(market_data)
        features_np = features.cpu().numpy().squeeze()
        
        # Check that we have all expected features
        assert features_np.shape == (13,)  # Total number of features
        
        # Check that no features are NaN or infinite
        assert not np.any(np.isnan(features_np))
        assert not np.any(np.isinf(features_np))
        
    def test_macd_calculation(self, live_inference):
        """Test MACD calculation specifically."""
        # Create sample market data with enough history for MACD
        market_data = {
            "open": np.array([100.0] * 50),
            "high": np.array([105.0] * 50),
            "low": np.array([95.0] * 50),
            "close": np.array([102.0] * 50),
            "volume": np.array([1000.0] * 50)
        }
        
        features = live_inference._prepare_features(market_data)
        features_np = features.cpu().numpy().squeeze()
        
        # MACD features should be at indices 5, 6, 7
        macd_line = features_np[5]
        signal_line = features_np[6]
        macd_hist = features_np[7]
        
        # Check that MACD components are calculated
        assert not np.isnan(macd_line)
        assert not np.isnan(signal_line)
        assert not np.isnan(macd_hist)
        
    def test_bollinger_bands(self, live_inference):
        """Test Bollinger Bands calculation specifically."""
        # Create sample market data with price variation
        close_prices = np.array([100.0 + i * 0.1 for i in range(50)])  # Upward trend
        high_prices = close_prices + 2.0  # 2 points above close
        low_prices = close_prices - 2.0   # 2 points below close
        
        market_data = {
            "open": close_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": np.array([1000.0] * 50)
        }
        
        features = live_inference._prepare_features(market_data)
        
        # Get raw values before normalization
        bb_width = live_inference.last_raw_features['bb_width']
        bb_position = live_inference.last_raw_features['bb_position']
        
        # Check that BB components are calculated
        assert not np.isnan(bb_width)
        assert not np.isnan(bb_position)
        
        # BB width should be positive
        assert bb_width > 0
        
        # BB position should be between 0 and 1
        assert 0 <= bb_position <= 1
        
    def test_invalid_market_data(self, live_inference):
        """Test handling of invalid market data."""
        with pytest.raises(Exception):
            live_inference._prepare_features({})  # Empty market data
            
        with pytest.raises(Exception):
            live_inference._prepare_features({
                "close": np.array([100.0])  # Missing required fields
            })

class TestInterpretPrediction:
    def test_high_confidence_long(self, live_inference):
        """Test interpretation of high confidence long prediction."""
        # Create prediction tensor with high confidence for long
        prediction = torch.tensor([[0.86, 0.07, 0.07]], dtype=torch.float32)
        live_inference.last_prediction = prediction
        signal = live_inference._interpret_prediction(prediction)
        
        assert signal == SignalType.LONG
        
    def test_high_confidence_short(self, live_inference):
        """Test interpretation of high confidence short prediction."""
        prediction = torch.tensor([[0.07, 0.86, 0.07]], dtype=torch.float32)
        live_inference.last_prediction = prediction
        signal = live_inference._interpret_prediction(prediction)
        
        assert signal == SignalType.SHORT
        
    def test_low_confidence(self, live_inference):
        """Test that low confidence predictions return neutral."""
        # Create prediction tensor with low confidence
        prediction = torch.tensor([[0.84, 0.08, 0.08]], dtype=torch.float32)
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
        
    def test_prediction_edge_cases(self, live_inference):
        """Test prediction interpretation with edge cases."""
        # Test equal probabilities
        prediction = torch.tensor([[0.33, 0.33, 0.34]], dtype=torch.float32)
        signal = live_inference._interpret_prediction(prediction)
        assert signal == SignalType.NEUTRAL
        
        # Test very high confidence
        prediction = torch.tensor([[0.99, 0.005, 0.005]], dtype=torch.float32)
        signal = live_inference._interpret_prediction(prediction)
        assert signal == SignalType.LONG
        
        # Test exactly at threshold
        prediction = torch.tensor([[0.85, 0.075, 0.075]], dtype=torch.float32)
        signal = live_inference._interpret_prediction(prediction)
        assert signal == SignalType.LONG

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

class TestTradeSpacing:
    def test_trade_spacing_enforcement(self, live_inference, monkeypatch):
        """Test that trade spacing is enforced."""
        # Mock _prepare_features to return valid tensor
        def mock_prepare_features(*args):
            return torch.tensor([[[0.8, 0.1, 0.1] + [0.0] * 10]], dtype=torch.float32)
            
        monkeypatch.setattr(live_inference, '_prepare_features', mock_prepare_features)
        
        # Mock model to return high confidence prediction
        def mock_forward(*args):
            return torch.tensor([[0.9, 0.05, 0.05]], dtype=torch.float32)
            
        monkeypatch.setattr(live_inference.model, 'forward', mock_forward)
        
        # Mock trend and RSI alignment to always return True
        def mock_trend_aligned(*args):
            return True
            
        def mock_rsi_aligned(*args):
            return True
            
        monkeypatch.setattr(live_inference, 'is_trend_aligned', mock_trend_aligned)
        monkeypatch.setattr(live_inference, 'is_rsi_aligned', mock_rsi_aligned)
        
        # Create market data with enough history for lookback window
        n_samples = live_inference.config["data"]["lookback_window"] + 10  # Add some extra samples
        market_data = {
            "open": np.array([100.0] * n_samples),
            "high": np.array([105.0] * n_samples),
            "low": np.array([95.0] * n_samples),
            "close": np.array([102.0] * n_samples),
            "volume": np.array([1000.0] * n_samples)
        }
        
        # Initialize price history with enough data
        live_inference.price_history = {
            "high": [105.0] * n_samples,
            "low": [95.0] * n_samples,
            "close": [102.0] * n_samples
        }
        
        # Reset base filter and candle index
        live_inference.base_filter.reset()
        live_inference.current_candle_index = -1  # Start at -1 so first increment makes it 0
        
        # First trade should be allowed
        signal = live_inference.process_market_data(market_data)
        assert signal == SignalType.LONG, "First trade should be allowed"
        assert live_inference.base_filter.last_trade_index == 0, "Last trade index should be updated after first trade"
        
        # Process candles until we reach the minimum spacing
        spacing = live_inference.base_filter.min_trade_spacing
        for i in range(spacing):  # Process exactly spacing candles
            signal = live_inference.process_market_data(market_data)
            assert signal is None, f"Trade should be blocked during spacing period"
            
        # After spacing period, trade should be allowed
        signal = live_inference.process_market_data(market_data)
        assert signal == SignalType.LONG, "Trade should be allowed after spacing period"
        
        # Verify that the last trade index was updated
        assert live_inference.base_filter.last_trade_index == live_inference.current_candle_index, "Last trade index should be updated after successful trade"
        
        # Verify that the spacing requirement was met
        current_spacing = live_inference.current_candle_index - live_inference.base_filter.last_trade_index
        assert current_spacing == 0, f"Spacing should be 0 after trade execution, got {current_spacing}"
        
    def test_trade_spacing_with_neutral_signals(self, live_inference, monkeypatch):
        """Test that neutral signals don't affect trade spacing."""
        # Mock _prepare_features to return valid tensor
        def mock_prepare_features(*args):
            return torch.tensor([[[0.33, 0.33, 0.34] + [0.0] * 10]], dtype=torch.float32)
            
        monkeypatch.setattr(live_inference, '_prepare_features', mock_prepare_features)
        
        # Mock trend and RSI alignment to always return True
        def mock_trend_aligned(*args):
            return True
            
        def mock_rsi_aligned(*args):
            return True
            
        monkeypatch.setattr(live_inference, 'is_trend_aligned', mock_trend_aligned)
        monkeypatch.setattr(live_inference, 'is_rsi_aligned', mock_rsi_aligned)
        
        # Create market data with enough history for lookback window
        n_samples = live_inference.config["data"]["lookback_window"] + 10  # Add some extra samples
        market_data = {
            "open": np.array([100.0] * n_samples),
            "high": np.array([105.0] * n_samples),
            "low": np.array([95.0] * n_samples),
            "close": np.array([102.0] * n_samples),
            "volume": np.array([1000.0] * n_samples)
        }
        
        # Initialize price history with enough data
        live_inference.price_history = {
            "high": [105.0] * n_samples,
            "low": [95.0] * n_samples,
            "close": [102.0] * n_samples
        }
        
        # Neutral signals should not affect spacing
        for _ in range(3):
            signal = live_inference.process_market_data(market_data)
            assert signal == SignalType.NEUTRAL
            
        # Mock high confidence prediction
        def mock_forward(*args):
            return torch.tensor([[0.9, 0.05, 0.05]], dtype=torch.float32)
            
        monkeypatch.setattr(live_inference.model, 'forward', mock_forward)
        
        # Should still be able to trade
        signal = live_inference.process_market_data(market_data)
        assert signal == SignalType.LONG 