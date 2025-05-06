"""
Tests for the configuration schema validation.
"""
import pytest
from config.schema import validate_config, ConfigError

@pytest.fixture
def valid_config():
    """Create a valid configuration dictionary."""
    return {
        'exchange': {
            'name': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        },
        'trading': {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'timeframe': '1h',
            'max_position_size': 1000,
            'max_daily_trades': 5
        },
        'risk': {
            'max_drawdown': 0.2,
            'stop_loss': 0.05,
            'take_profit': 0.1
        },
        'model': {
            'type': 'lstm',
            'input_dim': 10,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2
        }
    }

@pytest.fixture
def invalid_config():
    """Create an invalid configuration dictionary."""
    return {
        'exchange': {
            'name': 'invalid_exchange'
        },
        'trading': {
            'symbols': 'BTCUSDT',  # Should be a list
            'timeframe': 'invalid_timeframe'
        }
    }

def test_valid_configuration(valid_config):
    """Test validation of valid configuration."""
    assert validate_config(valid_config) is True

def test_invalid_configuration(invalid_config):
    """Test validation of invalid configuration."""
    with pytest.raises(ConfigError) as exc_info:
        validate_config(invalid_config)
    assert "Invalid configuration" in str(exc_info.value)

def test_missing_required_fields():
    """Test validation with missing required fields."""
    config = {
        'exchange': {
            'name': 'binance'
        }
    }
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)
    assert "Missing required fields" in str(exc_info.value)

def test_invalid_exchange_name():
    """Test validation of invalid exchange name."""
    config = {
        'exchange': {
            'name': 'invalid_exchange',
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        },
        'trading': {
            'symbols': ['BTCUSDT'],
            'timeframe': '1h'
        }
    }
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)
    assert "Invalid exchange name" in str(exc_info.value)

def test_invalid_timeframe():
    """Test validation of invalid timeframe."""
    config = {
        'exchange': {
            'name': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        },
        'trading': {
            'symbols': ['BTCUSDT'],
            'timeframe': 'invalid_timeframe'
        }
    }
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)
    assert "Invalid timeframe" in str(exc_info.value)

def test_invalid_risk_parameters():
    """Test validation of invalid risk parameters."""
    config = {
        'exchange': {
            'name': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        },
        'trading': {
            'symbols': ['BTCUSDT'],
            'timeframe': '1h'
        },
        'risk': {
            'max_drawdown': 2.0,  # Should be between 0 and 1
            'stop_loss': -0.05    # Should be positive
        }
    }
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)
    assert "Invalid risk parameters" in str(exc_info.value)

def test_invalid_model_parameters():
    """Test validation of invalid model parameters."""
    config = {
        'exchange': {
            'name': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret'
        },
        'trading': {
            'symbols': ['BTCUSDT'],
            'timeframe': '1h'
        },
        'model': {
            'type': 'invalid_model',
            'input_dim': -10,  # Should be positive
            'hidden_dim': 0    # Should be positive
        }
    }
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)
    assert "Invalid model parameters" in str(exc_info.value) 