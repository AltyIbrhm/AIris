"""
Tests for the configuration schema validation.
"""
import pytest
from config.schema import Config, ConfigError

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
            'exchange': 'binance',
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'interval': '1h',
            'poll_interval': 60,
            'risk_config': {
                'min_confidence': 0.3,
                'max_open_positions_total': 3,
                'max_open_positions_per_symbol': 1,
                'max_drawdown_percent': 10.0,
                'max_daily_loss': 300.0,
                'default_sl_percent': 2.0,
                'default_tp_percent': 4.0,
                'duplicate_signal_block_minutes': 5,
                'max_position_size_percent': 10.0,
                'max_leverage': 1.0,
                'risk_free_rate': 0.02,
                'volatility_lookback': 20,
                'position_sizing_method': 'kelly',
                'emergency_stop_loss_percent': 5.0
            },
            'ai_model_config': {
                'model_type': 'lstm',
                'input_features': ['open', 'high', 'low', 'close', 'volume'],
                'output_features': ['direction', 'confidence'],
                'sequence_length': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'validation_split': 0.2
            },
            'paper_trading': True,
            'log_level': 'INFO'
        },
        'risk': {
            'min_confidence': 0.3,
            'max_open_positions_total': 3,
            'max_open_positions_per_symbol': 1,
            'max_drawdown_percent': 10.0,
            'max_daily_loss': 300.0,
            'default_sl_percent': 2.0,
            'default_tp_percent': 4.0,
            'duplicate_signal_block_minutes': 5,
            'max_position_size_percent': 10.0,
            'max_leverage': 1.0,
            'risk_free_rate': 0.02,
            'volatility_lookback': 20,
            'position_sizing_method': 'kelly',
            'emergency_stop_loss_percent': 5.0
        },
        'model': {
            'model_type': 'lstm',
            'input_features': ['open', 'high', 'low', 'close', 'volume'],
            'output_features': ['direction', 'confidence'],
            'sequence_length': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'validation_split': 0.2
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
            'symbols': 'BTCUSDT',
            'timeframe': 'invalid_timeframe'
        }
    }

@pytest.fixture
def invalid_timeframe_config():
    """Create a configuration with invalid timeframe."""
    config = valid_config()
    config['trading']['interval'] = 'invalid'
    return config

@pytest.fixture
def invalid_risk_config():
    """Create a configuration with invalid risk parameters."""
    config = valid_config()
    config['risk']['max_drawdown_percent'] = -10.0  # Invalid negative value
    return config

@pytest.fixture
def invalid_model_config():
    """Create a configuration with invalid model parameters."""
    config = valid_config()
    config['model']['learning_rate'] = -0.001  # Invalid negative value
    return config

def test_valid_configuration(valid_config):
    """Test validation of valid configuration."""
    config = Config(**valid_config)
    assert isinstance(config, Config)
    assert config.exchange.name == 'binance'
    assert config.trading.symbols == ['BTC/USDT', 'ETH/USDT']
    assert config.trading.interval == '1h'

def test_invalid_configuration(invalid_config):
    """Test validation of invalid configuration."""
    with pytest.raises(ConfigError) as exc_info:
        Config(**invalid_config)
    assert "Invalid configuration" in str(exc_info.value)

def test_missing_required_fields():
    """Test validation of missing required fields."""
    config = {
        'exchange': {
            'name': 'binance'
        }
    }
    with pytest.raises(ConfigError) as exc_info:
        Config(**config)
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
        Config(**config)
    assert "Invalid exchange name" in str(exc_info.value)

def test_invalid_timeframe(valid_config):
    """Test that invalid timeframe raises error."""
    config = valid_config.copy()
    config['trading']['interval'] = 'invalid'
    with pytest.raises(ConfigError) as exc_info:
        Config(**config)
    assert "Invalid timeframe" in str(exc_info.value)

def test_invalid_risk_parameters(valid_config):
    """Test that invalid risk parameters raise error."""
    config = valid_config.copy()
    config['risk']['max_drawdown_percent'] = -10.0  # Invalid negative value
    with pytest.raises(ConfigError) as exc_info:
        Config(**config)
    assert "Input should be greater than 0" in str(exc_info.value)

def test_invalid_model_parameters(valid_config):
    """Test that invalid model parameters raise error."""
    config = valid_config.copy()
    config['model']['learning_rate'] = -0.001  # Invalid negative value
    with pytest.raises(ConfigError) as exc_info:
        Config(**config)
    assert "Input should be greater than 0" in str(exc_info.value) 