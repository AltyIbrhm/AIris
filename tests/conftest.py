import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    }, index=dates)
    return data

@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        'trading': {
            'symbol': 'BTC/USD',
            'timeframe': '1h',
            'max_position_size': 1.0
        },
        'risk': {
            'max_drawdown': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.04
        },
        'strategy': {
            'ema_fast': 12,
            'ema_slow': 26
        }
    }

@pytest.fixture
def mock_strategy_signal():
    """Generate a mock strategy signal."""
    return {
        'timestamp': datetime.now(),
        'symbol': 'BTC/USD',
        'signal': 1,  # 1 for buy, -1 for sell, 0 for hold
        'confidence': 0.85,
        'price': 50000.0
    }

@pytest.fixture
def mock_position():
    """Create a mock trading position."""
    return {
        'symbol': 'BTC/USD',
        'side': 'long',
        'entry_price': 50000.0,
        'size': 0.1,
        'stop_loss': 49000.0,
        'take_profit': 52000.0,
        'entry_time': datetime.now()
    } 