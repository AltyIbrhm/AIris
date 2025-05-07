"""
Tests for market data components.
"""
import pytest
from datetime import datetime

from market_data.mock import MockMarketDataFetcher

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "symbol": "BTC/USD",
        "interval": "1m",
        "base_price": 50000.0,
        "volatility": 0.02,
        "trend": 0.0,
        "volume_range": (1.0, 10.0)
    }

@pytest.fixture
def mock_fetcher(mock_config):
    """Create a mock market data fetcher for testing."""
    return MockMarketDataFetcher(mock_config)

@pytest.mark.asyncio
async def test_fetch_latest(mock_fetcher):
    """Test fetching latest market data."""
    data = await mock_fetcher.fetch_latest()
    
    # Check data structure
    assert isinstance(data, dict)
    assert "timestamp" in data
    assert "symbol" in data
    assert "interval" in data
    assert "open" in data
    assert "high" in data
    assert "low" in data
    assert "close" in data
    assert "volume" in data
    
    # Check data types
    assert isinstance(data["timestamp"], (int, float))
    assert isinstance(data["symbol"], str)
    assert isinstance(data["interval"], str)
    assert isinstance(data["open"], float)
    assert isinstance(data["high"], float)
    assert isinstance(data["low"], float)
    assert isinstance(data["close"], float)
    assert isinstance(data["volume"], float)
    
    # Check data validity
    assert data["symbol"] == "BTC/USD"
    assert data["interval"] == "1m"
    assert data["high"] >= data["low"]
    assert data["high"] >= data["open"]
    assert data["high"] >= data["close"]
    assert data["low"] <= data["open"]
    assert data["low"] <= data["close"]
    assert data["volume"] >= 0

@pytest.mark.asyncio
async def test_fetch_data(mock_fetcher):
    """Test fetching historical market data."""
    data = await mock_fetcher.fetch_data("BTC/USD", "1h")
    
    # Check data structure
    assert isinstance(data, dict)
    assert "symbol" in data
    assert "interval" in data
    assert "candles" in data
    
    # Check candles
    candles = data["candles"]
    assert isinstance(candles, list)
    assert len(candles) == 100
    
    # Check first candle
    first_candle = candles[0]
    assert isinstance(first_candle, dict)
    assert "timestamp" in first_candle
    assert "open" in first_candle
    assert "high" in first_candle
    assert "low" in first_candle
    assert "close" in first_candle
    assert "volume" in first_candle

def test_reset(mock_fetcher):
    """Test resetting the mock data generator."""
    # Generate some data
    mock_fetcher.current_price = 60000.0
    mock_fetcher.last_candle = {"close": 60000.0}
    
    # Reset
    mock_fetcher.reset()
    
    # Check reset state
    assert mock_fetcher.current_price == 50000.0
    assert mock_fetcher.last_candle == {}

@pytest.mark.asyncio
async def test_trend_influence(mock_config):
    """Test that trend parameter influences price movement."""
    # Create fetcher with strong uptrend
    mock_config["trend"] = 1.0
    uptrend_fetcher = MockMarketDataFetcher(mock_config)
    
    # Create fetcher with strong downtrend
    mock_config["trend"] = -1.0
    downtrend_fetcher = MockMarketDataFetcher(mock_config)
    
    # Fetch multiple candles
    uptrend_prices = []
    downtrend_prices = []
    
    for _ in range(10):
        uptrend_data = await uptrend_fetcher.fetch_latest()
        downtrend_data = await downtrend_fetcher.fetch_latest()
        uptrend_prices.append(uptrend_data["close"])
        downtrend_prices.append(downtrend_data["close"])
    
    # Check that uptrend prices generally increase
    uptrend_increases = sum(1 for i in range(1, len(uptrend_prices))
                          if uptrend_prices[i] > uptrend_prices[i-1])
    assert uptrend_increases > len(uptrend_prices) / 2
    
    # Check that downtrend prices generally decrease
    downtrend_decreases = sum(1 for i in range(1, len(downtrend_prices))
                            if downtrend_prices[i] < downtrend_prices[i-1])
    assert downtrend_decreases > len(downtrend_prices) / 2 