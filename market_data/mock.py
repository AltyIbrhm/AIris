"""
Mock market data fetcher for testing.
"""
import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from market_data.base import BaseMarketDataFetcher

class MockMarketDataFetcher(BaseMarketDataFetcher):
    """Mock market data fetcher that generates simulated market data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock market data fetcher.
        
        Args:
            config: Configuration dictionary with additional mock-specific options:
                - base_price: Starting price for the mock data
                - volatility: Price volatility (0-1)
                - trend: Price trend (-1 to 1, where -1 is downtrend, 1 is uptrend)
                - volume_range: Tuple of (min_volume, max_volume)
        """
        super().__init__(config)
        self.base_price = config.get("base_price", 50000.0)
        self.volatility = config.get("volatility", 0.02)
        self.trend = config.get("trend", 0.0)
        self.volume_range = config.get("volume_range", (1.0, 10.0))
        self.current_price = self.base_price
        self.last_candle: Dict[str, Any] = {}
        
    async def fetch_latest(self) -> Dict[str, Any]:
        """Generate and return the latest mock market data.
        
        Returns:
            Dictionary containing simulated market data
        """
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        # Generate new candle
        candle = self._generate_candle()
        self.last_candle = candle
        return candle
    
    async def fetch_data(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Generate historical mock market data.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            
        Returns:
            Dictionary containing simulated historical market data
        """
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        # Generate a series of candles
        candles = []
        current_price = self.base_price
        
        for _ in range(100):  # Generate 100 historical candles
            candle = self._generate_candle(current_price)
            candles.append(candle)
            current_price = candle["close"]
        
        return {
            "symbol": symbol,
            "interval": interval,
            "candles": candles
        }
    
    def _generate_candle(self, start_price: float = None) -> Dict[str, Any]:
        """Generate a single mock candle.
        
        Args:
            start_price: Starting price for the candle (uses current_price if None)
            
        Returns:
            Dictionary containing candle data
        """
        if start_price is None:
            start_price = self.current_price
        
        # Calculate price movement
        price_change = random.uniform(-self.volatility, self.volatility)
        price_change += self.trend * self.volatility  # Add trend bias
        
        # Generate OHLC prices
        open_price = start_price
        close_price = open_price * (1 + price_change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, self.volatility/2))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, self.volatility/2))
        
        # Generate volume
        volume = random.uniform(*self.volume_range)
        
        # Update current price
        self.current_price = close_price
        
        # Create candle
        candle = {
            "timestamp": datetime.utcnow().timestamp(),
            "symbol": self.symbol,
            "interval": self.interval,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        }
        
        return self._format_candle(candle)
    
    def reset(self):
        """Reset the mock data generator to initial state."""
        self.current_price = self.base_price
        self.last_candle = {} 