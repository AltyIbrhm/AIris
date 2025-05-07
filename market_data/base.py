"""
Base market data fetcher implementation.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from core.interfaces import MarketDataFetcher

logger = logging.getLogger("airis")

class BaseMarketDataFetcher(MarketDataFetcher):
    """Base implementation of market data fetcher with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the market data fetcher.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading pair symbol (e.g., "BTC/USD")
                - interval: Candle interval (e.g., "1m", "5m", "1h")
                - max_retries: Maximum number of retries for failed requests
                - retry_delay: Delay between retries in seconds
        """
        self.config = config
        self.symbol = config.get("symbol", "BTC/USD")
        self.interval = config.get("interval", "1m")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)
        self.last_fetch_time: Optional[datetime] = None
        
    async def fetch_latest(self) -> Dict[str, Any]:
        """Fetch the latest market data.
        
        Returns:
            Dictionary containing the latest market data with keys:
                - timestamp: Current timestamp
                - symbol: Trading pair symbol
                - interval: Candle interval
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Trading volume
        """
        raise NotImplementedError("Subclasses must implement fetch_latest()")
    
    async def fetch_data(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Fetch market data for a given symbol and interval.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            
        Returns:
            Dictionary containing market data
        """
        raise NotImplementedError("Subclasses must implement fetch_data()")
    
    def _validate_candle(self, candle: Dict[str, Any]) -> bool:
        """Validate a candle data structure.
        
        Args:
            candle: Candle data to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
        return all(field in candle for field in required_fields)
    
    def _format_candle(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw candle data into a standardized structure.
        
        Args:
            raw_data: Raw candle data from the data source
            
        Returns:
            Formatted candle data
        """
        return {
            "timestamp": raw_data.get("timestamp"),
            "symbol": self.symbol,
            "interval": self.interval,
            "open": float(raw_data.get("open", 0)),
            "high": float(raw_data.get("high", 0)),
            "low": float(raw_data.get("low", 0)),
            "close": float(raw_data.get("close", 0)),
            "volume": float(raw_data.get("volume", 0))
        } 