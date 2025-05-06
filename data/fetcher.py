"""
Market data fetcher for BinanceUS or other data APIs.
Handles data retrieval and basic error handling.
"""
from typing import Dict, Any, List
from core.interfaces import MarketDataFetcher
import requests
import time

class BinanceUSFetcher(MarketDataFetcher):
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        """Initialize the BinanceUS data fetcher."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def fetch_data(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Fetch market data for a given symbol and interval."""
        endpoint = f"/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 100
        }
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return {} 