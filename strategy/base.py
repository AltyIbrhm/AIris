"""
Base class for all trading strategies.
Defines the interface for signal generation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.interfaces import Strategy

class BaseStrategy(Strategy):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base strategy with configuration."""
        self.config = config

    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary containing signal information
        """
        pass

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate the generated signal."""
        required_fields = ['action', 'price', 'timestamp']
        return all(field in signal for field in required_fields) 