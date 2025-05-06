"""
Base interface for risk management modules.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.interfaces import RiskManager

class BaseRiskManager(RiskManager):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base risk manager with configuration."""
        self.config = config

    @abstractmethod
    def evaluate_risk(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk for a given position.
        
        Args:
            position: Dictionary containing position information
            
        Returns:
            Dictionary containing risk evaluation results
        """
        pass

    def validate_position(self, position: Dict[str, Any]) -> bool:
        """
        Validate position data.
        
        Args:
            position: Dictionary containing position information
            
        Returns:
            True if position data is valid, False otherwise
        """
        required_fields = ['symbol', 'size', 'entry_price', 'side']
        return all(field in position for field in required_fields) 