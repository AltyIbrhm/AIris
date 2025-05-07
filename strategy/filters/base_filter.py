from typing import List
import logging

class BaseFilter:
    def __init__(self, min_trade_spacing: int = 5):
        """
        Initialize base filter with trade spacing logic
        
        Args:
            min_trade_spacing (int): Minimum number of periods between trades
        """
        self.min_trade_spacing = min_trade_spacing
        self.last_trade_index = None  # Initialize to None to allow first trade
        self.logger = logging.getLogger(__name__)
        
    def check_trade_spacing(self, current_index: int) -> bool:
        """
        Check if enough time has passed since last trade
        
        Args:
            current_index (int): Current period index
            
        Returns:
            bool: True if spacing requirement is met, False otherwise
        """
        # First trade is always allowed
        if self.last_trade_index is None:
            return True
            
        # Calculate spacing
        spacing = current_index - self.last_trade_index
        
        # Check if spacing requirement is met (strictly greater than minimum)
        if spacing <= self.min_trade_spacing:  # Changed from < to <= to enforce strict spacing
            self.logger.info(f"Trade spacing check failed: {spacing} <= {self.min_trade_spacing}")
            return False
            
        return True
        
    def update_last_trade_index(self, current_index: int):
        """Update the last trade index."""
        self.last_trade_index = current_index
        self.logger.info(f"Updated last trade index to {current_index}")
        
    def reset(self):
        """Reset filter state."""
        self.last_trade_index = None
        self.logger.info("Base filter reset") 