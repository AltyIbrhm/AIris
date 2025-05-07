from typing import List
import time

class BaseFilter:
    def __init__(self, min_trade_spacing: int = 5):
        """
        Initialize base filter with trade spacing logic
        
        Args:
            min_trade_spacing (int): Minimum number of periods between trades
        """
        self.min_trade_spacing = min_trade_spacing
        self.last_trade_index = -min_trade_spacing  # Initialize to allow first trade
        
    def check_trade_spacing(self, current_index: int) -> bool:
        """
        Check if enough time has passed since last trade
        
        Args:
            current_index (int): Current period index
            
        Returns:
            bool: True if spacing requirement is met, False otherwise
        """
        spacing = current_index - self.last_trade_index
        if spacing >= self.min_trade_spacing:
            self.last_trade_index = current_index
            return True
        return False 