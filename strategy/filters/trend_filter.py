from typing import List
import numpy as np

class TrendFilter:
    def __init__(self, fast_period: int = 8, slow_period: int = 21):
        """
        Initialize Trend Filter using EMA crossover
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """
        Calculate EMA for the given price series
        
        Args:
            prices (List[float]): List of closing prices
            period (int): EMA period
            
        Returns:
            float: EMA value
        """
        if len(prices) < period:
            return prices[-1] if prices else 0.0
            
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
            
        return float(ema)
    
    def check_trend(self, prices: List[float]) -> tuple[float, float, str]:
        """
        Calculate EMAs and determine trend direction
        
        Args:
            prices (List[float]): List of closing prices
            
        Returns:
            tuple[float, float, str]: (fast_ema, slow_ema, trend_direction)
        """
        fast_ema = self.calculate_ema(prices, self.fast_period)
        slow_ema = self.calculate_ema(prices, self.slow_period)
        
        trend = "uptrend" if fast_ema > slow_ema else "downtrend"
        return fast_ema, slow_ema, trend
    
    def check_entry(self, fast_ema: float, slow_ema: float, signal: str) -> bool:
        """
        Check if entry is allowed based on trend alignment
        
        Args:
            fast_ema (float): Current fast EMA value
            slow_ema (float): Current slow EMA value
            signal (str): Trading signal ('BUY' or 'SELL')
            
        Returns:
            bool: True if entry is allowed, False otherwise
        """
        if signal == "BUY" and fast_ema > slow_ema:
            return True
        if signal == "SELL" and fast_ema < slow_ema:
            return True
        return False
    
    def __str__(self) -> str:
        return f"TrendFilter(fast_period={self.fast_period}, slow_period={self.slow_period})" 