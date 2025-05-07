from typing import List
import numpy as np

class RSIFilter:
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize RSI Filter
        
        Args:
            period (int): RSI calculation period
            oversold (float): RSI threshold for oversold condition (buy signal)
            overbought (float): RSI threshold for overbought condition (sell signal)
        """
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def calculate_rsi(self, prices: List[float]) -> float:
        """
        Calculate RSI value for the given price series
        
        Args:
            prices (List[float]): List of closing prices
            
        Returns:
            float: RSI value
        """
        if len(prices) < self.period + 1:
            return 50.0  # Default neutral value if not enough data
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        
        if avg_loss == 0:
            return 100.0
            
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def check_entry(self, rsi_value: float, signal: str) -> bool:
        """
        Check if entry is allowed based on RSI value and signal
        
        Args:
            rsi_value (float): Current RSI value
            signal (str): Trading signal ('BUY' or 'SELL')
            
        Returns:
            bool: True if entry is allowed, False otherwise
        """
        if signal == "BUY" and rsi_value < self.oversold:
            return True
        if signal == "SELL" and rsi_value > self.overbought:
            return True
        return False
    
    def __str__(self) -> str:
        return f"RSIFilter(period={self.period}, oversold={self.oversold}, overbought={self.overbought})" 