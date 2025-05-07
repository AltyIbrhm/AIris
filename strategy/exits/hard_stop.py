from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

class HardStop:
    def __init__(
        self,
        stop_loss_atr_mult: float = 2.0,  # Stop loss at 2 × ATR from entry
        atr_period: int = 14,
        max_loss_pct: float = 0.01,  # Maximum loss of 1% of balance
        max_holding_time: int = 72,  # Maximum holding time in hours
        balance_override: Optional[float] = None  # Optional balance for max loss calculation
    ):
        """
        Initialize Hard Stop exit strategy
        
        Args:
            stop_loss_atr_mult (float): Multiple of ATR for stop loss distance (default: 2.0)
            atr_period (int): Period for ATR calculation (default: 14)
            max_loss_pct (float): Maximum loss percentage of balance (default: 1%)
            max_holding_time (int): Maximum holding time in hours (default: 72)
            balance_override (Optional[float]): Optional balance for max loss calculation
        """
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.atr_period = atr_period
        self.max_loss_pct = max_loss_pct
        self.max_holding_time = max_holding_time
        self.balance_override = balance_override
        
        # State variables
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.position_side = ""
        self.entry_time = None
        
    def calculate_atr(self, high_prices: List[float], low_prices: List[float], close_prices: List[float]) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high_prices (List[float]): List of high prices
            low_prices (List[float]): List of low prices
            close_prices (List[float]): List of close prices
            
        Returns:
            float: ATR value
        """
        if len(high_prices) < 2:
            return 0.0
            
        # Convert to numpy arrays
        high_arr = np.array(high_prices)
        low_arr = np.array(low_prices)
        close_arr = np.array(close_prices)
            
        # Calculate True Range
        prev_close = close_arr[:-1]
        curr_high = high_arr[1:]
        curr_low = low_arr[1:]
        
        tr1 = np.abs(curr_high - curr_low)
        tr2 = np.abs(curr_high - prev_close)
        tr3 = np.abs(curr_low - prev_close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR using simple moving average
        atr = np.mean(tr[-self.atr_period:])
        return float(atr)
        
    def initialize_trade(
        self,
        entry_price: float,
        side: str,
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float]
    ) -> None:
        """
        Initialize a new trade and set stop loss
        
        Args:
            entry_price (float): Entry price of the trade
            side (str): Position side ("BUY" or "SELL")
            high_prices (List[float]): List of high prices for ATR
            low_prices (List[float]): List of low prices for ATR
            close_prices (List[float]): List of close prices for ATR
        """
        self.entry_price = entry_price
        self.position_side = side
        self.entry_time = datetime.now()
        
        # Calculate ATR-based stop loss distance
        atr = self.calculate_atr(high_prices, low_prices, close_prices)
        stop_distance = self.stop_loss_atr_mult * atr
        
        # Set stop loss price based on position side
        if side == "BUY":
            self.stop_loss_price = entry_price - stop_distance
        else:  # SELL
            self.stop_loss_price = entry_price + stop_distance
            
        # Adjust stop loss if it exceeds max loss percentage of balance
        if self.balance_override:
            max_loss_amount = self.balance_override * self.max_loss_pct
            max_loss_price = (
                entry_price * (1 - self.max_loss_pct) if side == "BUY"
                else entry_price * (1 + self.max_loss_pct)
            )
            # Use the more conservative stop loss
            if side == "BUY":
                self.stop_loss_price = max(self.stop_loss_price, max_loss_price)
            else:
                self.stop_loss_price = min(self.stop_loss_price, max_loss_price)
        
    def check_time_exit(self) -> bool:
        """
        Check if position should be exited based on holding time
        
        Returns:
            bool: True if time-based exit is triggered
        """
        if not self.entry_time:
            return False
            
        holding_time = datetime.now() - self.entry_time
        return holding_time > timedelta(hours=self.max_holding_time)
        
    def check_exit(self, current_price: float) -> Tuple[bool, float, str]:
        """
        Check if position should be exited
        
        Args:
            current_price (float): Current market price
            
        Returns:
            Tuple[bool, float, str]: (exit_signal, exit_price, exit_reason)
        """
        if not self.position_side:
            return False, 0.0, ""
            
        # Check time-based exit first
        if self.check_time_exit():
            return True, current_price, "time_exit"
            
        # Check stop loss hit
        if self.position_side == "BUY":
            if current_price <= self.stop_loss_price:
                return True, self.stop_loss_price, "stop_loss"
        else:  # SELL
            if current_price >= self.stop_loss_price:
                return True, self.stop_loss_price, "stop_loss"
                
        return False, self.stop_loss_price, ""
        
    def __str__(self) -> str:
        return (f"HardStop(stop_mult={self.stop_loss_atr_mult}, "
                f"atr_period={self.atr_period}, "
                f"max_loss={self.max_loss_pct}, "
                f"max_time={self.max_holding_time}h)") 