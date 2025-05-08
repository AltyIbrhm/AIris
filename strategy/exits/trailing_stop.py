from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

class TrailingStop:
    def __init__(
        self,
        activation_threshold: float = 1.0,  # Activates at 1.0 × TP
        trail_distance_atr_mult: float = 1.5,  # Trail 1.5 × ATR behind price
        atr_period: int = 14,
        min_profit_threshold: float = 0.005,  # Minimum profit to start trailing (0.5%)
        max_loss_pct: float = 0.015,  # Maximum loss of 1.5%
        max_holding_time: int = 48,  # Maximum holding time in hours
        min_holding_time: int = 5,  # Minimum holding time in bars
        position_size_mult: float = 1.0  # Position size multiplier for trail distance
    ):
        """
        Initialize Trailing Stop exit strategy
        
        Args:
            activation_threshold (float): Multiple of take-profit to activate trailing (default: 1.0)
            trail_distance_atr_mult (float): Multiple of ATR to trail behind price (default: 1.5)
            atr_period (int): Period for ATR calculation (default: 14)
            min_profit_threshold (float): Minimum profit required to activate trailing (default: 0.5%)
            max_loss_pct (float): Maximum loss percentage before forced exit (default: 1.5%)
            max_holding_time (int): Maximum holding time in hours (default: 48)
            min_holding_time (int): Minimum holding time in bars (default: 5)
            position_size_mult (float): Position size multiplier for trail distance (default: 1.0)
        """
        self.activation_threshold = activation_threshold
        self.trail_distance_atr_mult = trail_distance_atr_mult
        self.atr_period = atr_period
        self.min_profit_threshold = min_profit_threshold
        self.max_loss_pct = max_loss_pct
        self.max_holding_time = max_holding_time
        self.min_holding_time = min_holding_time
        self.position_size_mult = position_size_mult
        
        # State variables
        self.trailing_active = False
        self.trail_price = 0.0
        self.entry_price = 0.0
        self.position_side = ""
        self.entry_time = None
        self.position_size = 0.0
        self.bars_held = 0  # Track number of bars held
        
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
        
    def initialize_trade(self, entry_price: float, side: str, position_size: float = 1.0) -> None:
        """
        Initialize a new trade
        
        Args:
            entry_price (float): Entry price of the trade
            side (str): Position side ("BUY" or "SELL")
            position_size (float): Size of the position (default: 1.0)
        """
        self.entry_price = entry_price
        self.position_side = side
        self.trailing_active = False
        self.trail_price = entry_price  # Initialize at entry price
        self.entry_time = datetime.now()
        self.position_size = position_size
        
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
        
    def check_max_loss(self, current_price: float) -> bool:
        """
        Check if position should be exited based on maximum loss
        
        Args:
            current_price (float): Current market price
            
        Returns:
            bool: True if max loss exit is triggered
        """
        if self.position_side == "BUY":
            loss_pct = (self.entry_price - current_price) / self.entry_price
        else:  # SELL
            loss_pct = (current_price - self.entry_price) / self.entry_price
            
        return loss_pct >= self.max_loss_pct
        
    def update_trail(
        self,
        current_price: float,
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        take_profit: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Update trailing stop and check for exit signal
        
        Args:
            current_price (float): Current market price
            high_prices (List[float]): List of high prices for ATR
            low_prices (List[float]): List of low prices for ATR
            close_prices (List[float]): List of close prices for ATR
            take_profit (Optional[float]): Take profit level, if any
            
        Returns:
            Tuple[bool, float, str]: (exit_signal, trail_price, exit_reason)
        """
        if not self.position_side:
            return False, 0.0, ""
            
        # Increment bars held
        self.bars_held += 1
            
        # Check time-based exit
        if self.check_time_exit():
            return True, current_price, "time_exit"
            
        # Check maximum loss
        if self.check_max_loss(current_price):
            return True, current_price, "max_loss"
            
        # Calculate ATR for dynamic trail distance
        atr = self.calculate_atr(high_prices, low_prices, close_prices)
        trail_distance = self.trail_distance_atr_mult * atr * self.position_size_mult
        
        # Calculate current profit
        if self.position_side == "BUY":
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if take_profit:
                activation_profit = (take_profit - self.entry_price) * self.activation_threshold / self.entry_price
            else:
                activation_profit = self.min_profit_threshold
        else:  # SELL
            profit_pct = (self.entry_price - current_price) / self.entry_price
            if take_profit:
                activation_profit = (self.entry_price - take_profit) * self.activation_threshold / self.entry_price
            else:
                activation_profit = self.min_profit_threshold
            
        # Check if trailing should be activated
        if not self.trailing_active and profit_pct >= activation_profit:
            self.trailing_active = True
            if self.position_side == "BUY":
                self.trail_price = max(current_price - trail_distance, self.entry_price)
            else:  # SELL
                self.trail_price = min(current_price + trail_distance, self.entry_price)
                
        # Update trail price if active
        if self.trailing_active:
            if self.position_side == "BUY":
                new_trail = current_price - trail_distance
                self.trail_price = max(self.trail_price, new_trail, self.entry_price)
                # Only exit if minimum holding time is met
                if current_price < self.trail_price and self.bars_held >= self.min_holding_time:
                    return True, self.trail_price, "trail_hit"
            else:  # SELL
                new_trail = current_price + trail_distance
                self.trail_price = min(self.trail_price, new_trail, self.entry_price)
                # Only exit if minimum holding time is met
                if current_price > self.trail_price and self.bars_held >= self.min_holding_time:
                    return True, self.trail_price, "trail_hit"
                
        return False, self.trail_price, ""
        
    def __str__(self) -> str:
        return (f"TrailingStop(activation={self.activation_threshold}, "
                f"trail_mult={self.trail_distance_atr_mult}, "
                f"atr_period={self.atr_period}, "
                f"max_loss={self.max_loss_pct}, "
                f"max_time={self.max_holding_time}h)") 