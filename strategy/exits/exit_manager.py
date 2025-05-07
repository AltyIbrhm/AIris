from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .trailing_stop import TrailingStop
from .hard_stop import HardStop

class ExitManager:
    def __init__(
        self,
        trailing_config: Optional[Dict] = None,
        hard_stop_config: Optional[Dict] = None
    ):
        """
        Initialize Exit Manager with trailing and hard stop configurations
        
        Args:
            trailing_config (Optional[Dict]): Configuration for TrailingStop
                {
                    "activation_threshold": float,  # Activates at X × TP (default: 0.5)
                    "trail_distance_atr_mult": float,  # Trail X × ATR behind price (default: 0.8)
                    "atr_period": int,  # Period for ATR calculation (default: 14)
                    "min_profit_threshold": float,  # Minimum profit to start trailing (default: 0.3%)
                    "max_loss_pct": float,  # Maximum loss percentage (default: 2%)
                    "max_holding_time": int,  # Maximum holding time in hours (default: 48)
                    "position_size_mult": float  # Position size multiplier (default: 1.0)
                }
            hard_stop_config (Optional[Dict]): Configuration for HardStop
                {
                    "stop_loss_atr_mult": float,  # Stop loss at X × ATR (default: 2.0)
                    "atr_period": int,  # Period for ATR calculation (default: 14)
                    "max_loss_pct": float,  # Maximum loss percentage (default: 1%)
                    "max_holding_time": int,  # Maximum holding time in hours (default: 72)
                    "balance_override": Optional[float]  # Optional balance for max loss
                }
        """
        # Use empty dict if config not provided
        trailing_config = trailing_config or {}
        hard_stop_config = hard_stop_config or {}
        
        self.trailing = TrailingStop(**trailing_config)
        self.hard_stop = HardStop(**hard_stop_config)
        
        # State variables
        self.position_side = ""
        self.entry_price = 0.0
        self.entry_time = None
        self.position_size = 0.0
        
    def initialize_trade(
        self,
        entry_price: float,
        side: str,
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        position_size: float = 1.0
    ) -> None:
        """
        Initialize a new trade with both trailing and hard stops
        
        Args:
            entry_price (float): Entry price of the trade
            side (str): Position side ("BUY" or "SELL")
            high_prices (List[float]): List of high prices for ATR
            low_prices (List[float]): List of low prices for ATR
            close_prices (List[float]): List of close prices for ATR
            position_size (float): Size of the position (default: 1.0)
        """
        self.entry_price = entry_price
        self.position_side = side
        self.entry_time = datetime.now()
        self.position_size = position_size
        
        # Initialize both exit strategies
        self.trailing.initialize_trade(
            entry_price=entry_price,
            side=side,
            position_size=position_size
        )
        
        self.hard_stop.initialize_trade(
            entry_price=entry_price,
            side=side,
            high_prices=high_prices,
            low_prices=low_prices,
            close_prices=close_prices
        )
        
    def check_exit(
        self,
        current_price: float,
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        take_profit: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Check all exit conditions and return the first one triggered
        
        Args:
            current_price (float): Current market price
            high_prices (List[float]): List of high prices for ATR
            low_prices (List[float]): List of low prices for ATR
            close_prices (List[float]): List of close prices for ATR
            take_profit (Optional[float]): Take profit level, if any
            
        Returns:
            Tuple[bool, float, str]: (exit_signal, exit_price, exit_reason)
        """
        if not self.position_side:
            return False, 0.0, ""
            
        # Check trailing stop first (highest priority)
        trail_signal, trail_price, trail_reason = self.trailing.update_trail(
            current_price=current_price,
            high_prices=high_prices,
            low_prices=low_prices,
            close_prices=close_prices,
            take_profit=take_profit
        )
        if trail_signal:
            return True, trail_price, trail_reason
            
        # Check hard stop next
        stop_signal, stop_price, stop_reason = self.hard_stop.check_exit(current_price)
        if stop_signal:
            # Use the hard stop price as the exit price, not the current price
            return True, self.hard_stop.stop_loss_price, stop_reason
            
        # Return the current stop levels if no exit
        # Use trailing stop price if active, otherwise hard stop
        if self.trailing.trailing_active:
            return False, trail_price, ""
        return False, self.hard_stop.stop_loss_price, ""
        
    def get_current_stops(self) -> Dict[str, float]:
        """
        Get current stop levels for both exit strategies
        
        Returns:
            Dict[str, float]: Dictionary with current stop levels
                {
                    "trailing_stop": float,  # Current trailing stop price
                    "hard_stop": float  # Current hard stop price
                }
        """
        return {
            "trailing_stop": self.trailing.trail_price,
            "hard_stop": self.hard_stop.stop_loss_price
        }
        
    def __str__(self) -> str:
        return (f"ExitManager(\n"
                f"  {self.trailing}\n"
                f"  {self.hard_stop}\n"
                f")") 