"""
Portfolio tracking and position management utilities.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
import os

class PortfolioTracker:
    def __init__(self, initial_capital: float = 10000.0):
        """Initialize portfolio tracker with initial capital."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.closed_positions: List[Dict[str, Any]] = []
        self.daily_pnl: Dict[str, float] = {}
        self.peak_capital = initial_capital
        self.logger = logging.getLogger(__name__)

    def open_position(self, symbol: str, entry_price: float, size: float, 
                     direction: str, timestamp: datetime) -> bool:
        """Open a new position."""
        if symbol in self.open_positions:
            self.logger.warning(f"Position already open for {symbol}")
            return False

        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'size': size,
            'direction': direction,
            'entry_time': timestamp,
            'current_price': entry_price,
            'unrealized_pnl': 0.0
        }
        
        self.open_positions[symbol] = position
        self.logger.info(f"Opened {direction} position for {symbol} at {entry_price}")
        return True

    def close_position(self, symbol: str, exit_price: float, 
                      timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Close an existing position."""
        if symbol not in self.open_positions:
            self.logger.warning(f"No open position found for {symbol}")
            return None

        position = self.open_positions.pop(symbol)
        pnl = self._calculate_pnl(position, exit_price)
        
        closed_position = {
            **position,
            'exit_price': exit_price,
            'exit_time': timestamp,
            'realized_pnl': pnl
        }
        
        self.closed_positions.append(closed_position)
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Update daily PnL
        date_str = timestamp.strftime('%Y-%m-%d')
        self.daily_pnl[date_str] = self.daily_pnl.get(date_str, 0.0) + pnl
        
        self.logger.info(f"Closed position for {symbol} at {exit_price}, PnL: {pnl:.2f}")
        return closed_position

    def update_position(self, symbol: str, current_price: float) -> None:
        """Update position with current market price."""
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        position['current_price'] = current_price
        position['unrealized_pnl'] = self._calculate_pnl(position, current_price)

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position details for a symbol."""
        return self.open_positions.get(symbol)

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all open positions."""
        return self.open_positions

    def get_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_capital == 0:
            return 0.0
        return ((self.peak_capital - self.current_capital) / self.peak_capital) * 100

    def get_daily_pnl(self, date: Optional[str] = None) -> float:
        """Get PnL for a specific date or today."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        return self.daily_pnl.get(date, 0.0)

    def get_total_pnl(self) -> float:
        """Calculate total realized PnL."""
        return sum(pos['realized_pnl'] for pos in self.closed_positions)

    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL."""
        return sum(pos['unrealized_pnl'] for pos in self.open_positions.values())

    def _calculate_pnl(self, position: Dict[str, Any], current_price: float) -> float:
        """Calculate PnL for a position."""
        if position['direction'] == 'long':
            return (current_price - position['entry_price']) * position['size']
        else:  # short
            return (position['entry_price'] - current_price) * position['size']

    def save_state(self, filepath: str) -> None:
        """Save portfolio state to file."""
        state = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'open_positions': self.open_positions,
            'closed_positions': self.closed_positions,
            'daily_pnl': self.daily_pnl
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, default=str)

    def load_state(self, filepath: str) -> None:
        """Load portfolio state from file."""
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.initial_capital = state['initial_capital']
        self.current_capital = state['current_capital']
        self.peak_capital = state['peak_capital']
        self.open_positions = state['open_positions']
        self.closed_positions = state['closed_positions']
        self.daily_pnl = state['daily_pnl'] 