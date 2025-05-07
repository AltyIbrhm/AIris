"""
Portfolio tracking and position management.
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Position data structure."""
    symbol: str
    entry_price: float
    size: float
    direction: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0

class PortfolioTracker:
    """Tracks positions and portfolio performance across multiple symbols."""
    
    def __init__(self, initial_capital: float = 10000.0):
        """Initialize portfolio tracker."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, List[Position]] = {}  # symbol -> list of positions
        self.closed_positions: Dict[str, List[Position]] = {}  # symbol -> list of closed positions
        self.daily_pnl: Dict[str, float] = {}  # symbol -> daily PnL
        self.total_pnl: Dict[str, float] = {}  # symbol -> total PnL
        self.max_drawdown: Dict[str, float] = {}  # symbol -> max drawdown
        self.peak_capital: Dict[str, float] = {}  # symbol -> peak portfolio value
        self.logger = logging.getLogger(__name__)
        
    def open_position(self, symbol: str, entry_price: float, size: float, direction: str,
                     stop_loss: float = None, take_profit: float = None) -> Position:
        """Open a new position."""
        if symbol not in self.positions:
            self.positions[symbol] = []
            self.closed_positions[symbol] = []
            self.daily_pnl[symbol] = 0.0
            self.total_pnl[symbol] = 0.0
            self.max_drawdown[symbol] = 0.0
            self.peak_capital[symbol] = self.current_capital
            
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            size=size,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now()
        )
        
        self.positions[symbol].append(position)
        logger.info(f"Opened {direction} position for {symbol} at {entry_price}")
        return position
        
    def close_position(self, symbol: str, position: Position, exit_price: float) -> None:
        """Close an existing position."""
        if symbol not in self.positions or position not in self.positions[symbol]:
            self.logger.warning(f"Position not found for {symbol}")
            return
            
        position.exit_time = datetime.now()
        position.exit_price = exit_price
        
        # Calculate PnL
        if position.direction == 'long':
            position.pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            position.pnl = (position.entry_price - exit_price) * position.size
        
        position.pnl_percent = (position.pnl / (position.entry_price * position.size)) * 100
        
        # Update portfolio metrics
        self.daily_pnl[symbol] += position.pnl
        self.total_pnl[symbol] += position.pnl
        self.current_capital += position.pnl
        
        # Update drawdown
        if self.current_capital > self.peak_capital[symbol]:
            self.peak_capital[symbol] = self.current_capital
        
        drawdown = ((self.peak_capital[symbol] - self.current_capital) / 
                   self.peak_capital[symbol]) * 100
        if drawdown > self.max_drawdown[symbol]:
            self.max_drawdown[symbol] = drawdown
            self.logger.warning(f"New max drawdown for {symbol}: {drawdown:.2f}%")
        
        # Move position to closed positions
        self.positions[symbol].remove(position)
        self.closed_positions[symbol].append(position)
        
        self.logger.info(f"Closed {position.direction} position for {symbol} at {exit_price}, PnL: {position.pnl:.2f}, Drawdown: {drawdown:.2f}%")
        
    def update_position(self, symbol: str, position: Position, current_price: float) -> None:
        """Update position with current market price."""
        if symbol not in self.positions or position not in self.positions[symbol]:
            raise ValueError(f"Position not found for {symbol}")
            
        position.pnl = self._calculate_pnl(position)
        position.pnl_percent = (position.pnl / (position.entry_price * position.size)) * 100
        
        # Update drawdown tracking
        self._update_drawdown(symbol)
        
    def get_position(self, symbol: str, position_id: int) -> Optional[Position]:
        """Get a specific position by ID."""
        if symbol in self.positions and 0 <= position_id < len(self.positions[symbol]):
            return self.positions[symbol][position_id]
        return None
        
    def get_open_positions(self, symbol: str = None) -> Dict[str, List[Position]]:
        """Get all open positions or positions for a specific symbol."""
        if symbol:
            return {symbol: self.positions.get(symbol, [])}
        return self.positions
        
    def get_closed_positions(self, symbol: str = None) -> Dict[str, List[Position]]:
        """Get all closed positions or positions for a specific symbol."""
        if symbol:
            return {symbol: self.closed_positions.get(symbol, [])}
        return self.closed_positions
        
    def get_drawdown(self, symbol: str = None) -> Dict[str, float]:
        """Get current drawdown for a symbol or maximum across all symbols."""
        if symbol:
            return {symbol: self.max_drawdown.get(symbol, 0.0)}
        return self.max_drawdown
        
    def get_daily_pnl(self, symbol: str = None) -> float:
        """Get daily PnL for a symbol or total."""
        if symbol:
            return self.daily_pnl.get(symbol, 0.0)
        return sum(self.daily_pnl.values())
        
    def get_total_pnl(self, symbol: str = None) -> float:
        """Get total PnL for a symbol or total."""
        if symbol:
            return self.total_pnl.get(symbol, 0.0)
        return sum(self.total_pnl.values())
        
    def _calculate_pnl(self, position: Position) -> float:
        """Calculate PnL for a position."""
        if not position.exit_price:
            return 0.0
            
        if position.direction == 'long':
            return (position.exit_price - position.entry_price) * position.size
        else:  # short
            return (position.entry_price - position.exit_price) * position.size
            
    def _update_drawdown(self, symbol: str) -> None:
        """Update drawdown tracking for a symbol."""
        current_value = self.current_capital
        if current_value > self.peak_capital[symbol]:
            self.peak_capital[symbol] = current_value
            
        drawdown = ((self.peak_capital[symbol] - current_value) / self.peak_capital[symbol]) * 100
        self.max_drawdown[symbol] = max(self.max_drawdown[symbol], drawdown)
        
    def save_state(self, filepath: str) -> None:
        """Save portfolio state to file."""
        state = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'positions': {symbol: [asdict(p) for p in positions] 
                         for symbol, positions in self.positions.items()},
            'closed_positions': {symbol: [asdict(p) for p in positions] 
                               for symbol, positions in self.closed_positions.items()},
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'peak_capital': self.peak_capital
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
    def load_state(self, filepath: str) -> None:
        """Load portfolio state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                self.initial_capital = state['initial_capital']
                self.current_capital = state['current_capital']
                self.positions = {
                    symbol: [Position(**p) for p in positions]
                    for symbol, positions in state['positions'].items()
                }
                self.closed_positions = {
                    symbol: [Position(**p) for p in positions]
                    for symbol, positions in state['closed_positions'].items()
                }
                self.daily_pnl = state['daily_pnl']
                self.total_pnl = state['total_pnl']
                self.max_drawdown = state['max_drawdown']
                self.peak_capital = state['peak_capital']
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {str(e)}")

    def reset_daily_metrics(self) -> None:
        """Reset daily metrics."""
        for symbol in self.daily_pnl:
            self.daily_pnl[symbol] = 0.0
        
    def get_current_drawdown(self, symbol: str = None) -> float:
        """Get current drawdown for a symbol or maximum across all symbols."""
        if symbol:
            if symbol not in self.peak_capital:
                return 0.0
            peak = self.peak_capital[symbol]
            current = self.current_capital
            return ((peak - current) / peak) * 100 if peak > 0 else 0.0
        
        # Calculate maximum current drawdown across all symbols
        max_current_dd = 0.0
        for sym in self.peak_capital:
            peak = self.peak_capital[sym]
            current = self.current_capital
            if peak > 0:
                dd = ((peak - current) / peak) * 100
                max_current_dd = max(max_current_dd, dd)
        return max_current_dd 