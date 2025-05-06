"""
Handles position sizing, daily limits, and exposure management.
"""
from typing import Dict, Any
from .base import BaseRiskManager
import time

class ExposureManager(BaseRiskManager):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the exposure manager with risk parameters."""
        super().__init__(config)
        self.max_position_size = float(config.get('max_position_size', 100000))  # In USDT
        self.max_daily_trades = int(config.get('max_daily_trades', 10))
        self.max_drawdown = float(config.get('max_drawdown', 0.2))
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.initial_balance = self.max_position_size
        self.last_reset = time.time()

    def validate_position(self, position: Dict[str, Any]) -> bool:
        """Validate position data."""
        required_fields = ['symbol', 'size', 'entry_price', 'side']
        return all(field in position for field in required_fields)

    def evaluate_risk(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate position risk and exposure limits."""
        if not self.validate_position(position):
            return {'error': 'Invalid position data'}

        # Reset daily counters if needed
        self._reset_daily_if_needed()

        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return {
                'allowed': False,
                'reason': 'Daily trade limit reached',
                'daily_trades': self.daily_trades
            }

        # Calculate drawdown
        if self.daily_pnl < 0:
            current_drawdown = abs(self.daily_pnl) / self.initial_balance
            if current_drawdown > self.max_drawdown:
                return {
                    'allowed': False,
                    'reason': 'Maximum drawdown reached',
                    'current_drawdown': current_drawdown
                }

        # Calculate position size in USDT
        position_size_btc = float(position['size'])
        entry_price = float(position['entry_price'])
        position_size_usdt = position_size_btc * entry_price

        # Check position size (in USDT)
        if position_size_usdt > self.max_position_size:
            return {
                'allowed': False,
                'reason': 'Position size exceeds maximum',
                'max_allowed': self.max_position_size,
                'position_size': position_size_usdt
            }

        return {
            'allowed': True,
            'position_size': position_size_usdt,
            'daily_trades': self.daily_trades,
            'current_drawdown': abs(self.daily_pnl) / self.initial_balance if self.daily_pnl < 0 else 0.0
        }

    def _reset_daily_if_needed(self):
        """Reset daily counters if 24 hours have passed."""
        current_time = time.time()
        if current_time - self.last_reset >= 86400:  # 24 hours in seconds
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.initial_balance = self.max_position_size
            self.last_reset = current_time

    def update_pnl(self, pnl: float):
        """Update the daily PnL and trade count."""
        self.daily_pnl += float(pnl)
        self.daily_trades += 1 