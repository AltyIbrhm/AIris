"""
Implements stop-loss and take-profit logic.
"""
from typing import Dict, Any
from .base import BaseRiskManager

class StopLossTakeProfit(BaseRiskManager):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the SL/TP manager."""
        super().__init__(config)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.03)
        self.take_profit_pct = config.get('take_profit_pct', 0.05)

    def evaluate_risk(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate position risk and calculate SL/TP levels."""
        if not self.validate_position(position):
            return {'error': 'Invalid position data'}

        entry_price = position['entry_price']
        side = position['side']

        # Calculate SL/TP levels
        if side == 'long':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # short
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        } 