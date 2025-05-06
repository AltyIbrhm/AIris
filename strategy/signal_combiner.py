"""
Combines outputs from multiple strategies into a single trading signal.
"""
from typing import Dict, Any, List
import logging
from .base import BaseStrategy

class SignalCombiner(BaseStrategy):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the signal combiner with strategy weights."""
        super().__init__(config)
        self.strategies = []
        self.strategy_weights = []  # Store weights alongside strategies
        self.hold_threshold = config.get('hold_threshold', 0.1)  # Threshold for considering signals as conflicting
        self.logger = logging.getLogger(__name__)

    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Add a strategy to the combiner with its weight."""
        self.strategies.append(strategy)
        self.strategy_weights.append(weight)

    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine signals from all strategies into a single signal."""
        if not self.strategies:
            return {
                'action': 'hold',
                'price': market_data['close'][-1],
                'timestamp': market_data['timestamp'][-1],
                'error': 'No strategies configured'
            }

        # Initialize signal collection
        buy_signals = []
        sell_signals = []
        total_weight = 0.0
        errors = []

        # Collect signals and check for errors
        for strategy, weight in zip(self.strategies, self.strategy_weights):
            signal = strategy.generate_signal(market_data)
            
            # Check for errors
            if 'error' in signal:
                errors.append(signal['error'])
                continue
                
            confidence = signal.get('confidence', 1.0)
            
            if signal['action'] == 'buy':
                buy_signals.append((weight, confidence))
                self.logger.debug(f"Added buy signal: weight={weight}, confidence={confidence}")
            elif signal['action'] == 'sell':
                sell_signals.append((weight, confidence))
                self.logger.debug(f"Added sell signal: weight={weight}, confidence={confidence}")
            
            total_weight += weight

        if errors:
            return {
                'action': 'hold',
                'price': market_data['close'][-1],
                'timestamp': market_data['timestamp'][-1],
                'error': '; '.join(errors)
            }

        if total_weight == 0:
            return {
                'action': 'hold',
                'price': market_data['close'][-1],
                'timestamp': market_data['timestamp'][-1],
                'error': 'Total weight is zero'
            }

        # Calculate weighted signals
        buy_weight = sum(w for w, _ in buy_signals)
        sell_weight = sum(w for w, _ in sell_signals)
        buy_signal = sum(w * c for w, c in buy_signals)
        sell_signal = sum(w * c for w, c in sell_signals)

        self.logger.debug(f"Buy weight: {buy_weight}, Sell weight: {sell_weight}")
        self.logger.debug(f"Buy signal: {buy_signal}, Sell signal: {sell_signal}")

        # For equal weights, use the hold threshold
        if abs(buy_weight - sell_weight) < 1e-6:
            self.logger.debug("Equal weights detected")
            if abs(buy_signal - sell_signal) < self.hold_threshold * total_weight:
                action = 'hold'
                confidence = 0.0
            else:
                action = 'buy' if buy_signal > sell_signal else 'sell'
                confidence = abs(buy_signal - sell_signal) / total_weight
        else:
            self.logger.debug("Unequal weights detected")
            # For unequal weights, higher weight wins
            if buy_weight > sell_weight:
                action = 'buy'
                confidence = buy_signal / total_weight
            else:
                action = 'sell'
                confidence = sell_signal / total_weight

        self.logger.debug(f"Final decision: action={action}, confidence={confidence}")

        return {
            'action': action,
            'price': market_data['close'][-1],
            'timestamp': market_data['timestamp'][-1],
            'confidence': confidence,
            'debug': {
                'buy_weight': buy_weight,
                'sell_weight': sell_weight,
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'total_weight': total_weight
            }
        } 