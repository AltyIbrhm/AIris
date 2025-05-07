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
        self.min_confidence = config.get('min_confidence', 0.3)  # Minimum confidence required for a signal
        self.conflict_threshold = config.get('conflict_threshold', 0.2)  # Threshold for detecting conflicting signals
        self.logger = logging.getLogger(__name__)

    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Add a strategy to the combiner with its weight."""
        self.strategies.append(strategy)
        self.strategy_weights.append(weight)

    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine signals from all strategies into a single signal."""
        if not self.strategies:
            return self._create_hold_signal(market_data, error='No strategies configured')

        # Initialize signal collection
        buy_signals = []
        sell_signals = []
        total_weight = 0.0
        errors = []

        # Collect signals and check for errors
        for strategy, weight in zip(self.strategies, self.strategy_weights):
            try:
                signal = strategy.generate_signal(market_data)
                
                # Check for errors
                if 'error' in signal:
                    errors.append(signal['error'])
                    continue
                    
                confidence = signal.get('confidence', 1.0)
                
                # Skip signals with low confidence
                if confidence < self.min_confidence:
                    self.logger.debug(f"Skipping low confidence signal: {confidence} < {self.min_confidence}")
                    continue
                    
                if signal['action'] == 'buy':
                    buy_signals.append((weight, confidence))
                    self.logger.debug(f"Added buy signal: weight={weight}, confidence={confidence}")
                elif signal['action'] == 'sell':
                    sell_signals.append((weight, confidence))
                    self.logger.debug(f"Added sell signal: weight={weight}, confidence={confidence}")
                
                total_weight += weight
            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"Error getting signal from strategy: {str(e)}", exc_info=True)
                continue

        if errors:
            return self._create_hold_signal(market_data, error='; '.join(errors))

        if total_weight == 0:
            return self._create_hold_signal(market_data, error='Total weight is zero')

        # Calculate weighted signals
        buy_weight = sum(w for w, _ in buy_signals)
        sell_weight = sum(w for w, _ in sell_signals)
        buy_signal = sum(w * c for w, c in buy_signals)
        sell_signal = sum(w * c for w, c in sell_signals)

        self.logger.debug(f"Buy weight: {buy_weight}, Sell weight: {sell_weight}")
        self.logger.debug(f"Buy signal: {buy_signal}, Sell signal: {sell_signal}")

        # Enhanced edge case handling
        action, confidence = self._resolve_signal_conflict(
            buy_weight, sell_weight,
            buy_signal, sell_signal,
            total_weight
        )

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
                'total_weight': total_weight,
                'conflict_threshold': self.conflict_threshold,
                'min_confidence': self.min_confidence
            }
        }

    def _resolve_signal_conflict(
        self,
        buy_weight: float,
        sell_weight: float,
        buy_signal: float,
        sell_signal: float,
        total_weight: float
    ) -> tuple[str, float]:
        """
        Resolve conflicts between buy and sell signals.
        
        Args:
            buy_weight: Total weight of buy signals
            sell_weight: Total weight of sell signals
            buy_signal: Weighted sum of buy signals
            sell_signal: Weighted sum of sell signals
            total_weight: Total weight of all signals
            
        Returns:
            Tuple of (action, confidence)
        """
        # Case 1: Equal weights - use signal strength comparison
        if abs(buy_weight - sell_weight) < 1e-6:
            self.logger.debug("Equal weights detected")
            signal_diff = abs(buy_signal - sell_signal)
            
            # If signals are too close, hold
            if signal_diff < self.hold_threshold * total_weight:
                return 'hold', 0.0
                
            # If signals are conflicting but clear winner
            if signal_diff < self.conflict_threshold * total_weight:
                self.logger.warning("Conflicting signals detected, but clear winner")
                
            action = 'buy' if buy_signal > sell_signal else 'sell'
            confidence = signal_diff / total_weight
            return action, confidence

        # Case 2: Unequal weights but potential conflict
        stronger_weight = max(buy_weight, sell_weight)
        weaker_weight = min(buy_weight, sell_weight)
        weight_ratio = weaker_weight / stronger_weight
        
        # If weights are close enough, check for conflicts
        if weight_ratio > (1 - self.conflict_threshold):
            stronger_signal = max(buy_signal, sell_signal)
            weaker_signal = min(buy_signal, sell_signal)
            signal_ratio = weaker_signal / stronger_signal if stronger_signal > 0 else 0
            
            # If signals are too close relative to their weights, hold
            if signal_ratio > (1 - self.conflict_threshold):
                self.logger.warning("Conflicting signals with similar strength detected")
                return 'hold', 0.0

        # Case 3: Clear winner based on weight
        if buy_weight > sell_weight:
            return 'buy', buy_signal / total_weight
        else:
            return 'sell', sell_signal / total_weight

    def _create_hold_signal(self, market_data: Dict[str, Any], error: str = None) -> Dict[str, Any]:
        """Create a hold signal with optional error message."""
        signal = {
            'action': 'hold',
            'price': market_data['close'][-1],
            'timestamp': market_data['timestamp'][-1],
            'confidence': 0.0
        }
        if error:
            signal['error'] = error
        return signal 