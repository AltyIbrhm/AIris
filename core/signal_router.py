"""
Implements the SignalRouter class for managing and routing trading signals.
"""
from typing import Dict, Any, List
import logging
from core.interfaces import SignalRouter, Strategy
from strategy.signal_combiner import SignalCombiner

class DefaultSignalRouter(SignalRouter):
    """
    Default implementation of SignalRouter that manages multiple strategies
    and combines their signals using SignalCombiner.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the signal router.
        
        Args:
            config: Configuration dictionary containing strategy settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.signal_combiner = SignalCombiner(config)
        self.strategies: Dict[str, Strategy] = {}
        
    async def get_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get trading signals from all active strategies.
        
        Args:
            market_data: Dictionary containing current market data
            
        Returns:
            List of signal dictionaries from active strategies
        """
        if not self.strategies:
            self.logger.warning("No active strategies registered")
            return []
            
        try:
            # Use SignalCombiner to get combined signal
            combined_signal = self.signal_combiner.generate_signal(market_data)
            
            # If there's an error in the combined signal, log and return empty list
            if 'error' in combined_signal:
                self.logger.error(f"Error generating signals: {combined_signal['error']}")
                return []
                
            # Return as a list since interface expects List[Dict[str, Any]]
            return [combined_signal]
            
        except Exception as e:
            self.logger.error(f"Error getting signals: {str(e)}", exc_info=True)
            return []
    
    def add_strategy(self, strategy: Strategy) -> None:
        """
        Add a strategy to the router.
        
        Args:
            strategy: Strategy instance to add
        """
        strategy_name = strategy.__class__.__name__
        if strategy_name in self.strategies:
            self.logger.warning(f"Strategy {strategy_name} already exists, updating")
            
        self.strategies[strategy_name] = strategy
        # Add to signal combiner with default weight 1.0
        self.signal_combiner.add_strategy(strategy, weight=1.0)
        self.logger.info(f"Added strategy: {strategy_name}")
    
    def remove_strategy(self, strategy_name: str) -> None:
        """
        Remove a strategy from the router.
        
        Args:
            strategy_name: Name of the strategy to remove
        """
        if strategy_name not in self.strategies:
            self.logger.warning(f"Strategy {strategy_name} not found")
            return
            
        del self.strategies[strategy_name]
        # Note: SignalCombiner doesn't currently support strategy removal
        # This is a potential enhancement to add
        self.logger.info(f"Removed strategy: {strategy_name}") 