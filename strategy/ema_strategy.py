"""
Sample technical strategy using Exponential Moving Averages (EMA).
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy

class EMAStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the EMA strategy."""
        super().__init__(config)
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)

    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on EMA crossover."""
        df = pd.DataFrame(market_data)
        
        # Calculate EMAs
        df['EMA_fast'] = df['close'].ewm(span=self.fast_period).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.slow_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['EMA_fast'] > df['EMA_slow'], 'signal'] = 1  # Buy signal
        df.loc[df['EMA_fast'] < df['EMA_slow'], 'signal'] = -1  # Sell signal
        
        # Get the latest signal
        latest = df.iloc[-1]
        
        return {
            'action': 'buy' if latest['signal'] == 1 else 'sell' if latest['signal'] == -1 else 'hold',
            'price': latest['close'],
            'timestamp': latest.name,
            'fast_ema': latest['EMA_fast'],
            'slow_ema': latest['EMA_slow']
        } 