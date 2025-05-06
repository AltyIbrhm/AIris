"""
Preprocesses raw market data: formatting, cleaning, and aligning for strategy input.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor."""
        pass

    def process_klines(self, klines: List[List[Any]]) -> pd.DataFrame:
        """Convert raw klines data to a pandas DataFrame with proper formatting."""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame."""
        # Example: Add Simple Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        return df 