import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Optional
from ta.volatility import AverageTrueRange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
LABEL_MAP = {
    1: 'BUY',
    0: 'HOLD',
    -1: 'SELL'
}

def calculate_future_returns(df: pd.DataFrame, future_bars: int) -> pd.Series:
    """
    Calculate future returns for each timestep.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        future_bars (int): Number of bars to look ahead
        
    Returns:
        pd.Series: Future returns for each timestep
    """
    # Calculate future price
    future_price = df['close'].shift(-future_bars)
    # Calculate future return
    future_return = (future_price - df['close']) / df['close']
    return future_return

def generate_labels(
    df: pd.DataFrame,
    future_bars: int = 10,
    atr_window: int = 14,
    atr_multiplier: float = 0.75
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate trading labels based on future price movement and ATR.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        future_bars (int): Number of bars to look ahead for label generation
        atr_window (int): Window size for ATR calculation
        atr_multiplier (float): Multiplier for ATR threshold
        
    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame with labels and label statistics
    """
    try:
        df = df.copy()
        
        # Calculate ATR
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=atr_window
        ).average_true_range()
        
        # Calculate ATR percentage
        atr_pct = atr / df['close']
        
        # Calculate future returns
        future_return = calculate_future_returns(df, future_bars)
        
        # Generate labels based on ATR thresholds
        df['label'] = 0  # Default to HOLD
        df.loc[future_return > atr_multiplier * atr_pct, 'label'] = 1  # BUY
        df.loc[future_return < -atr_multiplier * atr_pct, 'label'] = -1  # SELL
        
        # Calculate label statistics
        label_stats = {
            'total_samples': len(df),
            'buy_samples': (df['label'] == 1).sum(),
            'sell_samples': (df['label'] == -1).sum(),
            'hold_samples': (df['label'] == 0).sum(),
            'buy_ratio': (df['label'] == 1).mean(),
            'sell_ratio': (df['label'] == -1).mean(),
            'hold_ratio': (df['label'] == 0).mean()
        }
        
        # Add label text for easier interpretation
        df['label_text'] = df['label'].map(LABEL_MAP)
        
        # Drop rows with NaN values (last future_bars rows)
        df = df.dropna()
        
        return df, label_stats
        
    except Exception as e:
        logger.error(f"Error generating labels: {str(e)}")
        raise

def process_all_symbols(
    input_path: str = "ml/data/processed",
    output_path: str = "ml/data/processed",
    future_bars: int = 10,
    atr_window: int = 14,
    atr_multiplier: float = 0.75
) -> None:
    """
    Process all symbol data files and generate labels.
    
    Args:
        input_path (str): Path to input feature files
        output_path (str): Path to save label files
        future_bars (int): Number of bars to look ahead
        atr_window (int): Window size for ATR calculation
        atr_multiplier (float): Multiplier for ATR threshold
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        
        for file in os.listdir(input_path):
            if not file.endswith("_features.csv"):
                continue
                
            symbol = file.split("_")[0]
            logger.info(f"Processing labels for {symbol}...")
            
            # Read feature data
            df = pd.read_csv(os.path.join(input_path, file))
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Generate labels
            df_labeled, label_stats = generate_labels(
                df,
                future_bars=future_bars,
                atr_window=atr_window,
                atr_multiplier=atr_multiplier
            )
            
            # Save labeled data
            output_file = os.path.join(output_path, f"{symbol}_5m_labels.csv")
            df_labeled.to_csv(output_file, index=False)
            
            # Log label statistics
            logger.info(f"[✓] Label statistics for {symbol}:")
            logger.info(f"    Total samples: {label_stats['total_samples']}")
            logger.info(f"    BUY ratio: {label_stats['buy_ratio']:.2%}")
            logger.info(f"    SELL ratio: {label_stats['sell_ratio']:.2%}")
            logger.info(f"    HOLD ratio: {label_stats['hold_ratio']:.2%}")
            logger.info(f"[✓] Saved labels to {output_file}")
            
    except Exception as e:
        logger.error(f"Error processing symbols: {str(e)}")
        raise

if __name__ == "__main__":
    process_all_symbols() 