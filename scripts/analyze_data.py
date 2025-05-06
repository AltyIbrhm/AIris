"""
Script for analyzing market data using Jupyter or CLI.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from data.preprocessor import DataPreprocessor
from utils.common import calculate_drawdown

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data for analysis."""
    df = pd.read_csv(file_path)
    preprocessor = DataPreprocessor()
    return preprocessor.add_technical_indicators(df)

def analyze_returns(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze trading returns and statistics."""
    returns = data['close'].pct_change()
    
    return {
        'total_return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
        'annualized_return': returns.mean() * 252 * 100,
        'volatility': returns.std() * np.sqrt(252) * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'max_drawdown': calculate_drawdown(data['close'].tolist()) * 100
    }

def plot_price_and_volume(data: pd.DataFrame, save_path: str = None):
    """Plot price and volume charts."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Price chart
    ax1.plot(data.index, data['close'], label='Close Price')
    ax1.plot(data.index, data['SMA_20'], label='20-day SMA')
    ax1.plot(data.index, data['SMA_50'], label='50-day SMA')
    ax1.set_title('Price Chart')
    ax1.legend()
    
    # Volume chart
    ax2.bar(data.index, data['volume'])
    ax2.set_title('Volume')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Example usage
    data = load_and_prepare_data('data/historical_data.csv')
    
    # Analyze returns
    stats = analyze_returns(data)
    print("\nTrading Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Plot charts
    plot_price_and_volume(data, 'analysis/price_volume.png') 